import psycopg2
import os
import json
import time
import re
import sqlparse
from typing import Dict, List, Any, Optional, Tuple
# from dataclasses import dataclass
# from anthropic import Anthropic


class ValidationResult:
    """Result of query validation"""
    is_valid: bool
    reason: str = ""
    sql: str = ""

class DatabaseAgent:
    def __init__(self, model_name: str, task: dict[str, str]):
        """Initialize the database agent.

        Args:
            model_name: Name of the LLM model to use
            task: a dictionary containing the user request, the directory of the files containing column meanings and business logic.
        """
        self.model_name = model_name
        self.task = task

        # Connect to the labor_certification database
        self.conn = psycopg2.connect(
            dbname="labor_certification",
            user=os.getenv("USER"),
            host="localhost",
            port="5432"
        )

        # Load all required resources
        self.schema = self._load_schema()
        self.column_meanings = self._load_column_meanings()
        self.knowledge_base = self._load_knowledge_base()
        self.critical_rules = self._extract_critical_rules()
        
        # Deletion threshold from knowledge base
        self.deletion_threshold = 100

    def handle_request(self) -> dict[str, str]:
        """Handle a user request and generate SQL query.

        Returns:
            Dictionary with action, SQL query, and elapsed time
        """
        start_time = time.time()
        
        try:
            # Intent Analysis + Early Rejection
            intent = self._analyze_intent()
            early_reject = self._check_early_rejection(intent)
            
            if early_reject:
                return self._format_result(
                    ValidationResult(False, early_reject), 
                    start_time
                )
            
            # Retrieve Relevant Context (RAG)
            relevant_context = self._retrieve_relevant_context()
            
            # Generate SQL with Chain-of-Thought
            generation_result = self._generate_sql_with_reasoning(relevant_context)
            
            # Multi-Layer Validation
            validation = self._validate_query(generation_result)
            
            # Format and Return
            return self._format_result(validation, start_time)
            
        except Exception as e:
            # If anything goes wrong, reject with error message
            return {
                "action": "reject",
                "sql": "",
                "elapsed_seconds": round(time.time() - start_time, 2)
            }

    def _load_schema(self) -> str:
        """Load database schema from the schema file"""
        schema_path = os.path.join(
            os.path.dirname(self.task.get('column_meaning', '')),
            '../labor_certification_applications_schema.txt'
        )
        
        # Try alternative paths
        possible_paths = [
            schema_path,
            'data/labor_certification_applications/labor_certification_applications_schema.txt',
            '/mnt/user-data/uploads/data/labor_certification_applications/labor_certification_applications_schema.txt'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return f.read()
        
        # Return minimal schema if file not found
        return "Schema file not found"

    def _load_column_meanings(self) -> Dict:
        """Load column meanings from JSON file"""
        column_meaning_path = self.task.get('column_meaning', '')
        
        try:
            with open(column_meaning_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def _load_knowledge_base(self) -> List[Dict]:
        """Load knowledge base from JSONL file"""
        kb_path = self.task.get('knowledge_base', '')
        
        try:
            knowledge_base = []
            with open(kb_path, 'r') as f:
                for line in f:
                    if line.strip():
                        knowledge_base.append(json.loads(line))
            return knowledge_base
        except Exception:
            return []

    def _extract_critical_rules(self) -> List[Dict]:
        """Extract critical policy and safety rules from knowledge base"""
        critical_rule_ids = [62, 64, 65]  # Attorney privacy, deletion threshold, cascade delete
        critical_rules = []
        
        for rule in self.knowledge_base:
            if rule.get('id') in critical_rule_ids:
                critical_rules.append(rule)
            elif rule.get('type') == 'policy_knowledge':
                critical_rules.append(rule)
        
        return critical_rules

    def _analyze_intent(self) -> Dict[str, Any]:
        """Analyze the user's request to understand intent"""
        request_lower = self.task['request'].lower()
        
        intent = {
            'operation': 'query',  # query, delete, update, insert
            'is_bulk': False,
            'involves_deletion': False,
            'involves_attorney_data': False,
            'keywords': []
        }
        
        # Detect operation type
        if any(word in request_lower for word in ['delete', 'remove', 'lay off', 'fire']):
            intent['operation'] = 'delete'
            intent['involves_deletion'] = True
        elif any(word in request_lower for word in ['update', 'modify', 'change']):
            intent['operation'] = 'update'
        elif any(word in request_lower for word in ['insert', 'add', 'create']):
            intent['operation'] = 'insert'
        
        # Detect bulk operations
        if any(word in request_lower for word in ['all', 'every', 'many', 'multiple', 'attorneys', 'employees']):
            intent['is_bulk'] = True
        
        # Detect attorney data access
        if 'attorney' in request_lower:
            intent['involves_attorney_data'] = True
        
        # Extract keywords
        keywords = re.findall(r'\b[a-z]{3,}\b', request_lower)
        intent['keywords'] = list(set(keywords))
        
        return intent

    def _check_early_rejection(self, intent: Dict[str, Any]) -> Optional[str]:
        """Check for obvious rule violations before generating SQL"""
        request = self.task['request']
        
        # Check for bulk deletion
        if intent['involves_deletion'] and intent['is_bulk']:
            # Use LLM to estimate impact
            estimation = self._estimate_deletion_impact_llm(request)
            
            if estimation.get('likely_exceeds_threshold', False):
                return f"Request likely exceeds 100 row deletion threshold. Estimated impact: {estimation.get('reasoning', 'bulk operation detected')}"
        
        # Check for attorney personal info requests
        if intent['involves_attorney_data']:
            if any(word in request.lower() for word in ['address', 'phone', 'contact', 'personal']):
                # Might be a privacy violation, validation layer will handle it
                pass
        
        return None

    def _estimate_deletion_impact_llm(self, request: str) -> Dict[str, Any]:
        """Use LLM to estimate if deletion will exceed threshold"""
        prompt = f"""Analyze this deletion request and estimate if it will affect more than 100 rows.

Request: "{request}"

Context:
- The database contains labor certification applications
- There are tables for employers, attorneys, cases, worksites
- Keywords like "all", "many", "attorneys" suggest bulk operations
- The deletion threshold is 100 rows (including CASCADE deletes)

Provide your analysis:
1. What entities are being deleted?
2. Approximate count estimate (rough order of magnitude)
3. Will this likely exceed 100 rows?

Respond in JSON format:
{{
  "entities": "...",
  "estimated_count": "...",
  "likely_exceeds_threshold": true/false,
  "reasoning": "..."
}}"""

        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception:
            pass
        
        return {"likely_exceeds_threshold": False}

    def _retrieve_relevant_context(self) -> Dict[str, Any]:
        """Retrieve relevant schema, columns, and rules (RAG approach)"""
        request = self.task['request']
        keywords = self._extract_keywords(request)
        
        # Get relevant tables from schema
        relevant_tables = self._get_relevant_tables(keywords)
        
        # Get relevant column meanings
        relevant_columns = self._get_relevant_columns(keywords)
        
        # Get relevant business rules
        relevant_rules = self._get_relevant_rules(keywords)
        
        # Always include critical rules
        all_rules = list(relevant_rules) + list(self.critical_rules)
        
        # Deduplicate rules by ID
        seen_ids = set()
        unique_rules = []
        for rule in all_rules:
            rule_id = rule.get('id')
            if rule_id not in seen_ids:
                seen_ids.add(rule_id)
                unique_rules.append(rule)
        
        return {
            'tables': relevant_tables,
            'columns': relevant_columns,
            'rules': unique_rules
        }

    def _extract_keywords(self, request: str) -> List[str]:
        """Extract meaningful keywords from request"""
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                      'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                      'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                      'should', 'could', 'may', 'might', 'can'}
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', request.lower())
        
        # Filter stop words and return unique keywords
        keywords = [w for w in words if w not in stop_words]
        return list(set(keywords))

    def _get_relevant_tables(self, keywords: List[str]) -> str:
        """Get relevant table schemas based on keywords"""
        # Table name to schema mapping (simplified)
        table_keywords = {
            'employer': ['employer', 'company', 'firm', 'organization', 'business'],
            'attorney': ['attorney', 'lawyer', 'legal', 'counsel'],
            'cases': ['case', 'application', 'petition', 'visa', 'h1b', 'certification'],
            'preparer': ['preparer', 'prepare'],
            'worksite': ['worksite', 'location', 'site', 'address'],
            'prevailing_wage': ['wage', 'salary', 'pay', 'compensation'],
            'employer_poc': ['contact', 'poc', 'point'],
            'case_attorney': ['represent', 'representation'],
            'case_worksite': ['worksite']
        }
        
        # Find matching tables
        relevant_tables = set()
        for keyword in keywords:
            for table, table_kws in table_keywords.items():
                if keyword in table_kws or keyword in table:
                    relevant_tables.add(table)
        
        # If no matches, include all major tables
        if not relevant_tables:
            relevant_tables = {'employer', 'attorney', 'cases', 'case_attorney'}
        
        # Extract relevant parts from schema
        schema_lines = self.schema.split('\n')
        relevant_schema_parts = []
        current_table = None
        capture = False
        
        for line in schema_lines:
            if 'CREATE TABLE' in line:
                # Check if this table is relevant
                for table in relevant_tables:
                    if f'"{table}"' in line:
                        capture = True
                        current_table = table
                        relevant_schema_parts.append(f"\n--- {table.upper()} TABLE ---")
                        break
                else:
                    capture = False
            
            if capture:
                relevant_schema_parts.append(line)
                if ');' in line or 'First 3 rows:' in line:
                    # Stop after table definition
                    capture = False
        
        return '\n'.join(relevant_schema_parts) if relevant_schema_parts else self.schema[:3000]

    def _get_relevant_columns(self, keywords: List[str]) -> Dict:
        """Get relevant column meanings based on keywords"""
        relevant = {}
        
        for col_path, meaning in self.column_meanings.items():
            # Check if any keyword matches column name or meaning
            col_lower = col_path.lower()
            meaning_str = str(meaning).lower() if not isinstance(meaning, dict) else json.dumps(meaning).lower()
            
            if any(kw in col_lower or kw in meaning_str for kw in keywords):
                relevant[col_path] = meaning
        
        # If too many, limit to most relevant
        if len(relevant) > 30:
            # Prioritize exact matches
            exact_matches = {k: v for k, v in relevant.items() 
                           if any(kw in k.lower() for kw in keywords)}
            if len(exact_matches) > 10:
                return dict(list(exact_matches.items())[:30])
            return dict(list(relevant.items())[:30])
        
        return relevant

    def _get_relevant_rules(self, keywords: List[str]) -> List[Dict]:
        """Get relevant business rules based on keywords"""
        relevant_rules = []
        
        for rule in self.knowledge_base:
            # Check if keywords match rule content
            rule_text = (rule.get('knowledge', '') + ' ' + 
                        rule.get('description', '') + ' ' + 
                        rule.get('definition', '')).lower()
            
            if any(kw in rule_text for kw in keywords):
                relevant_rules.append(rule)
        
        # Limit to most relevant rules
        return relevant_rules[:15]

    def _generate_sql_with_reasoning(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SQL query with explicit chain-of-thought reasoning"""
        
        # Build the prompt with relevant context
        prompt = self._build_generation_prompt(context)
        
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=4000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            
            # Parse the structured response
            result = self._parse_llm_response(content)
            return result
            
        except Exception as e:
            return {
                'decision': 'reject',
                'reasoning': f'Error generating SQL: {str(e)}',
                'sql': '',
                'rule_concerns': []
            }

    def _build_generation_prompt(self, context: Dict[str, Any]) -> str:
        """Build the complete prompt for SQL generation"""
        
        # Format rules in a clear way
        rules_text = self._format_rules(context['rules'])
        
        # Format column meanings
        columns_text = self._format_columns(context['columns'])
        
        prompt = f"""You are an expert SQL query generator for a labor certification database. You must follow business rules strictly.

DATABASE SCHEMA (relevant tables):
{context['tables']}

COLUMN MEANINGS (relevant columns):
{columns_text}

BUSINESS RULES (MUST FOLLOW - CRITICAL):
{rules_text}

IMPORTANT NOTES:
1. The 'employer' table has a composite primary key: (corphandle, zipref)
2. When joining to employer, ALWAYS use BOTH columns: ON c.homefirm = e.corphandle AND c.homezip = e.zipref
3. The 'statustag' field has inconsistent casing - use LOWER(statustag) for comparisons
4. JSONB fields (attorney_profile, employer_contact_info, wage_details) use -> for objects and ->> for text
5. For attorney_profile, you can access: firm, name, highest_court (PUBLIC) but NOT address or contact (PRIVATE)

USER REQUEST:
{self.task['request']}

YOUR TASK:
Generate a response following this EXACT structure:

<analysis>
[Explain what the user wants to do - what is the goal of the query?]
</analysis>

<tables_needed>
[List which tables you need and why]
</tables_needed>

<rule_check>
[Check each relevant business rule. For each rule, state:
 - Rule ID and name
 - Does this request violate it? Yes/No
 - Reasoning]
</rule_check>

<decision>
[Write EXACTLY ONE WORD: "execute" or "reject"]
</decision>

<reasoning>
[Explain your decision. If rejecting, explain which rule is violated and why. If executing, confirm no rules are violated.]
</reasoning>

<sql>
[If decision is "execute", write the complete SQL query here. If "reject", leave empty.]
</sql>

CRITICAL RULE REMINDERS:
- Rule 64: Cannot DELETE more than 100 rows total (including cascades)
- Rule 62: Cannot access attorney address/contact info unless same firm
- Always check if a DELETE or UPDATE will affect too many rows before allowing it

Generate your response now:"""

        return prompt

    def _format_rules(self, rules: List[Dict]) -> str:
        """Format rules in a readable way"""
        formatted = []
        for rule in rules:
            rule_text = f"""
Rule ID {rule.get('id')}: {rule.get('knowledge', 'Unknown')}
Type: {rule.get('type', 'unknown')}
Description: {rule.get('description', '')}
Definition: {rule.get('definition', '')}
"""
            formatted.append(rule_text)
        
        return '\n'.join(formatted)

    def _format_columns(self, columns: Dict) -> str:
        """Format column meanings in a readable way"""
        formatted = []
        for col_path, meaning in list(columns.items())[:20]:  # Limit to avoid token overflow
            if isinstance(meaning, dict):
                formatted.append(f"{col_path}: {json.dumps(meaning, indent=2)}")
            else:
                formatted.append(f"{col_path}: {meaning}")
        
        return '\n\n'.join(formatted)

    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """Parse structured response from LLM"""
        result = {
            'analysis': '',
            'rule_concerns': [],
            'decision': 'reject',
            'reasoning': '',
            'sql': ''
        }
        
        # Extract sections using regex
        analysis_match = re.search(r'<analysis>(.*?)</analysis>', content, re.DOTALL | re.IGNORECASE)
        if analysis_match:
            result['analysis'] = analysis_match.group(1).strip()
        
        rule_check_match = re.search(r'<rule_check>(.*?)</rule_check>', content, re.DOTALL | re.IGNORECASE)
        if rule_check_match:
            result['rule_concerns'] = rule_check_match.group(1).strip()
        
        decision_match = re.search(r'<decision>(.*?)</decision>', content, re.DOTALL | re.IGNORECASE)
        if decision_match:
            decision = decision_match.group(1).strip().lower()
            if 'execute' in decision:
                result['decision'] = 'execute'
            else:
                result['decision'] = 'reject'
        
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', content, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            result['reasoning'] = reasoning_match.group(1).strip()
        
        sql_match = re.search(r'<sql>(.*?)</sql>', content, re.DOTALL | re.IGNORECASE)
        if sql_match:
            result['sql'] = sql_match.group(1).strip()
        
        return result

    def _validate_query(self, generation_result: Dict[str, Any]) -> ValidationResult:
        """Multi-layer validation of generated query"""
        
        # If LLM already rejected, accept that decision
        if generation_result['decision'] == 'reject':
            return ValidationResult(
                is_valid=False,
                reason=generation_result.get('reasoning', 'Query rejected by generator'),
                sql=''
            )
        
        sql = generation_result['sql']
        
        if not sql or sql.strip() == '':
            return ValidationResult(
                is_valid=False,
                reason='No SQL query generated'
            )
        
        # Validation Layer 1: Parse SQL
        try:
            parsed = sqlparse.parse(sql)
            if not parsed:
                return ValidationResult(
                    is_valid=False,
                    reason='Failed to parse SQL query'
                )
            parsed_query = parsed[0]
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                reason=f'SQL parsing error: {str(e)}'
            )
        
        # Validation Layer 2: Check deletion threshold
        if self._is_delete_query(parsed_query):
            try:
                count = self._estimate_deletion_count(sql)
                if count > self.deletion_threshold:
                    return ValidationResult(
                        is_valid=False,
                        reason=f'Would delete {count} rows (max {self.deletion_threshold} allowed per Rule 64)'
                    )
            except Exception as e:
                # If we can't estimate, be conservative and reject
                return ValidationResult(
                    is_valid=False,
                    reason=f'Cannot safely estimate deletion impact: {str(e)}'
                )
        
        # Validation Layer 3: Check attorney privacy
        if 'attorney_profile' in sql.lower():
            if not self._uses_safe_jsonb_access(sql):
                return ValidationResult(
                    is_valid=False,
                    reason='Attorney privacy violation: Cannot access personal address/contact info (Rule 62)'
                )
        
        # Validation Layer 4: Syntax check with EXPLAIN
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"EXPLAIN {sql}")
            cursor.close()
        except psycopg2.Error as e:
            return ValidationResult(
                is_valid=False,
                reason=f'SQL syntax error: {str(e)}'
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                reason=f'Query validation error: {str(e)}'
            )
        
        # All validations passed
        return ValidationResult(
            is_valid=True,
            sql=sql
        )

    def _is_delete_query(self, parsed_query) -> bool:
        """Check if query is a DELETE statement"""
        query_str = str(parsed_query).upper()
        return 'DELETE' in query_str

    def _estimate_deletion_count(self, delete_sql: str) -> int:
        """Estimate how many rows will be deleted by converting to COUNT query"""
        
        # Replace DELETE FROM table with SELECT COUNT(*) FROM table
        count_sql = re.sub(
            r'DELETE\s+FROM\s+(\w+)',
            r'SELECT COUNT(*) FROM \1',
            delete_sql,
            flags=re.IGNORECASE
        )
        
        # Execute count query
        cursor = self.conn.cursor()
        try:
            cursor.execute(count_sql)
            result = cursor.fetchone()
            count = result[0] if result else 0
            return count
        finally:
            cursor.close()

    def _uses_safe_jsonb_access(self, sql: str) -> bool:
        """
        Check if query accesses attorney_profile JSONB safely.
        
        UNSAFE: SELECT attorney_profile FROM attorney
        UNSAFE: SELECT attorney_profile->'address' FROM attorney
        UNSAFE: SELECT attorney_profile->'contact' FROM attorney
        
        SAFE: SELECT attorney_profile->>'firm' FROM attorney
        SAFE: SELECT attorney_profile->'name' FROM attorney
        SAFE: SELECT attorney_profile->'highest_court' FROM attorney
        """
        
        # Check if selecting full attorney_profile without field extraction
        if re.search(r'attorney_profile\s*(?:,|FROM)', sql, re.IGNORECASE):
            # Selecting the entire JSONB object
            return False
        
        # Check if accessing sensitive fields
        sensitive_fields = ['address', 'contact']
        for field in sensitive_fields:
            # Check for -> or ->> access to sensitive fields
            if re.search(rf"attorney_profile\s*-[>]+\s*['\"]?{field}['\"]?", sql, re.IGNORECASE):
                return False
        
        return True

    def _format_result(self, validation: ValidationResult, start_time: float) -> dict[str, str]:
        """Format the final result dictionary"""
        elapsed = time.time() - start_time
        
        if validation.is_valid:
            return {
                "action": "execute",
                "sql": validation.sql,
                "elapsed_seconds": round(elapsed, 2)
            }
        else:
            return {
                "action": "reject",
                "sql": "",
                "elapsed_seconds": round(elapsed, 2)
            }

    def __del__(self):
        """Cleanup database connection"""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()


def main():
    """Test your DatabaseAgent class."""
    # Read task from example_task_1.json
    with open('example_task_1.json', 'r') as f:
        task = json.load(f)

    print(f"Loaded task from example_task_1.json:")
    print(f"  Request: {task['request']}")
    print(f"  Database: {task['database']}")
    print(f"  Column Meaning: {task['column_meaning']}")
    print(f"  Knowledge Base: {task['knowledge_base']}")
    print()

    agent = DatabaseAgent(model_name="gpt-4", task=task)
    # Test the gold query
    with open('example_task_1_gold_query.sql', 'r') as f:
        gold_query = f.read()
    cursor = agent.conn.cursor()
    cursor.execute(gold_query)

    # Fetch and display results
    results = cursor.fetchall()
    print(f"\nQuery returned {len(results)} results:\n")

    # Print header
    print(f"{'Employer':<40} {'Attorney Email':<35} {'Cases':<8} {'Certified':<10} {'Success %':<10}")
    print("-" * 110)

    # Print first 10 results
    for row in results[:10]:
        employer, email, case_load, certified, success_rate = row
        employer_short = employer[:38] if len(employer) > 38 else employer
        email_short = email[:33] if len(email) > 33 else email
        print(f"{employer_short:<40} {email_short:<35} {case_load:<8} {certified:<10} {success_rate:<10}")

    if len(results) > 10:
        print(f"\n... and {len(results) - 10} more results")

    cursor.close()
    agent.conn.close()
    print("\n" + "="*80)


if __name__ == "__main__":
    main()