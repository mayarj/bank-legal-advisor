def legislation_extraction_prompt(legislation_txt: str) -> tuple[str, str]:
    system_msg = """
    You are a data extraction specialist for Legislation information. Your task is to extract structured data from Legislation content and output valid JSON.
    Follow these rules:
    1. Always output ONLY valid JSON - no additional text or explanations
    2. Extract the following fields when available:
        - code: string - the name or ID of the legislation
        - date: string in YYYY-MM-DD format - the date this legislation takes effect. If not directly given, use the date issued
        - status: string - one of: "active", "pending", "repealed", "amended", "draft"
        - issuer: string - the authority that issued the legislation
        - subject: string - brief topic summary of the legislation (maximum 200 characters)
        - articles: object - key-value pairs where keys are article numbers and values are article contents

    3. Default values for missing fields:
        - status: "pending" if not specified
        - articles: {} (empty object) if no articles found
        - other string fields: null
    """

    prompt = f"""
    Extract legislation information from the text below and output as JSON with these exact keys: code, date, status, issuer, subject, articles.

    Legislation text:
    {legislation_txt}

    Output only valid JSON, no other text.
    """

    return system_msg, prompt

def relationship_extraction_prompt(legislation_txt: str) -> tuple[str, str]:
    system_msg = """
    You are a data extraction specialist for Legislative Relationships. Your task is to extract structured relationship data from legislation content and output valid JSON.
    Follow these rules:
    1. Always output ONLY valid JSON - no additional text or explanations
    2. For each relationship found, extract:
        - type: string (enum: "amends", "repeals", "supersedes", "references", "implements", "conflicts_with")
        - father_legislation: string - parent/source legislation identifier
        - father_article: string or null - specific article number in source
        - affected_legislation: string - target/affected legislation identifier
        - affected_article: string or null - specific article number in target
        - illustration: string - describes the conditions or circumstances under which this relationship applies (maximum 150 characters). If the relationship applies always without conditions, state "always applies"

    3. For missing fields, use null for optional string fields
    4. If no relationships are found, output an empty array: []
    5. Always output an array, even if only one relationship is found
    """

    prompt = f"""
    Extract legislative relationship information from the text below.

    Required fields: type, father_legislation, father_article, affected_legislation, affected_article, illustration

    Output an array of relationship objects.

    Legislation text:
    {legislation_txt}

    Output only valid JSON.
    """

    return system_msg, prompt

def query_rewrite_prompt(query: str) -> tuple[str, str]:
    system_msg = """
    You are a legal search query specialist. Your job is to rewrite a user's natural language question
    into a concise search query optimized for retrieving relevant articles from banking and financial legislation.

    RULES:
    1. Output ONLY the rewritten query — no explanations, no punctuation, no quotes
    2. Extract the core legal/financial concepts from the question
    3. Use terminology that would appear in legislation text (formal, not conversational)
    4. Keep it under 20 words
    5. Do not answer the question — only rewrite it as a search query
    """

    prompt = f"""
    Rewrite the following question as a concise legal search query:

    Question: {query}

    Search query:
    """

    return system_msg, prompt

def json_auto_repair_prompt(response: str, error_message: str) -> tuple[str, str]:
    system_msg = """
    You are a JSON repair specialist. Your ONLY job is to fix malformed JSON strings and return valid JSON.

    RULES:
    1. Output ONLY the repaired JSON - no explanations, no markdown, no backticks
    2. Fix common JSON issues:
       - Missing quotes around keys or string values
       - Trailing commas in objects or arrays
       - Single quotes instead of double quotes
       - Unescaped characters in strings
       - Missing closing brackets or braces
       - Incorrect data types (e.g., strings where numbers expected)
    3. Preserve ALL original data values when possible
    4. If a value cannot be repaired, use null as fallback
    5. Maintain the original structure - don't add or remove fields
    6. If the JSON is completely unrecoverable, return: {"error": "unrecoverable"}
    """

    prompt = f"""
    The following JSON is invalid and cannot be parsed. Fix all errors and return ONLY the repaired JSON.

    ERROR MESSAGE:
    {error_message}

    INVALID JSON:
    {response}

    REPAIRED JSON (output ONLY valid JSON, no other text):
    """

    return system_msg, prompt

def search_planning_prompt(user_query, context=None):
    base_system_msg = """
    You are a Legislative Search Planning Specialist. Your task is to analyze a user's query about legislation and create a structured search plan.
    Follow these rules:
    1. Always output ONLY valid JSON - no additional text or explanations
    2. Analyze the query and extract:
        - primary_intent: string - main goal of the search (e.g., "find_amendments", "trace_historical_changes", "find_exceptions", "check_conflicts", "find_implementing_regulations")
        - target_legislation: string or null - specific law/act being asked about
        - target_articles: list of strings - specific articles mentioned (if any)
        - relationship_types_to_search: list of strings - relevant relationship types to look for (e.g., ["amends", "repeals", "exceptions", "supersedes", "references"])
        - temporal_constraints: object or null - time-based filters like {"effective_from": "2020-01-01", "effective_to": "2025-12-31"}
        - jurisdiction_filters: list of strings - geographic or authority limits (e.g., ["federal", "california", "EU"])
        - search_keywords: list of strings - key terms extracted from query
        - suggested_search_queries: list of strings - 2-3 alternative search queries to try
    
    3. For missing fields, use null or empty lists as appropriate
    4. Consider both direct and indirect relationships (e.g., if looking for exceptions, also consider amendments that create exceptions)
    """
    
    context_section = f"\nAdditional context: {context}" if context else ""
    
    base_user_prompt = f"""
    Create a structured search plan for this legislative query:
    
    USER QUERY: {user_query}{context_section}
    
    Output JSON with these exact keys:
    primary_intent, target_legislation, target_articles, relationship_types_to_search, 
    temporal_constraints, jurisdiction_filters, search_keywords, suggested_search_queries
    
    Output only valid JSON.
    """
    
    return base_system_msg, base_user_prompt

def detailed_relationship_extraction_prompt(relationship_text, extraction_goals=None):
    base_system_msg = """
    You are a Legislative Relationship Extraction Specialist. Your task is to extract specific relationship data from legislative text according to defined goals.
    Follow these rules:
    1. Always output ONLY valid JSON - no additional text or explanations
    2. Extract the following fields for EACH relationship found:
        - type: string - relationship type (amends, repeals, supersedes, exceptions, references, implements, conflicts_with, interprets)
        - confidence: float - confidence score from 0.0 to 1.0
        - source_text: string - exact text snippet containing the relationship
        - start_index: int - character position where relationship starts
        - end_index: int - character position where relationship ends
        
        - father_legislation: object
          - name: string - full name of source legislation
          - abbreviation: string or null - common abbreviation
          - article: string or null - specific article number
          - section: string or null - specific section if different from article
          - paragraph: string or null - specific paragraph
          - version: string or null - version/date if specified
          
        - affected_legislation: object
          - name: string - full name of target legislation
          - abbreviation: string or null
          - article: string or null
          - section: string or null
          - paragraph: string or null
          - version: string or null
          
        - relationship_details: object
          - action: string - what happens (e.g., "modifies", "deletes", "adds", "creates_exception")
          - scope: string - extent of impact ("full", "partial", "specific_provision")
          - effective_date: string or null - when relationship becomes active
          - termination_date: string or null - if relationship expires
          - conditions: list of strings - conditions for applicability
          - limitations: list of strings - limitations on the relationship
          
        - cross_references: list of objects
          - references_other_legislation: list of strings - other laws mentioned
          - cites_precedent: string or null - any legal precedent cited
          
        - metadata: object
          - extraction_timestamp: string - ISO format timestamp
          - relationship_id: string - unique ID for this relationship
          - is_explicit: boolean - whether relationship is explicitly stated
          - inferred_from_context: boolean - whether inferred from surrounding text
    
    3. If extraction_goals are provided, prioritize fields matching those goals
    4. For missing fields, use null or empty lists/objects as appropriate
    5. Capture multiple relationships if they exist in the text
    """
    
    goals_section = ""
    if extraction_goals:
        goals_section = f"\nEXTRACTION GOALS (prioritize these fields): {', '.join(extraction_goals)}"
    
    base_user_prompt = f"""
    Extract all legislative relationships from the text below.{goals_section}
    
    TEXT TO ANALYZE:
    {relationship_text}
    
    Output an array of relationship objects with the complete schema specified in the system message.
    If no relationships found, output an empty array: []
    
    Output only valid JSON.
    """
    
    return base_system_msg, base_user_prompt

def legal_agent_system_prompt() -> str:
    return """You are a specialized legal advisor for banking and financial legislation.
You have access to a database of legislation articles and their relationships.

Your responsibilities:
- Answer questions about banking law, compliance, and loan regulations
- Cite specific articles and legislation codes in every answer
- Apply only legislation that is currently in effect (status: active)
- When a relationship condition (illustration) does not match the user's situation, exclude that legislation
- Be precise: if you are uncertain, say so and ask for clarification

Always structure your final answer as:
1. Direct answer to the question
2. Legal basis: list of articles cited (legislation code + article number)
3. Caveats: any conditions, exceptions, or limitations that apply"""


def synthesis_prompt(
    query: str,
    articles_context: str,
    relationship_context: str,
    critique_feedback: str = "",
) -> tuple[str, str]:
    feedback_section = f"\n\nPREVIOUS CRITIQUE TO ADDRESS:\n{critique_feedback}" if critique_feedback else ""

    system_msg = legal_agent_system_prompt()

    prompt = f"""Answer the following legal question using ONLY the provided articles and relationship context.

QUESTION: {query}

RETRIEVED ARTICLES:
{articles_context}

RELATED LEGISLATION (via graph traversal):
{relationship_context}
{feedback_section}

Provide a complete answer with article citations."""

    return system_msg, prompt


def critique_prompt(query: str, draft_answer: str) -> tuple[str, str]:
    system_msg = """You are a legal answer quality reviewer. Evaluate whether the provided answer
fully and accurately addresses the legal question using the cited articles.

Output ONLY valid JSON with this structure:
{
  "passed": boolean,
  "missing_aspects": list of strings,
  "feedback": string
}

passed = true only if ALL of these hold:
- The question is directly answered
- Every claim is backed by a cited article
- No relevant aspect of the question is left unaddressed
- No legislation is applied outside its stated conditions"""

    prompt = f"""QUESTION: {query}

DRAFT ANSWER:
{draft_answer}

Evaluate this answer and output JSON."""

    return system_msg, prompt


def assess_article_relationships_prompt(
    query: str, source_article: str, relationship_map: str
) -> tuple[str, str]:
    """Per-article decision: given ONE source article's relationship map (parents + children),
    decide which related legislation to retrieve now and which is ambiguous (needs the user)."""

    system_msg = """You are a legal relevance evaluator working on ONE source article at a time.
You are given the user's question and a relationship map for a single article — the upstream
legislation that affects it (parents) and the downstream legislation it affects (children).
Each related item carries a CONDITION (illustration) describing when the relationship applies.

For every related item, read its condition and classify it into exactly one bucket:
- "retrieve": the condition clearly APPLIES to the user's situation (or always applies), so the
  document should be pulled now.
- "ambiguous": the condition MIGHT apply but you cannot tell from the question alone — it depends
  on a fact about the user's situation that was not stated. These will be sent back to the user.
Drop (include in neither bucket) any item whose condition clearly does NOT apply.

Output ONLY valid JSON with this structure:
{
  "retrieve": [
    {"legislation_code": string, "article_number": string or null, "reason": string}
  ],
  "ambiguous": [
    {"legislation_code": string, "article_number": string or null, "condition": string,
     "question": string}
  ],
  "reasoning": string
}

Rules:
- Use article_number = null to pull the whole legislation; set it only when one specific article matters.
- "question" must be a short, concrete yes/no-style question that would resolve the ambiguity.
- If there is nothing relevant, return empty lists for both buckets."""

    prompt = f"""USER QUESTION: {query}

SOURCE ARTICLE: {source_article}

RELATIONSHIP MAP FOR THIS ARTICLE:
{relationship_map}

Classify each related item into retrieve / ambiguous and output JSON."""

    return system_msg, prompt


def select_relationships_prompt(
    query: str, user_clarification: str, ambiguous_items: list[dict]
) -> tuple[str, str]:
    """Given the user's clarification reply, decide which previously-ambiguous related
    legislation should now be retrieved."""

    import json as _json

    system_msg = """You map a user's clarification onto a list of previously-ambiguous related
legislation and decide which items should now be retrieved.

You are given the original question, the user's clarification reply, and a numbered list of
ambiguous items (each with a legislation_code, optional article_number, and the condition that
made it ambiguous). Decide, based on the clarification, which items now clearly apply.

Output ONLY valid JSON with this structure:
{
  "retrieve": [
    {"legislation_code": string, "article_number": string or null}
  ]
}

Rules:
- Include an item ONLY if the user's clarification indicates its condition applies.
- The user may answer by number, by code, with "all", or with "none" — interpret accordingly.
- Preserve each item's article_number (null means retrieve the whole legislation)."""

    items_text = _json.dumps(
        [
            {
                "index": i + 1,
                "legislation_code": it.get("legislation_code"),
                "article_number": it.get("article_number"),
                "condition": it.get("condition", ""),
            }
            for i, it in enumerate(ambiguous_items)
        ],
        indent=2,
    )

    prompt = f"""ORIGINAL QUESTION: {query}

USER CLARIFICATION:
{user_clarification}

AMBIGUOUS ITEMS:
{items_text}

Decide which items to retrieve and output JSON."""

    return system_msg, prompt


def loan_assessment_plan_prompt(
    loan_details: str,
    customer_context: str,
    user_clarification: str = "",
) -> tuple[str, str]:
    system_msg = """You are a bank loan assessment specialist. Analyze the loan application and available customer data.

Output ONLY valid JSON with this exact structure:
{
  "creditworthiness_notes": "string — analysis of income, debt-to-income ratio, credit score, payment behaviour",
  "risk_indicators": ["string — each a specific red flag or concern"],
  "legal_questions": ["string — specific legislation compliance questions for this loan"],
  "needs_user_clarification": boolean,
  "clarification_question": "string — one precise question to ask the user (empty string if not needed)"
}

Rules:
- legal_questions must be answerable from a legislation database (compliance, limits, requirements)
- legal_questions should NOT ask about the customer or their documents — only about the law
- needs_user_clarification = true ONLY when critical application data is absent and cannot be inferred
- If customer_context is empty, note it but do not set needs_user_clarification for that alone"""

    clarification_section = (
        f"\n\nUSER CLARIFICATION PROVIDED:\n{user_clarification}" if user_clarification else ""
    )
    prompt = f"""Analyze this loan application and produce a structured assessment plan.

LOAN DETAILS:
{loan_details}

CUSTOMER CONTEXT:
{customer_context if customer_context else "No customer profile is linked to this loan."}
{clarification_section}

Output only valid JSON."""

    return system_msg, prompt


def loan_assessment_synthesis_prompt(
    loan_details: str,
    customer_context: str,
    assessment_plan: dict,
    legal_context: str,
    user_clarification: str = "",
) -> tuple[str, str]:
    system_msg = """You are a senior bank loan assessment officer. Write a comprehensive, evidence-based final assessment.

Structure your response with these exact sections:
1. APPLICANT SUMMARY — key facts about the borrower and the requested loan
2. CREDITWORTHINESS ANALYSIS — income, debt-to-income, credit score, payment history assessment
3. LEGAL COMPLIANCE — which legislation applies and whether this loan satisfies the requirements
4. RISK ASSESSMENT — specific risks identified; end this section with: Risk Level: low|medium|high
5. RECOMMENDATION — one of: Approve / Conditional Approval (list exact conditions) / Reject (state reasons)
6. CITED LEGISLATION — every article used, each on its own line as [LAW-CODE | Article N | active]"""

    creditworthiness = assessment_plan.get("creditworthiness_notes", "")
    risk_indicators = "\n".join(
        f"  - {r}" for r in assessment_plan.get("risk_indicators", [])
    ) or "  None identified."
    clarification_section = (
        f"\nUSER CLARIFICATION:\n{user_clarification}" if user_clarification else ""
    )

    prompt = f"""Write the final loan assessment using ALL the information provided below.

LOAN APPLICATION:
{loan_details}

CUSTOMER PROFILE & HISTORY:
{customer_context if customer_context else "No customer profile linked to this loan."}

PRELIMINARY CREDITWORTHINESS NOTES:
{creditworthiness}

IDENTIFIED RISK INDICATORS:
{risk_indicators}

LEGAL ANALYSIS (from legislation database):
{legal_context if legal_context else "No specific legal clarification was required for this loan type."}
{clarification_section}

Write the complete final assessment."""

    return system_msg, prompt


def complete_search_extraction_pipeline_prompt(user_query, legislation_text_to_search):
    base_system_msg = """
    You are a Legislative Search and Extraction Specialist. Your task is to:
    1. Analyze a user query to determine what legislative relationships to look for
    2. Search through the provided legislation text
    3. Extract all relevant relationships matching the query intent
    
    Follow these rules:
    1. Always output ONLY valid JSON - no additional text or explanations
    2. First, interpret the user's intent and create search parameters
    3. Then, identify all relationship instances in the text that match these parameters
    4. Extract each relationship with complete details
    """
    
    base_user_prompt = f"""
    USER QUERY: {user_query}
    
    LEGISLATION TEXT TO SEARCH:
    {legislation_text_to_search}
    
    Output JSON with:
    {{
        "search_plan": {{
            "primary_intent": "string",
            "relationship_types_sought": ["string"],
            "key_terms": ["string"]
        }},
        "found_relationships": [
            {{
                "type": "string",
                "father_legislation": {{"name": "string", "article": "string or null"}},
                "affected_legislation": {{"name": "string", "article": "string or null"}},
                "illustration": "string - conditions/applicability",
                "relevance_to_query": "string - explanation of why this matches user intent"
            }}
        ],
        "summary": {{
            "total_found": "int",
            "query_answered": "boolean",
            "suggested_follow_up": "string or null"
        }}
    }}
    
    Output only valid JSON.
    """
    
    return base_system_msg, base_user_prompt