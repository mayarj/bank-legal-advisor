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
