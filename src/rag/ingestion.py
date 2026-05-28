from concurrent.futures import ThreadPoolExecutor
from pydantic import TypeAdapter

from src.core.llm import invoke
from src.core.prompts import (
    legislation_extraction_prompt,
    relationship_extraction_prompt,
    json_auto_repair_prompt,
)
from src.db.schemas import Legislation, Relationship

_MAX_RETRIES = 3
_relationship_list_adapter = TypeAdapter(list[Relationship])


def generate_legislation_data(legislation_content: str) -> Legislation | None:
    system_msg, prompt = legislation_extraction_prompt(legislation_content)
    response = invoke(system_msg, prompt)

    for attempt in range(_MAX_RETRIES):
        try:
            return Legislation.model_validate_json(response)
        except Exception as e:
            if attempt == _MAX_RETRIES - 1:
                return None
            repair_system, repair_prompt = json_auto_repair_prompt(response, str(e))
            response = invoke(repair_system, repair_prompt)

    return None


def generate_relationships_data(legislation_content: str) -> list[Relationship]:
    system_msg, prompt = relationship_extraction_prompt(legislation_content)
    response = invoke(system_msg, prompt)

    for attempt in range(_MAX_RETRIES):
        try:
            return _relationship_list_adapter.validate_json(response)
        except Exception as e:
            if attempt == _MAX_RETRIES - 1:
                return []
            repair_system, repair_prompt = json_auto_repair_prompt(response, str(e))
            response = invoke(repair_system, repair_prompt)

    return []


def run_ingestion_workflow(legislation_content: str) -> Legislation | None:
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_leg = executor.submit(generate_legislation_data, legislation_content)
        future_rel = executor.submit(generate_relationships_data, legislation_content)
        legislation = future_leg.result()
        relationships = future_rel.result()

    if legislation is not None:
        legislation.relationships = relationships

    return legislation