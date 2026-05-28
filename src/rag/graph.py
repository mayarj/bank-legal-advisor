from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Relationship as RelationshipModel


@dataclass
class GraphNode:
    legislation_code: str
    article_number: str | None
    relationship_type: str
    illustration: str
    depth: int


async def get_parents(
    session: AsyncSession,
    legislation_code: str,
    article_number: str | None = None,
    k_depth: int = 2,
) -> list[GraphNode]:
    """
    Traverse upward: who affects the given legislation/article?
    At each depth, the article filter is propagated from the previous level's father_article,
    so traversal follows the exact relationship chain rather than broadening to the whole law.
    """
    nodes: list[GraphNode] = []
    visited: set[str] = {legislation_code}
    current_level: list[tuple[str, str | None]] = [(legislation_code, article_number)]

    for depth in range(1, k_depth + 1):
        next_level: list[tuple[str, str | None]] = []

        for code, article in current_level:
            stmt = select(RelationshipModel).where(
                RelationshipModel.affected_legislation == code
            )
            if article:
                stmt = stmt.where(RelationshipModel.affected_article == article)

            result = await session.execute(stmt)
            relationships = result.scalars().all()

            for rel in relationships:
                if rel.father_legislation not in visited:
                    visited.add(rel.father_legislation)
                    nodes.append(GraphNode(
                        legislation_code=rel.father_legislation,
                        article_number=rel.father_article,
                        relationship_type=rel.type.value,
                        illustration=rel.illustration,
                        depth=depth,
                    ))
                    next_level.append((rel.father_legislation, rel.father_article))

        current_level = next_level
        if not current_level:
            break

    return nodes


async def get_children(
    session: AsyncSession,
    legislation_code: str,
    article_number: str | None = None,
    k_depth: int = 2,
) -> list[GraphNode]:
    """
    Traverse downward: what does the given legislation/article affect?
    At each depth, the article filter is propagated from the previous level's affected_article.
    """
    nodes: list[GraphNode] = []
    visited: set[str] = {legislation_code}
    current_level: list[tuple[str, str | None]] = [(legislation_code, article_number)]

    for depth in range(1, k_depth + 1):
        next_level: list[tuple[str, str | None]] = []

        for code, article in current_level:
            stmt = select(RelationshipModel).where(
                RelationshipModel.father_legislation == code
            )
            if article:
                stmt = stmt.where(RelationshipModel.father_article == article)

            result = await session.execute(stmt)
            relationships = result.scalars().all()

            for rel in relationships:
                if rel.affected_legislation not in visited:
                    visited.add(rel.affected_legislation)
                    nodes.append(GraphNode(
                        legislation_code=rel.affected_legislation,
                        article_number=rel.affected_article,
                        relationship_type=rel.type.value,
                        illustration=rel.illustration,
                        depth=depth,
                    ))
                    next_level.append((rel.affected_legislation, rel.affected_article))

        current_level = next_level
        if not current_level:
            break

    return nodes


def format_for_llm(nodes: list[GraphNode], direction: str = "parent") -> str:
    """
    Converts graph traversal results into a string an LLM can reason about.
    The model reads the illustration of each node to decide whether to retrieve
    the full document based on the user's question.
    """
    if not nodes:
        return "No related legislation found."

    by_depth: dict[int, list[GraphNode]] = {}
    for node in nodes:
        by_depth.setdefault(node.depth, []).append(node)

    relation_label = "affects" if direction == "parent" else "is affected by"

    lines = [
        "Related legislation found via graph traversal.",
        f"Read each illustration and decide if it applies to the user's question before retrieving the document.\n",
    ]

    for depth in sorted(by_depth):
        label = "Direct" if depth == 1 else f"Depth {depth} (indirect)"
        lines.append(f"[{label}]")

        for node in by_depth[depth]:
            article_ref = f", Article {node.article_number}" if node.article_number else ""
            lines.append(f"  - [{node.relationship_type}] {node.legislation_code}{article_ref}")
            lines.append(f"    Condition: \"{node.illustration}\"")

        lines.append("")

    return "\n".join(lines)