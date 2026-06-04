"""Pure status-propagation policy — no I/O, no DB, trivially unit-testable.

This module encodes *what* an article's status should be given its baseline (from its
own legislation) and the relationships that target it. The decisions here are deliberately
conservative: only direct, unambiguous effects move the flag. Chained effects (a law that
repeals a repealer) and genuine conflicts are NOT resolved here — they are left as
`CONFLICT`/in-force so the agent's query-time graph traversal can read each relationship's
illustration and decide. See STATUS_PROPAGATION_PLAN.md §4.
"""

from datetime import date

from src.db.schemas import ArticleStatus, LegislationStatus, RelationshipType


# A relationship type that targets an article maps to (resulting status, is_in_force).
# Only these three types move an article's status. `references` and `implements` never do;
# `conflicts_with` is handled separately (it flags, it does not decide).
EFFECT_POLICY: dict[RelationshipType, tuple[ArticleStatus, bool]] = {
    RelationshipType.REPEALS: (ArticleStatus.REPEALED, False),
    RelationshipType.SUPERSEDES: (ArticleStatus.REPEALED, False),
    RelationshipType.AMENDS: (ArticleStatus.AMENDED, True),
}


def article_baseline(status: LegislationStatus) -> tuple[ArticleStatus, bool]:
    """The status an article starts from, derived from its own legislation, before any
    relationship effects are applied. Pending/draft legislation is not yet in force."""
    if status == LegislationStatus.REPEALED:
        return (ArticleStatus.REPEALED, False)
    if status == LegislationStatus.AMENDED:
        return (ArticleStatus.AMENDED, True)
    if status in (LegislationStatus.PENDING, LegislationStatus.DRAFT):
        return (ArticleStatus.ACTIVE, False)
    return (ArticleStatus.ACTIVE, True)


def fold_status(
    baseline: tuple[ArticleStatus, bool],
    incoming: list[tuple[date, RelationshipType]],
) -> tuple[ArticleStatus, bool]:
    """Resolve an article's status from its baseline and the relationships targeting it.

    `incoming` is a list of (affecting_legislation_date, relationship_type) — only DIRECT
    relationships, never transitively walked. The latest effect by date wins. If the latest
    date carries two different effect types, the result is CONFLICT (kept in force — we never
    silently hide law). With no status-changing effects, a lone `conflicts_with` flags
    CONFLICT, otherwise the baseline stands.
    """
    effects = [(d, t) for d, t in incoming if t in EFFECT_POLICY]

    if not effects:
        if any(t == RelationshipType.CONFLICTS_WITH for _, t in incoming):
            return (ArticleStatus.CONFLICT, True)
        return baseline

    latest_date = max(d for d, _ in effects)
    latest_types = {t for d, t in effects if d == latest_date}

    if len(latest_types) > 1:
        return (ArticleStatus.CONFLICT, True)

    return EFFECT_POLICY[next(iter(latest_types))]