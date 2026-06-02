from datetime import date

import pytest

from src.db import crud
from src.db.schemas import (
    ArticleStatus,
    Legislation,
    LegislationStatus,
    Relationship,
    RelationshipType,
)
from src.rag.reconcile import recompute_article_status, reconcile_after_ingest
from src.rag.status_policy import article_baseline, fold_status
from tests.conftest import make_relationship


# ── Test helpers ────────────────────────────────────────────────────────────--

def make_legislation(
    code: str,
    date_str: str,
    status: LegislationStatus = LegislationStatus.ACTIVE,
    articles: dict | None = None,
    relationships: list[Relationship] | None = None,
) -> Legislation:
    return Legislation(
        code=code,
        date=date.fromisoformat(date_str),
        issuer="Central Bank",
        subject="Test legislation subject",
        status=status,
        articles=articles or {},
        relationships=relationships or [],
    )


def repeals(father: str, affected: str, affected_article: str | None) -> Relationship:
    return Relationship(
        type=RelationshipType.REPEALS,
        father_legislation=father,
        affected_legislation=affected,
        affected_article=affected_article,
        illustration="repeals the affected article",
    )


async def seed(session, legislation: Legislation) -> None:
    """Persist a legislation's metadata + article rows (mirrors the pipeline's Postgres save)."""
    await crud.upsert_legislation(session, legislation)
    await crud.upsert_articles(session, legislation)


async def save_rels(session, relationships: list[Relationship]) -> None:
    for r in relationships:
        session.add(make_relationship(
            r.father_legislation, r.affected_legislation, r.type,
            father_article=r.father_article,
            affected_article=r.affected_article,
            illustration=r.illustration,
        ))
    await session.flush()


@pytest.fixture(autouse=True)
def _no_vectorstore_sync(monkeypatch):
    """Reconciliation mirrors status into ChromaDB; stub it so DB tests need no vector store."""
    from unittest.mock import MagicMock
    mock = MagicMock()
    monkeypatch.setattr("src.rag.reconcile.update_article_status", mock)
    return mock


# ── Pure fold logic (no DB) ──────────────────────────────────────────────────-

class TestFoldStatus:

    def test_no_effects_returns_baseline(self):
        baseline = (ArticleStatus.ACTIVE, True)
        assert fold_status(baseline, []) == baseline

    def test_single_amend_keeps_article_in_force(self):
        status, in_force = fold_status(
            (ArticleStatus.ACTIVE, True),
            [(date(2010, 1, 1), RelationshipType.AMENDS)],
        )
        assert status == ArticleStatus.AMENDED
        assert in_force is True

    def test_single_repeal_removes_from_force(self):
        status, in_force = fold_status(
            (ArticleStatus.ACTIVE, True),
            [(date(2010, 1, 1), RelationshipType.REPEALS)],
        )
        assert status == ArticleStatus.REPEALED
        assert in_force is False

    def test_latest_effect_wins_repeal_then_amend(self):
        # Repealed in 2010, then amended in 2020 → amended (back in force).
        status, in_force = fold_status(
            (ArticleStatus.ACTIVE, True),
            [
                (date(2010, 1, 1), RelationshipType.REPEALS),
                (date(2020, 1, 1), RelationshipType.AMENDS),
            ],
        )
        assert status == ArticleStatus.AMENDED
        assert in_force is True

    def test_latest_effect_wins_amend_then_repeal(self):
        status, in_force = fold_status(
            (ArticleStatus.ACTIVE, True),
            [
                (date(2010, 1, 1), RelationshipType.AMENDS),
                (date(2020, 1, 1), RelationshipType.REPEALS),
            ],
        )
        assert status == ArticleStatus.REPEALED
        assert in_force is False

    def test_conflicting_effects_same_date_flag_conflict(self):
        status, in_force = fold_status(
            (ArticleStatus.ACTIVE, True),
            [
                (date(2020, 1, 1), RelationshipType.REPEALS),
                (date(2020, 1, 1), RelationshipType.AMENDS),
            ],
        )
        assert status == ArticleStatus.CONFLICT
        assert in_force is True  # never silently hide law

    def test_conflicts_with_only_flags_conflict(self):
        status, in_force = fold_status(
            (ArticleStatus.ACTIVE, True),
            [(date(2020, 1, 1), RelationshipType.CONFLICTS_WITH)],
        )
        assert status == ArticleStatus.CONFLICT
        assert in_force is True

    def test_references_and_implements_are_ignored(self):
        baseline = (ArticleStatus.ACTIVE, True)
        result = fold_status(baseline, [
            (date(2020, 1, 1), RelationshipType.REFERENCES),
            (date(2021, 1, 1), RelationshipType.IMPLEMENTS),
        ])
        assert result == baseline


class TestArticleBaseline:

    def test_active_is_in_force(self):
        assert article_baseline(LegislationStatus.ACTIVE) == (ArticleStatus.ACTIVE, True)

    def test_amended_is_in_force(self):
        assert article_baseline(LegislationStatus.AMENDED) == (ArticleStatus.AMENDED, True)

    def test_repealed_is_not_in_force(self):
        assert article_baseline(LegislationStatus.REPEALED) == (ArticleStatus.REPEALED, False)

    def test_pending_is_not_yet_in_force(self):
        assert article_baseline(LegislationStatus.PENDING) == (ArticleStatus.ACTIVE, False)


# ── recompute_article_status (DB) ─────────────────────────────────────────────-

class TestRecomputeArticleStatus:

    async def test_no_relationships_keeps_baseline(self, db_session):
        await seed(db_session, make_legislation("LAW-A", "2003-01-01", articles={"1": "a1"}))

        article = await recompute_article_status(db_session, "LAW-A", "1")

        assert article.is_in_force is True
        assert article.status == ArticleStatus.ACTIVE

    async def test_returns_none_for_missing_article(self, db_session):
        result = await recompute_article_status(db_session, "LAW-NONE", "1")
        assert result is None

    async def test_repeal_marks_article_out_of_force(self, db_session):
        await seed(db_session, make_legislation("LAW-A", "2003-01-01", articles={"1": "a1"}))
        await seed(db_session, make_legislation("LAW-B", "2010-01-01", articles={"1": "b1"}))
        await save_rels(db_session, [repeals("LAW-B", "LAW-A", "1")])

        article = await recompute_article_status(db_session, "LAW-A", "1")

        assert article.is_in_force is False
        assert article.status == ArticleStatus.REPEALED

    async def test_syncs_status_to_vectorstore(self, db_session, _no_vectorstore_sync):
        await seed(db_session, make_legislation("LAW-A", "2003-01-01", articles={"1": "a1"}))
        await seed(db_session, make_legislation("LAW-B", "2010-01-01", articles={"1": "b1"}))
        await save_rels(db_session, [repeals("LAW-B", "LAW-A", "1")])

        await recompute_article_status(db_session, "LAW-A", "1")

        _no_vectorstore_sync.assert_called_with("LAW-A_article_1", "repealed", False)


# ── reconcile_after_ingest — both directions & ordering ───────────────────────-

class TestReconcileBothDirections:

    async def test_backward_new_law_repeals_existing(self, db_session):
        await seed(db_session, make_legislation("LAW-A", "2003-01-01", articles={"1": "a1", "2": "a2"}))
        law_b = make_legislation(
            "LAW-B", "2010-01-01", articles={"1": "b1"},
            relationships=[repeals("LAW-B", "LAW-A", "2")],
        )
        await seed(db_session, law_b)
        await save_rels(db_session, law_b.relationships)

        await reconcile_after_ingest(db_session, law_b)

        affected = await crud.get_article_row(db_session, "LAW-A", "2")
        untouched = await crud.get_article_row(db_session, "LAW-A", "1")
        assert affected.is_in_force is False
        assert untouched.is_in_force is True

    async def test_forward_law_ingested_before_the_one_that_repeals_it(self, db_session):
        # B (which repeals A) is ingested FIRST, while A does not yet exist.
        law_b = make_legislation(
            "LAW-B", "2010-01-01", articles={"1": "b1"},
            relationships=[repeals("LAW-B", "LAW-A", "2")],
        )
        await seed(db_session, law_b)
        await save_rels(db_session, law_b.relationships)
        await reconcile_after_ingest(db_session, law_b)  # no-op for A — A absent

        # A arrives later; reconciliation must pick up B's pending repeal.
        law_a = make_legislation("LAW-A", "2003-01-01", articles={"1": "a1", "2": "a2"})
        await seed(db_session, law_a)
        await reconcile_after_ingest(db_session, law_a)

        affected = await crud.get_article_row(db_session, "LAW-A", "2")
        assert affected.is_in_force is False
        assert affected.status == ArticleStatus.REPEALED

    async def test_whole_legislation_repeal_affects_all_articles(self, db_session):
        await seed(db_session, make_legislation(
            "LAW-A", "2003-01-01", articles={"1": "a1", "2": "a2", "3": "a3"}
        ))
        law_b = make_legislation(
            "LAW-B", "2010-01-01", articles={"1": "b1"},
            relationships=[repeals("LAW-B", "LAW-A", None)],  # whole legislation
        )
        await seed(db_session, law_b)
        await save_rels(db_session, law_b.relationships)

        await reconcile_after_ingest(db_session, law_b)

        for number in ("1", "2", "3"):
            article = await crud.get_article_row(db_session, "LAW-A", number)
            assert article.is_in_force is False

    async def test_repealing_a_repealer_does_not_revive_the_original(self, db_session):
        # A ← repealed by B(2010) ← repealed by C(2020). A must STAY repealed.
        await seed(db_session, make_legislation("LAW-A", "2000-01-01", articles={"1": "a1"}))
        law_b = make_legislation(
            "LAW-B", "2010-01-01", articles={"1": "b1"},
            relationships=[repeals("LAW-B", "LAW-A", "1")],
        )
        await seed(db_session, law_b)
        await save_rels(db_session, law_b.relationships)
        await reconcile_after_ingest(db_session, law_b)

        law_c = make_legislation(
            "LAW-C", "2020-01-01", articles={"1": "c1"},
            relationships=[repeals("LAW-C", "LAW-B", "1")],
        )
        await seed(db_session, law_c)
        await save_rels(db_session, law_c.relationships)
        await reconcile_after_ingest(db_session, law_c)

        article_a = await crud.get_article_row(db_session, "LAW-A", "1")
        article_b = await crud.get_article_row(db_session, "LAW-B", "1")
        assert article_a.is_in_force is False           # not revived
        assert article_a.status == ArticleStatus.REPEALED
        assert article_b.is_in_force is False           # B itself now repealed

    async def test_later_amendment_overrides_earlier_repeal(self, db_session):
        await seed(db_session, make_legislation("LAW-A", "2000-01-01", articles={"1": "a1"}))
        law_b = make_legislation(
            "LAW-B", "2010-01-01", articles={"1": "b1"},
            relationships=[repeals("LAW-B", "LAW-A", "1")],
        )
        await seed(db_session, law_b)
        await save_rels(db_session, law_b.relationships)

        law_c = make_legislation(
            "LAW-C", "2020-01-01", articles={"1": "c1"},
            relationships=[Relationship(
                type=RelationshipType.AMENDS,
                father_legislation="LAW-C",
                affected_legislation="LAW-A",
                affected_article="1",
                illustration="amends the article back into force",
            )],
        )
        await seed(db_session, law_c)
        await save_rels(db_session, law_c.relationships)
        await reconcile_after_ingest(db_session, law_c)

        article = await crud.get_article_row(db_session, "LAW-A", "1")
        assert article.is_in_force is True
        assert article.status == ArticleStatus.AMENDED