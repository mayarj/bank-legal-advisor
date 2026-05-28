import uuid
from typing import Optional

from sqlalchemy import Enum as SAEnum, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base
from src.db.schemas import RelationshipType


class Relationship(Base):
    __tablename__ = "relationships"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    type: Mapped[RelationshipType] = mapped_column(SAEnum(RelationshipType))
    father_legislation: Mapped[str] = mapped_column(String)
    father_article: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    affected_legislation: Mapped[str] = mapped_column(String)
    affected_article: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    illustration: Mapped[str] = mapped_column(String(150))

    __table_args__ = (
        Index("ix_father_legislation", "father_legislation"),
        Index("ix_affected_legislation", "affected_legislation"),
    )
