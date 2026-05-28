from pydantic import BaseModel, Field
from datetime import date
from enum import Enum
from typing import Optional


class RelationshipType(str, Enum):
    AMENDS = "amends"
    REPEALS = "repeals"
    SUPERSEDES = "supersedes"
    REFERENCES = "references"
    IMPLEMENTS = "implements"
    CONFLICTS_WITH = "conflicts_with"


class LegislationStatus(str, Enum):
    ACTIVE = "active"
    REPEALED = "repealed"
    AMENDED = "amended"
    PENDING = "pending"
    DRAFT = "draft"


class Relationship(BaseModel):
    type: RelationshipType
    father_legislation: str
    father_article: Optional[str] = None
    affected_legislation: str
    affected_article: Optional[str] = None
    illustration: str = Field(max_length=150)


class Legislation(BaseModel):
    code: str
    date: date
    issuer: str
    subject: str = Field(max_length=200)
    status: LegislationStatus
    articles: dict[str, str] = Field(default_factory=dict)
    relationships: list[Relationship] = Field(default_factory=list)
