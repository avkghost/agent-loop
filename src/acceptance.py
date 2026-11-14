"""Typed, bounded acceptance criteria for a single agent run.

Criteria are supplied by the runtime or caller.  Model-proposed criteria may be
represented, but are never trusted by the verification gate.
"""

import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List


class CriterionType(str, Enum):
    FILE_EXISTS = "file_exists"
    FILE_CONTENT = "file_content"
    COMMAND_EXIT = "command_exit"
    JOB_STATE = "job_state"


class CriterionOrigin(str, Enum):
    USER = "user"
    TRUSTED = "trusted"
    MODEL = "model"


_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
MAX_DESCRIPTION = 300
MAX_PARAMETER_CHARS = 2000


class CriterionError(ValueError):
    """A criterion is malformed or cannot be safely trusted."""


@dataclass(frozen=True)
class AcceptanceCriterion:
    criterion_id: str
    run_id: str
    criterion_type: str
    expected: Any
    origin: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    @property
    def trusted(self) -> bool:
        return self.origin in (CriterionOrigin.USER.value, CriterionOrigin.TRUSTED.value)

    def __post_init__(self) -> None:
        if not isinstance(self.criterion_id, str) or not _ID.fullmatch(self.criterion_id):
            raise CriterionError("criterion_id must be 1-64 simple characters")
        if not isinstance(self.run_id, str):
            raise CriterionError("criterion run_id must be a string")
        try:
            CriterionType(self.criterion_type)
            CriterionOrigin(self.origin)
        except ValueError as exc:
            raise CriterionError("unknown criterion type or origin") from exc
        if not isinstance(self.description, str) or len(self.description) > MAX_DESCRIPTION:
            raise CriterionError("criterion description is too large")
        if not isinstance(self.parameters, dict):
            raise CriterionError("criterion parameters must be an object")
        if len(repr(self.parameters)) > MAX_PARAMETER_CHARS:
            raise CriterionError("criterion parameters are too large")
        if not isinstance(self.expected, (str, int, float, bool, type(None), dict, list)):
            raise CriterionError("criterion expected value is not serializable")

    def bind(self, run_id: str) -> "AcceptanceCriterion":
        return AcceptanceCriterion(self.criterion_id, run_id, self.criterion_type, self.expected,
                                   self.origin, self.description, dict(self.parameters))

    def to_dict(self) -> Dict[str, Any]:
        return {"criterion_id": self.criterion_id, "run_id": self.run_id,
                "criterion_type": self.criterion_type, "expected": self.expected,
                "origin": self.origin, "description": self.description,
                "parameters": self.parameters}

    @classmethod
    def from_dict(cls, data: Any) -> "AcceptanceCriterion":
        if not isinstance(data, dict):
            raise CriterionError("criterion must be an object")
        required = {"criterion_id", "run_id", "criterion_type", "expected", "origin", "description", "parameters"}
        if set(data) != required:
            raise CriterionError("criterion has missing or unexpected fields")
        return cls(**{key: data[key] for key in required})


def new_criterion_id() -> str:
    return "criterion_" + uuid.uuid4().hex


def validate_criteria(criteria: Any) -> List[AcceptanceCriterion]:
    if not isinstance(criteria, list):
        raise CriterionError("criteria must be a list")
    parsed = [AcceptanceCriterion.from_dict(item) for item in criteria]
    if any(not item.run_id for item in parsed):
        raise CriterionError("persisted criteria require an owning run_id")
    ids = [item.criterion_id for item in parsed]
    if len(ids) != len(set(ids)):
        raise CriterionError("criterion identifiers must be unique")
    return parsed
