"""Typed, bounded acceptance criteria for a single agent run.

Criteria are supplied by the runtime or caller.  Model-proposed criteria may be
represented, but are never trusted by the verification gate.
"""

import hashlib
import json
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


# --- declarative criteria documents (the CLI's --criteria input) -------------------------------------------------

MAX_DOCUMENT_BYTES = 64 * 1024
MAX_RESUME_ID_CHARS = 64
# job_state is not declarable: its job id exists only at run time.
DOCUMENT_CRITERION_PARAMETERS = {
    CriterionType.FILE_EXISTS.value: {"path", "type"},
    CriterionType.FILE_CONTENT.value: {"path", "contains"},
    CriterionType.COMMAND_EXIT.value: {"command", "timeout_seconds"},
}
_DOCUMENT_FIELDS = {"criterion_id", "criterion_type", "expected", "parameters", "description"}
_REQUIRED_FIELDS = {"criterion_id", "criterion_type", "expected", "parameters"}
_TIMEOUT_RANGE = (1, 600)


class CriterionDocumentError(ValueError):
    """A criteria document cannot be used. Carries every problem found, each with its location."""

    def __init__(self, problems: List[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


def _no_duplicate_keys(pairs):
    seen = {}
    for key, value in pairs:
        if key in seen:
            raise CriterionDocumentError([f"duplicate key {key!r}; a key may appear once"])
        seen[key] = value
    return seen


def _check_parameters(where: str, kind: str, expected: Any, params: Any) -> List[str]:
    problems = []
    if not isinstance(params, dict):
        return [f"{where}.parameters must be an object"]
    unknown = sorted(set(params) - DOCUMENT_CRITERION_PARAMETERS[kind])
    if unknown:
        problems.append(f"{where}.parameters has unknown fields {unknown}; {kind} accepts "
                        f"{sorted(DOCUMENT_CRITERION_PARAMETERS[kind])}")
    if kind in ("file_exists", "file_content"):
        path = params.get("path")
        if not isinstance(path, str) or not path:
            problems.append(f"{where}.parameters.path must be a non-empty string")
    if kind == "file_exists":
        if params.get("type", "file") not in ("file", "directory"):
            problems.append(f"{where}.parameters.type must be \"file\" or \"directory\"")
        if not isinstance(expected, bool):
            problems.append(f"{where}.expected must be true (present) or false (absent) for file_exists")
    if kind == "file_content":
        contains = params.get("contains")
        if not isinstance(contains, str) or not contains:
            problems.append(f"{where}.parameters.contains must be a non-empty string")
        if expected is not True:
            problems.append(f"{where}.expected must be true for file_content (the text must be present)")
    if kind == "command_exit":
        command = params.get("command")
        if not isinstance(command, str) or not command.strip():
            problems.append(f"{where}.parameters.command must be a non-empty string")
        timeout = params.get("timeout_seconds")
        if timeout is not None and (not isinstance(timeout, int) or isinstance(timeout, bool)
                                    or not _TIMEOUT_RANGE[0] <= timeout <= _TIMEOUT_RANGE[1]):
            problems.append(f"{where}.parameters.timeout_seconds must be an integer from {_TIMEOUT_RANGE[0]} "
                            f"to {_TIMEOUT_RANGE[1]}")
        if not isinstance(expected, int) or isinstance(expected, bool):
            problems.append(f"{where}.expected must be the expected exit code (an integer) for command_exit")
    return problems


def parse_criteria_document(text: str, max_criteria: int) -> List[AcceptanceCriterion]:
    """Parse and validate a criteria document. Nothing is defaulted or reinterpreted: every problem is collected, and
    the document is rejected as a whole if any exists. Criteria produced here are trusted: origin is always "user"."""
    if len(text.encode("utf-8")) > MAX_DOCUMENT_BYTES:
        raise CriterionDocumentError([f"the document is larger than {MAX_DOCUMENT_BYTES} bytes"])
    try:
        document = json.loads(text, object_pairs_hook=_no_duplicate_keys)
    except CriterionDocumentError:
        raise
    except ValueError as exc:
        raise CriterionDocumentError([f"not valid JSON: {exc}"]) from exc
    if not isinstance(document, dict) or set(document) != {"criteria"}:
        raise CriterionDocumentError(['the document must be a JSON object whose only key is "criteria"'])
    items = document["criteria"]
    if not isinstance(items, list) or not items:
        raise CriterionDocumentError(['"criteria" must be a non-empty list'])
    if len(items) > max_criteria:
        raise CriterionDocumentError([f"{len(items)} criteria; at most {max_criteria} can be checked. Extra criteria "
                                      "would be recorded as not checked, so the document is refused instead"])

    problems: List[str] = []
    parsed: List[AcceptanceCriterion] = []
    ids: set = set()
    for index, item in enumerate(items):
        where = f"criteria[{index}]"
        if not isinstance(item, dict):
            problems.append(f"{where} must be an object")
            continue
        if "run_id" in item or "origin" in item:
            problems.append(f"{where}: run_id and origin are set by the runtime and cannot be declared in a file")
        unknown = sorted(set(item) - _DOCUMENT_FIELDS - {"run_id", "origin"})
        if unknown:
            problems.append(f"{where}: unknown fields {unknown}")
        missing = sorted(_REQUIRED_FIELDS - set(item))
        if missing:
            problems.append(f"{where}: missing fields {missing}")
            continue
        criterion_id, kind = item["criterion_id"], item["criterion_type"]
        if not isinstance(criterion_id, str) or not criterion_id:
            problems.append(f"{where}: criterion_id must be a non-empty string")
            continue
        if criterion_id in ids:
            problems.append(f"{where}: criterion_id {criterion_id!r} is used more than once")
        ids.add(criterion_id)
        if not isinstance(kind, str):
            problems.append(f"{where}: criterion_type must be a string")
            continue
        if kind == CriterionType.JOB_STATE.value:
            problems.append(f"{where}: job_state cannot be declared in a file, because its job id exists only at run time")
            continue
        if kind not in DOCUMENT_CRITERION_PARAMETERS:
            problems.append(f"{where}: unsupported criterion_type {kind!r}; supported: "
                            f"{sorted(DOCUMENT_CRITERION_PARAMETERS)}")
            continue
        check = _check_parameters(where, kind, item["expected"], item["parameters"])
        if check:
            problems.extend(check)
            continue
        try:
            parsed.append(AcceptanceCriterion(criterion_id=criterion_id, run_id="", criterion_type=kind,
                                              expected=item["expected"], origin=CriterionOrigin.USER.value,
                                              description=item.get("description", ""),
                                              parameters=dict(item["parameters"])))
        except CriterionError as exc:
            problems.append(f"{where}: {exc}")
    if problems:
        raise CriterionDocumentError(problems)
    return parsed


def _content(criterion: AcceptanceCriterion) -> Dict[str, Any]:
    """What a criterion checks, without its identity: the criterion's type, expectation, parameters and origin."""
    return {"criterion_type": criterion.criterion_type, "expected": criterion.expected, "origin": criterion.origin,
            "description": criterion.description, "parameters": criterion.parameters}


def contract_of(criteria: List[AcceptanceCriterion]) -> List[AcceptanceCriterion]:
    """The trusted contract a session carries: the criteria of its most recent run that owns criteria, unbound.
    Criteria are appended per run, so the last criterion's run is the latest group."""
    if not criteria:
        return []
    latest = criteria[-1].run_id
    return [item.bind("") for item in criteria if item.run_id == latest]


def contract_fingerprint(contract: List[AcceptanceCriterion]) -> str:
    """A short digest of the contract's content, shown to the operator so a resumed contract can be checked by eye."""
    text = json.dumps([_content(item) for item in contract], sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def same_contract(supplied: List[AcceptanceCriterion], persisted: List[AcceptanceCriterion]) -> bool:
    """Equal content, in the same order. Criterion identifiers are labels and may be renamed by the operator."""
    return [_content(item) for item in supplied] == [_content(item) for item in persisted]


def resumed_copies(contract: List[AcceptanceCriterion], existing_ids: set, generation: int) -> List[AcceptanceCriterion]:
    """Unbound copies of a contract for a resumed run. Identifiers must stay unique within a session, so each copy is
    renamed `<id>:r<generation>`; the content, which is what is trusted, is unchanged."""
    copies = []
    for item in contract:
        new_id = f"{item.criterion_id}:r{generation}"
        if len(new_id) > MAX_RESUME_ID_CHARS or new_id in existing_ids:
            raise CriterionDocumentError([f"criterion {item.criterion_id!r} cannot be carried into a resumed run: "
                                          f"its resumed identifier would be {new_id!r}; shorten the identifier"])
        copies.append(AcceptanceCriterion(criterion_id=new_id, run_id="", criterion_type=item.criterion_type,
                                          expected=item.expected, origin=item.origin, description=item.description,
                                          parameters=dict(item.parameters)))
    return copies
