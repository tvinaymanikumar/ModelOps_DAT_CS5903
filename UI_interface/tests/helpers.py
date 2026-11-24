from config.task_config import TaskConfig, default_prompt
from validation.rules import rules_for_task
from validation.schema_validator import schema_from_fields, template_from_fields

CLAIM_FIELDS = [
    {"key": "name", "type": "string", "required": True, "description": "Claimant name"},
    {"key": "age", "type": "integer", "required": True, "description": ""},
    {"key": "policy_number", "type": "string", "required": True, "description": ""},
    {"key": "claim_amount", "type": "number", "required": False, "description": ""},
    {"key": "date_of_incident", "type": "date", "required": False, "description": ""},
]


def make_task(mode: str, fields=None, **kw) -> TaskConfig:
    fields = fields if fields is not None else (CLAIM_FIELDS if mode == "schema_based" else [])
    extra = {}
    if mode == "schema_based":
        extra = dict(fields=fields, template=template_from_fields(fields), schema=schema_from_fields(fields))
    else:
        extra = dict(strict_keys=False)
    extra["rules"] = rules_for_task(mode, fields)
    extra.update(kw)
    return TaskConfig(mode=mode, name="Test task", **extra)


def prompt_for(mode: str) -> str:
    return default_prompt(mode)
