"""
JSON Schema Validation for IntentLog Configuration

Provides schema definitions and validation for:
- config.json (project configuration)
- intents.json (intent data)
- Branch files
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .logging import get_logger


# JSON Schema definitions
CONFIG_SCHEMA = {
    "type": "object",
    "required": ["project_name"],
    "properties": {
        "project_name": {
            "type": "string",
            "minLength": 1,
            "maxLength": 128,
            "pattern": "^[a-zA-Z0-9][a-zA-Z0-9._-]*$",
            "description": "Name of the IntentLog project"
        },
        "created_at": {
            "type": "string",
            "format": "date-time",
            "description": "ISO 8601 timestamp of project creation"
        },
        "current_branch": {
            "type": "string",
            "default": "main",
            "pattern": "^[a-zA-Z0-9][a-zA-Z0-9._-]*$",
            "description": "Currently active branch"
        },
        "version": {
            "type": "string",
            "pattern": "^\\d+\\.\\d+\\.\\d+$",
            "description": "IntentLog version"
        },
        "llm": {
            "type": "object",
            "properties": {
                "provider": {
                    "type": "string",
                    "enum": ["", "openai", "anthropic", "ollama"],
                    "description": "LLM provider"
                },
                "model": {
                    "type": "string",
                    "description": "Model name"
                },
                "api_key_env": {
                    "type": "string",
                    "description": "Environment variable for API key"
                },
                "embedding_provider": {
                    "type": "string",
                    "description": "Provider for embeddings"
                },
                "embedding_model": {
                    "type": "string",
                    "description": "Embedding model name"
                },
                "base_url": {
                    "type": "string",
                    "format": "uri",
                    "description": "Custom API endpoint"
                }
            },
            "additionalProperties": False
        }
    },
    "additionalProperties": False
}

INTENT_SCHEMA = {
    "type": "object",
    "required": ["intent_id", "intent_name", "intent_reasoning", "timestamp"],
    "properties": {
        "intent_id": {
            "type": "string",
            "pattern": "^[a-f0-9-]{36}$",
            "description": "UUID of the intent"
        },
        "intent_name": {
            "type": "string",
            "minLength": 1,
            "maxLength": 256,
            "description": "Name/title of the intent"
        },
        "intent_reasoning": {
            "type": "string",
            "minLength": 1,
            "description": "Reasoning/explanation"
        },
        "timestamp": {
            "type": "string",
            "format": "date-time",
            "description": "ISO 8601 timestamp"
        },
        "parent_intent_id": {
            "type": ["string", "null"],
            "description": "Parent intent UUID"
        },
        "metadata": {
            "type": "object",
            "description": "Additional metadata"
        },
        "hash": {
            "type": "string",
            "pattern": "^[a-f0-9]+$",
            "description": "Content hash"
        },
        "chain_hash": {
            "type": "string",
            "pattern": "^[a-f0-9]+$",
            "description": "Chain hash"
        },
        "prev_hash": {
            "type": "string",
            "pattern": "^[a-f0-9]+$",
            "description": "Previous hash in chain"
        },
        "sequence": {
            "type": "integer",
            "minimum": 0,
            "description": "Sequence number in chain"
        },
        "signature": {
            "type": "object",
            "description": "Cryptographic signature"
        }
    },
    "additionalProperties": True
}

INTENTS_FILE_SCHEMA = {
    "type": "object",
    "required": ["branch", "intents"],
    "properties": {
        "branch": {
            "type": "string",
            "pattern": "^[a-zA-Z0-9][a-zA-Z0-9._-]*$",
            "description": "Branch name"
        },
        "intents": {
            "type": "array",
            "items": INTENT_SCHEMA,
            "description": "List of intents"
        },
        "created_from": {
            "type": "string",
            "description": "Parent branch name"
        },
        "created_at": {
            "type": "string",
            "format": "date-time",
            "description": "Branch creation timestamp"
        },
        "chain_version": {
            "type": "string",
            "description": "Chain format version"
        },
        "root_hash": {
            "type": "string",
            "pattern": "^[a-f0-9]+$",
            "description": "Merkle root hash"
        }
    },
    "additionalProperties": False
}


@dataclass
class ValidationError:
    """Represents a single validation error."""
    path: str
    message: str
    value: Any = None

    def __str__(self) -> str:
        if self.value is not None:
            return f"{self.path}: {self.message} (got: {self.value!r})"
        return f"{self.path}: {self.message}"


@dataclass
class ValidationResult:
    """Result of schema validation."""
    valid: bool
    errors: List[ValidationError]

    @property
    def error_messages(self) -> List[str]:
        return [str(e) for e in self.errors]

    def __bool__(self) -> bool:
        return self.valid


class SchemaValidator:
    """
    JSON Schema validator for IntentLog configuration files.

    Uses a lightweight validation approach without external dependencies.
    """

    def __init__(self):
        self.schemas = {
            "config": CONFIG_SCHEMA,
            "intent": INTENT_SCHEMA,
            "intents_file": INTENTS_FILE_SCHEMA,
        }

    def validate(
        self,
        data: Dict[str, Any],
        schema_name: str,
    ) -> ValidationResult:
        """
        Validate data against a named schema.

        Args:
            data: Data to validate
            schema_name: Name of schema ("config", "intent", "intents_file")

        Returns:
            ValidationResult with errors if any
        """
        if schema_name not in self.schemas:
            return ValidationResult(
                valid=False,
                errors=[ValidationError("", f"Unknown schema: {schema_name}")]
            )

        schema = self.schemas[schema_name]
        errors = self._validate_object(data, schema, "")
        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def _validate_object(
        self,
        data: Any,
        schema: Dict[str, Any],
        path: str,
    ) -> List[ValidationError]:
        """Validate an object against a schema."""
        errors = []

        # Check type
        expected_type = schema.get("type")
        if expected_type:
            if not self._check_type(data, expected_type):
                errors.append(ValidationError(
                    path or "root",
                    f"Expected type {expected_type}",
                    type(data).__name__
                ))
                return errors  # Can't continue validation if type is wrong

        # Object-specific validation
        if expected_type == "object" and isinstance(data, dict):
            # Check required fields
            for field in schema.get("required", []):
                if field not in data:
                    errors.append(ValidationError(
                        f"{path}.{field}" if path else field,
                        "Required field missing"
                    ))

            # Validate properties
            properties = schema.get("properties", {})
            for key, value in data.items():
                field_path = f"{path}.{key}" if path else key
                if key in properties:
                    errors.extend(self._validate_object(
                        value, properties[key], field_path
                    ))
                elif schema.get("additionalProperties") is False:
                    errors.append(ValidationError(
                        field_path,
                        "Additional property not allowed"
                    ))

        # Array validation
        elif expected_type == "array" and isinstance(data, list):
            items_schema = schema.get("items")
            if items_schema:
                for i, item in enumerate(data):
                    item_path = f"{path}[{i}]"
                    errors.extend(self._validate_object(
                        item, items_schema, item_path
                    ))

        # String validation
        elif expected_type == "string" and isinstance(data, str):
            # minLength
            min_len = schema.get("minLength")
            if min_len is not None and len(data) < min_len:
                errors.append(ValidationError(
                    path,
                    f"String too short (min {min_len})",
                    len(data)
                ))

            # maxLength
            max_len = schema.get("maxLength")
            if max_len is not None and len(data) > max_len:
                errors.append(ValidationError(
                    path,
                    f"String too long (max {max_len})",
                    len(data)
                ))

            # pattern
            pattern = schema.get("pattern")
            if pattern:
                import re
                if not re.match(pattern, data):
                    errors.append(ValidationError(
                        path,
                        f"Does not match pattern {pattern}",
                        data[:50] if len(data) > 50 else data
                    ))

            # enum
            enum = schema.get("enum")
            if enum and data not in enum:
                errors.append(ValidationError(
                    path,
                    f"Must be one of: {enum}",
                    data
                ))

            # format
            fmt = schema.get("format")
            if fmt == "date-time":
                try:
                    # Try parsing ISO format
                    datetime.fromisoformat(data.replace("Z", "+00:00"))
                except ValueError:
                    errors.append(ValidationError(
                        path,
                        "Invalid date-time format",
                        data
                    ))

        # Integer validation
        elif expected_type == "integer" and isinstance(data, int):
            minimum = schema.get("minimum")
            if minimum is not None and data < minimum:
                errors.append(ValidationError(
                    path,
                    f"Value below minimum ({minimum})",
                    data
                ))

        return errors

    def _check_type(self, data: Any, expected: Union[str, List[str]]) -> bool:
        """Check if data matches expected type(s)."""
        if isinstance(expected, list):
            return any(self._check_type(data, t) for t in expected)

        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }

        expected_class = type_map.get(expected)
        if expected_class:
            return isinstance(data, expected_class)
        return True


def validate_config_file(path: Path) -> ValidationResult:
    """
    Validate a config.json file.

    Args:
        path: Path to config.json

    Returns:
        ValidationResult
    """
    logger = get_logger()

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return ValidationResult(
            valid=False,
            errors=[ValidationError("", f"Invalid JSON: {e}")]
        )
    except FileNotFoundError:
        return ValidationResult(
            valid=False,
            errors=[ValidationError("", f"File not found: {path}")]
        )

    validator = SchemaValidator()
    result = validator.validate(data, "config")

    if not result.valid:
        logger.warning(f"Config validation failed: {path}")
        for error in result.errors:
            logger.debug(f"  {error}")

    return result


def validate_intents_file(path: Path) -> ValidationResult:
    """
    Validate an intents.json or branch file.

    Args:
        path: Path to intents file

    Returns:
        ValidationResult
    """
    logger = get_logger()

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return ValidationResult(
            valid=False,
            errors=[ValidationError("", f"Invalid JSON: {e}")]
        )
    except FileNotFoundError:
        return ValidationResult(
            valid=False,
            errors=[ValidationError("", f"File not found: {path}")]
        )

    validator = SchemaValidator()
    result = validator.validate(data, "intents_file")

    if not result.valid:
        logger.warning(f"Intents file validation failed: {path}")
        for error in result.errors:
            logger.debug(f"  {error}")

    return result


def validate_project(project_path: Path) -> Tuple[bool, List[str]]:
    """
    Validate all configuration files in a project.

    Args:
        project_path: Path to project root

    Returns:
        Tuple of (all_valid, list_of_issues)
    """
    from .storage import INTENTLOG_DIR, CONFIG_FILE, INTENTS_FILE, BRANCHES_DIR

    issues = []
    intentlog_dir = project_path / INTENTLOG_DIR

    if not intentlog_dir.is_dir():
        return False, ["No .intentlog directory found"]

    # Validate config.json
    config_path = intentlog_dir / CONFIG_FILE
    if config_path.exists():
        result = validate_config_file(config_path)
        if not result.valid:
            issues.extend([f"config.json: {e}" for e in result.error_messages])
    else:
        issues.append("config.json: File not found")

    # Validate main intents.json
    intents_path = intentlog_dir / INTENTS_FILE
    if intents_path.exists():
        result = validate_intents_file(intents_path)
        if not result.valid:
            issues.extend([f"intents.json: {e}" for e in result.error_messages])

    # Validate branch files
    branches_dir = intentlog_dir / BRANCHES_DIR
    if branches_dir.is_dir():
        for branch_file in branches_dir.glob("*.json"):
            result = validate_intents_file(branch_file)
            if not result.valid:
                issues.extend([f"{branch_file.name}: {e}" for e in result.error_messages])

    return len(issues) == 0, issues


# Convenience function for CLI
def check_project_health(project_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Check the health of an IntentLog project.

    Args:
        project_path: Path to project root (default: current directory)

    Returns:
        Dict with health check results
    """
    from .storage import find_project_root, IntentLogStorage

    if project_path is None:
        project_path = find_project_root()
        if project_path is None:
            return {
                "healthy": False,
                "project_found": False,
                "issues": ["No IntentLog project found"],
            }

    storage = IntentLogStorage(project_path)

    results = {
        "healthy": True,
        "project_found": True,
        "project_path": str(project_path),
        "issues": [],
        "checks": {},
    }

    # Schema validation
    valid, issues = validate_project(project_path)
    results["checks"]["schema_validation"] = {
        "passed": valid,
        "issues": issues,
    }
    if not valid:
        results["healthy"] = False
        results["issues"].extend(issues)

    # Chain verification
    try:
        chain_result = storage.verify_chain()
        results["checks"]["chain_integrity"] = {
            "passed": chain_result.valid,
            "issues": [chain_result.error] if chain_result.error else [],
        }
        if not chain_result.valid:
            results["healthy"] = False
            results["issues"].append(f"Chain integrity: {chain_result.error}")
    except Exception as e:
        results["checks"]["chain_integrity"] = {
            "passed": False,
            "issues": [str(e)],
        }

    # Count intents
    try:
        intents = storage.load_intents()
        results["checks"]["intent_count"] = {
            "passed": True,
            "count": len(intents),
        }
    except Exception as e:
        results["checks"]["intent_count"] = {
            "passed": False,
            "error": str(e),
        }

    return results


__all__ = [
    "CONFIG_SCHEMA",
    "INTENT_SCHEMA",
    "INTENTS_FILE_SCHEMA",
    "ValidationError",
    "ValidationResult",
    "SchemaValidator",
    "validate_config_file",
    "validate_intents_file",
    "validate_project",
    "check_project_health",
]
