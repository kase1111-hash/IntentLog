"""
Input Validation for IntentLog

Provides security-focused input validation to prevent:
- Path traversal attacks
- Injection vulnerabilities
- Malformed input handling
"""

import re
import os
from pathlib import Path
from typing import Optional, List


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


class PathTraversalError(ValidationError):
    """Raised when path traversal is detected."""
    pass


class InvalidNameError(ValidationError):
    """Raised when a name contains invalid characters."""
    pass


# Allowed characters for names (conservative whitelist)
NAME_PATTERN = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_\-\.]*$')
NAME_MAX_LENGTH = 64

# Dangerous path components
DANGEROUS_PATH_COMPONENTS = {'..', '~', '$'}


def validate_name(
    name: str,
    field_name: str = "name",
    max_length: int = NAME_MAX_LENGTH,
    allow_dots: bool = True,
) -> str:
    """
    Validate a name (branch name, project name, key name, etc.).

    Args:
        name: The name to validate
        field_name: Name of the field for error messages
        max_length: Maximum allowed length
        allow_dots: Whether to allow dots in the name

    Returns:
        The validated name (stripped)

    Raises:
        InvalidNameError: If name is invalid
    """
    if not name:
        raise InvalidNameError(f"{field_name} cannot be empty")

    name = name.strip()

    if len(name) > max_length:
        raise InvalidNameError(
            f"{field_name} too long: {len(name)} > {max_length} characters"
        )

    if not allow_dots:
        if '.' in name:
            raise InvalidNameError(f"{field_name} cannot contain dots")

    if not NAME_PATTERN.match(name):
        raise InvalidNameError(
            f"{field_name} contains invalid characters. "
            f"Use only letters, numbers, underscores, hyphens, and dots."
        )

    # Additional safety checks
    if name.startswith('.') or name.startswith('-'):
        raise InvalidNameError(f"{field_name} cannot start with '.' or '-'")

    if '..' in name:
        raise InvalidNameError(f"{field_name} cannot contain '..'")

    return name


def validate_branch_name(name: str) -> str:
    """
    Validate a branch name.

    Args:
        name: Branch name to validate

    Returns:
        Validated branch name

    Raises:
        InvalidNameError: If branch name is invalid
    """
    return validate_name(name, "Branch name", max_length=64)


def validate_project_name(name: str) -> str:
    """
    Validate a project name.

    Args:
        name: Project name to validate

    Returns:
        Validated project name

    Raises:
        InvalidNameError: If project name is invalid
    """
    return validate_name(name, "Project name", max_length=64)


def validate_key_name(name: str) -> str:
    """
    Validate a cryptographic key name.

    Args:
        name: Key name to validate

    Returns:
        Validated key name

    Raises:
        InvalidNameError: If key name is invalid
    """
    return validate_name(name, "Key name", max_length=32, allow_dots=False)


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize a filename by removing dangerous characters.

    Args:
        filename: The filename to sanitize
        max_length: Maximum allowed length

    Returns:
        Sanitized filename
    """
    # Remove null bytes
    filename = filename.replace('\0', '')

    # Replace path separators
    filename = filename.replace('/', '_').replace('\\', '_')

    # Remove dangerous prefixes
    while filename.startswith('.') or filename.startswith('-'):
        filename = filename[1:]

    # Limit length
    if len(filename) > max_length:
        filename = filename[:max_length]

    return filename or "unnamed"


def validate_path_within_directory(
    path: Path,
    base_directory: Path,
    allow_symlinks: bool = False,
) -> Path:
    """
    Validate that a path is within a base directory.

    Prevents path traversal attacks by ensuring the resolved path
    stays within the allowed directory.

    Args:
        path: Path to validate
        base_directory: Directory that path must be within
        allow_symlinks: Whether to allow symlinks (default False)

    Returns:
        The resolved, validated path

    Raises:
        PathTraversalError: If path escapes base directory
    """
    try:
        # Resolve both paths to absolute
        resolved_path = path.resolve()
        resolved_base = base_directory.resolve()

        # Check if path is relative to base
        resolved_path.relative_to(resolved_base)

        # Check for symlinks if not allowed
        if not allow_symlinks:
            # Walk up the path checking for symlinks
            check_path = resolved_path
            while check_path != resolved_base:
                if check_path.is_symlink():
                    raise PathTraversalError(
                        f"Symlinks not allowed: {check_path}"
                    )
                parent = check_path.parent
                if parent == check_path:
                    break
                check_path = parent

        return resolved_path

    except ValueError:
        raise PathTraversalError(
            f"Path '{path}' escapes base directory '{base_directory}'"
        )


def validate_file_path(
    file_path: str,
    base_directory: Path,
    must_exist: bool = False,
) -> Path:
    """
    Validate a file path string.

    Args:
        file_path: File path string to validate
        base_directory: Directory path must be within
        must_exist: Whether the file must exist

    Returns:
        Validated Path object

    Raises:
        PathTraversalError: If path is invalid or escapes directory
        FileNotFoundError: If must_exist=True and file doesn't exist
    """
    # Check for dangerous components
    for component in DANGEROUS_PATH_COMPONENTS:
        if component in file_path:
            raise PathTraversalError(
                f"Path contains dangerous component: '{component}'"
            )

    path = Path(file_path)
    validated = validate_path_within_directory(path, base_directory)

    if must_exist and not validated.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return validated


def validate_intent_reasoning(
    reasoning: str,
    min_length: int = 1,
    max_length: int = 100000,
) -> str:
    """
    Validate intent reasoning text.

    Args:
        reasoning: The reasoning text to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length

    Returns:
        Validated reasoning text

    Raises:
        ValidationError: If reasoning is invalid
    """
    if not reasoning:
        raise ValidationError("Reasoning cannot be empty")

    reasoning = reasoning.strip()

    if len(reasoning) < min_length:
        raise ValidationError(
            f"Reasoning too short: {len(reasoning)} < {min_length} characters"
        )

    if len(reasoning) > max_length:
        raise ValidationError(
            f"Reasoning too long: {len(reasoning)} > {max_length} characters"
        )

    # Check for null bytes (could be injection attempt)
    if '\0' in reasoning:
        raise ValidationError("Reasoning contains invalid characters")

    return reasoning


def validate_metadata(metadata: dict, max_depth: int = 5) -> dict:
    """
    Validate metadata dictionary.

    Args:
        metadata: Metadata dict to validate
        max_depth: Maximum nesting depth

    Returns:
        Validated metadata

    Raises:
        ValidationError: If metadata is invalid
    """
    def check_depth(obj, depth=0):
        if depth > max_depth:
            raise ValidationError(
                f"Metadata too deeply nested: depth > {max_depth}"
            )

        if isinstance(obj, dict):
            for key, value in obj.items():
                if not isinstance(key, str):
                    raise ValidationError(
                        f"Metadata keys must be strings, got: {type(key)}"
                    )
                check_depth(value, depth + 1)
        elif isinstance(obj, list):
            for item in obj:
                check_depth(item, depth + 1)

    check_depth(metadata)
    return metadata


# Convenience exports
__all__ = [
    "ValidationError",
    "PathTraversalError",
    "InvalidNameError",
    "validate_name",
    "validate_branch_name",
    "validate_project_name",
    "validate_key_name",
    "sanitize_filename",
    "validate_path_within_directory",
    "validate_file_path",
    "validate_intent_reasoning",
    "validate_metadata",
]
