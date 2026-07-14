"""User-level AbstraK configuration and credential loading."""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError, field_validator

from abstrak.providers.manifests import (
    ENV_PATTERN,
    IDENTIFIER_PATTERN,
    ManifestBundle,
    ModelManifest,
    ProviderManifest,
)

CONFIG_ENV = "ABSTRAK_CONFIG"
AUTH_ENV = "ABSTRAK_AUTH"


class ConfigurationError(ValueError):
    """Raised when a user-level configuration file is missing or invalid."""


class ConfigProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: ProviderManifest
    model: ModelManifest

    def bundle(self) -> ManifestBundle:
        return ManifestBundle(provider=self.provider, model=self.model)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["config.v1"] = "config.v1"
    default_profile: str
    profiles: dict[str, ConfigProfile]

    @field_validator("default_profile")
    @classmethod
    def validate_default_profile_name(cls, value: str) -> str:
        if re.fullmatch(IDENTIFIER_PATTERN, value) is None:
            raise ValueError("default_profile must be a lowercase identifier")
        return value

    @field_validator("profiles")
    @classmethod
    def validate_profiles(cls, value: dict[str, ConfigProfile]) -> dict[str, ConfigProfile]:
        if not value:
            raise ValueError("profiles cannot be empty")
        invalid = [name for name in value if re.fullmatch(IDENTIFIER_PATTERN, name) is None]
        if invalid:
            raise ValueError("profile names must be lowercase identifiers")
        for profile in value.values():
            profile.bundle()
        return value

    def bundle(self, profile_name: str | None = None) -> ManifestBundle:
        selected = profile_name or self.default_profile
        try:
            profile = self.profiles[selected]
        except KeyError as error:
            available = ", ".join(sorted(self.profiles))
            raise ConfigurationError(
                f"unknown profile {selected!r}; available profiles: {available}"
            ) from error
        return profile.bundle()


class AuthStore(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["auth.v1"] = "auth.v1"
    environment: dict[str, SecretStr]

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, value: dict[str, SecretStr]) -> dict[str, SecretStr]:
        for name, secret in value.items():
            if re.fullmatch(ENV_PATTERN, name) is None:
                raise ValueError("authentication keys must be environment variable names")
            if not secret.get_secret_value():
                raise ValueError("authentication values cannot be empty")
        return value

    def reveal_environment(self) -> dict[str, str]:
        return {name: secret.get_secret_value() for name, secret in self.environment.items()}


def default_config_path() -> Path:
    return Path.home() / ".abstrak" / "config.yaml"


def default_auth_path() -> Path:
    return Path.home() / ".abstrak" / "auth.json"


def resolve_path(
    explicit: str | Path | None,
    *,
    environment_name: str,
    default: Path,
    environment: Mapping[str, str] | None = None,
) -> tuple[Path, bool]:
    values = os.environ if environment is None else environment
    configured = values.get(environment_name)
    selected = explicit if explicit is not None else configured or default
    return Path(selected).expanduser(), explicit is not None or bool(configured)


def _validation_summary(error: ValidationError) -> str:
    issues: list[str] = []
    for issue in error.errors(include_url=False, include_context=False, include_input=False):
        location = ".".join(str(segment) for segment in issue["loc"])
        issues.append(f"{location}: {issue['msg']} ({issue['type']})")
    return "; ".join(issues)


def load_app_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError(f"cannot read {config_path}: {error}") from error
    if not isinstance(payload, dict):
        raise ConfigurationError(f"{config_path} must contain one YAML mapping")
    try:
        config = AppConfig.model_validate(payload)
        config.bundle()
        return config
    except ValidationError as error:
        raise ConfigurationError(
            f"invalid {config_path}: {_validation_summary(error)}"
        ) from None
    except ValueError as error:
        raise ConfigurationError(f"invalid {config_path}: {error}") from error


def load_auth_store(path: str | Path, *, missing_ok: bool = False) -> AuthStore:
    auth_path = Path(path).expanduser()
    try:
        metadata = auth_path.stat()
    except FileNotFoundError:
        if missing_ok:
            return AuthStore(environment={})
        raise ConfigurationError(f"cannot read {auth_path}: file does not exist") from None
    except OSError as error:
        raise ConfigurationError(f"cannot inspect {auth_path}: {error}") from error

    if not stat.S_ISREG(metadata.st_mode):
        raise ConfigurationError(f"{auth_path} must be a regular file")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise ConfigurationError(f"{auth_path} must have permissions 0600 or stricter")

    try:
        payload = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"cannot read {auth_path}: invalid JSON") from error
    if not isinstance(payload, dict):
        raise ConfigurationError(f"invalid {auth_path}: authentication data must be one object")
    try:
        return AuthStore.model_validate(payload)
    except ValidationError:
        # Pydantic validation details may contain the submitted credential value.
        raise ConfigurationError(
            f"invalid {auth_path}: authentication schema validation failed"
        ) from None


def runtime_environment(
    auth: AuthStore,
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Merge disk credentials with non-empty process values taking precedence."""

    process_values = os.environ if environment is None else environment
    merged = auth.reveal_environment()
    merged.update({name: value for name, value in process_values.items() if value})
    return merged
