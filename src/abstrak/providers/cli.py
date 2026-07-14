"""CLI for offline manifest validation and explicitly opted-in live smoke probes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence

from pydantic import ValidationError

from abstrak.config import (
    AUTH_ENV,
    CONFIG_ENV,
    ConfigurationError,
    default_auth_path,
    default_config_path,
    load_app_config,
    load_auth_store,
    resolve_path,
    runtime_environment,
)
from abstrak.providers.artifacts import ArtifactError
from abstrak.providers.client import ProviderClient, ProviderConfigurationError
from abstrak.providers.conformance import LiveProbeConfigurationError, run_live_probe
from abstrak.providers.manifests import (
    ManifestBundle,
    ManifestLoadError,
    MissingEnvironmentError,
    ModelManifest,
    ProviderManifest,
    load_manifest,
    manifest_sha256,
    required_environment,
)

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_TRANSPORT = 3
EXIT_CONTRACT = 4
EXIT_ARTIFACT = 5


def _load_bundle(arguments: argparse.Namespace) -> ManifestBundle:
    has_provider = arguments.provider is not None
    has_model = arguments.model is not None
    if has_provider != has_model:
        raise ConfigurationError("--provider and --model must be supplied together")
    if has_provider:
        if arguments.config is not None or arguments.profile is not None:
            raise ConfigurationError(
                "--provider/--model cannot be combined with --config or --profile"
            )
        return ManifestBundle(
            provider=load_manifest(arguments.provider, ProviderManifest),
            model=load_manifest(arguments.model, ModelManifest),
        )

    config_path, _ = resolve_path(
        arguments.config,
        environment_name=CONFIG_ENV,
        default=default_config_path(),
    )
    return load_app_config(config_path).bundle(arguments.profile)


def _emit(value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "smoke"):
        command = subparsers.add_parser(name)
        command.add_argument(
            "--config",
            help=f"user configuration YAML (default: ${CONFIG_ENV} or ~/.abstrak/config.yaml)",
        )
        command.add_argument("--profile", help="profile from the user configuration")
        command.add_argument("--provider", help="standalone provider manifest YAML")
        command.add_argument("--model", help="standalone model manifest YAML")
    smoke = subparsers.choices["smoke"]
    smoke.add_argument(
        "--auth",
        help=f"credential JSON (default: ${AUTH_ENV} or ~/.abstrak/auth.json)",
    )
    smoke.add_argument(
        "--live",
        action="store_true",
        help="acknowledge that this command performs one billable network request",
    )
    smoke.add_argument(
        "--artifact-root",
        default="artifacts/provider-conformance",
        help="ignored-by-Git directory for the immutable run bundle",
    )
    return parser


def _validate(bundle: ManifestBundle) -> int:
    _emit(
        {
            "status": "valid",
            "provider_id": bundle.provider.id,
            "model_id": bundle.model.id,
            "provider_manifest_sha256": manifest_sha256(bundle.provider),
            "model_manifest_sha256": manifest_sha256(bundle.model),
            "required_environment": required_environment(bundle.provider),
        }
    )
    return EXIT_OK


def _smoke(
    bundle: ManifestBundle, artifact_root: str, environment: dict[str, str] | None = None
) -> int:
    if not bundle.model.allow_live_probe:
        raise ManifestLoadError(
            "model manifest must set allow_live_probe: true before a billable smoke request"
        )
    client = ProviderClient(bundle, environment=environment)
    report, store = run_live_probe(client, artifact_root=artifact_root)
    _emit(
        {
            "status": report.status,
            "provider_id": report.provider_id,
            "model_id": report.model_id,
            "transport_ready": report.transport_ready,
            "action_protocol_ready": report.action_protocol_ready,
            "pilot_ready": report.pilot_ready,
            "artifact_directory": str(store.run_directory),
            "checks": [check.model_dump(mode="json") for check in report.checks],
        }
    )
    if report.error is not None:
        return EXIT_TRANSPORT if report.error.request_submitted else EXIT_CONFIG
    return EXIT_OK if report.status == "pass" else EXIT_CONTRACT


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        bundle = _load_bundle(arguments)
        if arguments.command == "validate":
            return _validate(bundle)
        if not arguments.live:
            parser.error("smoke requires --live because it performs a billable network request")
        auth_path, auth_was_configured = resolve_path(
            arguments.auth,
            environment_name=AUTH_ENV,
            default=default_auth_path(),
        )
        auth = load_auth_store(auth_path, missing_ok=not auth_was_configured)
        return _smoke(bundle, arguments.artifact_root, runtime_environment(auth, os.environ))
    except (
        ConfigurationError,
        LiveProbeConfigurationError,
        ManifestLoadError,
        MissingEnvironmentError,
        ProviderConfigurationError,
        ValidationError,
    ) as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return EXIT_CONFIG
    except ArtifactError as error:
        print(f"artifact error: {error}", file=sys.stderr)
        return EXIT_ARTIFACT
    except Exception as error:
        print(f"harness error: {type(error).__name__}", file=sys.stderr)
        return EXIT_ARTIFACT


if __name__ == "__main__":
    sys.exit(main())
