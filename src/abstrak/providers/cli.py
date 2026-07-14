"""CLI for offline manifest validation and explicitly opted-in live smoke probes."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from pydantic import ValidationError

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


def _load_bundle(provider_path: str, model_path: str) -> ManifestBundle:
    return ManifestBundle(
        provider=load_manifest(provider_path, ProviderManifest),
        model=load_manifest(model_path, ModelManifest),
    )


def _emit(value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "smoke"):
        command = subparsers.add_parser(name)
        command.add_argument("--provider", required=True, help="provider manifest YAML")
        command.add_argument("--model", required=True, help="model manifest YAML")
    smoke = subparsers.choices["smoke"]
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


def _smoke(bundle: ManifestBundle, artifact_root: str) -> int:
    if not bundle.model.allow_live_probe:
        raise ManifestLoadError(
            "model manifest must set allow_live_probe: true before a billable smoke request"
        )
    client = ProviderClient(bundle)
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
        bundle = _load_bundle(arguments.provider, arguments.model)
        if arguments.command == "validate":
            return _validate(bundle)
        if not arguments.live:
            parser.error("smoke requires --live because it performs a billable network request")
        return _smoke(bundle, arguments.artifact_root)
    except (
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
