# Experiment Manifests

Only declarative, reviewable inputs belong here. Secrets are injected through
the runner environment and must never appear in a manifest.

The planned manifest groups are:

- `providers/`: endpoint behavior, timeout, retry, and usage-accounting rules.
- `models/`: exact model identifiers, decoding settings, and context limits.
- `hardware/`: GPU SKU, host, driver, CUDA, clocks, and isolation policy.
- `targets/`: compiler version, documentation pack, toolchain, and allowed libraries.
- `tasks/`: public specification, development inputs, and qualification contract.
- `studies/`: frozen task-model-target matrix, budgets, seeds, and stop rules.

Directories and schemas will be added with their conformance tests. Do not add
a production manifest until the corresponding adapter can validate it.

The `examples/` manifests are non-runnable templates. Copy them to an ignored
or private configuration location, replace every example identifier, and run
the offline `abstrak-provider validate` command before any live probe. A model
manifest becomes a production manifest only after its exact contents and live
conformance artifact have been reviewed.
