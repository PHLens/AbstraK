# Runtime Configuration and Experiment Manifests

AbstraK separates local runtime selection from versioned experiment inputs.
Secrets must never appear in a provider, model, or experiment manifest.

## Default local files

Provider commands use these files unless their paths are overridden:

- `~/.abstrak/config.yaml`: non-secret profile configuration;
- `~/.abstrak/auth.json`: local credential values for live requests.

`config.yaml` uses `config.v1`. A profile embeds one complete `provider.v1`
manifest and one complete `model.v1` manifest:

```yaml
schema_version: config.v1
default_profile: deepseek-v4-flash
profiles:
  deepseek-v4-flash:
    provider:
      schema_version: provider.v1
      id: deepseek-official
      adapter: litellm
      protocol: chat_completions
      litellm_provider: deepseek
      base_url_env: ABSTRAK_DEEPSEEK_BASE_URL
      api_key_env: ABSTRAK_DEEPSEEK_API_KEY
      timeout_seconds: 180
      retry:
        max_attempts: 1
      transport:
        stream: false
        candidates: 1
        allow_fallback: false
        allow_cache: false
        drop_unsupported_params: false
    model:
      schema_version: model.v1
      id: deepseek-v4-flash
      provider: deepseek-official
      api_model: deepseek/replace-with-provider-model-id
      model_id_policy: mutable_alias
      expected_returned_model: null
      interface: chat_completions
      allow_live_probe: false
      generation:
        max_completion_tokens: 128
        temperature: null
        top_p: null
        api_seed: null
        stop: []
        reasoning_effort: null
      output_contract: plain_json
      capabilities:
        usage_reporting: required
        returned_model: required
        system_messages: required
      pricing_ref: null
```

Replace all placeholder and policy values only after checking the endpoint's
actual contract. In particular, a rolling model alias must use
`model_id_policy: mutable_alias`; an exact identifier may use `exact` and must
set `expected_returned_model` to the exact identity returned by the endpoint.

`auth.json` uses `auth.v1` and contains only values that should be exposed as
environment variables to the provider transport:

```json
{
  "schema_version": "auth.v1",
  "environment": {
    "ABSTRAK_DEEPSEEK_BASE_URL": "https://api.deepseek.com/v1",
    "ABSTRAK_DEEPSEEK_API_KEY": "replace-with-local-secret"
  }
}
```

Keep `~/.abstrak/` private to the local account, store `auth.json` with mode
0600 or stricter, and never commit it. A non-empty process environment variable
wins over a value with the same name in the auth file. `abstrak-provider
validate` reads the configuration but not the auth file; `abstrak-provider
smoke --live` reads both.

Override the defaults with `--config`, `--auth`, and `--profile`. The legacy
`--provider` and `--model` standalone manifest arguments remain available when
both are passed, so existing scripts do not need an immediate migration.

## Versioned experiment inputs

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
or private configuration location, or embed their contents in a local profile.
Replace every example identifier and run the offline
`abstrak-provider validate` command before any live probe. A model manifest
becomes a production manifest only after its exact contents and live
conformance artifact have been reviewed.
