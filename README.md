# AbstraK

AbstraK is the controlled experimental harness for studying how an agent's
effective GPU-kernel authoring abstraction depends on model capability,
workload structure, hardware, and resource budget.

The repository starts deliberately small. The first milestone is environment
and provider conformance, not an optimization policy.

## Quick start

AbstraK is pinned to Python 3.10 to share one runtime with the pinned
KernelBench evaluator. With `uv`, the tracked `.python-version` selects the
compatible interpreter automatically.

Create the local configuration described in `configs/README.md`, then run:

```bash
uv sync
uv run abstrak-doctor
uv run abstrak-provider validate
uv run abstrak-canary validate
uv run pytest
uv run ruff check .
```

Provider commands read `~/.abstrak/config.yaml` by default. Offline validation
does not read credentials. A live probe additionally reads
`~/.abstrak/auth.json`:

```bash
uv run abstrak-provider smoke --live
```

Use `--config` or `--profile` to change configuration selection, and `--auth`
on a smoke command to change the credential file. Existing scripts may continue
to pass the legacy `--provider` and `--model` manifests as a pair. Non-empty
process environment variables take precedence over values loaded from the auth
file.

Run the stricter probe on a GPU worker:

```bash
scripts/update-worker.sh
scripts/bootstrap-a100.sh
source scripts/activate-a100.sh
abstrak-doctor --require-gpu
```

The local controller does not require a GPU. Compilation, qualification, and
performance measurement must run on a declared GPU worker with a frozen
software manifest. The current A100 worker uses Python 3.10, PyTorch 2.13.0,
and the CUDA 12.6 wheel index. Source, locks, wheels, and artifacts stay on the
persistent volume; the GPU venv is rebuilt on container-local storage by
`scripts/bootstrap-a100.sh` so imports and JIT compilation do not run over NFS.
The persistent worker checkout uses the public HTTPS remote, so later container
refreshes update with `scripts/update-worker.sh` instead of copying source files.

## Repository layout

- `configs/`: versioned provider, model, hardware, target, task, and study manifests.
- `src/abstrak/`: shared controller and experiment infrastructure.
- `tests/`: unit and conformance tests that do not require hidden benchmark data.
- `artifacts/`: local run output; only its storage contract is tracked by Git.

## Experimental invariants

- The controlled track uses one orchestration loop and tool schema for every model.
- Provider retries, fallback, context compaction, and tool execution must be explicit.
- Every request, response, patch, compiler result, and budget event is recorded.
- Development feedback and sealed qualification run in separate processes.
- Model credentials, generated kernels, traces, and raw benchmark results are not committed.
- Published summaries must be reproducible from an immutable artifact bundle and manifest hash.

## Current scope

The default dependency set contains only the API transport and structured-config
building blocks. The optional `gpu` dependency set freezes Triton, TileLang,
CuTe DSL, and the CUDA Python stack in one validated A100 environment. The
controller remains usable without those packages; separate target environments
will be introduced only if a reproducible ABI or runtime conflict requires them.

P0.1 provider conformance is implemented as a strict, single-attempt boundary.
See `docs/p0.1-provider-conformance.md`. Controlled probes require exact model IDs,
resolved credentials, and an explicit `--live` acknowledgement.

The GPU-independent skeleton for the single-turn KernelBench screen is also
available. It uses a pinned external checkout and keeps generated kernels out of
Git. See `docs/kernelbench-naive-screen.md` and start with:

```bash
uv run abstrak-kernelbench validate \
  --study configs/studies/kernelbench-naive-smoke.yaml \
  --kernelbench-root /path/to/KernelBench
```

The first R1 canary vertical slice can execute a hash-bound trusted oracle on
an SSH A100 worker before any model request:

```bash
uv run abstrak-canary run-trusted \
  --ssh-host <a100-host> \
  --worker-root /path/to/AbstraK
```

After that bundle passes, one supervised four-turn trajectory uses the same
worker transport:

```bash
uv run abstrak-canary run-cell \
  --ssh-host <a100-host> \
  --worker-root /path/to/AbstraK \
  --profile deepseek-v4-flash \
  --expected-max-requests 4 \
  --live
```

`run-cell` executes model-generated Python and therefore only accepts the SSH
worker path. Each job runs under bubblewrap with no network, a read-only root,
temporary home/cache paths, a cleared environment, remote and controller
timeouts, GPU health quarantine, and A100/SM80 checks. This remains a
supervised canary rather than an adversarial security boundary; formal runs
must use a dedicated worker and independently verify its bubblewrap policy.
