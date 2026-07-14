# Artifact Storage Contract

Raw artifacts are intentionally ignored by Git. Each completed trajectory will
eventually write an immutable directory containing:

- a frozen study and environment manifest;
- append-only model and tool events;
- source snapshots and patches with content hashes;
- compiler, verifier, profiler, and qualification output;
- token, API-cost, wall-time, and GPU-time accounting;
- a machine-readable terminal status.

Artifact bundles intended for a paper must be stored in a content-addressed or
versioned external location. Git should contain only schemas, analysis code, and
the bundle index with checksums.
