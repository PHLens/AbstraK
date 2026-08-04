"""Non-recursive raw-byte pins for the reviewed M8 study manifests.

This module is deliberately excluded from the M8 source-asset inventory.  The
three values are patched only after the generated JSON bytes have been reviewed;
including this file in that inventory would create a self-referential digest.
"""

PINNED_FORMAL_STUDY_SHA256 = "89c789c31d1340984f0180189a678334207dbf0cbda6a33abc8af48d085a04b9"
PINNED_SHAKEOUT_STUDY_SHA256 = "6554fff331cd6b6707b8d55dd0f484987ee67afe58ef12bef02d3461770ac5c5"
PINNED_OFFLINE_FREEZE_SHA256 = "4c45ec8fd753d1a50fbff4896f21234807a0a2e324de8e6ecc96bc66691a3bef"
