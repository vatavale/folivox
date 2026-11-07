"""Folivox — local audiobook maker (book file → chaptered m4b) powered by CosyVoice 3."""

# 0.1.0.<commit number>: bumped by .githooks/pre-commit (enable: git config core.hooksPath .githooks)
__version__ = "0.1.0.340"

import os as _os

# Windows without Developer Mode cannot symlink: the hub cache still works, only the warning is noise.
_os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
