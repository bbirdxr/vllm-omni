"""Shared helpers for the vllm-omni MiniCPM-o-4.5 lecture series.

Each lecture imports from here so it can be run standalone:

    python tools/lectures/01_config_registration.py
"""

from __future__ import annotations

import os

MODEL_DIR = "/cache/wangjinxu/projects/models/MiniCPM-o-4_5"
DEPLOY_YAML = "/cache/wangjinxu/projects/vllm-omni/vllm_omni/deploy/minicpmo_4_5_8x4090.yaml"

SEP = "=" * 70


def _hr(title: str) -> None:
    print()
    print(SEP)
    print(f"  {title}")
    print(SEP)


def assert_paths_exist() -> None:
    assert os.path.isdir(MODEL_DIR), f"MODEL_DIR not found: {MODEL_DIR}"
    assert os.path.isfile(DEPLOY_YAML), f"DEPLOY_YAML not found: {DEPLOY_YAML}"
