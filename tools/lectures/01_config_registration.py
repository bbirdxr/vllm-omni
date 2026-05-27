"""Lecture 01 — Config registration & loading
==============================================

学习目标
--------
1. 看到 ``import vllm_omni`` 的副作用：往 transformers 的 ``CONFIG_MAPPING``
   里注册了哪些自定义 ``model_type``。
2. 用 ``AutoConfig.from_pretrained`` 读取 MiniCPM-o-4.5 的 ``config.json``，
   看 ``model_type`` / ``architectures`` / ``version`` 字段。

为什么重要
----------
这两个字段是 vllm-omni 选 pipeline 的输入：
- ``model_type`` 先去 ``_OMNI_PIPELINES`` 注册表里查（第 02 课）。
- 没命中就 fall through 到 ``architectures`` 跟每个 PipelineConfig 的
  ``hf_architectures`` 求交集（也是第 02 课）。

依赖
----
无（这是入口课）。

运行
----
    python tools/lectures/01_config_registration.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import MODEL_DIR, _hr, assert_paths_exist  # noqa: E402


def main() -> None:
    assert_paths_exist()

    _hr("Step 0  pre-import sanity")
    print(f"sys.path[0:3]={sys.path[:3]}")
    print(f"MODEL_DIR exists: {os.path.isdir(MODEL_DIR)}")

    # ---------------------------------------------------------------- #
    # Step B: import vllm_omni → 触发 configs/__init__.py 里的           #
    #         AutoConfig.register(...) 调用。这是个全局副作用。           #
    # ---------------------------------------------------------------- #
    # 在transformer中注册vllm-omni有的model但是transformer中看你没有
    _hr("Step B  import vllm_omni → 自定义 config 注册")
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    import vllm_omni  # noqa: F401  side-effect: registers configs

    interesting = ("voxcpm2", "qwen3_tts", "fish_speech", "cosyvoice3")
    omni_models = [m for m in interesting if m in CONFIG_MAPPING]
    print(f"vllm-omni 注册到 transformers.CONFIG_MAPPING 的 model_type: {omni_models}")
    print(f"(被检查的候选: {list(interesting)})")

    # 注意: minicpmo 没在这里注册, 因为它走 trust_remote_code 路线
    # (config.json 里有 auto_map 指向 hub 上的 configuration_minicpmo.py)
    print(f"\nminicpmo 在 CONFIG_MAPPING 里? {'minicpmo' in CONFIG_MAPPING}")
    print("→ 不在也正常: MiniCPM-o-4.5 走 trust_remote_code, config 类是从 model 目录里动态加载的")

    # ---------------------------------------------------------------- #
    # Step A→B/C: AutoConfig.from_pretrained                            #
    # ---------------------------------------------------------------- #
    _hr("Step A→B/C  AutoConfig.from_pretrained(MODEL_DIR)")
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
    print(f"type:           {type(hf_config).__name__}")
    print(f"model_type:     {getattr(hf_config, 'model_type', None)}")
    print(f"architectures:  {getattr(hf_config, 'architectures', None)}")
    print(f"version:        {getattr(hf_config, 'version', None)}")

    print(
        "\n小结: 这两个字段会在第 02 课用来选 pipeline。\n"
        f"  - model_type='{hf_config.model_type}' → 先查 _OMNI_PIPELINES\n"
        f"  - architectures={hf_config.architectures} → 再跟 hf_architectures 求交集"
    )


if __name__ == "__main__":
    main()
