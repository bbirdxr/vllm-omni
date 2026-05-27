"""Lecture 02 — Pipeline resolution
==============================================

学习目标
--------
1. 看 ``_OMNI_PIPELINES`` 注册表里都有哪些 pipeline。
2. 用 ``model_type`` 直接查注册表（MiniCPM-o-4.5 这条路径会落空）。
3. Fall through 到 ``hf_architectures`` 集合求交集，找到匹配的 PipelineConfig。
4. 看 PipelineConfig 内部的 stage 拓扑（thinker + talker）。

为什么重要
----------
这是 vllm-omni 多 stage 系统的"入口判别"：一个 model_path 来了，
通过这一步选定走 1-stage 还是 N-stage、每个 stage 是 LLM_AR 还是 DIFFUSION。

依赖
----
- 第 01 课的产物：``hf_config.model_type`` 和 ``hf_config.architectures``。

运行
----
    python tools/lectures/02_pipeline_resolution.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import MODEL_DIR, _hr, assert_paths_exist  # noqa: E402


def main() -> None:
    assert_paths_exist()

    from transformers import AutoConfig

    import vllm_omni  # noqa: F401  确保 config 注册副作用已发生

    hf_config = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)

    # ---------------------------------------------------------------- #
    # Step C: peek at _OMNI_PIPELINES                                   #
    # ---------------------------------------------------------------- #
    _hr("Step C  pipeline_registry → _OMNI_PIPELINES 总览")
    from vllm_omni.config.pipeline_registry import _OMNI_PIPELINES

    print(f"_OMNI_PIPELINES keys ({len(_OMNI_PIPELINES)}):")
    for k in sorted(_OMNI_PIPELINES.keys()):
        print(f"  - {k}")

    mt = hf_config.model_type
    print(f"\nmodel_type='{mt}' 在注册表里? {mt in _OMNI_PIPELINES}")
    if mt not in _OMNI_PIPELINES:
        print("→ 不在: 会 fall through 到 hf_architectures 匹配 (Step D)")

    # ---------------------------------------------------------------- #
    # Step D: 直接 import minicpmo_4_5 的 PipelineConfig                 #
    # ---------------------------------------------------------------- #
    _hr("Step D  pipeline.py → MINICPMO_4_5_PIPELINE")
    mod = importlib.import_module("vllm_omni.model_executor.models.minicpmo_4_5.pipeline")
    pipeline_cfg = getattr(mod, "MINICPMO_4_5_PIPELINE")

    print(f"model_type:        {pipeline_cfg.model_type}")
    print(f"model_arch:        {pipeline_cfg.model_arch}")
    print(f"hf_architectures:  {pipeline_cfg.hf_architectures}")
    print(f"# stages:          {len(pipeline_cfg.stages)}")
    for st in pipeline_cfg.stages:
        print(
            f"  stage {st.stage_id}: model_stage={st.model_stage}  "
            f"execution_type={st.execution_type}  model_arch={st.model_arch}"
        )

    # ---------------------------------------------------------------- #
    # Step C+D: 手动复现 stage_config.py 的自动检测逻辑                  #
    # ---------------------------------------------------------------- #
    _hr("Step C+D  手动跑一遍 hf_architectures 交集匹配")
    hf_archs = set(getattr(hf_config, "architectures", None) or [])
    pl_archs = set(pipeline_cfg.hf_architectures)
    intersection = hf_archs & pl_archs

    print(f"hf_archs from config.json:                  {hf_archs}")
    print(f"pipeline.hf_architectures:                  {pl_archs}")
    print(f"intersection (must be non-empty to match):  {intersection}")
    print(f"→ pipeline {'WILL' if intersection else 'WILL NOT'} be selected")

    if not intersection:
        print("\n警告: 没匹配上意味着引擎启动时会报'No pipeline matched architectures=...'")


if __name__ == "__main__":
    main()
