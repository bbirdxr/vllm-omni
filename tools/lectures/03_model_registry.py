"""Lecture 03 — vLLM ModelRegistry & vllm-omni model registration
==================================================================

学习目标
--------
1. 看 ``register_omni_models_to_vllm()`` 把哪些架构名注册进了 vLLM 的
   ``ModelRegistry``。
2. 用探针验证四类名字在不在注册表里：
   - 顶层 wrapper:  ``MiniCPMO45OmniForConditionalGeneration``
   - 子 stage 全名: ``MiniCPMO45OmniLLMForConditionalGeneration`` /
                    ``MiniCPMO45OmniTTSForConditionalGeneration``
   - 旧的短名:       ``MiniCPMO45OmniLLMModel`` / ``MiniCPMO45OmniTTSModel``
                    (mt 重构后应当 MISSING)
   - 上游冲突名:     ``MiniCPMO`` (vLLM 自带, 必须被 vllm-omni 覆盖)

为什么重要
----------
这是 worker 子进程加载模型类的最后一步：vLLM 拿到 architectures 列表后,
会去 ``ModelRegistry`` 里查类。如果短名/全名/冲突没处理好, worker 就会
加载到错的类 (比如上游 ``MiniCPMO`` 不支持 TP)。

依赖
----
- 第 01 课: ``import vllm_omni``。

运行
----
    python tools/lectures/03_model_registry.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import _hr, assert_paths_exist  # noqa: E402


def main() -> None:
    assert_paths_exist()

    from vllm.model_executor.models import ModelRegistry

    import vllm_omni  # noqa: F401  ensure config side-effects
    from vllm_omni.engine.arg_utils import register_omni_models_to_vllm

    # 模拟 worker 子进程加载 vllm_omni plugin 的入口
    _hr("Step E  ModelRegistry — 调用 register_omni_models_to_vllm()")
    register_omni_models_to_vllm()

    archs = ModelRegistry.get_supported_archs()
    relevant = sorted(a for a in archs if "minicpm" in a.lower())
    print(f"MiniCPM 相关 archs ({len(relevant)}):")
    for a in relevant:
        print(f"  - {a}")

    # ---------------------------------------------------------------- #
    # 探针: 确认每个我们关心的名字到底在不在                              #
    # ---------------------------------------------------------------- #
    _hr("Step E  名称探针")
    probes = (
        ("顶层 wrapper", "MiniCPMO45OmniForConditionalGeneration"),
        ("thinker 全名", "MiniCPMO45OmniLLMForConditionalGeneration"),
        ("talker 全名 ", "MiniCPMO45OmniTTSForConditionalGeneration"),
        ("旧短名 LLM  ", "MiniCPMO45OmniLLMModel"),
        ("旧短名 TTS  ", "MiniCPMO45OmniTTSModel"),
        ("上游冲突名  ", "MiniCPMO"),
    )
    for label, name in probes:
        status = "YES" if name in archs else "MISSING"
        print(f"  [{label}]  {name:55s} {status}")

    print(
        "\n要点:\n"
        "  - 顶层 + thinker 全名 + talker 全名 必须 YES (worker 才能加载到正确类)\n"
        "  - 旧短名 MISSING 是正常的 (mt 重构后已切换为全名)\n"
        "  - MiniCPMO 必须被 vllm-omni 覆盖 (否则 worker 跑上游 MiniCPMO, 不支持 TP)"
    )

    # ---------------------------------------------------------------- #
    # 真实 serve 流后续步骤 (这里不跑)                                   #
    # ---------------------------------------------------------------- #
    _hr("Done — registration walk-through complete")
    print(
        "真实 serve 流后续步骤 (本课不模拟):\n"
        "  1. ModelConfig._auto_detect_model_type() → 选 pipeline (第 02 课)\n"
        "  2. AsyncOmniEngine 启动每个 stage 的 vLLM 子进程 (第 04 课)\n"
        "  3. worker 进程 load_general_plugins() → 触发本课的 register\n"
        "  4. ModelRegistry.resolve_model_cls(architectures) → 找到类\n"
        "  5. model.__init__(vllm_config=...) → 实例化\n"
        "  6. load_weights(...) → 'tts_obj.' 前缀对齐"
    )


if __name__ == "__main__":
    main()
