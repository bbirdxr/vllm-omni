"""Lecture 04 — Stage configs from deploy YAML
==============================================

学习目标
--------
1. 直接调 ``load_and_resolve_stage_configs(...)``，把 deploy yaml 解析成
   ``list[StageConfig]``，**不用启动任何 vLLM 引擎**。
2. 看每个 stage 的关键字段：``stage_id`` / ``stage_type`` /
   ``runtime.devices`` / ``engine_args``。
3. （可选）打印完整 OmegaConf 树，看 yaml 没写但被默认填上的字段。

为什么重要
----------
这是 ``AsyncOmniEngine.__init__`` 真正在做的事 ——
``self.stage_configs = self._resolve_stage_configs(...)`` 这一行展开后
就是本课内容。后面 ``_initialize_stages`` 会拿这个 list 去为每个
StageConfig spawn 一个 vLLM 子进程。

依赖
----
- 第 01 课: ``import vllm_omni``。

运行
----
    python tools/lectures/04_stage_configs.py

注意: 这一课需要 GPU 环境 (current_omni_platform.device_name 不能为 None)。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import DEPLOY_YAML, MODEL_DIR, _hr, assert_paths_exist  # noqa: E402


def main() -> None:
    assert_paths_exist()

    import vllm_omni  # noqa: F401
    from vllm_omni.entrypoints.utils import load_and_resolve_stage_configs

    # ---------------------------------------------------------------- #
    # Step F: 解析 deploy yaml                                          #
    # ---------------------------------------------------------------- #
    _hr("Step F  load_and_resolve_stage_configs(deploy_yaml)")
    config_path, stage_configs = load_and_resolve_stage_configs(
        model=MODEL_DIR,
        stage_configs_path=None,
        kwargs={},
        deploy_config_path=DEPLOY_YAML,
    )
    print(f"deploy yaml:    {DEPLOY_YAML}")
    print(f"resolved path:  {config_path}")
    print(f"# stage_configs: {len(stage_configs)}")

    # ---------------------------------------------------------------- #
    # 关键字段速览                                                       #
    # ---------------------------------------------------------------- #
    _hr("每个 stage 的关键字段")
    for cfg in stage_configs:
        ea = getattr(cfg, "engine_args", None)
        runtime = getattr(cfg, "runtime", None)
        stage_id = getattr(cfg, "stage_id", None)
        stage_type = getattr(cfg, "stage_type", None)
        devices = getattr(runtime, "devices", None)
        tp = getattr(ea, "tensor_parallel_size", None) if ea else None
        mml = getattr(ea, "max_model_len", None) if ea else None
        gmu = getattr(ea, "gpu_memory_utilization", None) if ea else None
        mns = getattr(ea, "max_num_seqs", None) if ea else None
        print(
            f"  stage {stage_id}: type={stage_type}\n"
            f"    devices={devices!r}  TP={tp}\n"
            f"    max_model_len={mml}  max_num_seqs={mns}\n"
            f"    gpu_memory_utilization={gmu}"
        )

    # ---------------------------------------------------------------- #
    # （可选）整树查看                                                   #
    # ---------------------------------------------------------------- #
    if "--full" in sys.argv:
        from omegaconf import OmegaConf

        for cfg in stage_configs:
            _hr(f"完整 stage_config (stage {cfg.stage_id})")
            print(OmegaConf.to_yaml(cfg))
    else:
        print("\n提示: 加 --full 参数可打印每个 stage 的完整 OmegaConf 树, 看 yaml 没写但被默认填上的字段。")

    # ---------------------------------------------------------------- #
    # 串到下一步                                                        #
    # ---------------------------------------------------------------- #
    _hr("下一步会发生什么 (本课不跑)")
    print(
        "AsyncOmniEngine 拿到这个 list 后:\n"
        "  1. compute_replica_layout(stage_configs) → 算每 stage 副本数 + 设备分配\n"
        "  2. _build_logical_stage_init_plans(...)   → 拆 plan\n"
        "  3. 对每个 plan: spawn_stage_core(vllm_config, ...)\n"
        "     → 每个 stage 各起一个 vLLM EngineCoreProc 子进程\n"
        "  4. 组装成 list[StagePool], 暴露给上层 OpenAI API"
    )


if __name__ == "__main__":
    main()
