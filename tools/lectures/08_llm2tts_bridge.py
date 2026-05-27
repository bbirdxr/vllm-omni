"""Lecture 08 — llm2tts: thinker → talker 的 stage 间数据桥
==============================================================

学习目标
--------
1. 看 MiniCPM-o-4.5 ``pipeline.py`` 里 ``custom_process_input_func`` 这个字段
   的声明位置, 理解 "stage 之间靠一个普通 Python 函数搭桥" 这个设计。
2. 读 MiniCPM-o-4.5 的 ``llm2tts`` 函数源码, 看它从 thinker 输出里**精确抽取
   了哪些字段** (prompt_token_ids, hidden_states, TTS region) 并塞进 talker
   的 ``OmniTokensPrompt``。
3. 弄清楚 ``OmniTokensPrompt`` 为什么是 ``[1, 0, 2]`` 这种**假 token ids** +
   ``additional_information`` 字典搭便车的奇怪结构 (vllm 的 prefill 必须有
   token, talker 真正需要的数据另走 channel)。
4. 看 framework 怎么调到 llm2tts (``StagePoolClient.process_engine_inputs``
   接口)。
5. 列一下仓库里所有的 stage_input_processors, 直观感受 "每个多 stage 模型
   都自己写一个 bridge" 这个事实。

为什么重要
----------
你在 07 课看到 ``engine_client.generate(prompt)`` 把请求送进 stage 0。本课
讲的是 *stage 0 出来的 token / hidden states 怎么变成 stage 1 能吃的输入*。
这是 vllm-omni 区别于普通 vllm 最关键的一段, 也是为什么 vllm-omni 能跑
omni 模型而 vllm 本体只能跑单 stage LLM 的根本机制。

依赖
----
- 第 01 课 + 第 04 课 (理解 ``PipelineConfig.stages`` 拓扑)。
- 不需要 GPU (纯读源码)。

运行
----
    python tools/lectures/08_llm2tts_bridge.py
"""

# Lecture script: keeps illustrative locals (e.g. extracted AST source) for
# pedagogical clarity even when not all are referenced downstream.
# ruff: noqa: F841

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import _hr, assert_paths_exist  # noqa: E402


def main() -> None:
    assert_paths_exist()

    import vllm_omni  # noqa: F401

    # ---------------------------------------------------------------- #
    # Step J.1: pipeline.py 里 custom_process_input_func 声明           #
    # ---------------------------------------------------------------- #
    _hr("Step J.1  pipeline.py 里 custom_process_input_func 字段")
    pipe_mod = importlib.import_module("vllm_omni.model_executor.models.minicpmo_4_5.pipeline")
    cfg = pipe_mod.MINICPMO_4_5_PIPELINE
    print(f"pipeline: {cfg.model_type}")
    print(f"stages 总数: {len(cfg.stages)}\n")

    for st in cfg.stages:
        print(f"  stage {st.stage_id} ({st.model_stage}):")
        print(f"    execution_type     : {st.execution_type}")
        print(f"    input_sources      : {st.input_sources}")
        print(f"    final_output_type  : {st.final_output_type}")
        custom = getattr(st, "custom_process_input_func", None)
        if custom:
            print(f"    custom_process_input_func : {custom}   ← 这就是桥")
        else:
            print("    custom_process_input_func : (none — stage 直接吃用户 prompt)")
        print()

    print(
        "→ 翻译: stage 0 是入口, 直接吃用户传来的 OmniTokensPrompt。\n"
        "  stage 1 有 input_sources=(0,), 也就是 *从 stage 0 拿输入*; 它的 input\n"
        "  通过 custom_process_input_func 指定的那个普通函数来构造。\n"
        "  这个函数 = bridge / converter, 由 stage 0 输出 → stage 1 输入。\n"
        "  vllm-omni 对它**只要求一个固定签名**, 实现完全是 per-model 的 Python 代码。"
    )

    # ---------------------------------------------------------------- #
    # Step J.2: llm2tts 函数本身 —— 关键步骤                            #
    # ---------------------------------------------------------------- #
    _hr("Step J.2  llm2tts 函数源码 (MiniCPM-o-4.5 的实现)")
    bridge_mod = importlib.import_module("vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni")
    llm2tts = bridge_mod.llm2tts
    src, start = inspect.getsourcelines(llm2tts)

    print(f"source: {inspect.getsourcefile(llm2tts)}")
    print(f"def llm2tts 起始行: L{start}, 共 {len(src)} 行")
    print(f"签名      : {inspect.signature(llm2tts)}\n")

    doc = inspect.getdoc(llm2tts) or "(no docstring)"
    print("docstring 摘要:")
    for line in doc.splitlines()[:6]:
        print(f"  {line}")
    print("  ...")

    # 用关键字 grep 找几个关键操作
    src_str = "".join(src)
    keywords = (
        "output.multimodal_output.get",
        "hidden_states",
        "tts_bos_id",
        "tts_eos_id",
        "OmniTokensPrompt",
        "additional_information",
        "prompt_token_ids=[1, 0, 2]",
    )
    print("\n关键操作行号:")
    for i, line in enumerate(src):
        for kw in keywords:
            if kw in line:
                print(f"  L{start + i:>4}  {line.strip()[:110]}")
                break

    print(
        "\n→ 翻译: llm2tts 做了 4 件事:\n"
        "    1. 从 thinker 每条 output 里捞 latent / hidden_states (talker 要用它做\n"
        "       speaker embedding 提取 + TTS region 的语义条件)\n"
        "    2. 找特殊 token <|tts_bos|>/<|tts_eos|> 的位置, 切出 TTS 区间的\n"
        "       token_ids 和 hidden_states\n"
        "    3. 把这些一股脑塞进 OmniTokensPrompt.additional_information 字典\n"
        "    4. 返回 list[OmniTokensPrompt] —— 接下来送进 stage 1 talker"
    )

    # ---------------------------------------------------------------- #
    # Step J.3: OmniTokensPrompt 的奇怪结构                              #
    # ---------------------------------------------------------------- #
    _hr("Step J.3  为什么 OmniTokensPrompt 的 prompt_token_ids = [1, 0, 2]?")
    print(
        "  llm2tts 返回的 OmniTokensPrompt 长这样:\n"
        "    OmniTokensPrompt(\n"
        "        prompt_token_ids=[1, 0, 2],          ← 假 token: BOS + PAD + EOS\n"
        "        additional_information={             ← 真数据藏在这里\n"
        "            'prompt_embeds':        <tensor [P, D]>,  # 来自 thinker prompt 区\n"
        "            'prompt_token_ids':     <list[int]>,       # thinker 原 prompt\n"
        "            'llm_output_token_ids': <list[int]>,       # thinker 生成的全部 token\n"
        "            'llm_output_text':      <str>,             # thinker decode 出的文本\n"
        "            'tts_token_ids':        <tensor>,          # ← TTS region 的 token\n"
        "            'tts_hidden_states':    <tensor>,          # ← TTS region 的 hidden\n"
        "        },\n"
        "        multi_modal_data=...,                          # 透传原 image/audio\n"
        "    )\n"
        "\n"
        "  为啥 prompt_token_ids 是假的?\n"
        "    vllm 的 PagedAttention prefill 步骤 *必须有* 至少 1 个 token id,\n"
        "    才能跑 forward。但 talker 实际上不靠这几个假 token 干活, 它从\n"
        "    additional_information 里读真数据 (hidden_states + 特殊 token slice)。\n"
        "    这个 [1, 0, 2] 只是 *最小占位*, 让 vllm 调度系统能正常算 slot 数。\n"
        "    talker 模型自己的 preprocess() 会把 hidden_states 当成它的 inputs_embeds\n"
        "    直接灌进 forward, 完全绕过 token embedding 这一步。"
    )

    # ---------------------------------------------------------------- #
    # Step J.4: framework 怎么调到 llm2tts —— StagePool 接口             #
    # ---------------------------------------------------------------- #
    _hr("Step J.4  framework 怎么调到 llm2tts: StagePoolClient.process_engine_inputs")
    from vllm_omni.engine.stage_client import StagePoolLLMClient

    src, start = inspect.getsourcelines(StagePoolLLMClient)
    print(f"source: {inspect.getsourcefile(StagePoolLLMClient)}")
    print(f"class StagePoolLLMClient (Protocol) 起始行: L{start}\n")

    for i, line in enumerate(src):
        if "custom_process_input_func" in line or "process_engine_inputs" in line:
            print(f"  L{start + i:>4}  {line.rstrip()[:110]}")

    print(
        "\n→ 翻译: StagePoolLLMClient (每 stage 一个) 上挂了 custom_process_input_func\n"
        "  + process_engine_inputs(source_outputs, prompt, streaming_context) 协议。\n"
        "  当框架要把 stage N-1 输出送进 stage N 时, 它会:\n"
        "    1) 拿到 stage N 的 stage_client\n"
        "    2) stage_client.custom_process_input_func ← 从 pipeline.py 的字符串\n"
        "       'vllm_omni....minicpmo_4_5_omni.llm2tts' importlib 动态加载\n"
        "    3) stage_client.process_engine_inputs(stage_(N-1)_outputs, ...) → 触发\n"
        "       call(custom_func, ...) → list[OmniTokensPrompt]\n"
        "    4) 用这些 OmniTokensPrompt 给 stage N 的 worker 发 add_request"
    )

    # ---------------------------------------------------------------- #
    # Step J.5: 仓库里所有 stage_input_processors 一览                    #
    # ---------------------------------------------------------------- #
    _hr("Step J.5  仓库里所有 stage_input_processors (per-model bridges)")
    proc_dir = Path(__file__).resolve().parents[2] / "vllm_omni/model_executor/stage_input_processors"
    py_files = sorted(p for p in proc_dir.glob("*.py") if p.stem != "__init__")
    print(f"目录: {proc_dir}")
    print(f"共 {len(py_files)} 个文件:\n")
    util_stems = {"tts_utils", "chunk_size_utils"}  # 通用工具, 不是 bridge
    for f in py_files:
        stem = f.stem
        tag = "  (utility, not a bridge)" if stem in util_stems else ""
        print(f"  {f.name:<35s}{tag}")

    print(
        "\n→ 观察:\n"
        "  - tts_utils.py / chunk_size_utils.py 是共享工具, 不是 bridge。\n"
        "  - 剩下的每个文件对应一个多 stage 模型 (qwen2_5_omni / qwen3_omni /\n"
        "    minicpmo_4_5_omni / ming_flash_omni / mimo_audio / cosyvoice3 / ...)。\n"
        "  - 每个文件至少定义一个跟 llm2tts 同签名的函数, 在该模型的 pipeline.py\n"
        "    里被 custom_process_input_func 字段引用。\n"
        "  - 这是 vllm-omni 的 *plugin-friendly 设计*: 加新 omni 模型时, 你只需\n"
        "    (a) 在 stage_input_processors/ 下放一个新的 bridge .py\n"
        "    (b) 在该模型的 pipeline.py StagePipelineConfig 里指向它\n"
        "    框架代码完全不动。"
    )

    # ---------------------------------------------------------------- #
    # 串到下一步                                                         #
    # ---------------------------------------------------------------- #
    _hr("Done — 接下来:")
    print(
        "  你已经看到 stage 0 thinker 的输出怎么被切片、打包, 送进 stage 1 talker。\n"
        "  下一步:\n"
        "    - Lecture 09: talker 输出的 audio token / latent 怎么变成 wav 波形\n"
        "      (token2wav / code2wav, streaming output)。\n"
        "    - Lecture 10: 多模态输入侧 (image / audio / video) 怎么进 multi_modal_data。"
    )


if __name__ == "__main__":
    main()
