"""Lecture 06 — apply_chat_template → engine_prompt
=====================================================

学习目标
--------
1. 看 ``apply_chat_template`` 在三种调用方式下分别返回什么:
   - ``tokenize=False``                       → ChatML 字符串
   - ``tokenize=False, use_tts_template=True`` → 末尾多一个 ``<|tts_bos|>``
   - ``tokenize=True,  use_tts_template=True`` → ``BatchEncoding`` (dict-like)
2. 把 ``BatchEncoding`` 解包成 ``list[int]``，构造 vLLM 的 ``TokensPrompt``。
3. 搞清楚三种"长得像 prompt"的对象到底有什么区别:
   - ChatML 字符串 (人看的)
   - HF ``BatchEncoding`` (Transformers 标准)
   - vLLM ``TokensPrompt`` (引擎吃的)

为什么重要
----------
这是 ``serving_chat.py`` 里 ``_preprocess_chat`` 函数干的事 —— 把用户传来的
messages 列表转成可以塞给 ``self.engine_client.generate(...)`` 的
``engine_prompt``。理解这一步, 后面看多模态 ``multi_modal_data`` 怎么塞才不
会懵。

依赖
----
- MODEL_DIR 里的 tokenizer (本课不需要 vllm-omni)。
- 概念上接续 Lecture 05 (stage spawn + API routing): 这一课你应当
  已经知道"04 → 06 之间", AsyncOmniEngine 已经 spawn 完 stage 子进程,
  FastAPI 已经把 ``/v1/chat/completions`` 路由到了 ``serving_chat.py``。
  本课讲的就是 ``serving_chat.py`` 接到请求后做的第一件事。

运行
----
    python tools/lectures/06_chat_to_engine_prompt.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import MODEL_DIR, _hr, assert_paths_exist  # noqa: E402


def main() -> None:
    assert_paths_exist()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    messages = [{"role": "user", "content": "你好"}]

    # ---------------------------------------------------------------- #
    # 三种 apply_chat_template                                          #
    # ---------------------------------------------------------------- #
    _hr("Step G.1  三种 apply_chat_template 模式对照")

    s_no_tts = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    s_with_tts = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        use_tts_template=True,
    )
    enc_with_tts = tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        use_tts_template=True,
    )

    print("--- (1) tokenize=False, 不带 use_tts_template ---")
    print(repr(s_no_tts))
    print("\n--- (2) tokenize=False, use_tts_template=True ---")
    print(repr(s_with_tts))
    print("    ↑ 末尾多一个 <|tts_bos|>, 告诉模型'下面要生成可被 talker 用的 token'")
    print("\n--- (3) tokenize=True, use_tts_template=True ---")
    print(f"    type:    {type(enc_with_tts).__name__}")
    print(f"    content: {enc_with_tts}")
    print(
        "    ↑ 注意: MiniCPM-o-4.5 自定义 tokenizer 在 tokenize=True 时\n"
        "       返回 BatchEncoding (dict-like, 含 input_ids+attention_mask),\n"
        "       不是单纯 list[int]。这是 vLLM 不能直接吃的关键原因。"
    )

    # ---------------------------------------------------------------- #
    # BatchEncoding → list[int] 解包                                    #
    # ---------------------------------------------------------------- #
    _hr("Step G.2  BatchEncoding 解包成 list[int]")
    if isinstance(enc_with_tts, dict) or hasattr(enc_with_tts, "get"):
        token_ids = enc_with_tts["input_ids"]
    else:
        token_ids = list(enc_with_tts)

    print(f"解包后类型:        {type(token_ids).__name__}")
    print(f"长度:              {len(token_ids)}")
    print(f"完整 token id:     {token_ids}")
    print(f"末尾 5 个:         {token_ids[-5:]}")
    print(f"最后一个 decode:   {tok.decode([token_ids[-1]])}  (应该是 <|tts_bos|>)")

    # ---------------------------------------------------------------- #
    # 构造 vLLM 的 TokensPrompt                                         #
    # ---------------------------------------------------------------- #
    _hr("Step G.3  构造 vLLM TokensPrompt")
    from vllm.inputs import TokensPrompt

    engine_prompt: TokensPrompt = {
        "prompt_token_ids": token_ids,
        # 没传图/音频, 所以 multi_modal_data 省略
        # 多模态时大致是: "multi_modal_data": {"image": [...], "audio": [...]}
    }
    print(engine_prompt)

    # ---------------------------------------------------------------- #
    # 三种 prompt 对象的对照表                                          #
    # ---------------------------------------------------------------- #
    _hr("Step G.4  三种 prompt 对象 cheat sheet")
    print(
        "┌────────────────────┬─────────────────────────┬───────────────────────┐\n"
        "│ 对象                │ 来源                     │ 谁用?                 │\n"
        "├────────────────────┼─────────────────────────┼───────────────────────┤\n"
        "│ ChatML 字符串       │ apply_chat_template      │ 人看 / 日志             │\n"
        "│                    │   tokenize=False         │                       │\n"
        "├────────────────────┼─────────────────────────┼───────────────────────┤\n"
        "│ BatchEncoding      │ apply_chat_template      │ HF model.generate()   │\n"
        "│   {input_ids,       │   tokenize=True          │                       │\n"
        "│    attention_mask}  │                          │                       │\n"
        "├────────────────────┼─────────────────────────┼───────────────────────┤\n"
        "│ TokensPrompt       │ _preprocess_chat 解包     │ vLLM engine.generate() │\n"
        "│   {prompt_token_ids,│ + multi_modal_data 包装   │                       │\n"
        "│    multi_modal_data}│                          │                       │\n"
        "└────────────────────┴─────────────────────────┴───────────────────────┘"
    )
    print(
        "\n核心要点:\n"
        "  - vLLM TokensPrompt 的 key 叫 prompt_token_ids, 不是 input_ids。\n"
        "  - 没有 attention_mask: vLLM 的 PagedAttention 内部自己管。\n"
        "  - 多模态走独立字段 multi_modal_data, 不是塞在文本 token 里。"
    )


if __name__ == "__main__":
    main()
