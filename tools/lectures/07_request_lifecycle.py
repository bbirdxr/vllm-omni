"""Lecture 07 — HTTP request → engine_client.generate() 全链路
================================================================

学习目标
--------
1. 看清 FastAPI route handler ↔ ``OpenAIServingChat`` ↔ ``engine_client`` 这
   三层 dispatch 怎么发生 (06 已经知道 messages 怎么变 TokensPrompt; 这一课
   补完"TokensPrompt 是怎么送到 engine 的, 以及响应又是怎么流回去的")。
2. 用 AST 静态扫描 ``serving_chat.py::create_chat_completion`` 的关键步骤
   (model check → tokenizer → preprocess_chat → ``engine_client.generate(...)``
   → stream collection)。
3. 弄清楚同一个 ``POST /v1/chat/completions`` endpoint 怎么按 client 想要的
   方式返回 3 种不同响应:
   - ``ErrorResponse``                    → JSON 4xx
   - ``ChatCompletionResponse``           → JSON 200 (一次性返回)
   - ``AsyncGenerator[str, None]``         → SSE 流 (``text/event-stream``)
4. 看 ``OmniEngineClient`` 跟原版 ``EngineClient`` 在 ``generate(...)`` 这一
   层差在哪 (多 stage routing 是关键差异)。

为什么重要
----------
你在 06 学完 "messages → TokensPrompt" 之后, 接下来 ``serving_chat.py`` 会调
``self.engine_client.generate(prompt, sampling_params, request_id)`` 把这个
TokensPrompt 送进引擎。这一步**就是"用户请求"跨越到"vLLM 内部"的边界**。
之后的 stage 间数据流 (lecture 08+) 全发生在 engine_client.generate 这个
``async for`` 循环内部。

依赖
----
- 第 01 课: ``import vllm_omni``。
- 不需要 GPU (纯读源码)。

运行
----
    python tools/lectures/07_request_lifecycle.py
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import _hr, assert_paths_exist  # noqa: E402


def _grep_lines(source: str, base_lineno: int, *keywords: str) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for i, line in enumerate(source.splitlines()):
        for kw in keywords:
            if kw in line:
                hits.append((base_lineno + i, line.rstrip()))
                break
    return hits


def main() -> None:
    assert_paths_exist()

    import vllm_omni  # noqa: F401

    # ---------------------------------------------------------------- #
    # Step I.1: api_server.py 是个 thin wrapper                          #
    # ---------------------------------------------------------------- #
    _hr("Step I.1  api_server.py 里的 chat 入口 — 一个 thin wrapper")
    # 注意: 用 inspect.getsourcelines 会被 @with_cancellation / @load_aware_call
    # 这些装饰器迷惑, 拿到 vllm 工具函数的源码。这里直接读文件、按行号截取。
    api_server_path = Path(__file__).resolve().parents[2] / "vllm_omni/entrypoints/openai/api_server.py"
    all_lines = api_server_path.read_text().splitlines()
    # 找 "async def create_chat_completion"
    target_line = next(
        (i for i, ln in enumerate(all_lines) if "async def create_chat_completion(" in ln),
        None,
    )
    if target_line is None:
        print("  (create_chat_completion not found in api_server.py)")
    else:
        print(f"source: {api_server_path}")
        print(f"create_chat_completion 起始行: L{target_line + 1}\n")
        # 截后续 70 行做关键字 grep
        chunk = "\n".join(all_lines[target_line : target_line + 75])
        hits = _grep_lines(
            chunk,
            target_line + 1,
            "Omnichat",
            "handler.create_chat_completion",
            "ErrorResponse",
            "ChatCompletionResponse",
            "StreamingResponse",
            "JSONResponse",
        )
        seen: set[str] = set()
        for ln, line in hits:
            key = line.strip()[:40]
            if key in seen:
                continue
            seen.add(key)
            print(f"  L{ln:>4}  {line.strip()[:100]}")

    print(
        "\n→ 直白翻译: api_server.py 只是 *拆信封* —\n"
        "    1) 拿出 ChatCompletionRequest\n"
        "    2) 包成 Omnichat handler (内部就是 OmniOpenAIServingChat)\n"
        "    3) await handler.create_chat_completion(...) 拿一个 generator/响应对象\n"
        "    4) 按类型 (Error/Json/Stream) wrap 成 FastAPI Response\n"
        "    真正的业务逻辑全在 serving_chat.py。"
    )

    # ---------------------------------------------------------------- #
    # Step I.2: serving_chat.OmniOpenAIServingChat 的关键步骤            #
    # ---------------------------------------------------------------- #
    _hr("Step I.2  OmniOpenAIServingChat._create_chat_completion 主流程速览")
    # 注意: 不能直接 inspect OpenAIServingChat — 那是 vllm 本体的基类。
    # vllm-omni 在 serving_chat.py 里定义了真正的子类 OmniOpenAIServingChat
    # (继承 vllm 的 OpenAIServingChat) 来加 omni-specific 逻辑。
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    src, start = inspect.getsourcelines(OmniOpenAIServingChat._create_chat_completion)
    src_str = "".join(src)
    print(f"source: {inspect.getsourcefile(OmniOpenAIServingChat)}")
    print(f"OmniOpenAIServingChat._create_chat_completion 起始行: L{start}, 共 {len(src)} 行")
    print("(2700+ 行的方法略复杂; 只挑关键步骤行号):\n")

    keywords = (
        "_create_diffusion_chat_completion",  # 分流到 diffusion 路径
        "_check_model",  # 模型存在性 / 权限检查
        "self.engine_client.errored",  # 引擎死活探活
        "_preprocess_chat(",  # ← Lecture 06 讲的内容
        "self.engine_client.generate(",  # ★ 把 prompt 送进引擎
        "chat_completion_stream_generator",  # 流式分支
        "chat_completion_full_generator",  # 非流式分支
    )
    hits = _grep_lines(src_str, start, *keywords)
    last_kw = None
    for ln, line in hits:
        stripped = line.strip()[:110]
        marker = ""
        for kw in keywords:
            if kw in line:
                if kw != last_kw:
                    last_kw = kw
                    marker = f"  ← {kw}"
                break
        print(f"  L{ln:>4}  {stripped}{marker}")

    print(
        "\n→ 把这些点连起来就是 chat 请求的一生:\n"
        "    if self._diffusion_mode:                       # 1. 分流: 文生图模型走另一条 path\n"
        "        return await self._create_diffusion_chat_completion(...)\n"
        "    await self._check_model(request)               # 2. 模型存在 + 权限\n"
        "    if self.engine_client.errored: raise ...       # 3. 引擎死活探活\n"
        "    tokenizer = renderer.get_tokenizer()           # 4. 拿 tokenizer\n"
        "    engine_prompt = await self._preprocess_chat(   # 5. ← Lecture 06\n"
        "                       request, tokenizer, messages, ...)\n"
        "    generator = self.engine_client.generate(       # 6. ★ 边界: TokensPrompt → engine\n"
        "                   engine_prompt, sampling_params, request_id, ...)\n"
        "    if request.stream:                             # 7. 按 stream 与否选 collector\n"
        "        return self.chat_completion_stream_generator(generator, ...)\n"
        "    else:\n"
        "        return await self.chat_completion_full_generator(generator, ...)"
    )

    # ---------------------------------------------------------------- #
    # Step I.3: 三种响应类型 + FastAPI 怎么分发                          #
    # ---------------------------------------------------------------- #
    _hr("Step I.3  同一个 endpoint 怎么返回 3 种不同响应")
    print(
        "  handler.create_chat_completion(...) 的返回类型 (从 type hint 看):\n"
        "    AsyncGenerator[str, None]   →  text/event-stream  (SSE 流式)\n"
        "    ChatCompletionResponse      →  application/json   (一次性 200)\n"
        "    ErrorResponse               →  application/json   (4xx)\n"
        "\n"
        "  api_server.create_chat_completion 看到不同类型, 包成不同 FastAPI Response:\n"
        "    isinstance(generator, ErrorResponse)         → JSONResponse(status=err.code)\n"
        "    isinstance(generator, ChatCompletionResponse) → JSONResponse(content=...)\n"
        "    else (generator 是 AsyncGenerator)            → StreamingResponse(\n"
        "                                                       content=generator,\n"
        "                                                       media_type='text/event-stream')\n"
        "\n"
        "  即: 流式 vs 一次性 vs 错误 都共用一个 POST endpoint, 通过返回类型分流。"
    )

    # ---------------------------------------------------------------- #
    # Step I.4: engine_client.generate() — request 跨入引擎的边界        #
    # ---------------------------------------------------------------- #
    _hr("Step I.4  engine_client.generate() — 跨入 engine 的边界")

    # engine_client 的真身: vllm_omni/entrypoints/async_omni.py 里的 AsyncOmni
    # (api_server.py 第 2115 行: engine_client = cast(AsyncOmni, engine_client))
    try:
        from vllm_omni.entrypoints.async_omni import AsyncOmni

        gen_method = AsyncOmni.generate
        gen_src_file = inspect.getsourcefile(gen_method)
        gen_lineno = inspect.getsourcelines(gen_method)[1]
        gen_doc = inspect.getdoc(gen_method) or "(no docstring)"
        first_doc_line = gen_doc.splitlines()[0] if gen_doc != "(no docstring)" else gen_doc
        bases = [b.__name__ for b in AsyncOmni.__mro__[1:] if b is not object]
        print("AsyncOmni.generate  (= engine_client.generate 的真身)")
        print(f"  source : {gen_src_file}")
        print(f"  L{gen_lineno}")
        print(f"  MRO    : AsyncOmni → {' → '.join(bases)}")
        print(f"  doc    : {first_doc_line}\n")
    except (ImportError, AttributeError, OSError) as e:
        print(f"  (AsyncOmni.generate not directly inspectable: {e})")
        print()

    print(
        "  跟 vllm 原版的差异 (AsyncOmni vs vllm.EngineClient):\n"
        "    vllm.AsyncLLM.generate  : 请求送进 *单个* engine core, 流式返回 token\n"
        "    vllm_omni.AsyncOmni.generate :\n"
        "      a) 决定这个请求要走几个 stage (text-only chat → 只走 stage 0 thinker;\n"
        "         text+audio chat → 走 thinker + talker + (可选) code2wav)\n"
        "      b) 通过 ZMQ 把 prompt 路由给 stage 0 的 StageEngineCoreProc 子进程\n"
        "      c) 每个 stage 内部还是 vllm v1 的标准 engine core (没重写)\n"
        "      d) stage 间桥 (e.g. thinker → talker) 走 StagePool/RequestRouter\n"
        "      e) 最末端 stage 的输出 (text + audio + ...) 合并成 OmniOutput 流回\n"
        "\n"
        "  这一步是 *整个请求生命周期最复杂的一段*, 后续 lecture 08-09 专门讲 stage 间\n"
        "  数据流 (llm2tts bridge) 和 audio 输出 (token2wav / streaming)。"
    )

    # ---------------------------------------------------------------- #
    # Step I.5: 完整请求生命周期图                                      #
    # ---------------------------------------------------------------- #
    _hr("Step I.5  完整请求生命周期 (chat 例子)")
    print(
        "  Client                                                          Server\n"
        "  ────────────────────────────────────────────────────────────────────────\n"
        "  POST /v1/chat/completions  ──┐\n"
        "  { messages: [...],            │\n"
        "    stream: true,               │\n"
        "    modalities: [text, audio] } │\n"
        "                                 ↓\n"
        "                                FastAPI route handler (api_server.py L996)\n"
        "                                  ├─ @validate_json_request (decorator)\n"
        "                                  ├─ @with_cancellation\n"
        "                                  ├─ @load_aware_call\n"
        "                                  └─ Omnichat(raw_request)\n"
        "                                       │\n"
        "                                       ↓\n"
        "                                  OpenAIServingChat.create_chat_completion\n"
        "                                    (serving_chat.py L206; Lecture 07 主线)\n"
        "                                       │\n"
        "                                       ├─ _check_model\n"
        "                                       ├─ get_tokenizer\n"
        "                                       ├─ _preprocess_chat ────────► Lecture 06\n"
        "                                       │   (messages → TokensPrompt)\n"
        "                                       ↓\n"
        "                                  engine_client.generate(\n"
        "                                    prompt, sampling_params, request_id) ← ★ 边界\n"
        "                                       │\n"
        "                                       ↓\n"
        "                                  AsyncOmniEngine.generate (async_omni_engine.py)\n"
        "                                    ├─ route to stage_pool[0] (thinker)\n"
        "                                    ├─ ZMQ → StageEngineCoreProc (stage 0)\n"
        "                                    │    └─ vllm v1 worker (实际 forward)\n"
        "                                    ├─ thinker 输出 → llm2tts bridge ── Lecture 08\n"
        "                                    ├─ ZMQ → StageEngineCoreProc (stage 1 talker)\n"
        "                                    │    └─ vllm v1 worker\n"
        "                                    └─ talker 输出 audio token ── Lecture 09\n"
        "                                       │\n"
        "                                       ↓ async for chunk in generator:\n"
        "                                  chat_completion_stream_generator\n"
        "                                    (把 OmniOutput chunk 转成 SSE 'data: {...}\\n\\n')\n"
        "                                       │\n"
        "  ←─── data: {...}\\n\\n  ── (HTTP chunk)─┘\n"
        "  ←─── data: {...}\\n\\n\n"
        "  ←─── data: [DONE]\\n\\n\n"
        "  (client 一路 async for 收 chunk)"
    )

    # ---------------------------------------------------------------- #
    # 串到下一步                                                         #
    # ---------------------------------------------------------------- #
    _hr("Done — 接下来:")
    print(
        "  请求已经 *进入* engine_client.generate, 真正的 stage 间数据流和音频生成\n"
        "  在 vllm-omni 这一侧:\n"
        "\n"
        "  - Lecture 08: llm2tts bridge — thinker 输出 token/hidden 怎么变成 talker 输入\n"
        "  - Lecture 09: token2wav / audio streaming — talker 输出怎么变 wav / SSE 流\n"
        "  - Lecture 10: 多模态输入预处理 — image/audio/video 怎么进 multi_modal_data"
    )


if __name__ == "__main__":
    main()
