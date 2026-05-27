"""Lecture 05 — Stage spawn 落地 + API routing
================================================

学习目标
--------
1. 看 ``AsyncOmniEngine.__init__`` 拿到 ``list[StageConfig]`` 之后的关键步骤:
   compute_replica_layout → build_logical_stage_init_plans → spawn_stage_core。
2. 静态扫描 ``api_server.py``，列出 FastAPI 暴露的所有 route 路径与对应文件位置,
   让"04 yaml 配好" 跟 "06 chat 转 TokensPrompt" 中间的"请求是怎么进来的"
   补上。
3. 给出 vllm serve 起来之后的进程树/端口拓扑速查图。

为什么重要
----------
04 课讲完后，``list[StageConfig]`` 是个安静的数据结构。06 课开头你会发现
"用户已经把请求发到了 ``serving_chat.py``"。中间发生的两件事是:
  a) AsyncOmniEngine 把 StageConfig 列表 spawn 成跑着的 worker 子进程；
  b) FastAPI/uvicorn 起来，把 HTTP route 挂到 OpenAIServing* 类的方法上。
本课**不真启动引擎**(纯静态 introspection)，但把这两件事的入口、关键函数
名、端口、进程树画清楚，让 04 → 06 不再断层。

依赖
----
- 第 01 课: ``import vllm_omni``。
- 不需要 GPU (纯读源码)。

运行
----
    python tools/lectures/05_stage_spawn_and_api_routing.py
"""

# Lecture script: keeps illustrative locals (e.g. parsed AST nodes) for
# pedagogical clarity even when not all are referenced downstream.
# ruff: noqa: F841

from __future__ import annotations

import ast
import importlib
import inspect
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import _hr, assert_paths_exist  # noqa: E402


def _extract_route_paths(api_server_path: Path) -> list[tuple[int, str, str]]:
    """静态 AST 扫描 api_server.py, 提取所有 @router.<verb>(...) 装饰器声明.

    返回 [(line_no, verb, path_or_kw), ...]。
    """
    src = api_server_path.read_text()
    tree = ast.parse(src)
    results: list[tuple[int, str, str]] = []

    decorator_pattern = re.compile(r"@(?P<obj>\w+)\.(?P<verb>get|post|put|delete|patch|websocket)\(")

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for deco in node.decorator_list:
            if not isinstance(deco, ast.Call):
                continue
            # 取装饰器名形如 router.post 或 app.websocket
            func = deco.func
            obj_name = verb = None
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                obj_name = func.value.id
                verb = func.attr
            if obj_name not in {"router", "profiler_router", "app"} or verb is None:
                continue
            if verb not in {"get", "post", "put", "delete", "patch", "websocket"}:
                continue
            # 取第一个位置参数作为 path
            path_str = "<dynamic>"
            if deco.args:
                first = deco.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    path_str = first.value
            results.append((deco.lineno, f"{obj_name}.{verb}", path_str))
    results.sort()
    return results


def main() -> None:
    assert_paths_exist()

    import vllm_omni  # noqa: F401

    # ---------------------------------------------------------------- #
    # Step H.1: AsyncOmniEngine.__init__ 调用链速览                      #
    # ---------------------------------------------------------------- #
    _hr("Step H.1  AsyncOmniEngine.__init__ 的关键调用链 (源码引用)")
    from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

    src_file = inspect.getsourcefile(AsyncOmniEngine.__init__)
    init_src, init_lineno = inspect.getsourcelines(AsyncOmniEngine.__init__)
    init_src_str = "".join(init_src)

    interesting = (
        "_resolve_stage_configs",
        "compute_replica_layout",
        "_build_logical_stage_init_plans",
        "_initialize_stages",
        "_initialize_stage_replicas",
        "_initialize_llm_replica",
        "spawn_stage_core",
        "complete_stage_handshake",
        "_wait_for_orchestrator_init",
    )
    hits: list[tuple[int, str]] = []
    for i, line in enumerate(init_src):
        for name in interesting:
            if name in line:
                hits.append((init_lineno + i, line.rstrip()))
                break

    print(f"source: {src_file}")
    print(f"AsyncOmniEngine.__init__ 起始行: {init_lineno}\n")
    print("(节选包含关键函数名的行, 看 __init__ 调用了什么:)")
    for lineno, line in hits:
        print(f"  L{lineno:>4}  {line.strip()[:110]}")
    print(
        "\n→ 翻译: 这就是 04 课产出 (stage_configs) 到 worker 子进程之间的全部代码路径:\n"
        "    _resolve_stage_configs           # 内部即 load_and_resolve_stage_configs (04 课)\n"
        "      ↓\n"
        "    compute_replica_layout           # 算每 stage 副本数 + 设备分配\n"
        "      ↓\n"
        "    _build_logical_stage_init_plans  # 拆成具体的 init plan 列表\n"
        "      ↓\n"
        "    _initialize_stages\n"
        "      └ _initialize_stage_replicas\n"
        "          └ _initialize_llm_replica (per replica)\n"
        "              └ spawn_stage_core   ← 真正 fork 子进程\n"
        "              └ complete_stage_handshake   ← 等子进程 READY"
    )

    # ---------------------------------------------------------------- #
    # Step H.2: spawn_stage_core 的 docstring 速览                       #
    # ---------------------------------------------------------------- #
    _hr("Step H.2  spawn_stage_core / StageEngineCoreProc 一句话")
    try:
        sec_mod = importlib.import_module("vllm_omni.engine.stage_engine_core_proc")
        for name in ("spawn_stage_core", "complete_stage_handshake", "StageEngineCoreProc"):
            obj = getattr(sec_mod, name, None)
            if obj is None:
                print(f"  - {name}: <not found>")
                continue
            doc = inspect.getdoc(obj) or "(no docstring)"
            first_line = doc.splitlines()[0]
            print(f"  • {name}\n      {first_line}")
    except Exception as e:
        print(f"  (skipped: {e})")

    print(
        "\n→ 直白说:\n"
        "    每个 StageConfig 对应一个 vllm.v1 子进程 (StageEngineCoreProc).\n"
        "    主进程 fork 它, 通过 ZMQ socket 握手等到 'READY' 信号, 才认为这个\n"
        "    stage 上线. 之后请求通过 ZMQ 在主进程 ↔ 子进程之间打路过.\n"
        "    多个 stage 之间 (e.g. thinker 输出送给 talker) 走专门的 StagePool /\n"
        "    RequestRouter, 后续 lecture 07 会讲."
    )

    # ---------------------------------------------------------------- #
    # Step H.3: API server 静态路由表                                    #
    # ---------------------------------------------------------------- #
    _hr("Step H.3  api_server.py 的 FastAPI 路由表 (AST 静态扫描)")
    api_server_path = Path("vllm_omni/entrypoints/openai/api_server.py").resolve()
    if not api_server_path.is_file():
        api_server_path = Path(__file__).resolve().parents[2] / "vllm_omni/entrypoints/openai/api_server.py"

    routes = _extract_route_paths(api_server_path)
    print(f"source: {api_server_path}")
    print(f"扫描到 {len(routes)} 个 route:\n")
    print(f"  {'line':>6}  {'verb':<22}  path")
    print(f"  {'-' * 6}  {'-' * 22}  {'-' * 40}")
    for lineno, verb, path in routes:
        print(f"  L{lineno:>4}  {verb:<22}  {path}")

    # ---------------------------------------------------------------- #
    # Step H.4: route → handler 速查                                     #
    # ---------------------------------------------------------------- #
    _hr("Step H.4  几条最关心的 route 落到哪个 OpenAIServing* 类")
    routing_cheatsheet = (
        ("/v1/chat/completions", "serving_chat.OpenAIServingChat.create_chat_completion"),
        ("/v1/completions", "serving_completion.OpenAIServingCompletion.create_completion"),
        ("/v1/audio/speech", "serving_speech.OpenAIServingSpeech.create_speech"),
        ("/v1/audio/speech/stream", "serving_speech (websocket; 增量 TTS 流式)"),
        ("/v1/audio/transcriptions", "serving_transcription.OpenAIServingTranscription"),
        ("/v1/images/generations", "serving_image.OpenAIServingImage"),
        ("/v1/videos", "serving_video.OpenAIServingVideo (异步视频任务)"),
        ("/v1/realtime", "serving_realtime (websocket; speech-in/speech-out)"),
        ("/v1/models", "OpenAIServingModels (元数据)"),
        ("/health", "返回 200, 探活"),
    )
    print(f"  {'path':<30}  →  handler")
    print(f"  {'-' * 30}     {'-' * 60}")
    for path, handler in routing_cheatsheet:
        print(f"  {path:<30}  →  {handler}")

    # ---------------------------------------------------------------- #
    # Step H.5: vllm serve 启动后的进程树/端口拓扑                       #
    # ---------------------------------------------------------------- #
    _hr("Step H.5  vllm serve --omni 起来后的进程拓扑 (示意)")
    print(
        "假设你 vllm serve --omni 一个 2-stage 模型 (e.g. minicpmo_4_5: thinker + talker):\n"
        "\n"
        "  主进程 (APIServer + AsyncOmniEngine)\n"
        "    │\n"
        "    ├─ FastAPI/uvicorn  (listen :8000 默认)\n"
        "    │   ├─ POST /v1/chat/completions     → serving_chat\n"
        "    │   ├─ POST /v1/audio/speech         → serving_speech\n"
        "    │   ├─ WS   /v1/realtime             → serving_realtime\n"
        "    │   └─ GET  /v1/models, /health\n"
        "    │\n"
        "    ├─ Orchestrator thread\n"
        "    │   └─ ZMQ socket ↔ stage 子进程\n"
        "    │\n"
        "    ├─ StageEngineCoreProc (stage 0 = thinker, pid=NNNN, GPU 0)\n"
        "    │   └─ vllm v1 worker (一份完整 vllm engine)\n"
        "    │\n"
        "    └─ StageEngineCoreProc (stage 1 = talker, pid=MMMM, GPU 1)\n"
        "        └─ vllm v1 worker\n"
        "\n"
        "请求生命周期 (chat 例子):\n"
        "  curl -> :8000 -> FastAPI route -> serving_chat\n"
        "  serving_chat: messages -> TokensPrompt   (← 这就是 Lecture 06)\n"
        "  -> engine_client.generate(TokensPrompt) -> ZMQ -> stage 0 worker\n"
        "  -> thinker 输出 -> 经 llm2tts bridge -> ZMQ -> stage 1 worker\n"
        "  -> talker 输出 audio 流 -> 主进程 -> stream 回客户端"
    )

    # ---------------------------------------------------------------- #
    # 串到下一步                                                         #
    # ---------------------------------------------------------------- #
    _hr("Done — 接下来:")
    print(
        "  - Lecture 06 (改名前是 05): 第 H.5 那条流程图里 'messages -> TokensPrompt'\n"
        "    这一步, 即 serving_chat._preprocess_chat 内部.\n"
        "  - 后续: llm2tts bridge / audio codec / token2wav."
    )


if __name__ == "__main__":
    main()
