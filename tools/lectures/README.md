# vllm-omni × MiniCPM-o-4.5 入门 lectures

跟着 mt 调试 minicpmo-4.5 PR 沉淀下来的一组**自包含、可单独运行**的小脚本，
从「import vllm_omni 发生了什么」一路讲到「prompt 怎么变成 vLLM 引擎可吃的
TokensPrompt」。每节脚本都不会启动真正的引擎，几秒钟就能跑完。

## 运行环境

需要 GPU 机器（lecture 04 调 `current_omni_platform.device_name`，CPU-only 沙盒会失败）。
模型路径写死在 `_common.py`：

```python
MODEL_DIR   = "/cache/wangjinxu/projects/models/MiniCPM-o-4_5"
DEPLOY_YAML = "/cache/wangjinxu/projects/vllm-omni/vllm_omni/deploy/minicpmo_4_5_8x4090.yaml"
```

换机器记得改这两个常量。

## 课程顺序

| # | 文件 | 学什么 | 依赖 |
|---|------|-------|------|
| 01 | `01_config_registration.py` | `import vllm_omni` 的副作用 + `AutoConfig.from_pretrained` | 无 |
| 02 | `02_pipeline_resolution.py` | `_OMNI_PIPELINES` 注册表 + `hf_architectures` 交集匹配 | 01 |
| 03 | `03_model_registry.py`      | `register_omni_models_to_vllm()` + `ModelRegistry` 探针 | 01 |
| 04 | `04_stage_configs.py`       | deploy yaml → `list[StageConfig]` | 01 |
| 05 | `05_stage_spawn_and_api_routing.py` | StageConfig list 怎么变成跑着的 worker 子进程 + FastAPI 路由表 | 04 |
| 06 | `06_chat_to_engine_prompt.py` | `apply_chat_template` → `BatchEncoding` → `TokensPrompt` | 仅需 tokenizer |
| 07 | `07_request_lifecycle.py` | HTTP → FastAPI → `OmniOpenAIServingChat` → `engine_client.generate` 全链路 | 01 |
| 08 | `08_llm2tts_bridge.py` | stage 0 thinker → stage 1 talker 的 bridge (`custom_process_input_func` + `OmniTokensPrompt.additional_information`) | 01 + 04 |

## 单独跑

```bash
# 单节
python tools/lectures/01_config_registration.py
python tools/lectures/04_stage_configs.py --full     # 04 支持 --full 参数

# 一键全跑
python tools/lectures/run_all.py
python tools/lectures/run_all.py --only 02 03         # 只跑指定课
python tools/lectures/run_all.py --skip 04            # 跳过指定课
```

## 加新课

1. 复制现有的某节作模板（推荐 `01_config_registration.py`，结构最简单）。
2. 顶部 docstring 写清楚：学习目标 / 为什么重要 / 依赖 / 运行命令。
3. 把代码包在 `def main(): ...`，并保留 `if __name__ == "__main__": main()`。
4. 在 `run_all.py` 的 `LECTURES` 列表里追加一行。
5. 在本 README 的课程表里加一行。

## 课程之间的串联（一句话总结）

```
01 把 model 的 config.json 读进来 (model_type, architectures)
   ↓
02 用 model_type / architectures 找到匹配的 PipelineConfig (拓扑: thinker + talker)
   ↓
03 确认 PipelineConfig 里要的模型类已经被 vllm-omni 注册进 vLLM ModelRegistry
   ↓
04 yaml 里写的 stage 配置被解析成 list[StageConfig]
   ↓
05 AsyncOmniEngine 把 list[StageConfig] 落地: 算 replica 布局 → spawn worker
   子进程 → 起 FastAPI → 暴露 /v1/chat/completions、/v1/audio/speech 等 route
   ↓
06 请求来了, serving_chat / serving_speech 接住, 经 apply_chat_template + 解包
   → vLLM TokensPrompt
   ↓
07 TokensPrompt 通过 engine_client.generate (AsyncOmni) 跨入 engine 边界, 主进程
   通过 ZMQ 把 prompt 路由到 stage 0 worker; SSE/JSON 三种响应类型按 typed
   return 自动分发回客户端
   ↓
08 stage 0 thinker 输出 (hidden states + 生成 token + TTS 区间标记) 经 per-model
   bridge 函数 (custom_process_input_func, 如 llm2tts) 打包成 OmniTokensPrompt
   送给 stage 1 talker; 真数据搭便车在 additional_information 里, 绕过 vllm
   标准 prompt 接口
   ↓ (送进 stage 1 talker; talker 输出再经 token2wav 变成 wav 波形, 后续课程)
```

## TODO 后续课程候选

- 09 audio codebook & token2wav: talker 输出怎么变 wav (token2wav / streaming)
- 10 多模态输入预处理: image / audio / video 怎么进 multi_modal_data
- 11 vllm-omni vs vllm v1: 哪些组件被覆盖、哪些复用
