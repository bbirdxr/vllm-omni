"""Debug walk-through: how vllm-omni resolves & registers MiniCPM-o-4.5.

Run this WITHOUT a running vllm-omni serve. It only walks the registration
side (B/C/D/E from the checklist) and stops before any GPU work.

Usage (interactive pdb at each stop):
    python tools/explore_minicpmo_registration.py

Or step through it in Cursor: Run > Python Debugger > Debug Python File.
"""

# Exploratory script: imports are interleaved with print/inspect blocks for
# step-by-step demonstration, which trips ruff's "import at top" rule.
# ruff: noqa: E402

from __future__ import annotations

import os
import sys

MODEL_DIR = "/cache/wangjinxu/projects/models/MiniCPM-o-4_5"

SEP = "=" * 70


def _hr(title: str) -> None:
    print()
    print(SEP)
    print(f"  {title}")
    print(SEP)


# ---------------------------------------------------------------------------
# Step 0: nothing imported yet
# ---------------------------------------------------------------------------
_hr("Step 0  pre-import sanity")
print(f"sys.path[0:3]={sys.path[:3]}")
print(f"MODEL_DIR exists: {os.path.isdir(MODEL_DIR)}")

# ---------------------------------------------------------------------------
# Step B: import vllm_omni → triggers configs/__init__.py → AutoConfig.register
# ---------------------------------------------------------------------------
_hr("Step B  import vllm_omni → custom config registration")
from transformers import AutoConfig

import vllm_omni  # noqa: F401   ← side-effect: registers configs

# AutoConfig has a private _model_mapping; check which model_type we now have
mapping = AutoConfig._mapping if hasattr(AutoConfig, "_mapping") else None
try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    omni_models = [m for m in ("voxcpm2", "qwen3_tts", "fish_speech", "cosyvoice3") if m in CONFIG_MAPPING]
    print(f"vllm-omni-registered model_types in transformers CONFIG_MAPPING: {omni_models}")
except Exception as e:
    print(f"(could not inspect CONFIG_MAPPING: {e})")

# breakpoint()  # ← uncomment to inspect interactively

# ---------------------------------------------------------------------------
# Step A→B/C: load this model's config.json via AutoConfig (trust_remote_code)
# ---------------------------------------------------------------------------
_hr("Step A→B/C  AutoConfig.from_pretrained(MODEL_DIR)")
hf_config = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
print(f"type:           {type(hf_config).__name__}")
print(f"model_type:     {getattr(hf_config, 'model_type', None)}")
print(f"architectures:  {getattr(hf_config, 'architectures', None)}")
print(f"version:        {getattr(hf_config, 'version', None)}")

# breakpoint()

# ---------------------------------------------------------------------------
# Step C: peek at PIPELINE registry and find which entry matches
# ---------------------------------------------------------------------------
_hr("Step C  pipeline_registry.py → which pipeline matches?")
from vllm_omni.config.pipeline_registry import _OMNI_PIPELINES

print(f"_OMNI_PIPELINES keys ({len(_OMNI_PIPELINES)}):")
for k in sorted(_OMNI_PIPELINES.keys()):
    print(f"  - {k}")

mt = getattr(hf_config, "model_type", None)
print(f"\nmodel_type='{mt}' in registry? {mt in _OMNI_PIPELINES}")
if mt not in _OMNI_PIPELINES:
    print("→ will fall through to hf_architectures matching (Step D below)")

# ---------------------------------------------------------------------------
# Step D: pipeline.py → load PipelineConfig and inspect hf_architectures
# ---------------------------------------------------------------------------
_hr("Step D  pipeline.py → MINICPMO_4_5_PIPELINE")
import importlib

mod = importlib.import_module("vllm_omni.model_executor.models.minicpmo_4_5.pipeline")
pipeline_cfg = getattr(mod, "MINICPMO_4_5_PIPELINE")
print(f"model_type:        {pipeline_cfg.model_type}")
print(f"model_arch:        {pipeline_cfg.model_arch}")
print(f"hf_architectures:  {pipeline_cfg.hf_architectures}")
print(f"# stages:          {len(pipeline_cfg.stages)}")
for st in pipeline_cfg.stages:
    print(
        f"  stage {st.stage_id}: model_stage={st.model_stage} "
        f"execution_type={st.execution_type} model_arch={st.model_arch}"
    )

# Manually verify the auto-detection logic from stage_config.py:1094-1107
_hr("Step C+D  manual auto-detection like stage_config.py does it")
hf_archs = set(getattr(hf_config, "architectures", None) or [])
print(f"hf_archs from config.json:                     {hf_archs}")
print(f"pipeline.hf_architectures:                     {set(pipeline_cfg.hf_architectures)}")
intersection = hf_archs & set(pipeline_cfg.hf_architectures)
print(f"intersection (must be non-empty to match):     {intersection}")
print(f"→ pipeline {'WILL' if intersection else 'WILL NOT'} be selected")

# breakpoint()

# ---------------------------------------------------------------------------
# Step E: ModelRegistry – are vllm-omni archs registered into vllm?
# ---------------------------------------------------------------------------
_hr("Step E  ModelRegistry – vllm-omni → vllm")
from vllm.model_executor.models import ModelRegistry

# Trigger the same plugin path that vllm worker subprocesses use
from vllm_omni.engine.arg_utils import register_omni_models_to_vllm

register_omni_models_to_vllm()

archs = ModelRegistry.get_supported_archs()
relevant = sorted(a for a in archs if "MiniCPM" in a or "minicpm" in a.lower())
print(f"MiniCPM-related archs in ModelRegistry ({len(relevant)}):")
for a in relevant:
    print(f"  - {a}")

# Probe specific names mentioned in pipeline.model_arch and the
# init_vllm_registered_model() short-name lookup.
_hr("Step E  probes")
for name in (
    "MiniCPMO45OmniForConditionalGeneration",
    "MiniCPMO45OmniLLMForConditionalGeneration",
    "MiniCPMO45OmniTTSForConditionalGeneration",
    "MiniCPMO45OmniLLMModel",  # the OLD short name — should be MISSING after leader's commit
    "MiniCPMO45OmniTTSModel",  # ditto
    "MiniCPMO",  # the upstream HF arch on this checkpoint
):
    print(f"  {name:55s} {'YES' if name in archs else 'MISSING'}")

# ---------------------------------------------------------------------------
# Done.
# ---------------------------------------------------------------------------
_hr("Done — registration walk-through complete")
print(
    "Next step in real serve flow would be (NOT exercised here):\n"
    "  1. ModelConfig._auto_detect_model_type() → picks pipeline\n"
    "  2. AsyncOmniEngine launches per-stage Engines\n"
    "  3. Each Engine spawns workers, workers load_general_plugins()\n"
    "  4. vllm.ModelRegistry.resolve_model_cls(architectures) → finds the class\n"
    "  5. model.__init__(vllm_config=...) → calls init_vllm_registered_model(\n"
    "       architectures=['MiniCPMO45OmniLLMForConditionalGeneration']) for thinker\n"
    "  6. load_weights(...) → reports loaded set (must use 'tts_obj.' prefix)"
)
# 在脚本末尾加
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

# 不带 use_tts_template
prompt_no_tts = tok.apply_chat_template(
    [{"role": "user", "content": "你好"}],
    tokenize=False,
    add_generation_prompt=True,
)

# 带 use_tts_template=True
prompt_with_tts = tok.apply_chat_template(
    [{"role": "user", "content": "你好"}],
    tokenize=False,
    add_generation_prompt=True,
    use_tts_template=True,
)


# 带 use_tts_template=True
prompt_with_tts_template = tok.apply_chat_template(
    [{"role": "user", "content": "你好"}],
    tokenize=True,
    add_generation_prompt=True,
    use_tts_template=True,
)

print("--- 不带 ---")
print(repr(prompt_no_tts))
print("--- 带 use_tts_template=True ---")
print(repr(prompt_with_tts))
print("--- 带 use_tts_template=True ---")
print(repr(prompt_with_tts_template))

_hr("Step F  load_and_resolve_stage_configs → config_path & stage_configs")
# 这就是 AsyncOmniEngine.__init__ 里 _resolve_stage_configs(...) 内部调的函数。
# 我们直接调它，看 yaml 怎么被解析成 list[StageConfig]，根本不用起引擎。
from vllm_omni.entrypoints.utils import load_and_resolve_stage_configs

DEPLOY_YAML = "/cache/wangjinxu/projects/vllm-omni/vllm_omni/deploy/minicpmo_4_5_8x4090.yaml"

config_path, stage_configs = load_and_resolve_stage_configs(
    model=MODEL_DIR,
    stage_configs_path=None,
    kwargs={},
    deploy_config_path=DEPLOY_YAML,
)
print(f"config_path:    {config_path}")
print(f"# stage_configs: {len(stage_configs)}")
for cfg in stage_configs:
    stage_id = getattr(cfg, "stage_id", None)
    stage_type = getattr(cfg, "stage_type", None)
    devices = getattr(getattr(cfg, "runtime", None), "devices", None)
    ea = getattr(cfg, "engine_args", None)
    tp = getattr(ea, "tensor_parallel_size", None) if ea is not None else None
    mml = getattr(ea, "max_model_len", None) if ea is not None else None
    gmu = getattr(ea, "gpu_memory_utilization", None) if ea is not None else None
    print(
        f"  stage {stage_id}: type={stage_type}  devices={devices!r}  TP={tp}  max_model_len={mml}  gpu_mem_util={gmu}"
    )

# 如果想看完整原始 yaml 字段，把整个 cfg 打出来：
# from omegaconf import OmegaConf
# print(OmegaConf.to_yaml(stage_configs[0]))


_hr("Step G  apply_chat_template → engine_prompt 的解包")

# MiniCPM-o-4.5 自定义 tokenizer 的 apply_chat_template(tokenize=True, ...)
# 返回的是 BatchEncoding(dict-like)，不是 list[int]！
# 这正是为什么 _preprocess_chat 必须做"解包"才能塞给 vLLM。
from vllm.inputs import TokensPrompt

raw = tok.apply_chat_template(
    [{"role": "user", "content": "你好"}],
    tokenize=True,
    add_generation_prompt=True,
    use_tts_template=True,
)
print(f"apply_chat_template 返回类型:  {type(raw).__name__}")
print(f"raw 内容: {raw}")

# 解包：取出 input_ids → list[int]
if isinstance(raw, dict) or hasattr(raw, "get"):
    token_ids = raw["input_ids"]
else:
    token_ids = list(raw)
print(f"\n解包后 token_ids 类型:        {type(token_ids).__name__}")
print(f"token_ids 长度:               {len(token_ids)}")
print(f"末尾 5 个 token id:           {token_ids[-5:]}")
print(f"最后一个 token decode:        {tok.decode([token_ids[-1]])}  (应该是 <|tts_bos|>)")

# 这才是真正可以塞给 self.engine_client.generate(...) 的 engine_prompt
engine_prompt: TokensPrompt = {
    "prompt_token_ids": token_ids,
    # 没有图/音频，所以 multi_modal_data 省略
}
print("\n=== engine_prompt (vllm TokensPrompt) ===")
print(engine_prompt)
print("\n核心要点: vLLM 的 TokensPrompt['prompt_token_ids'] 必须是 list[int],")
print("          不是 BatchEncoding，也不是 {input_ids, attention_mask} dict。")
print("          attention_mask 在 vLLM 里由 PagedAttention 内部处理，无需手传。")
