# Spike notes: forced aligner as a 3rd Omni stage (`explore/mfa-3stage`)

> Exploration only — **do not merge**. This branch probes whether the streaming
> TTS word-timestamp feature (issue #3631) can be realized by running the
> forced aligner (`Qwen3-ForcedAligner-0.6B`, a `runner="pooling"` /
> `token_classify` model) as a real Omni pipeline stage, instead of the
> in-process sidecar in PR #4034.
>
> Goal of the spike: run the whole path on real hardware to surface the actual
> engine-level problems, so we can ask maintainers concrete questions.

## TL;DR

The 3-stage path **works end-to-end on real hardware** for a non-stream request:

```
POST /v1/audio/speech {word_timestamps: true, stream: false}
  -> HTTP 200
     body:   WAV audio
     header: X-Word-Timestamps: [{"word":"Hello","start_ms":80,"end_ms":480}, ... 8 words]
```

Pipeline: **Talker (LLM_AR) → Code2Wav (LLM_GENERATION, audio) → ForcedAligner
(LLM_POOLING, on its own GPU)**. Verified on 3×4090 for
`"Hello world, this is a forced aligner test."` (8 correct, monotonic word times).

The biggest **open problem** is streaming (async_chunk) — see below.

## What had to change (per area)

| Area | Change | Notes |
|------|--------|-------|
| `config/stage_config.py` | `StageExecutionType.LLM_POOLING`; `_inject_forced_aligner_stage` appends the aligner stage when `--forced-aligner` is set | injection reuses `deploy/qwen3_tts_forced_aligner.yaml` defaults via `build_forced_aligner_config` |
| `engine/stage_init_utils.py` | `resolve_worker_cls` pooling branch; `extract_stage_metadata` defaults `PoolingParams(task="token_classify")`; **per-stage `model` override** | per-stage model override is the key enabler: omni stages normally share one checkpoint; the aligner is separate |
| `worker/gpu_pooling_worker.py` | `GPUPoolingWorker` (currently a thin `GPUARWorker` subclass) | reuses the AR runner + the two patches below |
| `worker/gpu_ar_model_runner.py` | pooling stage skips the omni HS prefix-cache + mm-extract, goes straight to `_pool()` | fixes a `[n,classify_num]` vs `[n,hidden]` cache mismatch |
| `worker/gpu_model_runner.py` | filter `get_mrope_input_positions` kwargs to the model signature | stock vLLM aligner lacks the `hf_config` kwarg omni passes |
| `engine/orchestrator.py` | `_build_mm_stage_request`: run the aligner stage's own input preprocessor (tokenize + audio features) for a downstream multimodal stage | solves intermediate-stage mm preprocessing + tokenizer placement in one place |
| `core/sched/omni_ar_scheduler.py` | generic per-stage `pooling_output_decoder` hook; decode happens worker-side before IPC | avoids shipping a bare pooler tensor across the omni `dict[str,Tensor]` `pooling_output` schema |
| `stage_input_processors/forced_aligner.py` | `code2wav2aligner` (Code2Wav audio + text → aligner prompt) and `decode_pooling_output` (logits → `word_timestamps_ms`) + shared constants | model-specific logic lives here, not in the generic scheduler |
| `config/model.py`, `engine/arg_utils.py` | declare `pooling_output_decoder` so it propagates to `model_config` | |
| `entrypoints/openai/serving_speech.py` | request word_timestamps → add aligner output modality (route through stage 2); parse aligner `PoolingOutput`; return `X-Word-Timestamps` header | |

Unit tests (GPU-free): decode, serving extractor, stage injection — all pass.

## Open problems / questions for maintainers

1. **Streaming (the actual #3631 use case).** `code2wav2aligner` is only invoked
   in the **non-async** path (`orchestrator._route_output` gates next-stage
   forwarding on `not self.async_chunk`). With `async_chunk: true` (real
   streaming), stage→stage transfer goes through the connector + worker-side
   `*_async_chunk` hooks, which the aligner stage doesn't have. And forced
   alignment is inherently whole-sentence / non-AR, so it can't align per chunk.
   **Q: what's the intended way to run a whole-sentence pooling stage under
   async_chunk?** (e.g. an async_chunk hook that buffers the sentence audio and
   aligns once at sentence end — essentially the sidecar's strategy inside a
   stage.)

2. **Dedicated pooling worker/runner vs. reusing the AR runner.** `GPUPoolingWorker`
   is a no-op subclass; correctness comes from two patches in the AR runner
   (`_pool()` direct path, mrope kwarg filtering). **Q: acceptable, or should
   there be a first-class `GPUPoolingModelRunner`?**

3. **Pooling stage scheduler.** `LLM_POOLING` currently reuses `OmniARScheduler`
   (sync; it finishes on `pooler_output`). **Q: is that the right scheduler, or
   should pooling get its own?**

4. **`pooling_output` payload is `dict[str, Tensor]`.** Word strings can't ride
   in it; this spike carries times as `word_timestamps_ms` and re-segments the
   request text for labels in serving. **Q: preferred channel for word strings
   / a richer payload?**

5. **Response protocol.** Non-stream returns timestamps via an `X-Word-Timestamps`
   header; streaming would need a frame/sidecar. **Q: desired public contract?**

6. **Audio across the stage boundary.** The aligner consumes Code2Wav audio as a
   multimodal input (re-preprocessed). Fine for whole-sentence; under streaming
   this interacts with (1).

## Sidecar (#4034) vs. stage (this branch)

- **Sidecar (#4034):** in-process `vllm.LLM` aligner, sentence-final alignment,
  smallest change, already works for streaming via the shared streaming hook.
- **Stage (this branch):** aligner is a managed Omni stage with its own GPU;
  cleaner long-term and matches "any TTS model benefits", but needs the
  engine-level pieces above and the streaming story (1) resolved.
