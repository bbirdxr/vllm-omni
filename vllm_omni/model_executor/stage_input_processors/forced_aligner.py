# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SPIKE (explore/mfa-3stage): stage input processor for a forced-aligner stage.

Bridges a Code2Wav (audio) stage to a pooling forced-aligner stage:

    Talker (LLM_AR) -> Code2Wav (LLM_GENERATION) -> ForcedAligner (LLM_POOLING)

``code2wav2aligner`` is a synchronous ``custom_process_input_func`` (the same
shape as ``qwen3_tts.talker2code2wav``). It runs in the orchestrator process,
reads the synthesized waveform from the upstream Code2Wav output and the
ground-truth text from the original request, then builds the aligner prompt
(``<timestamp>`` markers per word) + audio multimodal input.

Open wiring gaps for the full stage path (tracked in the spike notes):
  * ``build_engine_core_request_from_tokens`` does not yet forward
    ``multi_modal_data`` into ``mm_features`` for a non-thinker stage.
  * the aligner stage needs ``PoolingParams`` (not ``SamplingParams``).
  * the pooling output (token_classify logits) needs a path back to the
    serving layer as word timestamps.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger

from vllm_omni.model_executor.stage_input_processors.tts_utils import (
    extract_language_from_prompt,
)
from vllm_omni.utils.qwen3_force_align_processor import build_prompt, segment_words

logger = init_logger(__name__)


def _extract_waveform(multimodal_output: dict[str, Any]) -> tuple[np.ndarray, int]:
    """Pull a mono float32 waveform + sample rate from a Code2Wav output."""
    wav = multimodal_output.get("audio")
    if wav is None:
        wav = multimodal_output.get("model_outputs")
    if isinstance(wav, list):
        wav = wav[0] if wav else None
    if isinstance(wav, torch.Tensor):
        wav_np = wav.detach().cpu().to(torch.float32).flatten().numpy()
    elif isinstance(wav, np.ndarray):
        wav_np = wav.astype(np.float32).flatten()
    else:
        wav_np = np.zeros(0, dtype=np.float32)

    sr_raw = multimodal_output.get("sr")
    if isinstance(sr_raw, list):
        sr_raw = sr_raw[-1] if sr_raw else None
    if isinstance(sr_raw, torch.Tensor):
        sr = int(sr_raw.reshape(-1)[-1].item()) if sr_raw.numel() > 0 else 24000
    elif sr_raw is None:
        sr = 24000
    else:
        sr = int(sr_raw)
    return wav_np, sr


def _extract_text(prompt: Any, index: int = 0) -> str:
    """Read the ground-truth text the aligner needs from the original prompt."""
    if not isinstance(prompt, dict):
        return ""
    info = prompt.get("additional_information") or {}
    text = info.get("text")
    if isinstance(text, list):
        return str(text[index]) if index < len(text) else (str(text[0]) if text else "")
    return str(text) if text else ""


def code2wav2aligner(
    source_outputs: list[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
    _streaming_context: Any | None = None,
) -> list[Any]:
    """Build forced-aligner stage inputs from finished Code2Wav audio.

    Emits a **text prompt** (``prompt`` string + ``multi_modal_data`` audio).
    Tokenization and audio feature extraction are deferred to the aligner
    stage's input preprocessor (run by the orchestrator), so this function
    needs neither the aligner tokenizer nor its mm processor and stays
    unit-testable on CPU.
    """
    from vllm_omni.inputs.data import OmniTextPrompt

    aligner_inputs: list[OmniTextPrompt] = []
    for i, src in enumerate(source_outputs):
        if not getattr(src, "finished", False):
            # Sentence-final alignment: only run once the audio is complete.
            continue
        mm = src.outputs[0].multimodal_output or {}
        wav_np, sample_rate = _extract_waveform(mm)
        if wav_np.size == 0:
            logger.warning("code2wav2aligner: empty waveform for request; skipping alignment.")
            continue

        text = _extract_text(prompt, index=i)
        language_list = extract_language_from_prompt(prompt, index=i)
        language = language_list[0] if language_list else None
        words = segment_words(text, language)
        prompt_str = build_prompt(words)

        additional_information: dict[str, Any] = {
            "text": [text],
            "aligner_words": [words],
        }
        if language_list is not None:
            additional_information["language"] = language_list

        aligner_inputs.append(
            OmniTextPrompt(
                prompt=prompt_str,
                multi_modal_data={"audio": (wav_np, sample_rate)},
                additional_information=additional_information,
            )
        )
    return aligner_inputs
