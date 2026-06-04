import numpy as np
import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.forced_aligner import (
    _extract_text,
    _extract_waveform,
    code2wav2aligner,
)


pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeCompletion:
    def __init__(self, multimodal_output):
        self.multimodal_output = multimodal_output


class _FakeOutput:
    def __init__(self, multimodal_output, finished=True):
        self.finished = finished
        self.outputs = [_FakeCompletion(multimodal_output)]


def _audio_mm(n=2400, sr=24000):
    return {"audio": torch.zeros(n, dtype=torch.float32), "sr": torch.tensor(sr, dtype=torch.int32)}


def test_extract_waveform_from_audio_key():
    wav, sr = _extract_waveform(_audio_mm(n=100, sr=16000))
    assert isinstance(wav, np.ndarray) and wav.dtype == np.float32
    assert wav.shape == (100,)
    assert sr == 16000


def test_extract_waveform_falls_back_to_model_outputs_and_default_sr():
    wav, sr = _extract_waveform({"model_outputs": [torch.ones(50)]})
    assert wav.shape == (50,)
    assert sr == 24000  # default when sr missing


def test_extract_text_unwraps_additional_information_list():
    prompt = {"additional_information": {"text": ["Hello world."]}}
    assert _extract_text(prompt) == "Hello world."


def test_code2wav2aligner_builds_prompt_and_audio_with_injected_tokenizer():
    captured = {}

    def fake_encode(s):
        captured["prompt"] = s
        return [1, 2, 3]

    prompt = {"additional_information": {"text": ["U.S.A test"], "language": ["English"]}}
    out = code2wav2aligner(
        [_FakeOutput(_audio_mm())],
        prompt,
        False,
        None,
        encode_prompt=fake_encode,
    )

    assert len(out) == 1
    item = out[0]
    assert item["prompt_token_ids"] == [1, 2, 3]
    # audio carried as (np.float32, sr) tuple
    audio, sr = item["multi_modal_data"]["audio"]
    assert isinstance(audio, np.ndarray) and sr == 24000
    # prompt built from official segmentation: "U.S.A" -> "USA"
    assert "USA<timestamp><timestamp>test" in captured["prompt"]
    assert item["additional_information"]["aligner_words"] == [["USA", "test"]]


def test_code2wav2aligner_defers_tokenization_when_no_tokenizer():
    prompt = {"additional_information": {"text": ["hello world"]}}
    out = code2wav2aligner([_FakeOutput(_audio_mm())], prompt)

    assert len(out) == 1
    item = out[0]
    # no tokenizer -> empty ids, prompt string deferred to the stage worker
    assert item["prompt_token_ids"] == []
    assert "aligner_prompt" in item["additional_information"]
    assert "<timestamp>" in item["additional_information"]["aligner_prompt"][0]


def test_code2wav2aligner_skips_unfinished_and_empty():
    # unfinished -> skipped
    assert code2wav2aligner([_FakeOutput(_audio_mm(), finished=False)], {}) == []
    # finished but empty waveform -> skipped
    empty = _FakeOutput({"audio": torch.zeros(0), "sr": torch.tensor(24000)})
    assert code2wav2aligner([empty], {"additional_information": {"text": ["hi"]}}) == []
