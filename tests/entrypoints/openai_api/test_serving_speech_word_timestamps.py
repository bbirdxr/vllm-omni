from types import SimpleNamespace

import pytest
import torch

from vllm_omni.entrypoints.openai.serving_speech import _extract_word_timestamps
from vllm_omni.model_executor.stage_input_processors.forced_aligner import WORD_TIMESTAMPS_MS_KEY


pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _PoolingOutput:
    """Mimics vLLM PoolingOutput: res.outputs is this object (not a list)."""

    def __init__(self, data):
        self.data = data


def _aligner_res(ms, additional_information=None):
    res = SimpleNamespace(outputs=_PoolingOutput({WORD_TIMESTAMPS_MS_KEY: ms}))
    res.additional_information = additional_information
    return res


def test_extract_word_timestamps_with_aligner_words():
    ms = torch.tensor([[0, 480], [480, 960]], dtype=torch.int32)
    res = _aligner_res(ms, additional_information={"aligner_words": [["Hello", "world"]]})

    out = _extract_word_timestamps(res)

    assert out == [
        {"word": "Hello", "start_ms": 0, "end_ms": 480},
        {"word": "world", "start_ms": 480, "end_ms": 960},
    ]


def test_extract_word_timestamps_uses_fallback_words():
    ms = torch.tensor([[0, 300], [300, 700]], dtype=torch.int32)
    res = _aligner_res(ms, additional_information=None)

    out = _extract_word_timestamps(res, fallback_words=["foo", "bar"])

    assert [o["word"] for o in out] == ["foo", "bar"]
    assert [(o["start_ms"], o["end_ms"]) for o in out] == [(0, 300), (300, 700)]


def test_extract_word_timestamps_returns_none_for_audio_output():
    # A normal audio CompletionOutput (list of outputs, no word_timestamps_ms).
    audio_res = SimpleNamespace(
        outputs=[SimpleNamespace(multimodal_output={"audio": torch.zeros(4)})],
        additional_information=None,
    )
    assert _extract_word_timestamps(audio_res) is None
