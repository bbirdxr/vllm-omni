# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU worker for a pooling/token_classify stage.

This is a feasibility skeleton for running a ``runner="pooling"`` model (e.g.
Qwen3-ForcedAligner) as an Omni pipeline stage rather than as an in-process
sidecar. It currently reuses :class:`GPUARModelRunner`, whose shared base
runner already has the ``is_pooling_model`` -> ``_pool()`` path, so a pooling
model can produce token-classify output through the AR forward loop.

It is intentionally a thin subclass for now; a dedicated pooling runner /
scheduler can replace this once the stage-level pooling path is validated.
"""

from vllm.logger import init_logger

from vllm_omni.worker.gpu_ar_worker import GPUARWorker

logger = init_logger(__name__)


class GPUPoolingWorker(GPUARWorker):
    """GPU worker for a pooling (token_classify/embedding) Omni stage.

    Reuses the AR worker's device init and model-runner construction. The
    underlying runner pools when ``model_config.runner == "pooling"``.
    """

    pass
