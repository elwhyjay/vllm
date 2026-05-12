# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end coverage for ``SpeculativeConfig.attention_backend`` overrides on
speculators-format models.

The field was added in https://github.com/vllm-project/vllm/pull/39930 but the
speculators auto-detect path silently dropped user-provided
``--speculative-config`` fields until https://github.com/vllm-project/vllm/pull/42376.
These tests pin the end-to-end override behavior so the regression cannot recur:

* passing ``attention_backend`` actually routes the drafter through that backend
* a backend incompatible with DFlash (causal-only) is rejected at engine init
  rather than silently ignored
"""

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.v1.attention.backends.registry import AttentionBackendEnum

MODEL_PATH = "nm-testing/dflash-qwen3-8b-speculators"
PROMPTS = ["The capital of France is", "Quantum entanglement is"]


def _load_dflash(attention_backend: str | None) -> LLM:
    spec_config = (
        {"attention_backend": attention_backend} if attention_backend else None
    )
    return LLM(
        model=MODEL_PATH,
        dtype=torch.bfloat16,
        quantization="fp8",
        enforce_eager=True,
        max_model_len=512,
        max_num_seqs=4,
        gpu_memory_utilization=0.85,
        speculative_config=spec_config,
        disable_log_stats=True,
    )


def _cleanup() -> None:
    torch.accelerator.empty_cache()
    cleanup_dist_env_and_memory()


@pytest.mark.slow_test
@pytest.mark.parametrize(
    "backend,expected_enum",
    [
        pytest.param(None, None, id="auto"),
        pytest.param("FLASH_ATTN", AttentionBackendEnum.FLASH_ATTN, id="flash_attn"),
        pytest.param(
            "FLEX_ATTENTION",
            AttentionBackendEnum.FLEX_ATTENTION,
            id="flex_attention",
        ),
    ],
)
def test_dflash_attention_backend_override(backend, expected_enum, monkeypatch):
    """User-provided attention_backend reaches SpeculativeConfig + greedy works."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    llm = _load_dflash(backend)
    try:
        spec_cfg = llm.llm_engine.vllm_config.speculative_config
        assert spec_cfg is not None
        assert spec_cfg.method == "dflash"
        assert spec_cfg.attention_backend == expected_enum

        outputs = llm.generate(PROMPTS, SamplingParams(temperature=0.0, max_tokens=8))
        assert all(o.outputs[0].text for o in outputs)
    finally:
        del llm
        _cleanup()


@pytest.mark.slow_test
def test_dflash_rejects_causal_only_backend(monkeypatch):
    """TRITON_ATTN lacks non-causal support; DFlash override must surface this.

    Before #42376, ``attention_backend=TRITON_ATTN`` was silently dropped and
    DFlash ran with the auto-selected non-causal backend, masking the
    incompatibility from the user.
    """
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    with pytest.raises((RuntimeError, ValueError)):
        _load_dflash("TRITON_ATTN")
    _cleanup()
