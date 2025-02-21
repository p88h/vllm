# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.config import LoadFormat
from vllm.device_allocator.cumem import CuMemAllocator
from vllm.utils import GiB_bytes

from ..conftest import MODEL_WEIGHTS_S3_BUCKET
from ..utils import fork_new_process_for_each_test

@fork_new_process_for_each_test
@pytest.mark.parametrize(
    "model, use_v1",
    [
        # sleep mode with pytorch checkpoint
        ("facebook/opt-125m", False),
    ])
def test_end_to_end(model: str, use_v1: bool):
    import os
    os.environ["VLLM_USE_V1"] = "1" if use_v1 else "0"
    free, total = torch.cuda.mem_get_info()
    used_bytes_baseline = total - free  # in case other process is running
    load_format = LoadFormat.AUTO
    if "Llama" in model:
        load_format = LoadFormat.RUNAI_STREAMER
    llm = LLM(model, load_format=load_format, enable_sleep_mode=True)
    prompt = "How are you?"
    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    output = llm.generate(prompt, sampling_params)

    # the benefit of `llm.sleep(level=2)` is mainly CPU memory usage,
    # which is difficult to measure in the test. therefore, we only
    # test sleep level 1 here.
    llm.sleep(level=1)

    free_gpu_bytes_after_sleep, total = torch.cuda.mem_get_info()
    used_bytes = total - free_gpu_bytes_after_sleep - used_bytes_baseline
    # now the memory usage is mostly cudagraph memory pool,
    # and it should be less than the model weights (1B model, 2GiB weights)
    assert used_bytes < 2 * GiB_bytes

    llm.wake_up()
    output2 = llm.generate(prompt, sampling_params)

    # cmp output
    assert output[0].outputs[0].text == output2[0].outputs[0].text

    del os.environ["VLLM_USE_V1"]
