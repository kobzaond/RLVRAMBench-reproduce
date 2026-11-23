"""Enable the paired-study UUID bridge in fresh Ray/vLLM Python subprocesses."""
import os

if os.environ.get("RLVRAM_GPU_UUIDS") is not None:
    from memory_tuner.vllm_uuid_compat import install

    install()
