"""Native LLMem initialization and supported OPT training, in separate processes."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("estimate", "measure"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import GPUtil
    import pynvml as nvml
    import torch
    import transformers
    import colossalai
    from colossalai.nn.optimizer import HybridAdam
    from colossalai.booster import Booster
    from colossalai.booster.plugin import GeminiPlugin
    source = Path(__file__).resolve().parents[1] / "third_party/llmem_estimator/size_estimator.py"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_hash == "e35a4f29e2e5d672813c1d536ea3fc06ebeff3db88c7ad016676bdb62389c8dd"
    spec = importlib.util.spec_from_file_location("upstream_llmem", source)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    nvml.nvmlInit()
    handle = nvml.nvmlDeviceGetHandleByUUID(args.gpu_uuid)
    def used():
        return nvml.nvmlDeviceGetMemoryInfo(handle).used / 2**20
    def gputil_used():
        matches = [g.memoryUsed for g in GPUtil.getGPUs() if g.uuid == args.gpu_uuid]
        assert len(matches) == 1
        return matches[0]
    baseline = used()
    assert baseline < 1000, f"GPU is not idle: {baseline} MiB"
    stop = threading.Event()
    samples = []
    def sample():
        while not stop.is_set():
            samples.append((time.time_ns(), used()))
            stop.wait(.01)
    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    started_ns = time.time_ns()
    result = dict(mode=args.mode, model=str(args.model), batch=args.batch, sequence=args.sequence,
                  seed=args.seed, gpu_uuid=args.gpu_uuid, baseline_mib=baseline,
                  hostname=socket.gethostname(), pid=os.getpid(), source_sha256=source_hash,
                  torch=torch.__version__, transformers=transformers.__version__,
                  colossalai=colossalai.__version__, status="unresolved",
                  measurement="NVML whole-device samples at 10 ms, allocator peaks separately",
                  input="fixed-shape synthetic nonpadding token IDs; no quality claim")
    try:
        colossalai.launch(config={}, rank=0, world_size=1, host="127.0.0.1",
                          port=args.port, backend="nccl", seed=args.seed)
        assert torch.cuda.device_count() == 1
        torch.cuda.init()
        assert os.getpid() in {p.pid for p in nvml.nvmlDeviceGetComputeRunningProcesses(handle)}
        # Same context correction and initialization procedure as dp_real.py.
        context = int(used()) - gputil_used()
        model = transformers.AutoModelForCausalLM.from_pretrained(args.model).to("cuda")
        model.half()
        torch.cuda.empty_cache()
        model.gradient_checkpointing_enable()
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        ids = torch.randint(4, model.config.vocab_size, (args.batch, args.sequence),
                            generator=generator).cuda()
        inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone())
        plugin = GeminiPlugin(device=torch.cuda.current_device(), placement_policy="cuda",
                              precision="fp16", pin_memory=False, strict_ddp_mode=False,
                              initial_scale=2**5)
        optimizer = HybridAdam(model.parameters(), lr=2e-5, weight_decay=0.0)
        if args.mode == "estimate":
            estimator = upstream.SizeEstimator(model, ids[:2], args.batch, bytes=2,
                                                bytes_input=8, gpu_n=1, tp=0, lm_fp32=False,
                                                m_total=nvml.nvmlDeviceGetMemoryInfo(handle).total/2**20)
            torch.cuda.empty_cache()
            before = gputil_used()
            estimator.get_output_sizes()
            torch.cuda.empty_cache()
            after = gputil_used()
        booster = Booster(plugin=plugin)
        model, optimizer, _, _, _ = booster.boost(model, optimizer)
        torch.cuda.empty_cache()
        if args.mode == "estimate":
            boosted = gputil_used()
            m_pbase = boosted + context - (after - before)
            estimate, batch = estimator.estimate_size(m_init=m_pbase)
            result.update(predicted_peak_mib=float(estimate), predicted_batch=int(batch),
                          m_pbase_mib=m_pbase, cuda_context_correction_mib=context,
                          output_probe_before_mib=before, output_probe_after_mib=after,
                          boosted_memory_mib=boosted, status="estimated")
        else:
            model.train()
            torch.cuda.reset_peak_memory_stats()
            losses = []
            observations = []
            measurement_start = time.time_ns()
            for step in range(3):
                model.zero_grad()
                a = used()
                outputs = model(use_cache=False, **inputs)
                b = used()
                loss = outputs["loss"]
                outputs = None
                booster.backward(loss, optimizer)
                c = used()
                if step == 0:
                    torch.cuda.empty_cache()
                optimizer.step()
                optimizer.zero_grad()
                # Upstream schedules learning rate; three steps are too short
                # for quality evaluation but optimizer state must be allocated.
                d = used()
                losses.append(float(loss.detach()))
                observations.append(dict(step=step+1, before=a, forward=b, backward=c, optimizer=d))
                assert torch.isfinite(loss).item(), "Non-finite training loss"
            torch.cuda.synchronize()
            result.update(status="completed", completed_steps=3, losses=losses,
                          stage_observations=observations,
                          measurement_start_ns=measurement_start,
                          allocator_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                          allocator_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        torch.distributed.destroy_process_group()
    except Exception as error:
        result.update(error=f"{type(error).__name__}: {error}")
        raise
    finally:
        stop.set()
        monitor.join(timeout=2)
        result.update(started_ns=started_ns, finished_ns=time.time_ns(),
                      observed_process_peak_mib=max((v for _, v in samples), default=baseline))
        if result.get("measurement_start_ns"):
            result["training_sampled_peak_mib"] = max(
                (v for t, v in samples if t >= result["measurement_start_ns"]), default=None)
        (args.output/"result.json").write_text(json.dumps(result, indent=2)+"\n")
        with (args.output/"nvml.csv").open("w") as handle_out:
            handle_out.write("timestamp_ns,used_mib\n")
            handle_out.writelines(f"{t},{v}\n" for t, v in samples)
        nvml.nvmlShutdown()


if __name__ == "__main__":
    main()
