"""Predeclared reservation, state-release, and iteration-cost follow-ups."""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import random

from memory_tuner import electronics_experiment as study
from memory_tuner import offload_factorial_protocol as prior


def base_row(template, *, panel, model, dataset, seed, block_index, period, condition,
             gpu_count=4, steps=1, full_logging=True, timeout=900):
    short = {"Qwen/Qwen2.5-7B-Instruct": "q7", "mistralai/Mistral-7B-Instruct-v0.3": "m7",
             "Qwen/Qwen2.5-3B-Instruct": "q3", "Qwen/Qwen2.5-1.5B-Instruct": "q15"}[model]
    block = f"{panel}-{short}-{dataset}-s{seed}"
    row = dict(template)
    row.update(experiment_id=block+"-"+condition, run_group=f"electronics_{panel}_20261016",
               study_stage="confirmatory", revision_study=f"electronics_{panel}",
               model=model, model_family="mistral" if short=="m7" else "qwen25",
               model_revision=study.REVISIONS[model], block_id=block, pair_id=block,
               block_index=str(block_index), period=str(period), condition=condition,
               configuration_id=f"{panel}-{short}-{dataset}-{condition}",
               configuration_level="independent-controls", case_id=f"{panel}-{short}-{dataset}",
               sequence_id=str(block_index), training_seed=str(seed),
               gpu_count=str(gpu_count), gpu_subset_ordinals=";".join(map(str,range(gpu_count))),
               lora_rank="64", lora_alpha="32", use_kl_loss="True",
               total_training_steps=str(steps), allocator_trace_enabled=str(full_logging),
               allocator_trace_sync=str(int(full_logging)), capture_response_lengths="True",
               per_process_timeout_seconds=str(timeout), save_freq="-1", test_freq="-1",
               val_before_train="False", novelty="fresh selected-grid follow-up")
    return row


def reservation_rows():
    templates = study.read_rows(study.ROOT/"benchmark/offload_factorial/matrix.csv")
    blocks = list(itertools.product(
        ("Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"),
        ("gsm8k","code_heavy_tail"), (451,452,453)))
    rng = random.Random(202610164)
    rng.shuffle(blocks)
    factors=list(itertools.product(("False","True"),(".45",".50",".55",".60",".65",".70")))
    rng.shuffle(factors)
    shifts=list(range(12))
    rng.shuffle(shifts)
    sequence=(0,1,11,2,10,3,9,4,8,5,7,6)
    rows=[]
    for index,(model,dataset,seed) in enumerate(blocks):
        template=next(r for r in templates if r["model"]==model and r["dataset"]==dataset)
        for period,factor_index in enumerate(sequence,1):
            offload,reservation=factors[(factor_index+shifts[index])%12]
            reservation="0"+reservation
            condition=f"off{int(offload=='True')}-r{round(float(reservation)*100)}"
            row=base_row(template,panel="reservation",model=model,dataset=dataset,seed=seed,
                         block_index=index,period=period,condition=condition)
            row.update(actor_micro_batch="8",parameter_offload=offload,
                       optimizer_offload="False",free_cache_engine="True",
                       vllm_gpu_memory_utilization=reservation,sequence_id=str(shifts[index]))
            rows.append(row)
    return rows


def reservation_predictions(rows):
    _, templates=prior.planned(study.ROOT)
    targets=[]
    seen=set()
    for row in rows:
        if row["configuration_id"] in seen:
            continue
        seen.add(row["configuration_id"])
        template=next(t for t in templates if t["model_id"]==row["model"]
                      and t["legacy_dataset_id"]==row["dataset"])
        target=dict(template)
        for key in ("configuration_id","case_id","actor_micro_batch","parameter_offload",
                    "vllm_gpu_memory_utilization","configuration_level","condition"):
            target[key]=row[key]
        targets.append(target)
    predictions=prior.predictions(study.ROOT,targets)
    # An unsupported target is not an implicit rejection or zero-valued peak.
    for target in targets:
        predictions.append(dict(configuration_id=target["configuration_id"],method="LLMem_native",
                                prediction_type="unsupported",predicted_state=None,
                                predicted_peak_mib=None,approved=None,
                                reason="Native OPT/Gemini training comparator has no validated LoRA/FSDP/vLLM mapping."))
    return predictions


def release_rows():
    old=study.read_rows(study.ROOT/"profiles/standard_grpo/same-node-trials.csv")
    template=study.read_rows(study.ROOT/"benchmark/offload_factorial/matrix.csv")[0]
    rows=[]
    rng=random.Random(202610165)
    anchors=[("Qwen/Qwen2.5-1.5B-Instruct","0.70",("0.60","0.70","0.75")),
             ("Qwen/Qwen2.5-3B-Instruct","0.65",("0.55","0.65","0.70"))]
    index=0
    for model,anchor,reservations in anchors:
        base=next(r for r in old if r["model"]==model and r["dataset"]=="gsm8k"
                  and float(r["vllm_gpu_memory_utilization"])==float(anchor) and r["condition"]=="release")
        settings=dict(template)
        for key in ("actor_micro_batch","rollout_logprob_micro_batch","ref_logprob_micro_batch",
                    "parameter_offload","optimizer_offload","max_prompt_length","max_response_length",
                    "max_model_len","max_num_seqs","rollout_n","train_batch_size","train_max_samples","val_max_samples"):
            settings[key]=base[key]
        settings.update(data_dir=str(study.ROOT/"data/gsm8k"),dataset="gsm8k",workload_name="gsm8k")
        for seed in (501,502,503):
            factors=list(itertools.product(("True","False"),reservations))
            rng.shuffle(factors)
            for period,(release,reservation) in enumerate(factors,1):
                condition=f"release{int(release=='True')}-r{round(float(reservation)*100)}"
                row=base_row(settings,panel="release",model=model,dataset="gsm8k",seed=seed,
                             block_index=index,period=period,condition=condition,gpu_count=2)
                row.update(free_cache_engine=release,vllm_gpu_memory_utilization=reservation)
                rows.append(row)
            index+=1
    return rows


def timing_rows():
    templates=study.read_rows(study.ROOT/"benchmark/offload_factorial/matrix.csv")
    rows=[]
    rng=random.Random(202610166)
    cases=[("Qwen/Qwen2.5-7B-Instruct","gsm8k"),
           ("mistralai/Mistral-7B-Instruct-v0.3","code_heavy_tail")]
    for index,(case,seed) in enumerate(itertools.product(cases,(471,472,473))):
        model,dataset=case
        template=next(r for r in templates if r["model"]==model and r["dataset"]==dataset)
        factors=[("False","0.45","True"),("True","0.45","True"),
                 ("True","0.70","True"),("True","0.45","False")]
        rng.shuffle(factors)
        for period,(offload,reservation,release) in enumerate(factors,1):
            condition=f"off{int(offload=='True')}-r{round(float(reservation)*100)}-release{int(release=='True')}"
            row=base_row(template,panel="timing",model=model,dataset=dataset,seed=seed,
                         block_index=index,period=period,condition=condition,
                         steps=20,full_logging=False,timeout=4200)
            row.update(actor_micro_batch="8",parameter_offload=offload,
                       optimizer_offload="False",vllm_gpu_memory_utilization=reservation,
                       free_cache_engine=release,train_max_samples="640")
            rows.append(row)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel",choices=("reservation","release","timing"))
    args=parser.parse_args()
    build={"reservation":reservation_rows,"release":release_rows,"timing":timing_rows}[args.panel]
    rows=build()
    caps={"reservation":dict(wall_seconds=14400,payload_timeout_seconds=900,max_concurrent_allocations=4),
          "release":dict(wall_seconds=7200,payload_timeout_seconds=900,max_concurrent_allocations=3),
          "timing":dict(wall_seconds=18000,payload_timeout_seconds=4200,max_concurrent_allocations=2)}
    study.freeze(study.SOURCE/study.DIRECTORY/args.panel,rows,study=args.panel,caps=caps[args.panel],
                 predictions=reservation_predictions(rows) if args.panel=="reservation" else None)


if __name__=="__main__":
    main()
