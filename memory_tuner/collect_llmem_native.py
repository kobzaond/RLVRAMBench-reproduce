"""Validate raw LLMem predictions and separately measured native training."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import mean

from memory_tuner.device_contract import digest
from memory_tuner.electronics_experiment import require


def read_trace(path):
    with path.open() as handle:
        rows=list(csv.DictReader(handle))
    values=[(int(r["timestamp_ns"]),float(r["used_mib"])) for r in rows]
    require(len(values)>1,"Missing memory samples")
    require(all(math.isfinite(v) and 0<=v<=40960 for _,v in values),"Invalid memory value")
    require(all(a[0]<b[0] for a,b in zip(values,values[1:])),"Nonmonotonic memory clock")
    return values


def collect(directory, output, source_commit, protocol_hash):
    require(digest(directory/"protocol.json")==protocol_hash,"Protocol anchor differs")
    protocol=json.loads((directory/"protocol.json").read_text())
    require(digest(directory/"matrix.csv")==protocol["matrix_sha256"],"Matrix differs")
    with (directory/"matrix.csv").open() as handle:
        rows=list(csv.DictReader(handle))
    observations=[]
    for row in rows:
        result=dict(row,status="unresolved",reason=None)
        block=output/("block-"+row["block_index"])
        pair=block/row["experiment_id"]
        try:
            claim=json.loads((block/"claim.json").read_text())
            require(claim["source_commit"]==source_commit and claim["protocol_sha256"]==protocol_hash,
                    "Source/protocol identity differs")
            require(row in claim["rows"],"Claim differs from matrix")
            terminal=json.loads((pair/"terminal.json").read_text())
            require(terminal["row"]==row,"Terminal identity differs")
            actual={str(p.relative_to(pair)):digest(p) for p in pair.rglob("*")
                    if p.is_file() and p.name!="terminal.json"}
            require(actual==terminal["files_sha256"],"Evidence changed after terminal receipt")
            records={}
            for mode in ("estimate","measure"):
                execution=terminal["executions"][mode]
                require(execution["exit_code"]==0 and not execution.get("timeout"),"Subprocess did not complete")
                data=json.loads((pair/mode/"result.json").read_text())
                require(data["mode"]==mode and data["gpu_uuid"]==claim["gpu_uuid"],"Mode/device differs")
                require(data["model"]==protocol["models"][row["model"]]["path"],"Model differs")
                require(all(data[k]==int(row[k]) for k in ("batch","sequence","seed")),"Input settings differ")
                require(data["source_sha256"]==protocol["upstream_estimator_sha256"],"Estimator source differs")
                require(data["torch"]=="2.0.1+cu117" and data["transformers"]=="4.29.2"
                        and data["colossalai"]=="0.3.0","Native runtime differs")
                require(execution["started_ns"]<=data["started_ns"]<data["finished_ns"]<=execution["finished_ns"],
                        "Subprocess lifetime differs")
                require(math.isfinite(data["baseline_mib"]) and 0<=data["baseline_mib"]<1000,
                        "Pre-workload baseline exceeds frozen gate")
                trace=read_trace(pair/mode/"nvml.csv")
                require(all(execution["started_ns"]<=t<=execution["finished_ns"] for t,_ in trace),
                        "Raw samples lie outside subprocess lifetime")
                require(max(v for _,v in trace)==data["observed_process_peak_mib"],"Reported/raw peak differs")
                records[mode]=data
                if mode=="measure":
                    require(data["started_ns"]<data["measurement_start_ns"]<data["finished_ns"],
                            "Invalid training-window start")
                    measured=max(v for t,v in trace if t>=data["measurement_start_ns"])
                    require(measured>0 and measured==data["training_sampled_peak_mib"],"Training window differs")
            prediction,measurement=records["estimate"],records["measure"]
            require(prediction["status"]=="estimated" and measurement["status"]=="completed","Native outcomes incomplete")
            require(prediction["finished_ns"]<measurement["started_ns"],"Prediction followed measured outcome")
            require(prediction["pid"]!=measurement["pid"],"Prediction and measurement must use fresh processes")
            require(measurement["completed_steps"]==3 and len(measurement["losses"])==3
                    and all(math.isfinite(v) for v in measurement["losses"]),"Invalid completed work")
            require([s["step"] for s in measurement["stage_observations"]]==[1,2,3],"Missing optimizer-step records")
            predicted=prediction["predicted_peak_mib"]
            require(math.isfinite(predicted) and predicted>0 and prediction["predicted_batch"]==int(row["batch"]),
                    "Invalid prediction")
            error=predicted-measured
            result.update(status="completed_pair",predicted_peak_mib=predicted,measured_training_peak_mib=measured,
                          measured_process_peak_mib=measurement["observed_process_peak_mib"],
                          signed_error_mib=error,absolute_error_mib=abs(error),relative_error_percent=100*error/measured,
                          absolute_relative_error_percent=100*abs(error)/measured,
                          underestimation=error<0,m_pbase_mib=prediction["m_pbase_mib"],
                          estimate_seconds=(prediction["finished_ns"]-prediction["started_ns"])/1e9,
                          measure_seconds=(measurement["finished_ns"]-measurement["started_ns"])/1e9,
                          allocator_peak_allocated_mib=measurement["allocator_peak_allocated_mib"],
                          allocator_peak_reserved_mib=measurement["allocator_peak_reserved_mib"],
                          source_commit=source_commit,job_id=claim["job_id"],gpu_uuid=claim["gpu_uuid"],
                          raw_path=str(pair))
        except (ValueError,KeyError,OSError,TypeError) as error:
            result["reason"]=f"{type(error).__name__}: {error}"
        observations.append(result)
    groups=defaultdict(list)
    for row in observations:
        groups[(row["model"],row["batch"],row["sequence"])].append(row)
    cases=[]
    for (model,batch,sequence), group in sorted(groups.items()):
        eligible=[r for r in group if r["status"]=="completed_pair"]
        case=dict(model=model,batch=int(batch),sequence=int(sequence),planned=len(group),complete=len(eligible))
        for key in ("predicted_peak_mib","measured_training_peak_mib","signed_error_mib","absolute_error_mib",
                    "relative_error_percent","absolute_relative_error_percent","estimate_seconds","measure_seconds"):
            case[key]=mean(r[key] for r in eligible) if eligible else None
        if eligible:
            case.update(min_measured_mib=min(r["measured_training_peak_mib"] for r in eligible),
                        max_measured_mib=max(r["measured_training_peak_mib"] for r in eligible))
        cases.append(case)
    complete=[r for r in cases if r["complete"]==r["planned"]]
    eligible=[r for r in observations if r["status"]=="completed_pair"]
    summary=dict(protocol_sha256=protocol_hash,source_commit=source_commit,planned_pairs=len(rows),
                 states=dict(Counter(r["status"] for r in observations)),
                 complete_cases=len(complete),
                 equal_case_mae_mib=mean(r["absolute_error_mib"] for r in complete) if complete else None,
                 equal_case_signed_error_mib=mean(r["signed_error_mib"] for r in complete) if complete else None,
                 equal_case_relative_error_percent=mean(r["relative_error_percent"] for r in complete) if complete else None,
                 equal_case_absolute_relative_error_percent=mean(r["absolute_relative_error_percent"] for r in complete) if complete else None,
                 underestimated_pairs=sum(r["underestimation"] for r in eligible),
                 native_only=True,rl_target_prediction_accuracy_evaluated=False)
    return summary,observations,cases


def write_rows(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[1]
    parser.add_argument("--directory",type=Path,default=root/"benchmark/electronics_revision_20261016/llmem_native")
    parser.add_argument("--output",type=Path,default=Path("/mnt/proj3/open-35-44/verl/rl/output/llmem_native_validation_20261016"))
    parser.add_argument("--results",type=Path,default=root/"profiles/electronics_revision/llmem_native")
    args=parser.parse_args()
    receipt=json.loads((args.directory/"submission.json").read_text())
    summary,rows,cases=collect(args.directory,args.output,receipt["source_commit"],receipt["protocol_sha256"])
    args.results.mkdir(parents=True,exist_ok=True)
    write_rows(args.results/"pairs.csv",rows)
    write_rows(args.results/"cases.csv",cases)
    (args.results/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
