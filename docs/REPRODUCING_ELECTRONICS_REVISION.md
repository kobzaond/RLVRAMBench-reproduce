# Reconstructing the Electronics revision

The revision has three distinct evidence sources:

1. New descriptive analyses of previously audited historical tables, with
   a raw-log re-audit of 105 memory failures.
2. Four frozen RL follow-ups: adaptation, reservation, release, and timing.
3. A separate native LLMem comparison, with 24 prediction/measurement pairs.

These are separate cohorts. Development runs are retained but cannot fill a
missing formal slot. Numerical peak contrasts require joint completion;
an observed partial peak from a failed process is not its completed demand.

## CPU reconstruction

Use Python 3.11 with NumPy, Matplotlib, pandas, scikit-learn, and psutil.
The collectors do not import the GPU training environment or need model
weights. In an extracted artifact, run from its root:

```sh
python -m memory_tuner.electronics_review_analysis --root .
python -m memory_tuner.electronics_followup_analysis --root . --output-root output
python -m memory_tuner.collect_llmem_native \
  --directory benchmark/electronics_revision_20261016/llmem_native \
  --output output/llmem_native_validation_20261016 \
  --results profiles/electronics_revision/llmem_native
python -m memory_tuner.electronics_followup_figures --root .
python -m memory_tuner.electronics_revision_tables --root .
```

The historical analysis needs the eight hashed source tables listed in
`profiles/electronics_revision/summary.json` and its referenced raw failure
logs. Historical absolute paths are remapped into the extracted artifact;
there is no fallback to the original project. Individual failure diagnostics
retain their raw-log hash and operation/error evidence.

Each new RL collector starts from `protocol.json`, `matrix.csv`, and the
submission receipt's source/protocol anchors. It checks block claims,
dispatch identities, terminal file inventories, the actual printed actor
configuration, the container/runtime record, physical GPU evidence,
NVML/phase-trace equality, durable completed-step records, and actual
parameter inventories. Legacy trial JSON lacks the new adaptation fields;
the applied Hydra configuration and parameter records provide those checks.
The configuration-printing process must match the durable metrics producer.
The final reassessment additionally requires an explicit GPU-memory
diagnostic, avoiding classification of host OOM from unrelated CUDA text.
These stronger checks affect analysis only; frozen executing source is
preserved separately.
`paper/electronics_measurement_audit.md` records three malformed phase-marker
exclusions, one prescribed timeout, and a subsequent atomic-write fix for
future execution. The fix does not repair or replace an excluded campaign
invocation. The audit also documents a collection-code correction for two
valid actor dictionaries whose following log heading had different spacing
or line breaks; incomplete dictionaries remain rejected.

For success, full training must report every actor parameter trainable and
no adapters. LoRA must report adapter-only trainable state. Reference-role
counts must match the native versus no-KL design. All requested steps must
complete. A timeout, contamination, missing proof, invalid cleanup, or
non-memory failure stays unresolved. Non-launched slots remain in the ledger.

Cost analysis validates durable metric text hashes, finite actor losses and
gradient norms, and per-response lengths.
It uses summed actual token counts and trainer step time; it does not average
per-step rates or reuse whole-node RAM as a job metric. Missing host-memory
evidence is missing, not zero. Host cgroup samples distinguish anonymous
memory and charged file cache, including cache retained from earlier
processes in the same allocation. Absolute host peaks are not measurements
of actor-owned memory alone.

The LLMem collector independently verifies prediction ordering, fresh process
identity, physical device UUID, frozen shapes/runtime, terminal inventories,
three finite completed training updates, and measured raw peaks. Unsupported
colocated targets are recorded as unsupported, not scored as rejections.

The outcome JSON files contain relocated `raw_path` values. Those path fields
are expected to change after extraction; the observations, labels, peaks,
costs, denominators, predictions, and paired contrasts must agree.

`paper/verify_electronics_reconstruction.py ARTIFACT --report REPORT.json`
checks the artifact manifest, then reconstructs results in fresh processes.
An audit hook rejects original source/evidence reads, reads of the artifact's
saved revision-derived results, writes to artifact inputs, and network connections.
The eight previously audited historical tables remain permitted historical
inputs; this does not regenerate every old experiment from its raw files.
Figure summaries and the supplement's generated tables are also rebuilt
from the freshly reconstructed revision outcomes and compared with the
packaged values.
Write the report outside the artifact directory. The author's CPU environment
libraries remain accessible and are explicitly identified as a dependency.
This is an author-performed relocation check, not independent replication.

Allocation costs are regenerated from the saved `sacct` snapshot with
`memory_tuner.electronics_resource_accounting`. Only outer job records count:
`batch` and `extern` steps describe the same allocation. GPU-hours reserved
by Slurm and GPU-hours spent in task invocations overlap and must not be
added. The revision's development jobs are costed separately from formal
experiments. A running allocation cannot produce a finalized cost report.

## Re-running GPU execution

Executing a new campaign is distinct from CPU reconstruction. The old
permanent block claims intentionally prevent overwriting/retrying old
invocations. Choose a new campaign directory and new identifiers, prepare
the exact model/data snapshots, and freeze a new protocol before execution.
Do not edit the archived matrices or reuse their claimed output directories.

The frozen execution sources are:

| Panel | Source commit | Scheduler array |
|---|---|---|
| Development adaptation pilot | `fcd019198906b7e91278243dc64ac58d76d96354` | 5214781 |
| Adaptation and no-KL control | `e6a8f0abd7790544362d97e4e633414f9528c241` | 5214838 |
| Native LLMem formal validation | `f933e933d63a976fb0a5c9395e27fcfd820f13d8` | 5214944 |
| Reservation | `cf86c7ae0f94df817531abd4b653645b1a2eb814` | 5215023 |
| Iteration cost | `cf86c7ae0f94df817531abd4b653645b1a2eb814` | 5215024 |
| Release | `cf86c7ae0f94df817531abd4b653645b1a2eb814` | 5215025 |

Full source/protocol hashes, model and dataset digests, exact matrix values,
execution order, and resource caps are in the protocol directories.
The Slurm launchers are site-specific; transferring them requires a new
allocation and device-identity check. The release panel reserves four GPUs
but uses two task GPUs. The other RL panels use four task GPUs. LLMem uses
one GPU per pair in a separate, pinned legacy environment.

The native LLMem environment and compatibility limitations are documented in
`paper/electronics_external_estimator_audit.md` and
`third_party/llmem_estimator/PROVENANCE.md`. The optional larger-model
one-step/100-step extension was not part of these frozen panels.

## Manuscript build

Run `python paper/build_electronics_revision.py` with pdfLaTeX, BibTeX,
and `pdftotext` on PATH. The helper builds both PDFs and verifies the source
ZIP after extraction. See the submission directory's README for the
equivalent manual commands. Final mode rejects pending reviewer-response
entries and unresolved build warnings.

Local packaging and source verification do not publish to GitHub,
Hugging Face, or the journal.
