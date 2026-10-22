# Measurement exclusions and subsequent tooling fixes

The completed campaign used its original frozen execution copies. Tooling
changes on the revision branch do not alter those copies, raw files, or
terminal receipts. No invocation is replaced to fill an excluded slot.

## Phase-marker read during a write

`adapt-q3-gsm8k-s311-rank64-off0-r65-kl1`, job 5215400, exited successfully
and recorded all five requested updates. Its phase/telemetry CSV was
malformed at timestamp `1792154923876556390`. The published marker contained
parts of the preceding `initializing` record and the new `rollout` record,
including a newline. The monitor inserted that text into CSV fields.
Consequently, four telemetry rows lack GPU indices, and the separate phase
trace has an extra line.

The ordinary memory trace and trainer records do not repair the missing
phase-integrity evidence required by the frozen eligibility contract.
The invocation remains **unresolved**, including in repeated configuration
labels and matched numerical contrasts. It is neither a diagnosed memory
failure nor an eligible within-margin completion. Its raw successful exit
and completed work remain visible.

Evidence is under
`output/electronics_adaptation_20261016/adapt-q3-gsm8k-s311-rank64-off0-r65-kl1/attempt-5215400/`.
The block's `period-6.json` preserves the original exclusion and hashes
of the raw files. Subsequent collection reports a descriptive malformed-row
error instead of the original `int(None)` error, without changing eligibility.

The same defect also occurs in
`adapt-q3-code_heavy_tail-s313-rank0-off1-r45-kl1`, job 5215805, at timestamp
`1792156400254354261`. It likewise exits zero and records five updates, but
four malformed telemetry rows invalidate its measurement contract.
It remains unresolved under the same rule.

`reservation-m7-code_heavy_tail-s453-off0-r60`, job 5215793, also exits
zero and records its single requested update. Its telemetry contains
296 malformed rows from timestamps `1792158181460078659` through
`1792158188760072973` (first affected CSV row 10690). The phase field
contains `actor_update`, a newline, and NUL bytes, so this corruption
persists across multiple samples. The raw trace remains unchanged.
The run and its incomplete three-repeat configuration remain unresolved,
outside primary numeric contrasts and prediction scores.

For future execution, both instrumented trainer variants now write a
temporary marker and atomically replace the published file. The monitor
also rejects malformed marker strings rather than inserting line breaks
or invalid characters into CSV fields. A concurrent two-writer/reader test checks complete
records and temporary-file cleanup. This fix has not been used to regenerate
or reinterpret the frozen campaign, and is not a GPU-tested performance
optimization.
The artifact keeps this later change in
`execution-source/post-campaign-marker-fix.patch`, relative to the controls
source `cf86c7ae0f94df817531abd4b653645b1a2eb814`. Applying it requires a new
execution-source identity and protocol; the archived campaign sources
themselves remain unmodified.

## Prespecified execution timeout

`release-q15-gsm8k-s502-release1-r75` reaches its 900-second payload
limit with no durable completed update. The wrapper returns 124 after
cleanup; its last recorded phase is `initializing`. Logs report engine
initialization completion but do not establish the cause of the subsequent
lack of progress. A generic warning about possible CPU-memory pressure is
not an observed OOM. This slot remains unresolved, without a replacement
or a completed-demand peak, under the original time cap.

## Configuration-log parsing correction

Final collection encountered two valid printed actor dictionaries followed
by differently formatted `algorithm` headings:
`release-q15-gsm8k-s502-release0-r70` has no leading space, and
`release-q15-gsm8k-s503-release0-r60` splits the key, colon, and value across
Ray records. The initial analysis parser expected one exact heading layout.
It now recognizes either heading while still requiring the entire preceding
actor dictionary to parse and its producer PID to match the completed-step
records. A test checks both layouts and rejects a truncated actor dictionary.
No configuration values are supplied from the planned matrix.

This is a collection-code correction, not a raw-data repair or replacement
run. The original executor labels and unchanged raw evidence remain the
comparison anchors. It is distinct from the invalid telemetry and timeout
exclusions above.
