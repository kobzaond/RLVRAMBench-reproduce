# Verified admission-study release

Verified on 15 September 2026.

- Scientific source snapshot: `7b534ef1d1e902ba490677acdccd8294e59e8bf5`
  in `kobzaond/rl`.
- Dataset: `kobzaond/RLVRAMBench`, public and ungated.
- Dataset revision: `f1172f166343c00b8397d5124d2b180f160a8d08`.
- Immutable release: `admission-20260915-7b534ef`.
- Evidence archive SHA-256:
  `a457d7169afbfb1980ec70a4af215035041d2e05eec9eb090e2c0b7c69e935b2`.

The archive was independently extracted into a new working directory.
All 20,145 evidence-manifest entries validated; the archive contains
20,147 members including its manifest and archive metadata.
CPU reconstruction matched 40 original result files, 27 control/audit
files, 14 curated benchmark files, and all 11 admission-panel outputs.
Four publication figures were regenerated. Reference outputs and known
project copies were hidden during analysis, with inaccessibility checked
inside each child filesystem namespace. Network isolation is not claimed.
This is a reconstruction check performed during the authors' workflow,
not external replication or a rerun of GPU training.

The new panel contains 48 validated memory outcomes: 32 completions and
16 diagnosed memory failures, with no retries or unresolved slots.
Twelve screening invocations and 36 independent-seed evaluation invocations
cover four cases and twelve candidates; they are not 48 independent cases.

After publication, all 121 uploaded payload files (187,363,632 bytes)
were downloaded and hashed without an authorization header. Every size
and hash matched the staged upload manifest. The public dataset viewer
returned all twelve candidates in its default `admission_candidates`
configuration: six within-margin, two above-margin, and four
memory-failure repeated evaluation labels.

Machine-readable source-build and reconstruction reports are included
under `verification/` in the pinned dataset revision. The 57-file paper
source ZIP was separately extracted, and both main and supplementary
documents compiled with text identical to the released PDFs.

`release.json` pins the verified download. `source-provenance.json`
records hashes of the copied scientific modules and frozen inputs.
Historical releases are preserved. Model checkpoint payloads are not
included, and no artifact DOI has been minted.
