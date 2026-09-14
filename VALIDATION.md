# Release validation — 2026-09-14

The standalone package was tested with a fresh CPython 3.11 environment and
the unchanged pinned publication requirements. `pip check` passed.

- 33 unit/regression tests passed.
- The frozen archive SHA-256 matched the release lock.
- All 19,226 evidence-manifest entries verified after extraction.
- Analysis reconstructed all 588 principal processes and the 16 separately
  counted historical source screens.
- All 40 regenerated CSV/JSON result files matched the archived references
  after normalizing only artifact-root path prefixes.
- All four publication figures were regenerated as PDF and PNG.
- During analysis, bubblewrap hid the original project and the directory
  holding precomputed reference profiles. No network-namespace isolation
  is claimed.
- 19,073 regular archive members were scanned for credential-like content.
  No model/optimizer checkpoint payloads were included.

The copied scientific source files are byte-identical to the original
`289747694f1a71052acc7977e742bfc0a45286a2` snapshot; their individual hashes
are recorded in `source-provenance.json`. The new wrapper and download helper
are separate packaging code.

This is numerical/raw-evidence reproduction, not a new GPU experiment,
learning-quality evaluation, author approval, license grant, or journal
acceptance certification.

## Initial remote deposit

The private dataset was uploaded to `kobzaond/RLVRAMBench` at commit
`68aa8e041b3248c2898ba8062c912e84a64b2c64`. All 52 published payload files
were downloaded at that exact revision and their SHA-256 digests matched
the staged files. The server-generated `.gitattributes` is separate from
these payload files. The initial download-helper release pinned this revision.
No DOI was created.

The complete isolated reproduction was then repeated from the downloaded
archive in a second fresh extraction. All 19,226 evidence files verified;
all 40 derived files matched and all four figures regenerated again.
The acceptance report is retained in
`validation/downloaded-reproduction.json`.

## Public release

On 2026-09-14, the repository owner requested public access for both the
benchmark and this reproduction repository. Both are now public; the dataset
is not gated. The current `release.json` pins dataset revision
`c9bd99f6c57676fca0f2829cd95d34599ef8a305`, which updates the access
documentation and manuscript-snapshot description without changing the
frozen evidence archive, tables, or scientific source. No DOI or license was
selected as part of this change.

All 52 payload files at the public revision were downloaded without
credentials and their SHA-256 digests matched the staged release. The archive
digest is identical to the archive used for both complete isolated
reconstructions above. See `validation/public-deposit-verification.json`.
The source snapshot, scientific dependencies, and analysis logic are unchanged.
