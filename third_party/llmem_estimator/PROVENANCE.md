`size_estimator.py` is copied without modification from Taeho Kim's LLMem
repository, commit `3d289b46ea7d969e23634745209e6ad02beacff5`.
SHA-256: `e35a4f29e2e5d672813c1d536ea3fc06ebeff3db88c7ad016676bdb62389c8dd`.
The accompanying upstream MIT license is retained.

Primary publication: Kim et al., *LLMem: Estimating GPU Memory Usage for
Fine-Tuning Pre-Trained LLMs*, IJCAI 2024, DOI 10.24963/ijcai.2024/699.
Primary implementation: `https://github.com/taehokim20/LLMem`.

The isolated native reproduction uses upstream ColossalAI commit
`d4fb7bfda7a2da5480e1187e8d3e40884b42ba11`, with the authors' `colossalai`
overlay as instructed by their README. PyTorch 2.0.1+cu117 and Transformers
4.29.2 are separate from the VERL environment. The native driver follows
the prediction initialization in `dp_real.py`, including measured
`m_pbase`, Gemini FP16, gradient checkpointing, and HybridAdam.

Uniform synthetic token batches provide controlled shapes instead of
Alpaca examples. Estimation and measurement run in separate fresh
processes. This is a supported-architecture, training-only numerical
comparison, not an end-to-end VERL/LoRA prediction or an exact rerun of
the paper's accuracy table. Software/hardware drift and input differences
must accompany its results.
