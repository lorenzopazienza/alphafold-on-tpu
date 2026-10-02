# AlphaFold3 and AlphaFold2 inference on Cloud TPUs

An independent research project measuring DeepMind's official AlphaFold3 inference, and AlphaFold2, on Cloud TPU v5e and v6e, with CPU and GPU baselines on fixed, recorded machine types, repeated runs and full provenance for every number. It follows up on the preprint arXiv:2609.34818.

## Status

In preparation. No results yet.

## AlphaFold3 on TPU

The official `run_alphafold.py` accepts only `gpu`, `cpu` and `mps` as JAX
backends. `af3_tpu/` holds a minimal patch, pinned to AlphaFold3 v3.0.4, that
adds a TPU backend with XLA attention. See
[`af3_tpu/README.md`](af3_tpu/README.md) for what the patch changes and for
the smoke test.

## Prior work

This project builds on the preprint *Accelerator Choice Is Not Enough:
AlphaFold2 Inference on Cloud TPUs*
([arXiv:2609.34818](https://arxiv.org/abs/2609.34818)). The code and data from
summer 2026 behind that preprint are archived at
[github.com/lorenzopazienza/alphafold-tpu-benchmark](https://github.com/lorenzopazienza/alphafold-tpu-benchmark).

## Layout

```
src/            AlphaFold2 forward-pass benchmark scripts (single, vmap, pmap, GSPMD, ensemble)
cloud/          tooling to run a measurement session on a Cloud TPU VM
  plans/        run plans: label, script, repetitions and arguments per configuration
af3_tpu/        AlphaFold3 TPU patch, pin, random-weight generator and test input
third_party/    the AlphaFold3 clone the patch is applied to (gitignored)
results/        session results, committed after review (created by the first session)
catalogue.md    one row per reported number, pointing to its result file
```

## Principles

- every reported number has a row in `catalogue.md` pointing to its result file;
- every configuration runs as repeated fresh processes, and sessions are repeated on different days;
- machine type, zone, driver, package versions and script hashes are recorded for every session;
- summer 2026 numbers stay in the archived repository and are never overwritten.

## Running

- AlphaFold2 sessions on Cloud TPU: see [`cloud/README.md`](cloud/README.md).
- AlphaFold3 on TPU, install and smoke test: see [`af3_tpu/README.md`](af3_tpu/README.md).

## Author and licence

Lorenzo Pazienza (LUISS Guido Carli University).

- Licence: MIT for this repository's code.
- `af3_tpu/af3_tpu.patch` modifies AlphaFold3's `run_alphafold.py`, which is
  under the Apache License 2.0.
- AlphaFold3 model parameters are never included in this repository and are
  subject to DeepMind's own terms of use.
