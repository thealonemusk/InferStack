# ADR-0005: Push the stack to the GPU, do not connect to it

- **Status:** accepted
- **Date:** 2026-09-19
- **Phase:** 1

## Context

The benchmark phases need a CUDA GPU. The available one is a free-tier Kaggle
session, which imposes three constraints that a normal deployment does not:

1. **No public ingress.** A Kaggle session cannot be reached from outside. There
   is no host to point a load generator at.
2. **Ephemeral.** Sessions are capped at roughly 12 hours and the filesystem is
   discarded on shutdown. Anything not exported is lost.
3. **Metered.** Roughly 30 GPU-hours per week. A wasted run is a real cost.

A tunnel (ngrok, cloudflared) could expose the endpoint, but then every latency
measurement would include a round trip over the public internet and whatever
jitter that path contributes. TTFT measured through a tunnel is a measurement of
the tunnel.

## Decision

The entire stack - engine, load generator, analysis - is **pushed into the
session and run there**. Results are written to the session's output directory
and pulled back afterwards as artefacts. Nothing connects in.

`inferstack.remote.kaggle` drives this over the Kaggle API: push a folder as a
kernel, wait for a terminal status, fetch the output.

## Consequences

- The load generator and the server share a loopback interface, so measured TTFT
  and inter-token latency are the server's, not the network's. This is better
  methodology, not a compromise.
- Every phase must be expressible as a script that runs unattended and exports
  its results. That discipline is what makes the runs reproducible.
- A session dying mid-sweep loses only the current run, provided results are
  written incrementally. Phase 4's harness must checkpoint.
- Interactive debugging is awkward: the feedback loop is push, wait, read the
  log. The `local-cpu` profile exists to keep that loop off the GPU.

## Kaggle-specific findings, recorded because they cost a run to learn

- **The API cannot choose the accelerator type.** `kernel-metadata.json` exposes
  only `enable_gpu` (a boolean). Choosing between one T4 and two is a setting on
  the notebook in the web UI, which subsequent API pushes then inherit.
- **Kaggle silently downgrades a kernel rather than rejecting it.** A push
  requesting `enable_gpu: true` is accepted, stored as `true`, and may still run
  on CPU with no error, no warning and no failed status. A run that looks
  successful can be CPU output wearing a GPU label.
- **Accelerators and internet are gated together** behind account phone
  verification. Losing internet is equally fatal: without it the session cannot
  install vLLM or download weights.
- `kernels_pull` echoes back what was *submitted*; it is not evidence of what was
  *granted*. The only authority on what the session actually got is the session
  itself, which is why `remote/kernels/gpu_probe.py` reports `nvidia-smi`,
  `torch.cuda` and internet reachability before any benchmark is trusted.

## Alternatives considered

- **Tunnel out of the session and drive it remotely.** Contaminates every
  latency measurement with public-internet jitter, and the tunnel becomes an
  uncontrolled variable across runs.
- **Manual notebook sessions.** Workable, but a Phase 5 parameter sweep is dozens
  of runs. Anything that needs a human watching a browser tab will not get done
  consistently, and cannot be re-run to check a result.
- **Rent a GPU.** Removes every constraint above and is what a funded project
  would do. Out of scope here by choice; the constraints are themselves
  instructive.
