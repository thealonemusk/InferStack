# Reviewing this before it reaches `main`

`main` is still the initial commit. Five branches stack on top of it, each cut
from the last, and nothing merges until you have read it. This is the order to
read it in and the things actually worth your attention.

```
main
 └── phase-00-foundations        2 commits    32 files   +2,419
      └── phase-01-baseline-serving   17      28 files   +5,383
           └── phase-02-gateway        9      15 files   +1,812
                └── phase-03-observability  29  53 files  +10,455
                     └── phase-04-bench       1  14 files   +2,543
```

Roughly 22,000 added lines, but a large fraction is documentation, ADRs,
captured artifacts and test bodies. The source under `src/` is about 5,000
lines. Read the ADRs first and the diffs second — every non-obvious choice has a
record explaining what was rejected and why, and if you disagree with a decision
the ADR is the thing to argue with, not the code implementing it.

## Read in this order (about 90 minutes)

### 1. The decisions, not the code — 20 minutes

```bash
ls docs/adr/
```

Eight records. Read them in order; they are short and each one is the argument
for a whole subsystem. The load-bearing ones:

- **ADR-0002** — hardware is a YAML profile, code never branches on it.
- **ADR-0003** — nothing outside `engine/` may import vLLM. This is what keeps
  the Phase 8 SGLang swap a config change.
- **ADR-0005** — push the stack to the GPU rather than tunnelling to it, because
  a tunnel puts WAN jitter inside every TTFT measurement.
- **ADR-0006** — the gateway is a pass-through; declaring a schema would be a
  second copy of a moving specification.
- **ADR-0007** — metrics are pulled per component, histograms not averages.
- **ADR-0008** — load is open-loop, capacity is goodput.

If you only read two: **0006 and 0008**.

### 2. Phase 0 — `main..phase-00-foundations` — 10 minutes

```bash
git diff main..phase-00-foundations -- src/ configs/
```

Config precedence, the hardware probe, the compatibility checker. Small and
self-contained. The thing to check is `config.py`: settings resolve
env > `.env` > profile YAML > defaults, and the `profile` field is a *label*
rather than a tunable — there is a comment explaining the bug that caused.

### 3. Phase 1 — the engine — 20 minutes

```bash
git log --oneline phase-00-foundations..phase-01-baseline-serving
git diff phase-00-foundations..phase-01-baseline-serving -- src/inferstack/engine/
```

Three files matter. `launcher.py` builds vLLM's argv and probes the installed
binary's `--help` rather than pinning a version table; the plausibility
threshold in the flag validator is there because an earlier version stripped
*every* flag and launched a bare `vllm serve`. `client.py` is the single place
TTFT is defined — note that it skips the role-only delta. `smoke.py` is the
1-vs-N batching proof, and is deliberately labelled a sanity check because it is
closed-loop.

Read `docs/phases/phase-01-baseline-serving.md` alongside: it records the three
runs it took and why each failure was real.

### 4. Phase 2 — the gateway — 15 minutes

```bash
git diff phase-01-baseline-serving..phase-02-gateway -- src/inferstack/gateway/
```

The whole phase turns on one thing, in `app.py`: a streaming response holds its
admission slot in the **body iterator**, not the handler. The handler returns
when upstream headers arrive, so `async with admission.slot():` would release
the slot before a single token was relayed — unlimited concurrent streams while
the counter read zero. Two tests pin it.

Also worth a look: `proxy.py`'s `finally` block, which closes the upstream
connection on client disconnect so vLLM aborts the sequence rather than
generating tokens nobody will read.

### 5. Phase 3 — observability — 20 minutes

The largest diff, and the most of it is documentation and captured artifacts.

```bash
git diff phase-02-gateway..phase-03-observability --stat -- src/
```

- `observability/promtext.py` — a hand-rolled exposition parser. The
  justification is that the read path must work on core dependencies alone;
  the defence is a test asserting it agrees with `prometheus_client`'s own
  parser sample by sample.
- `observability/histograms.py` — quantiles from bucket counts, mirroring
  Prometheus. There is a test pinning our answers to Prometheus' own on the
  same bytes.
- `observability/metrics.py` — the gateway's registry. Admission counts are
  *collected from* the controller at scrape time rather than mirrored, which is
  the direct lesson of the Phase 2 bug.

**The thing to actually check here** is `tests/fixtures/vllm_metrics_real.txt`
and the test that binds `ENGINE_SIGNALS` to it. This phase shipped asking vLLM
for a TPOT metric by a name vLLM does not use, and nothing failed — the signal
read as missing, the Grafana panel read "No data", the alert never fired. The
fix is that test.

### 6. Phase 4 — the benchmark harness — 15 minutes

```bash
git diff phase-03-observability..phase-04-bench -- src/inferstack/bench/
```

Read `arrivals.py` and `load.py` together; they are the two halves of the
open-loop claim. The schedule is computed before the run so it cannot adapt to
the server, and every record carries both the send clock and the schedule clock
so coordinated omission is visible rather than assumed away.

Then `tests/test_bench_sweep.py`, which is the capstone: a simulated engine with
a capacity known by arithmetic, and the sweep has to find the knee without being
told. It caught a real denominator bug on its first run.

## What to check, concretely

Run this on any branch; it is what CI runs:

```bash
uv venv && uv pip install -e ".[dev,gateway,bench]"
pytest -p no:warnings        # 364 tests, ~70s
ruff check . && ruff format --check src tests scripts
mypy
```

Then the two things that are not unit tests:

```bash
python scripts/measure_phase03.py --out /tmp/check      # real sockets, ~40s
python scripts/verify_observability.py --prometheus <path> --grafana <path> \
    --engine-metrics tests/fixtures/vllm_metrics_real.txt
```

## Questions worth asking me

Rather than a line-by-line read, these are the places where the design could
reasonably have gone the other way:

1. **The hand-rolled Prometheus parser.** Is "the read path must work on core
   dependencies" worth a parser we maintain? It is ~150 lines and checked
   against the reference, but a dependency would have been zero lines.
2. **`/metrics` is unauthenticated.** Deliberate and argued in ADR-0007, and it
   is a standing constraint on every future metric rather than a one-off.
3. **Bypassing the gateway in the Phase 4 sweep.** Clean attribution, at the
   cost of the numbers not describing the full stack.
4. **Stopping the sustainable rate at the first failure.** Gives up a higher
   number the data sometimes supports.
5. **`local-cpu` will never run a real vLLM.** Written off as needing a source
   build; if you disagree it is a real piece of work, not a small one.

## Merging, when you are ready

The branches are a linear chain, so each merge is a fast-forward:

```bash
git checkout main
git merge --ff-only phase-00-foundations
git merge --ff-only phase-01-baseline-serving
git merge --ff-only phase-02-gateway
git merge --ff-only phase-03-observability
git merge --ff-only phase-04-bench
git push origin main
```

Nothing is lost by doing it in stages — stopping after Phase 3 leaves `main` at
a coherent, fully verified point, and Phase 4 can follow later.

One thing to know: **until this lands, the repository looks empty to anyone who
opens it.** The landing page renders `main`'s README, and `main` has one file.
Every measurement, chart and ADR is invisible unless a visitor knows to switch
branches.
