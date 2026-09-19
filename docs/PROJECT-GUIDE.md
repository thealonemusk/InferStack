# InferStack — the guide

Everything you need to explain, justify and defend this project.

Read sections 1–3 to *understand* it. Read section 8 before anyone asks you
about it. Section 9 is the vocabulary.

> **Honesty rule used throughout this document:** anything stated as a
> *measurement* was actually measured on real hardware, and says where. Anything
> stated as *expected* is theory that has not yet been confirmed on this stack.
> Never present the second kind as the first — that is the single fastest way to
> lose credibility in a performance conversation.

---

## 1. What this project is, in one paragraph

InferStack is a self-hosted LLM inference stack. It serves an open-weights model
through vLLM behind an OpenAI-compatible API with continuous batching enabled,
then instruments and tunes it. The point is not "I made a chatbot endpoint" —
anyone can do that in ten lines. The point is the engineering discipline
underneath: knowing *why* p99 latency degrades when concurrency rises, what
`max_num_batched_tokens` actually trades away, and being able to prove any claim
with a reproducible measurement rather than a vibe.

The guiding line from the brief:

> **You cannot optimize what you have never served.**

That is why the project is built in phases: serve it first, measure it second,
optimise it third. In that order, deliberately.

---

## 2. The theory you actually need

This is the part to internalise. Most people who "use an LLM API" cannot explain
any of it, and it is what separates a serving project from a demo.

### 2.1 Inference has two completely different phases

When you send a prompt and get a response, the model does two different kinds of
work, with different performance characteristics. Confusing them is the most
common mistake in this field.

**Prefill** — processing your prompt.

The model reads all *N* prompt tokens **in parallel**, in one forward pass. It's
a big matrix multiply: large GEMMs over an `N × d` activation matrix. There is
plenty of arithmetic per byte of weight loaded, so the GPU's tensor cores are
the bottleneck.

> Prefill is **compute-bound**. It produces exactly one token — the first one.

**Decode** — generating the response, one token at a time.

Each step processes a **single** token, but must attend to every previous token.
To produce that one token the GPU reads *the entire model weights* from memory.
The arithmetic-per-byte ratio collapses to roughly 1.

> Decode is **memory-bandwidth-bound**. It runs once per output token.

**Why this matters so much:** these two phases have opposite optimisation
strategies, and they map directly onto the two latency metrics:

| Phase | Bound by | Metric it drives | Scales with |
|---|---|---|---|
| Prefill | Compute (FLOPs) | **TTFT** — time to first token | Prompt length, queue wait |
| Decode | Memory bandwidth | **TPOT** — time per output token | Batch size, KV cache size |

If someone asks "why is your first token slow but the rest fast?" — that's
prefill vs decode, and you can now answer it precisely.

### 2.2 The KV cache, and why it dominates everything

Recomputing attention over the whole sequence at every decode step would be
absurdly wasteful. So for every token, at every layer, the model caches its Key
and Value vectors. That's the **KV cache**.

Its size, per sequence:

```
kv_bytes = 2 (K and V)
         × num_layers
         × num_kv_heads
         × head_dim
         × bytes_per_element
         × sequence_length
```

Worked example — Qwen2.5-1.5B-Instruct, the model this project serves:

```
28 layers, 2 KV heads (grouped-query attention), head_dim 128, fp16 (2 bytes)

per token = 2 × 28 × 2 × 128 × 2 bytes ≈ 28 KB
4096-token sequence ≈ 115 MB
```

Now the crucial part. On a 15 GB T4:

```
model weights (1.5B × 2 bytes)   ≈  3 GB
leaves for KV cache               ≈ 11 GB
11 GB ÷ 115 MB                    ≈ 96 concurrent 4096-token sequences
```

**That number is your real concurrency limit.** Not CPU, not the network — KV
cache capacity. This is why the project deliberately serves a *small* model on a
*big* batch: a 7B model would eat most of the VRAM in weights and leave room for
only a handful of sequences, which would make continuous batching invisible.

This is also why `gpu_memory_utilization` matters: whatever is left after weights
*becomes* the KV cache. Set it too low and you starve the batch.

### 2.3 Batching: the actual subject of this project

**No batching.** One request at a time. To generate one token the GPU reads all
3 GB of weights. Generating that token for 1 sequence or for 50 sequences costs
*almost the same* memory traffic. Serving one at a time wastes ~98% of the GPU.

**Static batching.** Collect N requests, run them together, return when done.
Better — but with a fatal flaw: **the batch runs until the longest sequence
finishes.** If one request generates 500 tokens and seven generate 20, those
seven slots sit idle for 480 steps. And no new request can join until the whole
batch completes.

```
Static batching — X = wasted slot

  req A ####################################  (long)
  req B #####XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
  req C ###XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
  req D ########XXXXXXXXXXXXXXXXXXXXXXXXXXXX
        └─ new requests wait here for the whole batch ─┘
```

**Continuous batching** (a.k.a. *iteration-level scheduling*, from the Orca
paper, 2022). The scheduler makes a decision **every single decode step**, not
per batch. When a sequence finishes, its slot is freed *immediately* and a
waiting request takes it on the very next iteration.

```
Continuous batching

  req A ####################################
  req B #####→ E joins here →###############
  req C ###→ F joins here →#################
  req D ########→ G joins →################
        └─ no idle slots, no waiting for the batch ─┘
```

This is the single biggest throughput win in modern LLM serving, and it is what
the brief asks to enable. **This is the concept you must be able to explain
cold.**

### 2.4 PagedAttention — what made it practical

Continuous batching has a hard memory problem: KV caches are per-sequence, start
small, grow unpredictably, and die at unpredictable times. Naively you'd
pre-allocate `max_model_len` of contiguous memory per sequence — reserving 4096
tokens' worth for a request that generates 30. The vLLM paper measured 60–80% of
KV memory wasted to this fragmentation.

**PagedAttention** (Kwon et al., 2023 — the paper vLLM comes from) applies an
idea from operating systems. Virtual memory solved exactly this problem for RAM:
split memory into fixed-size **pages**, keep a **page table** mapping logical
addresses to physical ones, and allocate on demand.

vLLM does the same for the KV cache:

- KV cache is split into fixed-size **blocks** (e.g. 16 tokens each)
- Each sequence has a **block table** mapping logical positions → physical blocks
- Blocks are allocated only as the sequence actually grows
- Blocks need not be contiguous

Results: near-zero fragmentation, far more concurrent sequences in the same
VRAM, and — because blocks are indirected — **sharing becomes free**. Two
requests with the same prefix can point at the same physical blocks. That's
prefix caching, and it's Phase 6.

> One-line version: *PagedAttention is virtual memory for the KV cache.* If you
> remember one sentence about vLLM, remember that one.

### 2.5 Chunked prefill — the subtlety worth knowing

Here's a problem continuous batching creates. Prefill is a big compute burst. If
a request with a 4000-token prompt arrives, its prefill occupies an entire engine
step — and **every other sequence in the batch stalls** for that step. Everyone's
inter-token latency spikes because one user sent a long prompt.

**Chunked prefill** splits a long prefill into pieces and interleaves them with
decode work in the same step, bounded by `max_num_batched_tokens`.

The trade-off, stated honestly:

- The long prompt's own **TTFT gets slightly worse** (its prefill is spread out)
- **Everyone else's ITL tail gets much better** (no more full-step stalls)

That is a *tail-latency* optimisation, and it's the kind of nuance that shows you
understand serving rather than benchmarking.

### 2.6 Preemption — where latency spikes come from

When the KV cache fills up, vLLM must evict something. It **preempts** running
sequences. Historically there were two strategies:

- **Recompute** — throw the KV cache away, recompute it later from the prompt.
  Cheap for short sequences.
- **Swap** — move KV blocks out to CPU RAM and back. Cheaper for long ones.

**Important currency detail, confirmed the hard way on this project:** vLLM's
**V1 engine** — the default since 0.8 — **removed CPU swap entirely.
Preemption is now recompute-only**, and `--swap-space` no longer exists as a
CLI flag. If you describe vLLM preemption as "swap or recompute" in 2026, you
are describing V0. Say *recompute-only in V1*, and you will sound current
rather than like you read a two-year-old blog post.

Either way, a preempted request's latency spikes. **If you see mysterious p99
blowups under load, look at the preemption counter first.** A throughput graph
without a preemption count beside it is hiding something.

### 2.7 The metrics, and what each one is for

| Metric | Definition | Why it matters |
|---|---|---|
| **TTFT** | Request sent → first *content* token | Perceived responsiveness. Dominated by queueing + prefill |
| **ITL** | Gap between consecutive tokens | The streaming "feel". Tail matters more than mean |
| **TPOT** | Mean of ITL | Steady-state generation speed |
| **E2E latency** | `TTFT + TPOT × output_tokens` | What the user actually waits |
| **Throughput** | Output tokens/sec across all requests | Capacity, i.e. cost per token |
| **Goodput** | Throughput of requests *meeting an SLO* | **The honest metric** |

**Goodput is the one that separates a real engineer from a benchmark-runner.**
You can always raise raw throughput by batching harder — right up until p99 TTFT
is 40 seconds and every user has left. Goodput counts only requests that met
their latency target (say TTFT < 1s *and* TPOT < 50ms). Optimising throughput
without an SLO is optimising the wrong thing, and saying so out loud is a strong
signal.

### 2.8 The fundamental trade-off

```
        ▲ throughput (tok/s)
        │                    ╭──────────  ← batch saturated:
        │              ╭─────╯               throughput flat,
        │        ╭─────╯                     latency still climbing
        │   ╭────╯
        │ ╭─╯   ← the useful region
        │╭╯
        └──────────────────────────────────▶ latency (p99 TTFT)
```

Bigger batches → better GPU utilisation → higher throughput, **and** longer
queues → worse latency. There is no single "best" setting; there is a **Pareto
frontier**, and where you sit on it is a product decision, not a technical one.
An interactive chatbot and an overnight batch job want opposite ends.

**Phase 5 of this project is literally about mapping that curve.**

---

## 3. The hardware story — why every config value is what it is

This project runs on free-tier hardware, and those constraints drive real
decisions. Being able to explain them is worth more than having better hardware.

**Development machine:** AMD Ryzen 5 3500U, 14 GB RAM, **no NVIDIA GPU**.
**Benchmark machine:** Kaggle free tier — **2× Tesla T4** *(measured: confirmed
on a real session)*.

The T4 is **Turing, compute capability 7.5**. That single number determines a
surprising amount:

| Feature | Needs | T4 (SM 7.5) | Consequence for this project |
|---|---|---|---|
| `bfloat16` | SM 8.0 | ❌ | Profiles pin `float16` explicitly |
| FlashAttention-2 | SM 8.0 | ❌ | Falls back to **TRITON_ATTN** *(measured)* — must be recorded with every result |
| Marlin int4 kernels | SM 8.0 | ❌ | AWQ/GPTQ run on slower generic kernels |
| Native FP8 | SM 8.9 | ❌ | **Phase 6 is int4, not FP8** |

**Why bf16 vs fp16 matters** (a question you may well get): both are 16-bit, but
they split the bits differently. `bfloat16` keeps float32's 8-bit exponent, so it
has the same *range* and rarely overflows. `float16` has a 5-bit exponent — much
narrower range, needs care. Most modern checkpoints are *published* in bf16.
Turing has no bf16 tensor cores, so vLLM **refuses** `--dtype bfloat16` below
SM 8.0 outright. Leaving dtype on `auto` with a bf16 checkpoint therefore fails
at load time — *after* the download. Hence: pin `float16`, explicitly, in the
profile.

**The 2×T4 detail worth mentioning unprompted:** those two GPUs talk over
**PCIe, not NVLink**. Tensor parallelism splits each layer across both cards,
which requires an all-reduce *every layer*. Over PCIe that collective is
expensive. **A sub-linear TP=2 speedup here is expected, not a bug** — and
saying that before someone "catches" you with it is exactly how you defend a
result.

---

## 4. How the repository is organised

```
configs/profiles/          one YAML per target machine
  local-cpu.yaml             CPU-only dev loop (no GPU)
  colab-t4.yaml              1× T4
  kaggle-2xt4.yaml           2× T4, tensor parallel

src/inferstack/
  config.py       typed settings; env > .env > profile YAML > defaults
  probe.py        hardware detection → named capabilities
  compat.py       "can this profile run on this machine?"
  cli.py          the inferstack command
  logging.py      structlog; console locally, JSON in containers
  engine/
    launcher.py   EngineConfig → vLLM argv; process supervision
    client.py     OpenAI-compatible client that measures TTFT/ITL/TPOT
    smoke.py      the continuous-batching proof
  remote/
    kaggle.py     push/poll/fetch Kaggle kernels
    kernels/      scripts that RUN on the remote GPU
  gateway/        Phase 2 (empty)
  bench/          Phase 4 (empty)

docs/adr/         architecture decision records
docs/phases/      what each phase built and how to verify it
tests/            120 tests
```

### The four ideas that hold it together

**1. Profiles, not `if cuda:`** — Three target machines with *incompatible*
capabilities. Rather than branching on hardware throughout the code, each machine
is one YAML file. Code reads a profile. Adding a rented A10G later is a new file,
not a code change.

**2. Fail before it's expensive** — `inferstack doctor` probes the machine,
validates the selected profile against it, and **exits non-zero** if the
combination cannot work. On metered free-tier GPU hours, discovering a dtype
mismatch *after* a 3 GB download is a real cost.

**3. HTTP only** — Per ADR-0003, nothing outside `engine/` may import vLLM's
Python classes. The engine is addressed purely through its OpenAI-compatible
API. That's what makes the Phase 8 SGLang swap a config change instead of a
rewrite — and it mirrors how production deployments actually work.

**4. Push the stack to the GPU, don't connect to it** — Per ADR-0005. Kaggle
sessions have no public ingress. You *could* tunnel out with ngrok — but then
every latency number includes public-internet jitter, and **TTFT measured through
a tunnel is a measurement of the tunnel.** Instead the whole stack (engine, load
generator, analysis) is pushed into the session, run there, and results pulled
back as artifacts. The load generator and server share loopback. That is better
methodology, not a workaround — say it that way.

---

## 5. The phase plan

| Phase | Deliverable | Status |
|---|---|---|
| 0 | Foundations: profiles, hardware probe, config, ADRs | ✅ done |
| 1 | vLLM serving + continuous batching proven on hardware | ✅ done |
| 2 | FastAPI gateway: auth, SSE streaming, timeouts, backpressure | |
| 3 | Prometheus + Grafana: TTFT, TPOT, queue depth, KV-cache util | |
| 4 | Benchmark harness: Poisson arrivals, sweeps, p50/p95/p99 | |
| 5 | Continuous batching tuning → latency/throughput Pareto curves | |
| 6 | AWQ/GPTQ int4, prefix caching, speculative decoding, TP=2 | |
| 7 | Rate limiting, admission control, drain, multi-replica routing | |
| 8 | SGLang on the identical harness, head to head | |
| 9 | Written benchmark report | |

**Why this order.** You cannot measure what you haven't served (1 before 3). You
cannot tune what you haven't measured (3 before 5). You cannot claim an
optimisation without a baseline (5 before 6). Every phase exists because the
next one depends on it.

---

## 6. Current status — be precise about this

Phases 0 and 1 are complete. Every component below has now run on real
hardware, not just against mocks.

**Measured on Kaggle, 19 Sep 2026** — Tesla T4 (SM 7.5, 15 GB), vLLM 0.29.0,
Qwen2.5-1.5B-Instruct in float16:

| Measurement | Value |
|---|---|
| TTFT (single request) | 26 ms |
| TPOT (single request) | 14.6 ms/token |
| 1 request, end to end (64 tokens) | 0.95 s |
| 8 concurrent requests, wall clock | 1.04 s |
| **Speedup over serial** | **7.3× (91% of the 8× ceiling)** |
| Output throughput | 493 tok/s |
| TTFT p50 / p95 under load | 59 ms / 61 ms |
| Attention backend | TRITON_ATTN (no FA2 on SM 7.5) |
| KV cache | 8.62 GiB → 322,944 tokens → 78.84× concurrency |

**How to present that number.** Eight requests arriving together finished in
1.04 s when one alone took 0.95 s. That is continuous batching: they decoded in
a single running batch rather than queueing. It is 91% of the theoretical 8×,
and the engine reported capacity for **78.84×** — so the batch was barely
one-tenth full. The remaining 9% is scheduling overhead and the prefill of eight
prompts competing for one step.

Also worth volunteering: TTFT went from **26 ms alone to 59 ms p50 under load**.
Even a batch this small costs something at the head of the queue. That is the
latency-throughput trade-off showing up at the smallest possible scale — and
pointing at it yourself is far stronger than being asked.

**Still not done, and say so:** only one concurrency point was measured, so
there are no percentile curves yet. The `local-cpu` profile has never run a real
vLLM. Tensor parallelism is untested. Phases 2–9 — gateway, observability,
benchmark harness, tuning — are ahead.

> Being precise about what is proven versus what is merely written is the single
> most credible thing you can do. Overstating it is the one thing that will sink
> you.

---

## 7. The bugs found — your best material

Interviewers care far more about *what went wrong and how you caught it* than
about clean code. Each of these is a genuine finding with a real lesson.

### 7.1 The profile label silently lied

`load_settings("colab-t4")` returned colab-t4's **values** under the name
**"local-cpu"**, because an `INFERSTACK_PROFILE` entry in a `.env` file was
allowed to override the label after the file had already been chosen.

Nothing crashed. **That's the problem.** From Phase 4 that label is stamped onto
every benchmark artifact — so a result would have been attributed to the wrong
hardware profile. A wrong number that looks right is worse than a crash.

*Fix:* `profile` is a label recording which file was read, not a tunable.
*Lesson:* in measurement systems, **metadata correctness is as important as the
measurement.**

### 7.2 A hardware check that was itself wrong

I wrote a guard that compared requested vs granted Kaggle capabilities and
**raised** on a mismatch. On the first real run it reported `enable_gpu: False`
for a kernel that then received **two T4s**. The API field is stale/unreliable.

Had it shipped as written, it would have blocked every valid GPU run.

*Fix:* demoted to a warning; the authority became the session's own `probe.json`
— a session reporting on itself cannot be stale.
*Lesson:* **know which of your signals is authoritative.** A validator that's
wrong is worse than no validator.

### 7.3 PyTorch says the T4 supports bfloat16. It does not.

`torch.cuda.is_bf16_supported()` returns `True` on a T4 — because it counts
**emulation**. The card has no bf16 tensor cores, and vLLM refuses
`--dtype bfloat16` below SM 8.0 regardless.

Trusting torch here sends you straight into a load-time failure after a full
model download.

*Fix:* record both torch's answer and the compute-capability-derived answer, so
the contradiction is visible.
*Lesson:* **a capability API may answer a subtly different question than the one
you asked.**

### 7.4 The package installed fine, then failed on first use

Profiles were located by walking *up* the filesystem for `configs/profiles/` —
which exists in a git checkout and nowhere else. `pip install` succeeded; the
first profile lookup from `site-packages` raised `FileNotFoundError`.

Caught while preparing the GPU run — it would have failed *after* the 3 GB
download.

*Fix:* the wheel now ships the profiles; verified by installing the built wheel
into a clean venv in a directory containing no `configs/`.
*Lesson:* **"works in the repo" is not "works installed."** Test the artifact you
actually ship.

### 7.5 The TTFT definition that would have been quietly wrong

OpenAI-compatible streams open with a **role-only delta** carrying no content.
Counting it as the first token understates TTFT by a full inter-token gap and
inflates the token count by one.

*Fix:* TTFT is measured to the first chunk carrying **actual content**, and
there's a regression test named for it.
*Lesson:* **define your metrics in code, with tests, before you collect data.**
Every later phase inherits this definition — an error here wouldn't show up as a
failure, it would show up as a plausible, wrong graph.

### 7.6 A flag that had been valid for years was gone

The first real GPU run got all the way through — vLLM installed, InferStack
installed from git, profiles resolved from the wheel, hardware validation passed
on a real 2× T4 — and then died in one line:

```
vllm: error: unrecognized arguments: --swap-space 4
```

vLLM's V1 engine removed CPU swap, so the flag was deleted. My launcher had been
written against V0's flag set. **No unit test could have caught this** — the
mock didn't know what vLLM 0.29 accepts, because only vLLM 0.29 knows that.

*Fix:* the launcher now reads the installed engine's own `vllm serve --help` and
reports which of its flags are unsupported, saving the help text as an artifact.
Stripping is recorded, never silent — a removed flag changes what was measured,
so the artifact carries the original argv, the adjusted argv, and what was
dropped.
*Lesson:* **ask the binary, don't maintain a version table.** And more
generally: integration points against fast-moving dependencies need a real
integration test, because that is the one category of bug mocks structurally
cannot find.

### 7.7 The fix that caused a worse failure than the bug

Having been burned by a deleted flag, I made the launcher read
`vllm serve --help` and strip anything unsupported. On the next run that help
invocation printed a bare usage line listing only `--help`, because the model
positional was missing.

One flag is not zero flags. So the validator concluded that the other ten — host,
port, dtype, max-model-len, max-num-seqs, everything — were unsupported, and
stripped them all. The engine launched as a bare `vllm serve <model>` on
defaults.

I had explicitly written that an *empty* result means "unknown, never nothing
supported" — and then failed to apply the same reasoning to an *implausible*
result. **A validator that fails open is worse than no validator**, because it
corrupts a command line that was already correct.

*Fix:* a plausibility threshold, plus several probe invocations, one of which
supplies a placeholder model positional. If none returns a credible help text,
nothing is stripped.
*Payoff:* on the successful run it parsed 372 flags and stripped exactly one —
`--device`, which V1 also removed. Without it, that run would have failed the
same way the first one did.
*Lesson:* **safety mechanisms need their own failure analysis.** Ask what your
guard does when its input is garbage, not just when its input is missing.

---

## 8. Questions you will get, and how to answer

**"What does this project actually do?"**
> Serves an open-weights LLM through vLLM behind an OpenAI-compatible API with
> continuous batching, then instruments and tunes it. It's built in phases —
> serve, measure, optimise — so every optimisation claim has a baseline to be
> measured against.

**"What is continuous batching?"**
> Iteration-level scheduling. Instead of fixing a batch and running it to
> completion, the scheduler decides at *every decode step* which sequences run.
> When one finishes, its slot is freed immediately and a queued request takes it
> on the next iteration. Static batching runs until the longest sequence
> finishes, so short requests hold idle slots and new arrivals wait for the whole
> batch. Continuous batching removes both problems.

**"Why vLLM and not SGLang / TGI / TensorRT-LLM?"**
> vLLM is primary because its scheduler internals are the most directly
> observable — `max_num_seqs`, `max_num_batched_tokens`, chunked prefill,
> preemption — and that's what Phase 5 tunes. SGLang isn't dismissed; it's
> Phase 8, evaluated on the identical harness and request trace. TensorRT-LLM is
> faster on NVIDIA but the engine-build step would burn limited free-tier GPU
> hours on compilation rather than measurement.

**"Why such a small model?"**
> Because KV cache capacity is the real concurrency limit, not parameter count.
> Qwen2.5-1.5B in fp16 is ~3 GB, leaving ~11 GB of KV cache on a 15 GB T4 —
> roughly 96 concurrent 4096-token sequences. A 7B model would consume most of
> the VRAM in weights and leave room for a handful of sequences, which would make
> batching behaviour invisible. The project is about the *scheduler*, so I
> optimised for making scheduler behaviour observable.

**"What's the hardest part of LLM serving?"**
> Memory, and specifically the KV cache. Decode is memory-bandwidth-bound, KV
> cache grows per token per sequence, and it fragments badly. That's the problem
> PagedAttention solves by treating the KV cache like OS virtual memory — fixed
> blocks and a block table instead of contiguous per-sequence allocation.

**"Walk me through your results."**
> On a free-tier T4, one request took 0.95 s end to end with a 26 ms TTFT.
> Eight concurrent requests took 1.04 s — 7.3× better than serving them
> serially, 91% of the theoretical ceiling. The engine reported capacity for
> 78.84× concurrency, so the batch was about a tenth full; that headroom is what
> Phase 5 maps. Worth noting the cost side too: TTFT rose from 26 ms alone to
> 59 ms p50 under load, which is the latency-throughput trade-off at the
> smallest possible scale.

**"How do you know your numbers are real?"**
> Three ways. The measurement definitions are pinned in code with regression
> tests — TTFT excludes the role-only delta, for instance. The load generator
> runs *inside* the GPU session over loopback, so no WAN jitter contaminates
> TTFT. And every run records the exact engine argv plus which attention backend
> was active, because on a T4 FlashAttention-2 isn't available and results aren't
> comparable across backends.

**"Your T4 doesn't support bf16 or FP8 — isn't that a problem?"**
> It's a constraint I designed around rather than discovered late. The
> capability thresholds are encoded in code, so `inferstack doctor` refuses an
> impossible config in under a second instead of failing after a model download.
> Practically it means Phase 6 quantisation targets AWQ/GPTQ int4 rather than
> FP8, and that every result records its attention backend.

**"Why not just use a bigger GPU?"**
> A funded project would. But the constraints turned out to be instructive — the
> capability system, the fail-fast preflight, and the artifact-based remote
> execution all exist *because* the hardware was constrained, and all of them
> are good practice on any hardware.

**"What would you do differently?"**
> Get to a real integration run sooner. Most of the bugs I hit were
> configuration, packaging and version-drift problems rather than algorithmic
> ones — a profile label overridden by the environment, profiles missing from
> the wheel, a validator trusting an unreliable API field, a vLLM flag deleted
> between major versions. Unit tests found none of those, because mocks agree
> with whatever you assumed. In a measurement system the plumbing *around* the
> measurement is where the dangerous bugs live, since they produce plausible
> wrong answers instead of crashes.

**"How do you handle a fast-moving dependency like vLLM?"**
> Don't encode its interface as an assumption. My launcher originally emitted
> `--swap-space`, which V1 removed when preemption became recompute-only. Now it
> reads the installed binary's own `--help`, reports unsupported flags, and
> records any it strips into the run artifact — because a dropped flag changes
> what was measured, so that can never be silent.

**"Is it finished?"**
> No. Phases 0 and 1's infrastructure are done and the remote GPU path is
> proven, but the first end-to-end serving run is what converts the launcher,
> client and smoke harness from unit-tested to actually verified. Phases 2–9 —
> the gateway, observability, the benchmark harness and the tuning work — are
> ahead.

---

## 9. Glossary

| Term | Meaning |
|---|---|
| **Prefill** | Processing the prompt in one parallel pass. Compute-bound. Produces the first token |
| **Decode** | Generating one token at a time. Memory-bandwidth-bound. Runs once per output token |
| **KV cache** | Cached Key/Value vectors per token per layer, so attention isn't recomputed. The dominant memory consumer |
| **TTFT** | Time to first token — request sent to first *content* token |
| **ITL** | Inter-token latency — the gap between consecutive tokens |
| **TPOT** | Time per output token — the mean of ITL |
| **Throughput** | Output tokens per second across all requests |
| **Goodput** | Throughput counting only requests that met their SLO |
| **Static batching** | Fixed batch that runs until the longest sequence finishes |
| **Continuous batching** | Iteration-level scheduling; batch membership changes every step |
| **PagedAttention** | vLLM's KV cache allocator. Fixed blocks + block table — virtual memory for the KV cache |
| **Chunked prefill** | Splitting long prefills so they don't stall decode for everyone else |
| **Preemption** | Evicting a running sequence when KV cache runs out, by recompute or swap |
| **Prefix caching** | Reusing KV blocks across requests sharing a prefix |
| **Tensor parallelism** | Splitting each layer across GPUs; needs an all-reduce per layer |
| **SM / compute capability** | NVIDIA's GPU feature-level number. T4 = 7.5, A100 = 8.0, L4 = 8.9 |
| **AWQ / GPTQ** | 4-bit post-training weight quantisation schemes |
| **SLO** | Service level objective, e.g. "p99 TTFT < 1s" |

---

## 10. Running it yourself

```bash
uv venv
uv pip install -e ".[dev]"

inferstack doctor          # what is this machine, can the profile run here?
inferstack profiles        # list execution profiles
inferstack config show     # fully resolved settings
inferstack serve --dry-run # print the exact vLLM command without running it
```

`doctor` exits non-zero when the selected profile can't work on the current
machine, so it's safe in front of a long benchmark or in CI:

```bash
inferstack doctor --profile colab-t4 --strict
```

Run the checks:

```bash
pytest              # 120 tests
ruff check .        # lint (incl. bandit security rules)
```

### Further reading

- **vLLM / PagedAttention** — Kwon et al., *Efficient Memory Management for Large
  Language Model Serving with PagedAttention* (SOSP 2023)
- **Continuous batching** — Yu et al., *Orca: A Distributed Serving System for
  Transformer-Based Generative Models* (OSDI 2022)
- The ADRs in `docs/adr/` — every decision in this project, with the reasoning
  and the rejected alternatives
