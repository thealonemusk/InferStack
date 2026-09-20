# Plugging InferStack into an existing workflow

Four ways to get value out of this project, ordered by how little you have to
change to get it.

| # | You want to… | You need | Works today |
|---|---|---|---|
| 1 | Measure an endpoint you already have | Nothing but this CLI | ✅ |
| 2 | Gate a deploy on serving behaviour in CI | This CLI | ✅ |
| 3 | Replace a hosted API with your own server | A CUDA GPU | ✅ |
| 4 | Validate hardware before provisioning | Nothing | ✅ |

> Honest scoping: the FastAPI gateway (auth, rate limiting, multi-replica
> routing) is Phase 2 and **does not exist yet**. Everything below works with
> what is built.

---

## 1. Measure an endpoint you already have

**No GPU required. Nothing to deploy.** The smoke harness speaks plain
OpenAI-compatible HTTP, so it measures anything that does: vLLM, SGLang, TGI,
llama.cpp's server, LM Studio, Ollama's OpenAI shim, or a hosted API.

```bash
pip install git+https://github.com/thealonemusk/InferStack@phase-03-observability

inferstack smoke \
  --base-url http://your-host:8000/v1 \
  --model your-model-id \
  --concurrency 8
```

What you get back that a `curl` cannot tell you:

```
Single request, end to end      0.95 s
Single request, TTFT            26 ms
Single request, TPOT            14.6 ms/token
8 requests, if serial           7.58 s
8 requests, measured            1.04 s
Speedup over serial             7.3x (ideal 8x)
Output throughput               493 tok/s
TTFT p50 under load             59 ms
TTFT p95 under load             61 ms

Verdict: requests are batched, well short of saturation
```

### And if that endpoint is a vLLM, read its own signals too

`smoke` measures the endpoint from outside. vLLM knows things no external
measurement can reach — how many sequences are in the running batch right now,
how many are queued behind them, how full the KV cache is, how often it has had
to preempt — and it already exports them. No Prometheus, no Grafana, no agent:

```bash
inferstack metrics --url http://your-host:8000
```

Load is printed above latency deliberately: a p99 of 4 s means one thing at
queue depth 60 and something entirely different at queue depth 0. Percentiles
come from bucket counts using the same interpolation as Prometheus'
`histogram_quantile`, so a number here and a number on a Grafana panel agree —
and the output says out loud that they are no finer than the engine's own
bucket boundaries.

Note the port: `/metrics` is on the **engine**, not on a gateway in front of
it. If a signal is missing from the output it is listed as missing rather than
shown as zero, because an idle engine and a wrong URL otherwise render
identically — and because that is how this project found it was asking vLLM for
a TPOT metric by a name vLLM does not use.

To capture a whole load episode rather than one instant — including from a
session nothing outside can scrape:

```bash
inferstack metrics --url http://your-host:8000   --duration 60 --interval 0.2 --out run.jsonl
```

One JSON object per line, buckets included, so a percentile can be recomputed
later. The summary reports output throughput from the *delta* between the first
and last token counter; dividing a cumulative total by uptime would average in
every idle second since the engine started.

**Why this is worth running against a service you already own.** A serialised
server and a batching one look identical from a single request. They diverge
completely under concurrency, and that difference is invisible until you look
for it. If the speedup comes back near **1.0×**, your requests are queueing —
which usually means batching is off, `max_num_seqs` is 1, or something in front
is serialising them.

For an authenticated endpoint:

```bash
export INFERSTACK_API_KEY=sk-...
inferstack smoke --base-url https://api.example.com/v1 --model gpt-4o-mini
```

Machine-readable output for storing alongside a release:

```bash
inferstack smoke --base-url ... --model ... --json > smoke-$(git rev-parse --short HEAD).json
```

**Caveat worth stating before you quote any of it.** This is a sanity check, not
a benchmark: one concurrency point, a burst arrival pattern, a short prompt.
Controlled arrival rates and real percentile curves are Phase 4. Run it from a
machine close to the endpoint, or you are partly measuring your own network.

---

## 2. Gate a deploy in CI

Both commands exit non-zero on failure, so they work as pipeline steps without
any glue.

```yaml
# .github/workflows/inference.yml
- name: Verify the model server batches
  run: |
    inferstack smoke \
      --base-url ${{ vars.INFERENCE_URL }} \
      --model ${{ vars.MODEL_ID }} \
      --concurrency 16
  env:
    INFERSTACK_API_KEY: ${{ secrets.INFERENCE_API_KEY }}
```

`smoke` fails the build when the server is unreachable, when any request
errors, **or when requests were served serially rather than batched**. That last
condition is the useful one: it catches a config regression — someone setting
`max_num_seqs: 1`, disabling continuous batching, or putting a serialising proxy
in front — that no unit test and no single-request health check would notice.

Guard a GPU job before it costs anything:

```yaml
- name: Refuse an impossible configuration
  run: inferstack doctor --profile colab-t4 --strict
```

---

## 3. Replace a hosted API with your own server

InferStack serves an **OpenAI-compatible** endpoint, so existing client code
changes by one line: the base URL.

Start it:

```bash
pip install "inferstack[engine] @ git+https://github.com/thealonemusk/InferStack@phase-03-observability"

inferstack doctor --profile colab-t4      # confirm the box can run it
inferstack serve  --profile colab-t4      # preflight, then launch
```

Then point your existing code at it.

**OpenAI SDK**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",   # the only change
    api_key="not-used-yet",                 # auth arrives in Phase 2
)

response = client.chat.completions.create(
    model="qwen2.5-1.5b",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)
```

**LangChain**

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-used-yet",
    model="qwen2.5-1.5b",
)
```

**LlamaIndex**

```python
from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(
    api_base="http://localhost:8000/v1",
    api_key="not-used-yet",
    model="qwen2.5-1.5b",
    is_chat_model=True,
)
```

### Tune it for your workload

Every knob is an environment variable, so nothing needs editing in the repo:

```bash
# Many short requests - a chat product. Favour concurrency.
INFERSTACK_ENGINE__MAX_NUM_SEQS=256 \
INFERSTACK_ENGINE__MAX_NUM_BATCHED_TOKENS=8192 \
inferstack serve -p colab-t4

# Long shared prefixes - RAG, or a large fixed system prompt.
INFERSTACK_ENGINE__ENABLE_PREFIX_CACHING=true \
inferstack serve -p colab-t4

# Latency-sensitive. Smaller batches, less queueing.
INFERSTACK_ENGINE__MAX_NUM_SEQS=32 \
inferstack serve -p colab-t4
```

Describe your own hardware once, as a profile:

```yaml
# configs/profiles/my-a10g.yaml
description: "Production: 1x A10G (SM 8.6, 24 GB)"
engine:
  model: meta-llama/Llama-3.1-8B-Instruct
  device: cuda
  dtype: bfloat16          # SM 8.6 has bf16 units, unlike a T4
  max_model_len: 8192
  gpu_memory_utilization: 0.90
  max_num_seqs: 256
  enable_chunked_prefill: true
```

```bash
inferstack doctor --profile my-a10g     # validates it against the real card
inferstack serve  --profile my-a10g
```

**Before you commit to a model size,** do the KV-cache arithmetic from
[the project guide](PROJECT-GUIDE.md#22-the-kv-cache-and-why-it-dominates-everything).
Weights are the easy part; KV cache capacity is what sets your real concurrency
limit, and an 8B model on a 24 GB card leaves far less room for it than people
expect.

---

## 4. Validate hardware before you provision it

`inferstack doctor` derives capabilities from CUDA compute capability and
refuses configurations that cannot work — in under a second, before a model
download.

```bash
inferstack doctor --json | jq .environment.capabilities
```

```json
{
  "cuda": true,
  "bfloat16": false,
  "flash_attention_2": false,
  "fp8_quantization": false,
  "int4_marlin": false,
  "tensor_parallel": true
}
```

Useful when picking an instance type. A T4 is cheap, and that table is why it
may still be the wrong choice: no bfloat16 means most modern checkpoints need an
explicit `float16` override, no FlashAttention-2 means a slower attention path,
and no FP8 means int4 is your only quantisation route.

---

## What is not here yet

Stated plainly so nothing below is a surprise:

- **`serve` on its own is unauthenticated.** vLLM's endpoint has no API-key
  check; that lives in `inferstack gateway`, which must be put in front of it.
  Do not expose `serve` directly. The gateway itself has now run in front of a
  real vLLM on a T4 and costs about 7 ms of TTFT.
- **No per-key rate limiting.** The gateway's admission control is *global*: a
  fixed number of in-flight requests and a fast 429 beyond it. Per-key quotas
  are Phase 7.
- **No tracing.** Request ids reach the logs, but nothing correlates a single
  request across the gateway and the engine. Deferred deliberately; see
  ADR-0007.
- **Metric names are confirmed against vLLM 0.29.0 only.** They come from a
  capture, not a guess (`tests/fixtures/vllm_metrics_real.txt`), and older
  spellings are accepted as aliases — but if `inferstack metrics` reports a
  signal as *missing* against your engine, that is a version difference worth
  reporting rather than an idle server.
- **The compose stack's scrape targets are `host.docker.internal`.** Fine on
  Docker Desktop, fine on Linux via the `extra_hosts` mapping, and wrong for
  anything real: edit `deploy/compose/prometheus/prometheus.yml`.
- **Alert thresholds are placeholders and say so.** A latency target is a
  product decision; these are starting points, not SLOs.
- **No multi-replica routing.** One engine per gateway. Phase 7.
- **Single concurrency point only.** `smoke` is a closed-loop sanity check, so
  it sends *fewer* requests when the server slows down. Poisson arrivals,
  percentile curves and goodput are Phase 4.

If you need per-key quotas today, keep an existing reverse proxy in front of the
gateway — that part is Phase 7, not done.
