# Stabilizing GLM-5.2 on 4× DGX Spark: Fixing the vLLM MTP Deadlock

**A field report on diagnosing and fixing a reproducible NCCL deadlock when serving GLM-5.2 (671B MoE) across four DGX Spark nodes with MTP speculative decoding.**

This guide builds directly on [tonyd2wild's GLM-5.2-QuantTrio-200K-4x-DGX-Spark recipe](https://github.com/tonyd2wild), which got us serving at all — full credit to Tony for the foundation. His recipe benchmarks at 28.8 tok/s, but under real interactive traffic our cluster deadlocked hard after a handful of requests. This guide documents the root causes, a copy-paste stability fix, and a step-by-step recovery ladder that gets you back to **~23 tok/s with zero deadlocks** — about 80% of the recipe's theoretical maximum. If you're hitting the same wall, you should be able to fix your cluster in about 30 minutes.

## Environment

| Component | Value |
|---|---|
| Hardware | 4× DGX Spark (GB10, 128 GB unified memory each) |
| Model | QuantTrio/GLM-5.2-Int4-Int8Mix (405 GB, 256 experts, unpruned) |
| vLLM | 0.23.1rc1 (July 12 dev build), image modded with CosmicRaisins' SM120 Triton kernels |
| Speculative decoding | MTP, k=4 |
| KV cache | fp8_ds_mla, 200K context |
| Fabric | RoCE v2 via MikroTik CRS812, 200 Gbps QSFP |
| Base recipe | tonyd2wild's GLM-5.2-QuantTrio-200K-4x-DGX-Spark |

If your setup differs (2 nodes, different quant, another large MoE model with MTP), the failure mode and fixes below likely still apply — the same symptoms are reported on 2-node Spark clusters ([vLLM #41530](https://github.com/vllm-project/vllm/issues/41530)).

## What This Fixes (Symptoms)

If you're seeing this, you're in the right place:

- The cluster serves **3–5 requests successfully, then permanently deadlocks**.
- All 4 GPUs pinned at **~96% utilization** with no forward progress — this is an NCCL spin-wait, not real work.
- No new log output, except EngineCore's shm_broadcast repeating every 60 seconds:

  ```
  No available shared memory broadcast block found in 60 seconds
  ```

- The only recovery is killing the containers and doing a full cluster restart.
- It **reproduces consistently** — this is not a transient network blip. We hit it on 4 consecutive restarts.

One honest note: we believe this deadlock exists in Tony's original config too. His 28.8 tok/s benchmark ran clean sequential traffic, which happens to avoid the degenerate batch shapes that trigger it. Interactive or concurrent traffic finds them within a few requests.

## Root Cause

Three compounding issues. Any one of them can wedge the cluster; together they make it near-certain.

### Issue 1: FULL cudagraph mode + MTP degenerate batch shapes → NCCL rank desync

FULL cudagraph mode bakes NCCL collective sizes into the graph at capture time. MTP speculative decoding then produces batch shapes the warmup never captured:

- **Bonus-token-only steps** — all draft tokens rejected, only the bonus token survives
- **Partial rejection mixes** — some requests accept drafts, others don't
- **End-of-wave drain steps** — the last few requests finishing out

When one rank dispatches a different graph than its peers, the collective sizes disagree across ranks, and every rank spins at ~96% SM forever, waiting on a collective that will never match. vLLM has **no TP-wide cudagraph dispatch consensus guard** — [#45610](https://github.com/vllm-project/vllm/pull/45610) added one for pipeline parallelism only.

This directly matches [vLLM #40969](https://github.com/vllm-project/vllm/issues/40969) (GB10: "serves first 5-6 requests then silently hangs, ~100% SM on both ranks") and [#41530](https://github.com/vllm-project/vllm/issues/41530) (2× DGX Spark, `mp --nnodes`, MTP, identical shm_broadcast symptoms).

### Issue 2: `--async-scheduling` + MTP race conditions

Async scheduling combined with MTP is currently the newest, buggiest intersection in vLLM:

- [#40610](https://github.com/vllm-project/vllm/issues/40610) — proposer event race: the next batch mutates block tables while the proposer is still reading them
- [#47928](https://github.com/vllm-project/vllm/pull/47928) — async spec-slot accounting leak specific to GLM-5.2 MTP (merged July 7, 2026)
- [#46669](https://github.com/vllm-project/vllm/issues/46669) — garbage output at concurrency >1

Worse, the `take_draft_token_ids` RPC has **no timeout** — if a worker wedges for any reason, EngineCore blocks on it forever. That's why the hang is silent rather than a crash.

### Issue 3: NCCL 2.30.4 NVLS silent hang with 256-expert MoE

[NCCL #2167](https://github.com/NVIDIA/nccl/issues/2167) documents NVLS-related silent hangs on DGX Spark with MoE models of ≥256 experts. GLM-5.2 has exactly 256 experts. The fix is free: `NCCL_NVLS_ENABLE=0` costs nothing on Spark because there's no NVLink for NVLS to use anyway.

## The Fix — Phase 1: Get Stable First

Apply these five changes on **all 4 nodes**, then relaunch. Don't skip straight to the performance tuning — establish a stable baseline first, so that when you tune you can tell exactly which change breaks things.

**Environment variables** (add to your container environment / launch script on every node):

```bash
export NCCL_NVLS_ENABLE=0                        # NVLS silent hang, NCCL #2167
export CUDA_DEVICE_MAX_CONNECTIONS=32            # improve GPU pipeline parallelism
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=600    # detect wedged workers in 10 min, not 30
```

Keep all the NCCL variables from Tony's recipe (`NCCL_NET=IB`, `NCCL_MAX_NCHANNELS=4`, etc.) — those are correct for the RoCE fabric and unrelated to this bug.

**vLLM serve arguments** — three changes relative to the base recipe:

```bash
# 1. Change cudagraph mode from FULL to PIECEWISE:
--compilation-config '{"cudagraph_mode":"PIECEWISE"}'

# 2. REMOVE --async-scheduling entirely (restored later in the ladder)

# 3. REMOVE draft_tensor_parallel_size from --speculative-config, keeping:
--speculative-config '{"method":"mtp","num_speculative_tokens":4,"attention_backend":"FLASHMLA_SPARSE"}'
```

**Result:** completely stable. 10+ consecutive requests without a hiccup, and GPUs properly idle at 0% between requests (if yours sit at 96% while idle, they're deadlocked, not busy). The cost: throughput drops from ~26 tok/s (unstable) to **~15 tok/s**. Phase 2 recovers most of that.

## Cherry-Pick vLLM PR #48572 (Critical Patch)

[vLLM #48572](https://github.com/vllm-project/vllm/pull/48572) (merged July 14, 2026 — after the 0.23.1rc1 build in the recipe) fixes two problems that directly feed this deadlock class:

1. **MTP vocab-collective token-count mismatch.** Under sequence-parallel MoE with TP>1, the MTP block reduce-scatters activations down to a per-rank token shard, but `shared_head`/`compute_logits` expects full tokens. The subsequent vocab-parallel `all_gather` sees inconsistent token counts across ranks → deadlock. The fix restores full tokens via `tensor_model_parallel_all_gather` before `compute_logits`.

2. **Spec-decode warmup at full MTP width.** Warmup only exercises `compute_logits` at 1 row per request, but real spec decode produces `max_num_reqs × (1 + K)` rows. On the first real decode step, NCCL lazily connects the extra channels — and fails. The fix warms `compute_logits` at the full spec-decode row count during init.

Both patches are **Python-only** — no image rebuild required. Bind-mount the patched files over the installed package:

```bash
# On each node, after fetching the patched files from the PR:
# EDIT: set PATCH_DIR to wherever you placed the patched .py files
docker run ... \
  -v "${PATCH_DIR}/<patched-file>.py:/usr/local/lib/python3.12/dist-packages/vllm/<path-from-PR>.py:ro" \
  ...
```

Check the PR's changed-files list for the exact paths in your vLLM version, and verify the in-container package location with `docker exec <container> python -c "import vllm; print(vllm.__file__)"`.

## The Recovery Ladder — Phase 2: Earn the Speed Back

The rule: **one change per relaunch, soak-tested with 10+ real requests before the next step.** If a step deadlocks, revert exactly that step — you now know your specific tripwire.

| Step | Change | Throughput | Status |
|---|---|---|---|
| Phase 1 baseline | PIECEWISE, no async-scheduling | ~15 tok/s | ✅ Stable |
| Step 0 | + #48572 patches + `CUDA_DEVICE_MAX_CONNECTIONS=32` | ~15.3 tok/s | ✅ Stable |
| Step 1 | `cudagraph_mode` → `FULL_DECODE_ONLY` | **~23.7 tok/s** | ✅ Stable |
| Step 2 | + `hf-overrides` with `index_topk_pattern` | ~22.6 tok/s | ✅ Stable |
| Step 3 | + `--async-scheduling` restored | **~22.7 tok/s** | ✅ Stable |

**The key insight is Step 1: `FULL_DECODE_ONLY` is the sweet spot.** It captures cudagraphs for pure-decode steps — where shapes are regular and Python re-entry overhead hurts most — while falling back to eager execution for mixed prefill-decode steps, which is exactly where MTP's shape diversity causes rank desync. You dodge the FULL/FULL_AND_PIECEWISE dispatch divergence that caused the deadlock while recovering roughly 60% of the throughput lost by dropping to PIECEWISE.

Restoring `--async-scheduling` (Step 3) is safe **only after** the #48572 patches are in place — the July 7 slot-leak fix (#47928) plus the two #48572 fixes remove the known GLM-5.2 MTP async races. If you can't apply the patches, stop at Step 2.

## Final Recommended Config

```bash
# ── Environment variables (ALL 4 nodes) ─────────────────────────
export NCCL_NVLS_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=32
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=600
# ...plus all existing NCCL vars from the base recipe
# (NCCL_NET=IB, NCCL_MAX_NCHANNELS=4, etc.) — unchanged.

# ── vLLM serve args (changes vs. the base recipe) ───────────────
# EDIT: substitute your own model path, node addresses, and rank args
vllm serve <MODEL_PATH> \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --async-scheduling \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4,"attention_backend":"FLASHMLA_SPARSE"}' \
  --hf-overrides '{"use_index_cache":true,"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}' \
  # ...all other args from the base recipe unchanged

# Plus the #48572 patches applied (bind-mount, or baked into your image).
```

**Result: ~23 tok/s single-stream, fully stable.** That's 80% of the theoretical 28.8 tok/s maximum, with zero deadlocks across extended soak testing — GPUs idle cleanly at 0% between requests, and the cluster survives the concurrent and bursty traffic that reliably killed the original config within 5 requests.

## Why Not the Full 28.8 tok/s?

The remaining ~20% gap is `FULL_DECODE_ONLY` vs `FULL` cudagraph mode, and it isn't recoverable safely today. `FULL` gives the extra speed but requires all TP ranks to provably agree on which graph they dispatch every step — a TP-wide cudagraph dispatch consensus guard that **does not exist in vLLM yet**. [#45610](https://github.com/vllm-project/vllm/pull/45610) added exactly this guard for pipeline parallelism, but tensor parallelism remains unguarded. With MTP in the mix, degenerate batch shapes *will* eventually cause a dispatch divergence, and the failure mode is the silent 96%-SM spin described above — possibly hours or days into an otherwise healthy run.

Until a TP consensus guard merges upstream, **`FULL_DECODE_ONLY` is the safe ceiling.** We'd rather run 23 tok/s forever than 28.8 tok/s until the next drain step. If you're benchmarking clean sequential traffic and understand the risk, `FULL` will still hit the recipe numbers — just don't put it in front of real users.

## Credits & References

- **[tonyd2wild](https://github.com/tonyd2wild)** — the GLM-5.2-QuantTrio-200K-4x-DGX-Spark recipe this entire setup is built on. None of this works without it.
- **CosmicRaisins** — SM120 Triton kernels for GB10.
- **QuantTrio** — the GLM-5.2-Int4-Int8Mix quantization.
- The vLLM contributors behind #48572, #47928, and #45610.

### vLLM issues & PRs — deadlock class

- [#40969](https://github.com/vllm-project/vllm/issues/40969) — GB10, FULL_AND_PIECEWISE + chunked prefill: "serves 5-6 requests then silently hangs, ~100% SM"
- [#41530](https://github.com/vllm-project/vllm/issues/41530) — 2× DGX Spark, `mp --nnodes`, MTP: identical symptoms; only `--enforce-eager` was fully stable
- [#42271](https://github.com/vllm-project/vllm/issues/42271) — DeepSeek-V4-Flash, MTP, FULL_AND_PIECEWISE: deterministic deadlock on bonus-token-only shapes; FULL_DECODE_ONLY stable
- [#45610](https://github.com/vllm-project/vllm/pull/45610) — cudagraph dispatch consensus guard (PP only; TP still unguarded)
- [#48572](https://github.com/vllm-project/vllm/pull/48572) — GLM-5.2 MTP vocab-collective mismatch + spec-verify warmup (merged July 14, 2026)

### vLLM issues & PRs — async-scheduling + MTP

- [#40610](https://github.com/vllm-project/vllm/issues/40610) — proposer event race (draft PR)
- [#47928](https://github.com/vllm-project/vllm/pull/47928) — async spec-slot accounting leak, GLM-5.2 MTP (merged July 7, 2026)
- [#46669](https://github.com/vllm-project/vllm/issues/46669) — MTP + async-scheduling garbage output at concurrency >1

### NCCL

- [NCCL #2167](https://github.com/NVIDIA/nccl/issues/2167) — NVLS silent hang on DGX Spark with ≥256-expert MoE

---

*Found a different tripwire, or got FULL mode stable? Open an issue — this is a living document.*
