# Multi-Node Inference Deadlocks on DGX Spark (GB10): A Community Root-Cause Analysis and Request for Engagement

**Audience:** NVIDIA DGX Spark platform, NCCL, and GPU driver engineering
**From:** DGX Spark community operators running multi-node LLM inference
**Date:** 2026-07-14
**Status:** Root cause analysis complete at our layer; remaining fixes require NVIDIA

---

## 1. Executive Summary

Multi-node LLM inference on DGX Spark (GB10) clusters deadlocks permanently after minutes to hours of sustained operation. The hang is silent: all GPUs report ~96% SM utilization, 0% memory utilization, and ~15 W power draw — the signature of NCCL device kernels spin-waiting forever on a collective that one rank never joins. It reproduces across models (GLM-5.2, MiniMax-M2.7), runtimes (vLLM; TRT-LLM reports exist but see caveats), executor backends (multiprocessing and Ray), and with CUDA graphs fully disabled (`--enforce-eager`). Our analysis — built on NCCL v2.30.4 source review, six instrumented community reproductions, and kernel logs from our own four-node cluster — resolves the failure into four stacked layers: an engine-level tensor-parallel control-flow race (upstream vLLM's to fix), **the absence of GPUDirect RDMA on stock DGX Spark** (which widens the race window 10–100× and is the single reason this platform fails where H100/A100 clusters do not), an unfixed GB10 UMA driver allocation failure (0x51) that occasionally produces the same symptom, and NCCL's lack of any collective timeout, which converts a transient desync into a permanent outage. The community has built detection and auto-restart mitigations, but three of the four layers are only fixable by NVIDIA. This report documents the evidence and makes five specific, actionable requests.

---

## 2. Platform & Scope

**Hardware:** DGX Spark (GB10 Grace Blackwell, sm_121), 128 GB unified LPDDR5x, one GPU per node, clustered 2–8 nodes over the ConnectX NIC. Our own deployment is 4 nodes running GLM-5.2 with tensor parallelism (TP=4) under vLLM.

**Software:** DGX OS 7.2.x–7.5.0, drivers 580.142 / 580.159.03 (590.x is unsupported on Spark and regresses UMA behavior), NCCL 2.29.3–2.30.7, vLLM v1 engine, inbox rdma-core.

**How widespread:** Independent reproductions span at least four vLLM issues (#46097, #41725, #46253, #43547), one NCCL issue (#2213), analogous SGLang issues on the same hardware (#27949, #29548), and multiple NVIDIA developer forum threads. This is not one operator's misconfiguration; it is the default experience of anyone running multi-node tensor-parallel inference on this platform.

Throughout this report we distinguish **[verified]** — reproduced or confirmed on our own cluster, or confirmed by our direct source review — from **[reported]** — credible community evidence we have not independently reproduced.

---

## 3. Symptom

A cluster serving traffic normally wedges without any error, log line, or exception:

- All ranks: SM utilization ~96%, memory utilization 0%, power ~15 W (vs. ~100 W under real load). **[verified]**
- All worker processes alive; py-spy shows every rank blocked in or around a collective launch. No rank has crashed. **[verified via #46097 captures; consistent with our own wedges]**
- Time-to-failure is load- and configuration-dependent: minutes with full CUDA graph capture, ~2 hours with decode-only graphs, ~5 hours in eager mode under heavy load. The failure probability curve moves; it never reaches zero. **[verified across our own soak testing and #46097]**
- Recovery requires killing and restarting the entire serving stack. NCCL device kernels never exit on their own (see §4, L4).

The most precise capture is vLLM #46097: six independent eager-mode wedges on a 4× Spark cluster in 36 hours, each with full py-spy stacks. All six show the identical pattern: **exactly one rank has completed the MoE-output tensor-parallel all-reduce and advanced into the next decoder layer's projection, while the other three ranks are still entering that all-reduce.** One rank is one collective ahead; everyone spins forever.

---

## 4. Root Cause Analysis

There is no single bug. Four layers stack, and all four must be understood to fix the platform experience.

### L1 — Engine-level TP control-flow divergence (trigger)

The desync originates above NCCL, in data/timing-dependent control flow in the inference engine's forward pass. Evidence:

- In all six #46097 captures, the "ahead" rank rotates among the *lightly loaded* nodes and is never the slowest one. A transport fault would strand a random or fixed rank and eventually surface an IB timeout; a control-flow race selects whichever rank runs fastest. **[reported, with stacks]**
- Eager mode still deadlocks. CUDA graphs are an **amplifier** (they add per-rank graph-dispatch decisions with no cross-rank consensus), not the cause. **[reported, 6 reproductions]**
- NCCL version is not causal at this layer: #46097 runs 2.30.7 and still wedges. **[reported]**
- Known concrete divergence mechanisms are already documented in vLLM's own tracker: per-rank cudagraph dispatch without a TP consensus guard, speculative-decoding drafter-skip decisions near length bounds (#44954), zero-token/drain steps skipping batch coordination (#43547, fix unmerged), and mid-generation client aborts (which we correlated with wedge onset in our own soak tests **[verified]**).

L1 is vLLM's bug to fix, and we are pursuing it there. It matters to NVIDIA because of L2.

### L2 — No GPUDirect RDMA on stock DGX Spark (amplifier — the platform-specific layer)

This is why the same engine code that runs for weeks on H100/A100 clusters fails within hours on Spark.

Stock DGX OS ships inbox rdma-core, which lacks `mlx5dv_reg_dmabuf_mr`. NCCL init on every stock Spark logs:

```
dlvsym failed on mlx5dv_reg_dmabuf_mr ... undefined symbol: mlx5dv_reg_dmabuf_mr, version MLX5_1.25
GPU Direct RDMA Disabled for HCA ...
GDR 0
```

**[verified on our cluster]**

Consequences of `GDR 0`:

1. **Every cross-node collective is host-staged**: chunks bounce through host buffers driven by NCCL's CPU proxy threads, with driver calls per chunk. On GDR-capable platforms the NIC DMAs GPU memory directly and the proxy is a lightweight coordinator.
2. **Collective timing becomes a function of host CPU scheduling.** The proxy threads race on the same 20 Grace cores that run the inference engine, the executor plumbing, and the OS. Host jitter directly modulates when each rank enters and exits a collective. Our estimate is that the race windows L1 bugs need are **10–100× wider** on Spark than on any GDR-capable NVIDIA platform.
3. **It is NCCL's least-exercised code path at scale.** NCCL #2213 — a SIGSEGV in `ncclLocalOpAppend()` after ~1.1M collectives on a dual-Spark system — occurs on exactly this CPU-staged path. **[reported; our review of 2.30.4 found no clean counter-wraparound explanation, suggesting a slower proxy-path accounting bug — it deserves NVIDIA triage; it currently has none]**
4. The continuously-active proxy also invalidates cross-node CUDA graph capture (`capture_error_mode="global"`), the subject of vLLM #46253. **[reported]**

Every multi-node DGX Spark deployment is running this degraded path today, and users only discover it by reading NCCL init logs.

Contributing platform factors, ranked below GDR: one GPU per node means *every* TP collective is cross-node (~17 GB/s measured all-reduce bus bandwidth vs. ~450 GB/s NVLink on an H100 box — long collectives, long exposure per step); UMA co-tenancy means CPU memory pressure perturbs the GPU driver directly; and the sm_121 software ecosystem is immature (PyTorch aarch64 wheels do not list sm_121), adding per-rank JIT variance. We also note what it is **not**: NCCL topology detection is correct on Spark (the Grace↔Blackwell C2C link is detected via NVML; NCCL PR #2202 was withdrawn), and NVLS is not involved (Spark has none; the claim in NCCL #2167 was retracted by its own reporter).

### L3 — GB10 UMA 0x51 driver leak (co-traveling bug, same disguise)

Independently of the deadlock, GB10 exhibits `NV_ERR_NO_MEMORY (0x51)` from `_memdescAllocInternal` (mem_desc.c:1359) — a **contiguous allocation failure under UMA fragmentation**, not memory exhaustion (community reports show it firing with 118 GB free). **[verified: our own kernel logs captured the byte-identical signature on 2026-07-10; reported: one forum user logged 508 occurrences over 6 days]**

Usually the system survives it. When descriptor allocation fails inside a serving worker, that worker stalls, its peers spin, and the result is indistinguishable from an L1 wedge at the GPU level.

The driver situation, as best the community can determine:

- 580.142: first 0x51 within minutes of idle-adjacent workloads. **[reported]**
- 580.159.03 (current, newest supported): release notes say "enhancements to OOM handling"; field measurements show the leak slowed ~30× (first hit ~67 min vs ~2.5 min) — **not fixed**. **[reported, consistent with our single-hit-in-6-weeks observation]**
- 590.x: unsupported on Spark and regresses UMA reclamation (~80 GB unreclaimed after CUDA exit). **[reported]**
- No known-issues entry, no fix timeline; direct forum questions about a real fix have gone unanswered since May.

The community currently mitigates with scheduled reboots and page-cache drops. That is not a durable answer for a product positioned for sustained AI workloads.

### L4 — NCCL has no collective timeout (converter: transient → permanent)

We reviewed NCCL v2.30.4 source directly. **[verified]**

- Device collective kernels wait in unbounded spin loops (`primitives.h:116`, `prims_ll.h:65`). There is no iteration ceiling and no wall-clock bound.
- The only exit is `abortFlag`, polled every 10,000 spins (`primitives.h:143–153`). Nothing in the stock inference stack ever sets it. (vLLM's custom NCCL path bypasses PyTorch's ProcessGroupNCCL watchdog entirely, and additionally removes `NCCL_ASYNC_ERROR_HANDLING` from the environment — so no watchdog exists at *any* layer.)
- The word "watchdog" appears zero times in NCCL's documentation. `NCCL_IB_TIMEOUT` and socket retries fire on transport *errors*, not on a peer that never issues the matching collective.

The 96%-SM/15W symptom is exactly what this code does by construction: SM occupancy with no work, forever. A one-collective desync from *any* cause — L1 race, L3 allocation failure, a crashed peer — becomes a permanent, silent outage.

One positive finding we want to highlight: **NCCL's RAS subsystem (on by default since 2.24, localhost:28028) already diagnoses this condition perfectly.** `echo verbose status | nc localhost 28028` prints per-rank collective operation counts with an explicit `MISMATCH` grouping showing which rank is ahead. We found this in source and docs; nobody in any of the community threads knew it existed. RAS is genuinely excellent engineering — it deserves to be surfaced in DGX Spark troubleshooting documentation.

---

## 5. What the Community Has Tried (and Why It Is Not Enough)

| Mitigation | Effect | Limitation |
|---|---|---|
| PIECEWISE cudagraph mode | MTBF from minutes to many hours **[verified]** | Moves along the probability curve; does not reach zero, and costs throughput |
| FULL_DECODE_ONLY graphs | Best throughput (~23 tok/s on our 4-node GLM-5.2) **[verified]** | Wedges on ~2 h soak **[verified]** |
| `--enforce-eager` | Removes graph-dispatch divergence | Still deadlocks (~5 h MTBF under load) **[reported, 6 captures]** |
| NCCL 2.30.7 upgrade | Fixes an idle-time proxy TCP-accept hang (PR #1834) **[verified in source]** | Does not touch L1; #46097 deadlocks on 2.30.7 |
| NCCL RAS polling | Detects and classifies wedges within a minute **[verified in source/docs]** | Detection only |
| External watchdog: gloo side-channel step counter + `ncclCommAbort` + auto-restart | Converts a permanent outage into a ~15 s detection and a ~5 min restart | A tourniquet. It aborts and restarts; it does not prevent, and it exists only because no layer NVIDIA ships has a timeout |

The pattern: everything available to us either shifts failure probability or automates recovery. Nothing available to us shrinks the race window (needs GDR — L2), fixes the allocator (L3), or bounds the spin (L4). Those are yours.

---

## 6. Specific Requests

Ordered by impact.

### 6.1 Ship dmabuf-capable rdma-core for DGX Spark (highest impact)

Provide an officially supported rdma-core with `mlx5dv_reg_dmabuf_mr` (and any needed kernel bits) so NCCL reports `GDR 1` on Spark — via DGX OS update, DOCA support for the platform, or a documented supported package. This single change moves every cross-node collective off the host-staged CPU-proxy path, removes host scheduling jitter from collective timing, and shrinks the L1 race windows by our estimated 10–100× — likely making the engine-level divergence practically non-triggerable on this platform, exactly as it is on H100/A100. It would also take the ecosystem off NCCL's least-hardened code path (see NCCL #2213) and unblock cross-node CUDA graph capture (vLLM #46253). If there is a hardware or firmware reason GB10 cannot support GDR, the community needs to know that explicitly (see 6.4).

### 6.2 Fix the GB10 UMA 0x51 memdesc allocation failure

580.159.03's ~30× deceleration is appreciated, but it is mitigation, not a fix; 590.x regresses. We ask for: (a) acknowledgment in the DGX Spark known-issues list, (b) a root-cause fix for the fragmentation-driven contiguous-allocation failure in `_memdescAllocInternal`, and (c) interim official guidance (reboot cadence, cache-drop practice, allocator tunables) so operators stop reverse-engineering it from forum threads.

### 6.3 Add a bounded-wait / timeout option to NCCL device kernels

The unbounded spin in `primitives.h` means any single-rank stall anywhere in the stack becomes a permanent, undiagnosable outage. We ask for an opt-in timeout (spin-count ceiling or wall-clock bound) that sets the async error state and aborts the communicator, so the host stack gets a real error instead of eternal silence. The plumbing largely exists: the kernels already poll `abortFlag` every 10k spins, and RAS already tracks per-rank progress — this is connecting two mechanisms NCCL already has. We understand the design tension (spurious aborts on legitimately long collectives), which is why we ask for it as opt-in and configurable, defaulting off. Inference operators will gladly trade a rare spurious abort-and-restart for the elimination of permanent hangs.

### 6.4 Document the GDR-disabled status officially

Today, the only way a Spark owner learns that GPUDirect RDMA is non-functional on their multi-node cluster is by spotting a `dlvsym failed` line in NCCL debug logs. The DGX Spark documentation should state plainly: GDR is currently unavailable on stock DGX OS, multi-node collectives are host-staged, the expected performance envelope, and a roadmap (or an explicit "not planned") for GDR support. While there, pointing operators at NCCL RAS for hang diagnosis would help enormously. This costs a documentation page and would have saved this community months of collective debugging.

### 6.5 Engage with vLLM #46097

This is the highest-quality reproduction of the bug class in existence: 4× Spark, six wedges in 36 hours, py-spy stacks for every rank in every wedge, eager mode, NCCL 2.30.7. It has zero NVIDIA response to date. Even a triage comment — confirming the GDR situation, pointing at RAS for diagnosis, stating whether the NCCL team sees anything actionable — would materially help. The same applies to NCCL #2213 (a proxy-path SIGSEGV on Spark with zero triage): it may be a real corruption bug on the code path every multi-node Spark cluster runs 100% of the time.

---

## 7. Closing

We want to be clear about attribution: the *trigger* (L1) is an inference-engine bug and we are pursuing it upstream with vLLM; the community has also built its own detection (RAS polling) and recovery (watchdog + `ncclCommAbort` + auto-restart) tooling, which we are publishing. But the reason this bug class is a daily operational reality on DGX Spark — and essentially unobservable on every other NVIDIA platform — is the stack Spark ships with: host-staged collectives (6.1), an unfixed allocator failure that mimics the hang (6.2), and no timeout at any layer to bound the damage (6.3). Each of those is in NVIDIA's hands, and each is independently worthwhile.

The DGX Spark is a genuinely compelling platform — 128 GB of unified memory per node at this price point is why this community exists and why we cluster these machines. We have done the work at our layer and documented it here so your engineers can start from evidence rather than symptoms. We would welcome direct engagement on any of the five items above.

---

## Appendix: Community Evidence Index

**vLLM / NCCL / SGLang issues (deadlock class):**
- vLLM #46097 — MiniMax-M2.7, 4× Spark, 6 eager-mode reproductions in 36 h with py-spy stacks proving one-rank desync at the MoE-output all-reduce; NCCL 2.30.7. *The canonical reproduction.*
- vLLM #41725 — MiniMax-M2.7, 2× Spark, same symptom; also captured a distinct single-GPU sm_12x CUDA event-sync hang (reproduces on one RTX 5090 with no NCCL) that shares the surface symptom — relevant when triaging reports.
- vLLM #46253 — cross-node CUDA graph capture failure on GB10 (proxy CUDA activity invalidates `capture_error_mode="global"`).
- vLLM #43547 / PR #44601 (unmerged) — zero-token/drain steps skip batch coordination (an L1 divergence mechanism).
- vLLM PR #45610 (unmerged) — cudagraph dispatch consensus guard, currently PP-only; a TP analog would remove one L1 mechanism outright.
- NCCL #2213 — SIGSEGV in `ncclLocalOpAppend()` after ~1.1M collectives on the CPU-staged (GDR 0) path, dual Spark. No NVIDIA triage.
- SGLang #27949, #29548 — same hang class on 4× and 2× Spark respectively (SGLang vendors the same distributed hot path).

**NVIDIA developer forum threads:**
- 369381 — DGX Spark hard hangs under vLLM load.
- 363989 — 0x51 UMA characterization (508 events over 6 days without lockup; contiguous-allocation failure with 118 GB free).
- 361643 — community frustration thread on multi-node vLLM/Spark; cited for prevalence, not analysis.
- 366127 — initially appeared to show vLLM *and* TRT-LLM both deadlocking on the same cluster; **ultimately resolved as launch-environment misconfiguration** (the same hardware then ran vLLM TP=2 with full CUDA graphs at 23 tok/s). We cite it for completeness and as a caution: not every Spark hang report is this bug — which is precisely why the RAS `MISMATCH` classifier matters.
- 376831 — 8× GB10 GLM-5.2 recipe (patched vLLM, 33–54 tok/s): evidence the platform performs well multi-node when configured carefully.

**Source-level findings (our verification, NCCL v2.30.4):**
- Unbounded device spin loops: `primitives.h:116`, `prims_ll.h:65`; `abortFlag` poll every 10k spins: `primitives.h:143–153`; no collective timeout anywhere in source or docs.
- RAS on by default since 2.24, localhost:28028, per-rank `seqNumber` counts with `MISMATCH` grouping (`ras/collectives.cc`).
- NCCL 2.30.7 / PR #1834: proxy TCP-accept hang fix (the accept path with no timeout is visible in 2.30.4 `proxy.cc:1716–1731`).
- Stock Spark NCCL init log: `dlvsym failed on mlx5dv_reg_dmabuf_mr` → `GDR 0`.
- Local kernel log: `NV_ERR_NO_MEMORY (0x51)` from `_memdescAllocInternal @ mem_desc.c:1359` on driver 580.159.03.
