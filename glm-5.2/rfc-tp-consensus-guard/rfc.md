# [RFC] TP-wide step-consensus guard with NCCL RAS integration — the TP analog of DP `_post_process_cudagraph_mode` min-reduce and PP #45610

**Status:** Proposal, field-validated (observe mode + two live-fire recoveries) on 4× DGX Spark GB10, TP=4, GLM-5.2
**Author:** DGX Spark GLM-5.2 stability effort (`marksunner/dgx-spark-guides`)
**vLLM version validated against:** `0.23.1rc1.dev893+gd3a66aa7e` (v1 engine, `mp` executor, `--nnodes 4`)
**Related:** #45610 (PP cudagraph-mode consensus, unmerged), #46097 (TP eager-mode deadlock, 6 captures), #43547 (zero-token/drain step divergence), #44954 (MTP drafter-skip divergence), #41725 (sm_12x event-sync hang), #46253 (graph-capture invalidation on host-staged transport), NCCL #2213

---

## 1. Summary

vLLM has **no cross-rank consensus, watchdog, or timeout on the TP dimension.** DP min-reduces the cudagraph-mode decision across DP ranks (`_post_process_cudagraph_mode`); PP grew an analogous consensus guard in #45610. TP has neither. When TP ranks make rank-local, data/timing-dependent decisions that change which collectives they issue (cudagraph dispatch, MTP drafter-skip near length bounds, zero-token/drain steps, mid-generation aborts), one rank can issue or skip a collective the others do not — and because NCCL device kernels spin with **no timeout of any kind** (only the abortFlag breaks the loop, and nothing in vLLM sets it), the result is a permanent wedge at ~96% SM occupancy. vLLM additionally pops `NCCL_ASYNC_ERROR_HANDLING` and runs its hot path on pynccl rather than ProcessGroupNCCL, so no watchdog exists at **any** layer today.

This RFC proposes a **TP-wide consensus guard** for the v1 worker, combining three graph-transparent detectors behind one env-gated module, with an observe → enforce rollout:

1. **A per-step counter** gathered across TP ranks over a dedicated CPU (gloo) side-channel — cheap lockstep telemetry plus dead-peer detection.
2. **An NCCL RAS probe** (rank 0, raw socket to the in-process RAS agent on `localhost:28028`, on by default since NCCL 2.24) — per-rank *launched-collective* counts for every communicator, which detects the desync at **collective granularity**, works under CUDA-graph replay, and names the ahead/behind ranks natively.
3. **A stuck-timer** on hot-path worker RPCs — the catcher for the second wedge signature (all ranks frozen *equal*), where neither counter diverges.

On sustained trigger, enforce mode converts the permanent hang into a detected failure and a clean restart via the engine's existing dead-worker machinery. We deployed exactly this as a three-layer bind-mount prototype on a 4-node GB10 cluster and captured **two complete live deadlocks with full telemetry on 2026-07-15** — including one of each signature — plus a multi-hour healthy-baseline soak. All numbers below are from those captures.

This is deliberately the minimal, mode-agnostic cousin of #45610: it does not require enumerating every divergence source (that is the long-term fix); it provides the missing floor — *detect the desync generically and convert a permanent hang into a recoverable error* — while per-decision consensus guards land one by one.

## 2. Problem statement

**What happens.** Multi-node TP serving wedges permanently: all TP ranks spin at ~96% SM, ~15 W, zero forward progress, forever. No error, no timeout, no log line. The engine's request queue backs up; every request times out; operator intervention (container kill + full relaunch, ~22 min on our cluster) is the only recovery. Time-to-wedge under agentic traffic (OpenCode driving multi-turn tool-use with streaming and mid-generation aborts): **~4 minutes to ~2 hours** depending on cudagraph mode. One live capture: cluster served correctly from 05:47 to 06:32 UTC, wedged 4 minutes after an OpenCode session started.

**Which hardware.** Reproduced continuously on 4× NVIDIA DGX Spark (GB10, sm_121, one GPU per node, ConnectX over a 10 GbE/QSFP switch), GLM-5.2 TP=4, vLLM v1 `mp` backend, NCCL 2.30.4. The same class is reported on GB10 by multiple independent users (#46097 with six eager-mode py-spy captures; SGLang #27949/#29548 — same vendored pynccl stack). The **root defect is not platform-specific** (§3): GB10 is an amplifier that makes a generic race easy to hit, which also makes it the ideal reproduction platform for a generic fix.

**Why it matters.** A permanent, silent, undetectable-from-inside hang is the worst failure mode a serving engine can have. Every other failure in vLLM converges to `EngineDeadError` and restartability; this one converges to a human noticing the API stopped answering. SGLang ships a scheduler watchdog precisely for this ("server will crash to prevent hanging", `watchdog_timeout=300`); vLLM has nothing.

## 3. Root cause: four stacked layers

Full analysis with source-level citations in the companion write-up (`analysis-phase4-kernel.md`, publishable on request); summary:

- **L1 — engine-level TP control-flow divergence (the bug).** A fastest-rank race in the forward pass: one rank takes a different path through data/timing-dependent control flow and issues/skips a collective. #46097's six eager-mode captures localize it: three ranks blocked entering the MoE-output TP all-reduce, exactly one rank already launching the *next* layer's `qkv_proj` — one sub-block ahead, past the only collective between the two points, and the ahead rank is always a *fast* (unloaded) one. Known unguarded rank-local decisions feeding this class: per-rank cudagraph dispatch (no TP analog of the DP min-reduce), MTP drafter-skip near length bounds (#44954), zero-token/drain steps (#43547, fix PR #44601 unmerged), mid-generation aborts, JIT/warmup phase drift (#46253). Eager mode reproducing it proves cudagraphs are an amplifier, not the cause.
- **L2 — GB10 amplifies it ~10–100×.** Stock DGX OS ships inbox rdma-core without `mlx5dv_reg_dmabuf_mr`, so NCCL logs `GDR 0` and **every cross-node collective is host-staged**: CPU proxy threads, bounce buffers, continuous driver calls per chunk, on a 20-core Grace CPU that is simultaneously running the engine. Host scheduling jitter directly modulates collective timing, and with one GPU per node every TP collective is cross-node (~156 all-reduces per token for GLM-5.2's 78 MoE layers). A100/H100 clusters run GDR and TP-over-NVLink and never exercise this path at this intensity.
- **L3 — co-traveling platform bugs wear the same disguise.** The GB10 UMA 0x51 memdesc leak (confirmed in our kernel logs; no fixed driver exists) and an sm_12x CUDA-13 event-sync hang in the async sampler (#41725 — reproduces on a single RTX 5090 with no NCCL) both produce the same all-ranks-frozen symptom. Any detector must **classify** before it acts — this drove the two-signature design in §4.
- **L4 — NCCL provides no floor.** Verified in the 2.30.4 source: device collective kernels wait in unbounded spin loops (`primitives.h`), checking only `abortFlag` every ~10k spins; "watchdog" appears zero times in NCCL's docs; the only NCCL timeouts are transport-level and never fire on a peer that simply never issues the matching collective. vLLM's `gpu_worker.py` actively pops `NCCL_ASYNC_ERROR_HANDLING`, and `TORCH_NCCL_*` watchdogs do not govern pynccl. A wedge is permanent *by construction*.

L1 is vLLM's to fix (this RFC is the floor; per-decision guards are the cure). L2/L3 are NVIDIA's. L4 is upstream-NCCL's. The guard proposed here is the only layer that is buildable entirely inside vLLM today and protects against all of them.

## 4. The two deadlock signatures (both captured live, 2026-07-15)

The single most operationally important finding: **there are two distinct wedge signatures, and they need different detectors.** NCCL's RAS subsystem (§5.2) distinguishes them in one query.

### Type A — collective desync (the L1 race)

Ranks have launched **different numbers of collectives** and all wait forever on the mismatched one. RAS shows a frozen `MISMATCH`; the ahead rank is a fast rank. Captured at 06:36:24 UTC (wedge onset 06:32:39, OpenCode traffic, ~45 min after a clean relaunch):

```
#0-0 (a3f5545bf4e7cf8c) MISMATCH            <- the TP all-reduce communicator
  Communicator ranks have different AllReduce operation counts
  Rank 1 has launched up to operation 27911 -- GPU 0, process 285, node 10.0.0.8
  Rank 2 has launched up to operation 27890 -- GPU 0, process 284, node 10.0.0.1
  Rank 3 has launched up to operation 27872 -- GPU 0, process 287, node 10.0.0.5
  Rank 0 has launched up to operation 27860 -- GPU 0, process 341, node 10.0.0.7
```

A **51-operation spread** (rank 1 ahead of rank 0), frozen across successive queries. Meanwhile the guard's per-step counters on the same ranks were **frozen equal**: the divergence lives *inside* one model step (~156 collectives deep), invisible at step granularity. This is the empirical result that forced the RAS integration (§5.2): a step counter alone cannot detect Type A before the wedge; per-collective Python counting can't either (pynccl wrappers only execute at graph capture, not replay). NCCL's own launched-op bookkeeping is the only collective-granularity counter that keeps counting under graph replay — and it's already running.

### Type B — equal-freeze (event-sync hang / same-collective block)

All ranks have launched the **same** number of collectives and all freeze: RAS counts frozen **equal**, step counters frozen **equal**, hot RPC in flight on every rank. No mismatch anywhere — consistent with all ranks stuck inside the same collective, or with the engine-level sm_12x event-sync hang (#41725) that involves no NCCL at all. Captured at 08:05–08:08 UTC (wedge during post-relaunch warmup):

```
08:04:56  steps [1, 1, 1, 1]     inflight [1,1,1,1]  ras_spread 0
08:05:56  steps [7, 7, 7, 7]     inflight [1,1,1,1]  ras_spread 0   <- frozen from here
08:06:56  steps [7, 7, 7, 7]     inflight [1,1,1,1]  ras_spread 0
08:07:56  steps [7, 7, 7, 7]     inflight [1,1,1,1]  ras_spread 0
```

No divergence trigger can fire on this — correctly, because there is nothing mismatched. **The stuck-timer is the catcher for Type B** (a hot-path RPC in flight > `stuck_s`), and it fired: `execute_model stuck 186s` at 08:08:16.

The signatures demand the composite design: RAS frozen-mismatch catches Type A in ~10–15 s; the stuck-timer catches Type B in `stuck_s`; the step counter provides the lockstep baseline, the dead-peer detector, and the in-flight gate that keeps both triggers quiet during compile/idle.

## 5. Design

### 5.1 The per-step counter (graph-transparent lockstep telemetry)

One integer per rank, incremented in the worker busy loop's `finally` after each completed hot-path RPC (`execute_model`, `sample_tokens`, `take_draft_token_ids`, `execute_dummy_batch` — never `load_model`/`compile_or_warm_up_model`, which legitimately run 10+ min). TP ranks must complete steps in lockstep by construction, so `max(steps) − min(steps)` is a valid divergence measure in every execution mode, including full-graph replay. An `in_flight` flag (set at RPC begin, cleared at end) distinguishes "mid-step" from "idle" and gates the other triggers.

A daemon thread per worker `all_gather`s `(steps, in_flight, ras_verdict)` every `poll_s` (5 s) over a **dedicated** gloo group created once at worker init. Never reuse the metadata `cpu_group`: a side thread issuing collectives on a group the main thread also uses can interleave mismatched ops across ranks — manufacturing the exact desync we detect. All ranks create the group at the same `__init__` point, so it's a safe collective; on creation failure the guard logs and stays unarmed (never a half-state).

Empirical caveat, measured (§6.1): on a healthy cluster step divergence is 0 with brief off-by-one blips, and in a real wedge the counters freeze **equal or off-by-one** — so the step counter is the *substrate* (baseline, dead-peer, in-flight gate), not the primary wedge detector. Its `>1 for 2 polls` trigger stays as belt-and-braces for the one case the others are slow on: a rank looping ahead while a peer stalls.

### 5.2 The NCCL RAS probe (collective-granularity detection, for free)

NCCL ≥ 2.24 runs a RAS agent in every rank by default (`NCCL_RAS_ENABLE=1`), listening on `localhost:28028`. The agent aggregates over its own out-of-band socket mesh, so **one local query returns every rank's per-communicator launched-collective counts** — populated from NCCL's internal `seqNumber` bookkeeping, which advances under CUDA-graph replay. Rank 0 of the guard sends `set format json\nverbose status\n` over a raw socket each poll and reduces the response to a signature: `{comm_hash: (per-rank total launched-op counts)}` for communicators with no missing ranks and all counts nonzero (the nonzero filter drops the benign NOCOMM shape of single-node comms).

**Trigger = frozen ∧ mismatched ∧ in-flight, sustained:** some communicator's per-rank counts are *identical to the previous poll* (frozen) yet *unequal across ranks* (mismatched), for `ras_strikes` consecutive polls (default 2 → 10–15 s time-to-detect), while a hot RPC is in flight on some rank. Each conjunct kills a false-positive class: healthy launch-pipelining skew is mismatched but not frozen; idle/compile is frozen but not mismatched (and not in-flight); Type B is frozen but equal (correctly left to the stuck-timer). RAS query failures reset the strike count — fail-open, never trigger off stale data.

Rank 0 samples RAS *before* the gather so its verdict rides the same gather round to all ranks, and every rank applies the same decision in the same round — required because recovery must be initiated on all ranks together.

### 5.3 The stuck-timer (Type B catcher)

Per-rank, pure-local: if a hot-path RPC has been in flight longer than `stuck_s` (default 180 s; nothing on the hot path legitimately runs 1/10th of that) for two consecutive polls, trigger. All ranks enter a wedged step within milliseconds of each other, so all ranks' timers fire within ~one poll — which matters for the recovery path below. In our prototype this detector is a separate module (layer D2, §7) for deployment-risk reasons; upstream they belong in one guard.

### 5.4 Recovery: what actually works on sm_121 (ncclCommAbort does not)

The by-the-book recovery is `ncclCommAbort` on every rank: device kernels poll `abortFlag` every ~10k spins, the wedged collective returns an error, the worker raises, `EngineDeadError`, restart. vLLM already binds `ncclCommAbort` (`pynccl_wrapper.py`) and `PyNcclCommunicator.destroy()` already aborts from a watchdog-style thread.

**Live-fire result: on GB10/sm_121 this does not unwedge the worker.** Two independent incidents, same outcome:

- 06:49:17 — stuck-timer fired, `aborting 2 pynccl comm(s)` (TP + world comms, correct handles, calls returned); worker still stuck; 06:50:07 — hard-exit fallback fired.
- 08:08:16 — same; RAS subsequently reported the communicator in `ABORT` state, i.e. the abort *registered* in NCCL's host-side state; worker still stuck; 08:09:06 — hard-exit fallback fired.

Working hypothesis: under graph-replayed collectives on this platform the abortFlag write does not reach the spinning device code (the flag is read via a host-memory mapping whose propagation under replay on sm_121 is evidently not guaranteed); possibly related to the host-staged (GDR 0) proxy path. We did not root-cause it inside NCCL — for this RFC the operational lesson suffices:

> **Recovery must not depend on ncclCommAbort succeeding.** The reliable sequence is: log + dump all-thread stacks (`faulthandler`) → attempt `ncclCommAbort` with a bounded join (it is correct where it works, and harmless where it doesn't) → if the rank is still stuck after `exit_after_abort_s` (30 s), **`os._exit(1)`**. A raw exit cannot be blocked by any wedged CUDA/NCCL state, takes the worker down, the executor's existing monitor sees the dead proc, and the normal dead-engine path runs. Both live incidents recovered exactly this way, end-to-end, unattended (§6.3). After the second incident we removed the futile 30–50 s abort detour from our deployment and exit directly; on platforms where abort is known-good the bounded attempt is still the right default.

Aborting/exiting on **all ranks together** (the shared verdict of §5.2, plus every rank's own stuck-timer firing within one poll) is what makes this clean: no half-dead communicator states, and the container-level supervisor sees one coherent death.

### 5.5 Modes and rollout posture

- **`off`** (default): the module is inert — no gloo group, no threads. Mounting/merging the code with no env set changes nothing. Because all ranks read the same launch env, the group is created on all ranks or none.
- **`observe`**: full detection, telemetry, and a `WOULD-ENFORCE` event whenever a trigger *would* fire — no intervention ever. This is how a fleet's real divergence distribution is characterized before enforcement is trusted; it is also the RFC's data source.
- **`enforce`**: detection + the §5.4 recovery.

Every threshold is an env var; the RAS probe can be disabled independently (`..._RAS=0`).

## 6. Live telemetry (the evidence, verbatim)

Deployment context: 4× DGX Spark GB10 (S7/S8/S1/S5 = TP ranks 0–3), GLM-5.2 TP=4, `mp --nnodes 4`, FULL_DECODE_ONLY cudagraphs, MTP K=4, NCCL 2.30.4, guard in **observe** mode, poll 5 s, telemetry heartbeat 60 s, one JSON line per rank-0 poll grepped from container logs.

### 6.1 Healthy baseline (the trigger is clean)

Armed on all 4 ranks 05:44:38. Under sustained synthetic load (2–3 concurrent 250–500-token completions every 3–5 min) plus interactive traffic, ~50 minutes and ~2,200 steps:

```
05:48:43  steps [288, 288, 288, 288]    divergence 0  step_rate_hz 4.67
06:11:46  steps [1813, 1813, 1812, 1812] divergence 1                  <- worst blip ever seen
06:12:46  steps [1904, 1904, 1904, 1904] divergence 0
06:16:47  steps [2181, 2181, 2181, 2181] divergence 0  max_divergence_seen 1
```

Across every healthy soak segment (multiple container generations, several hours total): **`max_divergence_seen` = 1, zero false `WOULD-ENFORCE` events, zero RAS strikes** (`ras_spread` 0 at every healthy heartbeat). The `>1 × 2 polls` and `frozen-mismatch × 2 polls` triggers never fired on benign traffic. Step rate tracked decode throughput (4–6 Hz busy, 0 idle), confirming the counter is live under graph replay.

### 6.2 Incident A — Type A collective desync (06:32–06:36 UTC)

OpenCode agentic session begins ~06:28. Last good telemetry 06:16 ([2181 ×4]). Wedge onset 06:32:39 (first client-visible failures). Then:

- **06:35:47** — stuck-timer fires on all ranks: `execute_model stuck 188s`; `aborting 2 pynccl comm(s)`.
- **06:36:24** — external RAS watchdog (host-side, §7 D1) captures the evidence bundle: the §4 Type A snapshot — TP comm `MISMATCH`, spread **51 ops** (27911/27890/27872/27860), frozen across queries; the guard's step counters frozen equal through the wedge.
- **06:36:29** — guard logs `WOULD-ENFORCE — gloo gather failed 3x (peer dead or wedged)`: dead-peer detection working as designed once workers began exiting.
- Auto-restart; serving again after the normal ~22 min load+warmup.

At this point the guard was step-counter-only; the incident is what proved step granularity blind to Type A (51 collectives of divergence inside one frozen step) and drove the §5.2 RAS integration, deployed the same morning. The Type A signature (frozen mismatch, in-flight) is exactly what the RAS trigger keys on; against this snapshot it fires in 2 polls ≈ 10–15 s vs the stuck-timer's 180 s.

### 6.3 Incident B — Type B equal-freeze, RAS-upgraded guard live (08:05–08:09 UTC)

First live incident with the RAS probe deployed (per-poll `ras_spread`/`ras_strikes`/`ras_fails` in telemetry). Wedge onset ~08:05:10, during warmup traffic:

```
08:05:56  steps [7,7,7,7]  inflight [1,1,1,1]  ras_spread 0   <- Type B: everything frozen EQUAL
08:07:56  steps [7,7,7,7]  inflight [1,1,1,1]  ras_spread 0
08:08:16  D2 stuck-timer: "execute_model stuck 186s" -> aborting 2 pynccl comm(s)
08:08:26  WOULD-ENFORCE: step divergence 2>1 for 2 polls, steps=[7,9,7,8]
08:08:31  WOULD-ENFORCE: RAS frozen collective-count MISMATCH x2 polls, inflight=[1,1,1,1]
08:08:57  heartbeat: ras_spread 87, ras_strikes 7, divergence 2, steps [7,9,7,8]
08:09:06  "still stuck 30s after abort -- os._exit(1)"      <- what actually recovered it
08:09:33  external watchdog captures evidence bundle; relaunch rc=0 08:09:47
```

Three findings in one capture:

1. **Type B is real and the stuck-timer is its only catcher.** For the entire pre-abort wedge, steps and RAS counts were frozen **equal** — both divergence triggers correctly silent, timer correctly fired.
2. **The RAS trigger works end-to-end at 5 s cadence, on live data, under graph replay.** The ncclCommAbort attempt tore the ranks out of their collective sequence asymmetrically, instantly manufacturing a genuine frozen 87-op MISMATCH (and a first-ever step divergence of 2, [7,9,7,8] — aborted RPCs raise and still advance the counter) — and the guard detected it within 2 polls and escalated strikes exactly per design. An artifact of the abort rather than an organic Type A onset, but a complete live-fire validation of the detection path: signature present → detected in 10–15 s → verdict distributed to all ranks over the gather.
3. **ncclCommAbort failed to unwedge; `os._exit(1)` recovered** (§5.4), second independent confirmation.

Both incidents: total wedge→serving-again time ≈ detection + ~22 min relaunch, zero human involvement. Before this system, each was an indefinite outage ended by a human noticing.

## 7. Prototype architecture: three independent layers (and what upstream should take)

Deployed as bind-mounts over a stock container — no image rebuild — which dictated a defense-in-depth split. Each layer is independently disableable and none depends on another's health:

| Layer | Where it runs | Detects | Acts | Status |
|---|---|---|---|---|
| **D1 — external RAS watchdog** | Head-node host shell (outside the failure domain) | RAS frozen counts / MISMATCH via `localhost:28028`, API health probe, container death | Captures evidence bundle (RAS json+text, all-node docker logs, per-node kernel logs), restarts cluster; cooldown + post-launch grace so it never races a relaunch or flaps | Proven: caught + recovered all live incidents |
| **D2 — worker stuck-timer** | In-worker daemon thread (bind-mounted module) | Hot RPC in flight > 180 s (Type B); gloo gather-fail ×3 (dead peer) | Stack dump → bounded ncclCommAbort → `os._exit(1)` (now direct exit, §5.4) | Proven live ×2 |
| **D3 — consensus guard** | In-worker daemon thread (bind-mounted module) | Step divergence >1 ×2 polls; RAS frozen-mismatch ×2 polls with RPC in flight (Type A); gather-fail (dead peer) | observe: telemetry + WOULD-ENFORCE; enforce: same recovery path (idempotent, shared trigger lock) | Observe-proven live; enforce enabled 08:50 UTC 2026-07-15 (first enforce live-fire pending) |

The layering is belt-and-braces for a production cluster; **upstream, D2+D3 collapse into one worker-side guard module** (three detectors, one recovery path, one env surface — §8), and D1's role is played by whatever supervises the engine (the existing dead-worker → `EngineDeadError` machinery in-process; k8s/systemd outside). The prototype's D1 remains the model for what *operators* should run externally, and its evidence-bundle-on-trigger behavior is worth copying into any enforcement path (the guard dumps `faulthandler` stacks and emits a structured event before exiting).

Why each exists / why complementary: D1 is immune to worker-side wedges by construction but slow (minutes) and can only restart everything; D2 is fast on Type B but blind to Type A before 180 s; D3 sees Type A in seconds and names the diverging ranks, but its divergence triggers are structurally silent on Type B. Together, every observed failure mode has a detector whose latency matches its blast radius.

## 8. Proposed upstream implementation

### 8.1 Placement and hooks

New module `vllm/v1/executor/tp_consensus_guard.py` (~400 lines, stdlib + torch.distributed only; the prototype `glm52_consensus_guard.py` is attached and is a working reference). Three touch-points in `v1/executor/multiproc_executor.py`, mirroring the prototype hooks:

```python
# WorkerProc.__init__, after distributed init completes:
tp_consensus_guard.start(self)          # creates the dedicated gloo group + poll thread; no-op when off

# worker_busy_loop, around the RPC dispatch:
tp_consensus_guard.note_begin(method)   # sets in_flight for hot methods
try:
    output = func(*args, **kwargs)
finally:
    tp_consensus_guard.note_step(method)  # advances step counter, clears in_flight
```

`note_step` counts attempts (in the `finally`): a raising hot RPC still advances the counter symmetrically on all ranks (all ranks raise on the same failed collective), so error paths don't manufacture false divergence. Zero hot-path GPU cost; the poll thread does one small CPU `all_gather_object` every 5 s and (rank 0) one localhost socket round-trip.

### 8.2 Env surface (all default-inert)

```
VLLM_TP_CONSENSUS_MODE            off | observe | enforce     (default off)
VLLM_TP_CONSENSUS_POLL_S          5
VLLM_TP_CONSENSUS_STUCK_S         180        # Type B stuck-timer
VLLM_TP_CONSENSUS_DIVERGENCE      1          # step trigger: strictly greater-than
VLLM_TP_CONSENSUS_STRIKES         2
VLLM_TP_CONSENSUS_GATHER_FAILS    3          # dead-peer trigger
VLLM_TP_CONSENSUS_RAS             1          # collective-granularity probe (Type A)
VLLM_TP_CONSENSUS_RAS_PORT        28028
VLLM_TP_CONSENSUS_RAS_STRIKES     2
VLLM_TP_CONSENSUS_EXIT_AFTER_ABORT_S  30    # bounded ncclCommAbort attempt before hard exit
VLLM_TP_CONSENSUS_TELEMETRY_EVERY_S   60
```

### 8.3 Telemetry contract

Rank 0 emits one greppable JSON line per heartbeat and an immediate event line on any trigger:

```
TP_CONSENSUS_TELEMETRY {"steps":[...],"divergence":0,"max_divergence_seen":1,"inflight":[...],
                        "step_rate_hz":4.67,"ras_spread":0,"ras_strikes":0,"ras_fails":0,
                        "strikes":0,"gather_fails":0,"mode":"observe","poll":N}
TP_CONSENSUS_TELEMETRY {"event":"would_enforce"|"enforce_abort","reason":"...","steps":[...],"rank":0}
```

Even in permanent-observe this pays for itself: it prints **which rank is ahead and by how much** — the diagnosis #46097's reporter reconstructs by hand-rolled py-spy across nodes — natively, per poll, in every deployment that enables it. It is also the telemetry substrate for prioritizing which per-decision consensus guards (§10) to build first.

### 8.4 Recovery integration

Enforce-mode trigger → structured event + `faulthandler` dump → bounded abort attempt → `os._exit(1)` (§5.4). The dead worker proc is already handled: the executor's monitor detects it, the engine reaches `EngineDeadError`, and the existing restart/serving-layer machinery proceeds — the guard adds no new recovery machinery, only the conversion of "hangs forever" into "dies detectably". A dedicated death-reason string (`consensus-guard: <trigger>`) in the worker's last log line keeps operator forensics one grep away.

### 8.5 Compatibility

- **DP/PP guards:** orthogonal and composable — DP min-reduce prevents one divergence source pre-hoc; #45610 does PP; this is the generic TP detection/recovery floor beneath both.
- **RAS availability:** on by default NCCL ≥ 2.24; probe degrades to disabled (with one log line) on connect failure, older NCCL, or non-NCCL backends. Multiple vLLM instances per node can collide on port 28028 — the probe validates that reported comm world-sizes match its own world before trusting a signature, and `..._RAS_PORT` is settable.
- **Async scheduling:** with async scheduling the busy-loop counter still advances per RPC but the GPU wait moves; wrap `AsyncModelRunnerOutput.get_output()` equivalently (same pattern, one more hook). Flagged as a review point rather than solved — our validation ran sync.
- **Non-CUDA platforms:** the step/stuck/gather detectors are backend-agnostic; only the RAS probe is NCCL-specific.

## 9. Test and reproduction procedure

**Reproducing the deadlock** (GB10 is the reliable testbed; any multi-node TP setup with host-staged NCCL should amplify similarly):

1. 4 nodes × 1 GPU, TP=4, `mp` backend `--nnodes 4`, a large MoE model (GLM-5.2 shown; #46097 used MiniMax-M2), cudagraphs FULL_DECODE_ONLY, MTP/spec-decode on.
2. Drive it with an **agentic coding client** (OpenCode against the OpenAI endpoint) running multi-turn tool-use tasks: streaming + long prompts + frequent mid-generation aborts. This is the empirically potent trigger — plain concurrent completions ran for hours; OpenCode wedged the cluster in ~4 minutes on first contact and reproduced twice more the same day.
3. Signature check at wedge: all ranks ~96% SM / ~15 W, then `echo verbose status | nc localhost 28028` (or `-f json`): frozen MISMATCH = Type A; frozen-equal counts with a hot RPC in flight = Type B. `coredumpctl list` empty and all worker PIDs alive distinguishes both from crash-class bugs (NCCL #2213).

**Validating the guard without a real wedge** — divergence injection: `SIGSTOP` one worker mid-decode. Expect: step divergence climbs monotonically as three ranks step ahead, `WOULD-ENFORCE (step divergence)` after 2 polls, gloo gather-fail dead-peer trigger if held past the gather timeout; `SIGCONT` before `stuck_s` and everything returns to baseline with no action taken (observe) — harmless on a live cluster. Then the acceptance gates we used: (a) ≥24 h observe soak under real traffic with zero false WOULD-ENFORCE, `max_divergence_seen` ≤ 1, `ras_strikes` 0 throughout; (b) injected divergence detected < 15 s; (c) only then flip enforce, re-inject, confirm all-rank exit + clean engine death + restart.

## 10. Alternatives considered

- **Per-decision TP min-reduce guards (the "correct" fix).** Right long-term: min-reduce/agree the cudagraph dispatch decision, the MTP drafter-skip, the drain-step participation — each divergence source individually, as DP already does for its one decision. Requires enumerating every rank-local decision and paying a cross-rank agreement round per decision per step on TP's latency-critical path. This RFC is the generic floor that holds while those land, the telemetry that says which to build first, and the backstop for the sources nobody has found yet. They compose.
- **Kernel-side NCCL timeout** (spin-count ceiling + abort in `checkAbort()`): would give NCCL the floor it lacks; needs a custom NCCL build and NVIDIA buy-in, and our sm_121 abortFlag findings (§5.4) suggest the flag path itself is not wholly trustworthy on this platform. Out of vLLM's hands; worth pursuing with NVIDIA in parallel.
- **Torch ProcessGroupNCCL watchdog (`TORCH_NCCL_*`)**: inapplicable — vLLM's hot path is pynccl, and vLLM pops `NCCL_ASYNC_ERROR_HANDLING`.
- **External-only watchdog** (our D1 alone): works, proven, and any serious operator should run one — but it detects in minutes not seconds, can only restart the world, cannot classify without RAS access, and belongs to deployments, not the engine. The in-engine guard is what makes every vLLM deployment safe by default.
- **SGLang-style fixed scheduler watchdog** (`watchdog_timeout=300`, crash on stall): strictly a subset of this proposal (≈ the stuck-timer alone, without classification, divergence telemetry, dead-peer detection, or the all-ranks-together exit).

## 11. Rollout / safety

- Defaults **off** → merged code changes nothing for anyone. `observe` is side-effect-free by construction (its only writes are log lines). `enforce` is opt-in after an observe soak on the operator's own fleet — the observe→enforce ladder is the rollout mechanism, validated end-to-end here.
- Failure containment: gloo-group creation failure → unarmed + logged, engine unaffected. RAS probe failure → probe disabled/strikes reset, other detectors unaffected. Guard thread death → engine unaffected (daemon thread, no locks shared with the hot path). The guard cannot block the busy loop: `note_begin`/`note_step` are two assignments and a set-membership test.
- Cost: one extra `new_group` at init (same hazard class as any init collective; all-or-none across ranks via the shared env), one 5 s CPU gather, one rank-0 localhost socket round-trip per poll. Zero GPU-side cost, zero hot-path syscalls.
- Escape hatches at every level: mode env kills everything; each detector has its own toggle; every threshold tunable without code changes.

## 12. Open questions for maintainers

1. Death-reason plumbing: is a structured "consensus-abort" reason on the dead-worker path (vs today's generic dead-engine) worth a small API addition for operator telemetry? We think yes — flagged, not assumed.
2. Async scheduling hook (§8.5): confirm `AsyncModelRunnerOutput.get_output()` is the right second anchor.
3. Should the RAS probe be a vLLM-blessed utility (`vllm.utils.nccl_ras`) usable by diagnostics beyond this guard? The `nc localhost 28028` technique is currently unknown in every related issue thread (#46097's reporter is reconstructing exactly this data by hand); first-classing it may be the highest-leverage 100 lines in this proposal.
4. sm_121 abortFlag propagation under graph replay: we can supply the full capture bundles (RAS snapshots, stack dumps, timelines) to whoever picks this up with NVIDIA. Should vLLM's own `PyNcclCommunicator.destroy()` grow the same bounded-abort-then-exit hardening?

## Appendix A — artifact inventory (available on request / for the PR)

- `glm52_consensus_guard.py` — step-counter guard (deployed 05:44 UTC 2026-07-15, observe).
- `glm52_consensus_guard_ras.py` + `consensus_guard_ras.diff` — the RAS-integrated guard (deployed same day; the §6.3 capture is this code live).
- `glm52_watchdog.py` — the stuck-timer layer, including the post-incident direct-exit revision (deployed 08:50 UTC; both prior incidents recovered via its `os._exit` fallback, validating the exit path before it became primary).
- Executor hook diffs (3 hunks), launch-script integration, external RAS watchdog, soak/loadgen harness.
- Three complete incident evidence bundles (2026-07-15 06:36 / 07:21 / 08:09 UTC): RAS text+JSON snapshots, all-rank container logs, per-node kernel logs, reason records.
- Companion root-cause analysis with source-level citations (NCCL 2.30.4, vLLM v1) backing §3.

## Appendix B — raw telemetry index

| Time (UTC, 2026-07-15) | Event |
|---|---|
| 05:44:38 | Guard + watchdog ARMED, 4/4 ranks, observe |
| 05:48:43 | Healthy: steps [288×4], 4.67 Hz, divergence 0 |
| 06:11:46 | Healthy worst-case blip: [1813,1813,1812,1812], divergence 1 |
| 06:32:39 | **Incident A onset** (OpenCode traffic) |
| 06:35:47 | Stuck-timer: `execute_model stuck 188s`; abort 2 comms |
| 06:36:24 | RAS: TP comm MISMATCH, **51-op spread**, steps frozen equal |
| 06:49:17→06:50:07 | Incident A′ (warmup re-wedge): abort ineffective → **os._exit recovered** |
| 08:05:56 | **Incident B onset**: steps [7×4] frozen, ras_spread 0 — Type B |
| 08:08:16 | Stuck-timer fires (186 s); abort 2 comms |
| 08:08:26–31 | WOULD-ENFORCE: step divergence 2 ([7,9,7,8]); **RAS frozen MISMATCH ×2 polls** |
| 08:08:57 | ras_spread **87**, ras_strikes 7 |
| 08:09:06 | Abort ineffective → **os._exit recovered** |
| 08:09:47 | External watchdog relaunch rc=0 |
