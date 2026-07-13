# So You've Unboxed Four DGX Sparks

## Running GLM 5.2 from Virgin Hardware to First Token — the Complete, Honest Guide

*Four NVIDIA DGX Sparks. One 671-billion-parameter reasoning model. Three days, five build attempts, four genuine bugs, one machine that ate itself under swap pressure, and a 05:12 BST moment when it finally spoke. This is the guide we wish we'd had when the boxes were still full of foam.*

---

## Before Anything Else: Whose Shoulders We're Standing On

This guide is built directly on **[tonyd2wild's QuantTrio recipe](https://github.com/tonyd2wild/GLM-5.2-QuantTrio-200K-4x-DGX-Spark)** — a meticulous, battle-tested recipe for serving the unpruned QuantTrio GLM-5.2 checkpoint across four DGX Sparks at 200K context. Every build command, every serve flag, every memory number in the launch configuration below comes from that recipe unless we say otherwise. Where we quote something directly from it, you'll see an inline note like *(from the QuantTrio recipe)*.

Tony's recipe in turn stands on the work of a whole community of DGX Spark pioneers, and his README credits them properly — so we won't duplicate that chain here. Read it; if you use this guide, their work is what you are using. The full roll of honour also appears in [Credits](#credits--acknowledgements) at the end of this document.

**What this document adds** to the recipe is the part that comes *before* it — taking four Sparks from shrink-wrap to a working, correctly-networked cluster — plus the real debugging journey we went through on top of it, including bugs that required original fixes. Tony's recipe assumes provisioned hardware with a working RoCE fabric. We had cardboard boxes.

---

## Introduction: Now What?

So you've unboxed everything, and you're staring at this gleaming hardware — now what?

Here's what you're holding: four NVIDIA DGX Sparks, each a GB10 Grace Blackwell system with 128 GB of unified memory. Individually, each one is a very capable desktop AI machine. Together, connected over a 200 Gbps fabric, they can serve **GLM 5.2** — Z.ai's flagship reasoning model — with **all 671 billion parameters present**: every layer, all 256 experts, nothing pruned away, 200,000 tokens of context, and speculative decoding making it feel genuinely responsive.

### "All the Parameters" Is Not "Full Precision" — an Honest Word About Quantisation

Before we go any further, one thing deserves to be said plainly, because it's the difference between you feeling informed and you feeling misled three weeks from now: **the version of GLM 5.2 we run is quantised.** The parameter count above is real. The precision is not the original.

Here's what that means. As Z.ai trains it, GLM 5.2 stores each of its 671 billion parameters as a 16-bit number. That full-precision model weighs well over a terabyte — more than double what four Sparks hold between them. To fit a model this size onto hardware this size, *something* has to give, and there are exactly two things you can give up:

1. **Some of the parameters** — *pruning*. Techniques like REAP analyse which of the model's 256 experts fire least often and delete them outright. The model gets smaller, but knowledge is genuinely removed — and it tends to show up later as odd, unpredictable blind spots.
2. **Some of the precision** — *quantisation*. Keep every single parameter, but store each one with fewer bits. The checkpoint we use, QuantTrio's **Int4-Int8Mix**, stores most weights as 4-bit integers and the more sensitive ones as 8-bit, bringing the model from over a terabyte down to ~405 GB.

Why does rounding every weight down to 4 bits not wreck the model? The intuition is worth having: a language model's knowledge isn't stored in any individual weight — it's smeared across billions of them, and it's the *aggregate* that carries the signal. Rounding each one is like saving a photograph as a high-quality JPEG instead of the camera's RAW file: information is genuinely discarded, but it's the information your eye was least using, and you have to go looking to notice. Careful quantisation schemes push this further by keeping the most sensitive parts of the network at higher precision — that's the "Int8Mix" half of the name.

So set your expectations honestly: a well-made 4-bit quant measures slightly below the full-precision original on benchmarks, and on the most exacting tasks you might occasionally feel it. What it is *not* is a different, smaller model wearing GLM 5.2's badge. Every expert is present, the full 200K context works, the reasoning traces are complete. When this guide says "the full model," it means **the full architecture at reduced precision** — which is both the largest-brained version of GLM 5.2 that four Sparks can physically hold, and a far better trade than surgically removing parts of the brain to make it fit.

The parameter count in the headline is real. The precision is the concession. Now you know exactly what you're getting.

### What the Journey Looks Like

The recipe we followed targets **28.8 tok/s single-stream at 200K context with MTP k=4 speculative decoding** *(from the QuantTrio recipe)*, and it delivers. But between "boxes on the floor" and "curl returns a reasoning trace" there is a long path: out-of-box setup with a gotcha that can brick your first hour, cluster networking with three-letter acronyms stacked four deep, a 405 GB download that will test your monitoring habits (and, depending on your connection, your patience), a custom container build, and — inevitably — bugs.

We're writing down all of it: what to run, why it works, what broke, how we diagnosed it, and what we learned. The bugs are the most valuable part of this document. Everything else is a recipe; the bugs are the education.

**Who this is for:** someone with four DGX Sparks and enthusiasm. We do not assume you've run a GPU cluster before. Where a concept matters — NCCL, tensor parallelism, RoCE, MLA, MTP — we stop and explain it, because knowing *why* you're typing a command is what saves you when it fails.

---

## What We Built

| Component | Details |
|---|---|
| **Compute** | 4 × NVIDIA DGX Spark (GB10 Grace Blackwell, 128 GB unified memory each) |
| **Cluster fabric** | MikroTik CRS812 switch, QSFP-DD at 200 Gbps, MTU 9000, RoCE v2 — configuration walkthrough in the companion guide, [MikroTik CRS812 RoCE Switch Setup](what-is-fabric.md) |
| **Management network** | Ordinary Ethernet LAN (ours happens to be 10G; 1G is perfectly fine — see [Know Your Ports](#know-your-ports)) |
| **Model** | [QuantTrio/GLM-5.2-Int4-Int8Mix](https://huggingface.co/QuantTrio) — ~405 GB, all 256 experts, unpruned |
| **Serving stack** | Custom-built vLLM image (eugr's build harness + CosmicRaisins' Triton kernels + community mods), vLLM native multi-node, no Ray |
| **Result** | GLM 5.2 on an OpenAI-compatible API at 200K context with MTP speculative decoding |

### Naming and Placeholder Conventions

We refer to our four nodes as **S1, S2, S3, and S4**. S1 is the **head node** — the machine that runs the API endpoint and from which we orchestrate everything. The other three are workers.

Because this guide is public, all IP addresses are placeholders. Substitute your own values:

| Placeholder | Meaning |
|---|---|
| `${QSFP_1}` … `${QSFP_4}` | Each node's static IP on the QSFP cluster fabric |
| `${HEAD_NODE_IP}` | The head node's fabric IP (same as `${QSFP_1}` for us) |
| `${SPARK_N_IP}` | A node's IP on the management LAN |

Pick any private subnet for the fabric (the QuantTrio recipe uses a `/24`) and make the last octet the node number — `10.0.0.1` for S1 through `10.0.0.4` for S4, say. That way an IP always tells you which machine you're talking to, which is a small kindness to your future 2 a.m. self. We use the generic username `sparkuser` throughout. **Use the same username on every node** — you'll see later (in the kernel-distribution section) why mismatched usernames and home directories cause real pain.

A handy pattern: put your real values in a `cluster.env` file (mode 600, never committed) and `source` it before running anything from this guide:

```bash
# cluster.env — your real values, kept out of version control
export QSFP_1=...
export QSFP_2=...
export QSFP_3=...
export QSFP_4=...
export HEAD_NODE_IP=${QSFP_1}
export WORKERS="${QSFP_2} ${QSFP_3} ${QSFP_4}"
export ALL_NODES="${QSFP_1} ${QSFP_2} ${QSFP_3} ${QSFP_4}"
```

---

## A Ten-Minute Primer: The Concepts You'll Need

You can skip this if you've run distributed inference before. If you haven't, ten minutes here will make everything downstream make sense.

**Why four machines at all?** GLM 5.2's weights, in the quantisation we're using, are ~405 GB. One Spark has 128 GB of unified memory. The model physically cannot fit on one node — so it must be *split* across four, and the four must act as one GPU. Everything in this guide serves that goal.

**Tensor parallelism (TP).** The way we split the model. With TP=4, every layer of the model is sliced four ways: each node holds a quarter of each weight matrix and computes a quarter of each matrix multiplication. The catch: after almost every layer, the four partial results must be recombined (an "all-reduce"), which means the nodes are chattering constantly — thousands of small synchronizations per generated token. This is why the network between them matters so much. 405 GB ÷ 4 ≈ 95 GiB of weights per node *(the recipe measured 98.07 GiB; we measured 97.95 GiB)*, leaving genuine headroom on each 128 GB Spark for the KV cache, CUDA graphs, and the OS.

**Mixture of Experts (MoE).** GLM 5.2 has 671B total parameters, but it's a mixture-of-experts model with 256 expert sub-networks, of which only a fraction fire per token (~37B active parameters per forward pass). You still need all 256 experts *resident in memory* — you never know which ones the next token will want — but per-token compute stays manageable. Some community checkpoints prune "less used" experts to shrink the model (the REAP technique from the introduction); the checkpoint we use keeps **all 256 experts intact**, which is the whole point of the QuantTrio recipe: every parameter present, not a surgically reduced model.

**Quantisation (Int4-Int8Mix).** Covered properly in the introduction; the one-line recap: weights are stored as a mix of 4-bit and 8-bit integers instead of 16-bit floats, cutting the model from well over a terabyte to ~405 GB with modest quality loss and nothing pruned. This is what makes four 128 GB nodes sufficient.

**MLA and `fp8_ds_mla`.** During generation, a model caches attention keys and values for every token of context — the **KV cache** — so it doesn't recompute them each step. At 200,000 tokens of context this cache gets enormous. GLM 5.2 uses **Multi-head Latent Attention (MLA)**: instead of caching full keys and values, it caches a small *compressed latent* per token and reconstructs what it needs on the fly. `fp8_ds_mla` is the storage format that additionally squeezes that latent into 8-bit floats. The combined effect: 200K tokens of context fits in roughly 10.5 GB per node *(from the QuantTrio recipe)*. Remember this format's name — it stars in the best bug of the whole deployment.

**MTP speculative decoding.** A big model produces one token per (expensive, four-node-synchronized) forward pass. **Multi-Token Prediction (MTP)** speeds this up: a tiny "drafter" head predicts the next few tokens cheaply, and the big model then *verifies* all of them in a single forward pass. Accepted tokens are free speedup; rejected ones cost nothing beyond the verify pass you were doing anyway. We run **k=4** (draft four tokens per step) and typically see 3.2–3.6 of them accepted *(recipe benchmark numbers)* — meaning each expensive cluster-wide step yields ~3+ tokens instead of 1. A lovely detail of this checkpoint: the MTP drafter is *inside* the checkpoint (layer 78) — there's no separate draft model to download or version-match *(from the QuantTrio recipe)*.

**NCCL, RDMA, and RoCE.** **NCCL** (NVIDIA Collective Communications Library, "nickel") is the library that performs those all-reduce synchronizations between GPUs. Within one machine it uses fast local paths; between machines it wants **RDMA** (Remote Direct Memory Access) — network transfers that go NIC-to-memory without the CPU touching each packet. Our Sparks' ConnectX-7 NICs do RDMA over Ethernet, called **RoCE** (RDMA over Converged Ethernet, v2). When RoCE works, cross-node all-reduce is nearly invisible. When it silently fails, NCCL falls back to plain TCP and your token rate roughly halves — a failure mode so sneaky the recipe calls it out explicitly, and we'll show you how to detect it.

**The GID index.** RoCE addresses endpoints by entries in a per-port **GID table** (roughly: the RDMA-layer address book, each entry binding a protocol version to an interface address). NCCL must be told *which entry* to use via `NCCL_IB_GID_INDEX`. The right index differs per node depending on configuration history — this is not hypothetical; our four supposedly identical nodes ended up with different values, and you'll see exactly how to discover yours.

**Unified memory on GB10.** On a normal PC, GPU VRAM and system RAM are separate. On a Spark, the CPU and GPU share one 128 GB pool. Wonderful for fitting big models; treacherous in one specific way: *file page cache competes with model memory*. Linux happily fills spare RAM with cached file data, and on a machine where "spare RAM" is also your GPU memory, a launch that worked yesterday can OOM today purely because of what the page cache looked like. The recipe's configuration has specific countermeasures for this, which we'll flag when we get there.

---

## Prerequisites

### Hardware

- **4 × NVIDIA DGX Spark** (GB10, 128 GB unified memory each)
- **A QSFP switch** for the cluster fabric — we use a MikroTik CRS812, which gives all four Sparks any-to-any 200 Gbps connectivity. Configuring it properly is its own small project, covered step by step in our companion guide, [MikroTik CRS812 RoCE Switch Setup](what-is-fabric.md). (A direct-cabled mesh can also work if NCCL can see the IB devices *(from the QuantTrio recipe)*, but a switch is far simpler to reason about — the switch guide explains the trade-off.)
- **QSFP-DD cables**, one per node
- **An ordinary Ethernet switch** for the management network. **1 Gigabit is plenty.** This network only carries SSH, apt traffic, monitoring, and your initial downloads; every heavy byte between nodes travels the QSFP fabric instead. Ours happens to be a 10G switch, which makes the one-time model download fan-out marginally comfier, but nothing in this guide requires it.
- **~420 GB free disk per node** — 405 GB of weights plus the container image, caches, and slack *(from the QuantTrio recipe)*
- A laptop or workstation to orchestrate from

### Accounts, Repos, and Downloads

- A **Hugging Face account** (and its CLI token) to download `QuantTrio/GLM-5.2-Int4-Int8Mix`
- **The recipe:** [tonyd2wild/GLM-5.2-QuantTrio-200K-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.2-QuantTrio-200K-4x-DGX-Spark) — mod scripts, the indexer patch, and `launch.sh` live here
- **The kernels:** [CosmicRaisins/glm-5.2-gb10](https://github.com/CosmicRaisins/glm-5.2-gb10) — the 10 Triton kernel files (Apache-2.0)
- **The build harness:** [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) — builds the vLLM container for GB10

### Time Budget

Most of the calendar time in this project is one download and one build — and the download depends entirely on *your* internet connection, so estimate it for yourself rather than taking our word for it. The number that matters is **405 GB of model weights**, downloaded once. The arithmetic:

> **hours ≈ 405,000 MB ÷ (your real download speed in MB/s) ÷ 3,600**

Use your *measured* speed (a quick speed test, converted from Mbps to MB/s by dividing by 8), not the number on your bill. Some worked examples:

| Your connection (real-world) | 405 GB takes about |
|---|---|
| ~1 Gbps (≈ 110 MB/s) | **1 hour** |
| ~100 Mbps (≈ 12 MB/s) | **9–10 hours** — start it before bed |
| ~20 Mbps (≈ 2.5 MB/s) | **~2 days** |

Our own connection sat at the painful end of that table — low single-digit MB/s — so the download dominated our timeline and we spent two days glancing at progress bars. That was *our* pipe, not a property of the project: most readers will land in the top two rows. Whatever your speed, the advice in Phase 5 stands — start the download early, run it under `tmux`, and read the stall-detection section, because a download that takes an hour can still hang at minute 40.

The rest of the budget is more predictable:

- Hardware setup + node provisioning: 4–6 hours
- Model download: *(your calculation above — runs unattended, mostly)*
- Image build + distribution + launch: 4–6 hours
- Debugging: variable. Budget a day and call it a gift if it's less.

With decent internet the whole thing is a comfortable weekend. Ours took three days, and now you know exactly which row of the table to blame.

---

## Phase 1: Hardware Setup

### Unboxing and First Boot — the Ethernet Gotcha

Each Spark walks you through an OOBE (Out-of-Box Experience) on first power-up: the machine broadcasts a temporary WiFi hotspot, you join it from your laptop, open the setup URL, create your user account, set your timezone, and it runs a firmware/software update.

> ⚠️ **Do NOT plug in the ethernet cable before completing the WiFi setup wizard.**
>
> If ethernet is connected when the OOBE reaches its update step, the Spark tries to fetch the update over ethernet — but the full networking stack isn't initialized yet at that point in the flow. The result is **"Updating Your DGX Spark" frozen at 0%, indefinitely.** You can wait an hour; it will still say 0%.

We learned this on our second Spark. The first we happened to do in the right order — WiFi setup fully finished, ethernet after. The second, we helpfully pre-plugged the ethernet cable, and the update sat at 0% until we accepted it was never going to move.

**Recovery, if you're already stuck:** power-cycle the machine *with ethernet unplugged*. If the OOBE had already created your user account before it hung (it does this before the update step), the Spark boots straight to a working OS and your OOBE credentials work. You've effectively skipped the update — catch up later with `sudo apt update && sudo apt upgrade -y` (you'll be doing this in Phase 4 anyway). Our recovery went cleanly, no reinstall needed.

So, for each of the four Sparks: power on, join the hotspot, complete the wizard over WiFi, create the `sparkuser` account, *then* connect ethernet.

### Know Your Ports

Each DGX Spark has:

- **One 10G-capable Ethernet port (RJ-45)** — your management network: SSH, apt, downloads. Despite the "10G" on the spec sheet, it negotiates happily down to 1 Gbps, and 1 Gbps is entirely sufficient here — nothing performance-critical ever crosses this port. Plug it into whatever Ethernet switch you already own.
- **A ConnectX-7 NIC with QSFP connectivity at 200 Gbps** — the cluster fabric. This is the NIC that speaks RoCE, and it's where all the inter-node model traffic flows.

Here's the detail that will matter enormously later: the ConnectX-7 exposes **two port interfaces**, which show up in Linux with names like `enP2p1s0f0np0` (port 0) and `enP2p1s0f1np1` (port 1). Which one is live **depends on which physical port your cable happens to be plugged into** — and on identical hardware, nodes in your cluster can end up on different ports. File this away; it becomes a whole section in Phase 8.

Cable everything up: all four Sparks to your management Ethernet switch (1G or 10G, both fine), and each Spark's QSFP port to the QSFP switch (fabric).

### The QSFP Switch

The fabric switch deserves its own walkthrough — because an unconfigured switch is the single most convincing impostor of a broken cluster. Links come up, pings pass, and RDMA still quietly fails or, worse, hangs mid-inference. The short version of what your switch must provide:

- **200 Gbps per fabric port**, actually negotiated (don't trust auto-negotiation — verify)
- **MTU 9000 (jumbo frames) on every fabric port** — on the *switch*, not just the NICs. A switch port left at MTU 1500 silently fragments jumbo traffic and craters fabric bandwidth *(from the QuantTrio recipe's troubleshooting)*
- Ideally, **PFC and ECN** — the "lossless Ethernet" features that RDMA was designed to lean on

The full step-by-step for our MikroTik CRS812 — what each feature actually does, the RouterOS commands, how to verify the fabric end-to-end, and the pitfalls that cost us dearly on a previous cluster — lives in our companion guide: **[MikroTik CRS812 RoCE Switch Setup](what-is-fabric.md)**. Nothing in it is GLM-specific, so it can serve any multi-Spark project you build on this fabric. If you're using a different switch, the principles (correct port speed, MTU 9000, PFC, ECN) apply universally — adapt the vendor-specific commands. If your switch is already configured, or you're running a two-node back-to-back setup with no switch at all, carry straight on to Phase 2.

---

## Phase 2: Node Provisioning

Repeat this on every Spark. It's ~20 minutes per node once you have the rhythm.

### SSH Keys

From your workstation, install your public key on each node so everything from here on is passwordless:

```bash
ssh-copy-id -i ~/.ssh/id_ed25519 sparkuser@${SPARK_N_IP}
```

### Change the Default Password

Immediately, before the machine spends any time on a shared network:

```bash
passwd
```

Store the new password in a password manager or a mode-600 secrets file. Not in a chat log, not in a README, not in this guide.

### Disable WiFi

The WiFi radio did its job during OOBE; in a cluster it's now a liability (surprise second IP addresses, boot delays, confusion):

```bash
sudo nmcli radio wifi off
sudo systemctl disable NetworkManager-wait-online
```

The second line matters more than it looks: `NetworkManager-wait-online` will otherwise stall every boot waiting for a WiFi interface that is never coming up again.

### Docker Group

DGX OS ships with Docker and the NVIDIA container toolkit pre-installed, but your user isn't in the `docker` group yet:

```bash
sudo usermod -aG docker $USER
```

Log out and back in (or `newgrp docker` for the current session), then verify that containers can see the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi
```

You should see the GB10 listed with its driver and CUDA versions. If this fails, stop and fix it now — everything downstream runs in containers.

### Node.js (Optional)

Not needed for serving GLM itself, but useful if you plan to point agent frameworks and tooling at your new endpoint:

```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo bash -
sudo apt-get install -y nodejs
```

---

## Phase 3: Cluster Networking

This is the phase where four computers become one cluster. It's also where the most *silent* failure modes live — things that don't error, they just make everything mysteriously slow. Take it methodically.

### Find Your QSFP Interface

On each node, identify which ConnectX-7 port interface is actually cabled:

```bash
ip link show | grep -E 'enP2p|enP7'
```

Look for the `enP2p1s0f*` interface showing `state UP` (or that comes up when you connect the cable). Note the exact name **per node** — as foreshadowed, it may be `...f0np0` on some nodes and `...f1np1` on others.

### Assign Static Fabric IPs

Each node gets a static IP on the fabric subnet. To test quickly (this does *not* survive reboot — persistence next):

```bash
# Example on S1 — adjust interface name and IP per node
sudo ip addr add ${QSFP_1}/24 dev enP2p1s0f1np1
sudo ip link set enP2p1s0f1np1 mtu 9000
sudo ip link set enP2p1s0f1np1 up
```

**MTU 9000 is mandatory, on both ends of every link.** Jumbo frames let the fabric move data in 9000-byte chunks instead of 1500-byte ones — at 200 Gbps, per-packet overhead is the difference between the NIC flying and the CPU drowning. A mismatch doesn't produce an error; it produces transfers that "work" at a fraction of the expected speed and RDMA connections that fail intermittently. NICs *and switch ports*, all 9000 — the [switch guide](what-is-fabric.md) covers the switch side.

### Make It Persistent — and Why That's Not Just Tidiness

The `ip addr add` approach evaporates on reboot. Persist it via NetworkManager:

```bash
sudo nmcli connection add \
  type ethernet \
  con-name cluster-link \
  ifname enP2p1s0f1np1 \
  ip4 ${QSFP_1}/24 \
  -- \
  ethernet.mtu 9000 \
  connection.autoconnect yes
```

There's a deeper reason to persist this properly than avoiding retyping. The QuantTrio recipe's troubleshooting carries a hard-won warning: if the fabric IP is configured ad hoc, then after a reboot or link-flap a **link-local address (169.254.x.x) can squat the port** before your IP arrives — which **shifts the GID table**, the RDMA address book we met in the primer. Your carefully discovered `NCCL_IB_GID_INDEX` now points at the wrong entry, and NCCL either fails outright or silently degrades. Persist the config, and re-verify GIDs after any reboot (discovery procedure below).

### Verify the Fabric

From the head node:

```bash
# Reachability
for node in ${QSFP_2} ${QSFP_3} ${QSFP_4}; do ping -c 3 ${node}; done

# Link speed — should report 200000Mb/s
ethtool enP2p1s0f1np1 | grep Speed

# Jumbo frames actually pass end-to-end (-M do = don't fragment; 8972 = 9000 minus headers)
ping -c 3 -M do -s 8972 ${QSFP_2}
```

If the jumbo ping fails while the normal ping works, something in the path (almost certainly a switch port) is still at MTU 1500. For deeper, RDMA-level verification — measuring actual RDMA bandwidth and latency with `ib_write_bw` and `ib_write_lat` — see the [switch guide's verification section](what-is-fabric.md#step-6-verify-the-fabric-layer-by-layer); it's worth doing once before you ever launch the model.

### The Mixed-Cabling Discovery

Here's something no recipe warned us about, because it's an artifact of *history*, not configuration: **if your Sparks have ever been cabled up before — a previous experiment, a different topology — nodes may be using different physical QSFP ports, and therefore different HCA interfaces.**

Two of our nodes (S3 and S4) had lived in an earlier two-node setup. When we recabled everything for the four-node cluster, their cables landed on the *other* physical port of the ConnectX-7. Perfectly functional — link up, pings fine — but now S3 and S4 talked through `enP2p1s0f0np0` (HCA `mlx5_0`) while S1 and S2 used `enP2p1s0f1np1` (HCA `mlx5_1`).

NCCL does not guess this. Every node must be told exactly which HCA, which network interface, and which GID index to use — and if any node's answer differs, your launch configuration must be **per-node**, not global. Here's the discovery procedure; run it on every node and write down the results:

```bash
# Which RDMA devices exist and which are up
rdma link

# HCA details
ibv_devinfo -v | grep -E 'hca_id|port|gid'

# Dump the GID table with types (substitute your HCA: mlx5_0 or mlx5_1)
for i in $(seq 0 15); do
    gid=$(cat /sys/class/infiniband/mlx5_1/ports/1/gids/$i 2>/dev/null || echo "none")
    type=$(cat /sys/class/infiniband/mlx5_1/ports/1/gid_attrs/types/$i 2>/dev/null || echo "none")
    echo "GID $i: $gid  ($type)"
done
```

You're looking for the entry that is **type RoCE v2** and **contains your fabric IP**. IPv4 addresses appear embedded in the GID's IPv6-mapped form — the last four bytes of the GID are your IP in hex. That entry's index is your `NCCL_IB_GID_INDEX` for that node. (If `show_gids` is available on your system, it prints this table in one friendly shot.)

Our cluster's final inventory — note how "four identical machines" turned out to have three distinct configurations:

| Node | HCA | Interface | GID Index |
|---|---|---|---|
| S1 (head) | `mlx5_1` | `enP2p1s0f1np1` | 5 |
| S2 | `mlx5_1` | `enP2p1s0f1np1` | 5 |
| S3 | `mlx5_0` | `enP2p1s0f0np0` | 5 |
| S4 | `mlx5_0` | `enP2p1s0f0np0` | **3** |

Why does S4 alone have GID index 3? Because the GID table is populated in the order addresses landed on the interface, and S4's configuration history differed from its siblings' — exactly the link-local-squatting mechanism the recipe warns about. The fix isn't to force them all to match; it's to *measure each node and configure each node with its own truth*. Our launch script does exactly that (Phase 8).

> **Lesson:** identical hardware does not mean identical configuration. Discover, don't assume — and re-discover after any reboot or recabling.

---

## Phase 4: Software Parity

Before touching the model, bring all four nodes to **identical** kernel, driver, and CUDA versions. Version skew between nodes produces failures that look like anything *except* version skew — NCCL hangs, container weirdness, performance mysteries — and you can lose hours attributing them to the wrong cause.

Check what each node is running:

```bash
uname -r                                                    # kernel
nvidia-smi --query-gpu=driver_version --format=csv,noheader # GPU driver
nvcc --version                                              # CUDA
docker --version                                            # Docker
nvidia-ctk --version                                        # container toolkit
```

Bring everything current (DGX OS is an Ubuntu variant; `apt upgrade` pulls NVIDIA-patched kernels automatically):

```bash
sudo apt update && sudo apt upgrade -y
sudo reboot   # required after a kernel upgrade — actually do it
```

After reboot, re-run `uname -r` and confirm every node reports the same version. For the record, our cluster's target state was: kernel `6.17.0-1026-nvidia`, driver `580.159.03`, CUDA `13.0.3`, nvidia-ctk `1.19.1`, Docker `29.2.1` — yours will be whatever is current when you do this; the point is that all four match.

Our S3 and S4 (the two veterans of the earlier setup) had been powered on for months and were several kernel versions behind the freshly-provisioned S1 and S2. If all your nodes are fresh from OOBE, this phase may be a five-minute confirmation. Do the confirmation anyway.

---

## Phase 5: Model Selection & Download

### Why This Checkpoint: NVFP4 vs Int4-Int8Mix

Two quantised flavours of GLM 5.2 circulate for DGX Spark clusters, and the choice is not cosmetic:

- **NVFP4 hybrid** (~429 GB): a different quantisation scheme explored by others in the community (see p33zy's work in the credits). It works in its own recipes — but at ~107 GB of weights per node on a Spark, it's what the QuantTrio recipe aptly calls "a knife-edge that OOMs the moment page cache or warmup allocations breathe on it."
- **QuantTrio/GLM-5.2-Int4-Int8Mix** (~405 GB): all 256 experts, unpruned, ~95 GiB per node at TP=4 — leaving genuine headroom for KV cache, CUDA graphs, and the OS on a 128 GB unified-memory machine *(memory analysis from the QuantTrio recipe)*.

We went with Int4-Int8Mix, both for the memory margin and because it's the checkpoint the entire recipe — kernels, patches, serve config — was built and benchmarked around. These are *different checkpoints with different quant schemes*; you cannot swap one in under the other's container and kernel setup.

### Clone the Repos

On the head node:

```bash
cd /var/tmp

# The recipe (mods, patches, launch.sh)
git clone https://github.com/tonyd2wild/GLM-5.2-QuantTrio-200K-4x-DGX-Spark glm52-recipe

# The Triton kernels
git clone https://github.com/CosmicRaisins/glm-5.2-gb10

# The vLLM build harness
git clone https://github.com/eugr/spark-vllm-docker

# Stage the 10 kernel files at a path that will exist IDENTICALLY on every node
mkdir -p /var/tmp/glm-triton
cp /var/tmp/glm-5.2-gb10/kernels/*.py /var/tmp/glm-triton/
```

> **Why `/var/tmp/glm-triton` and not `~/glm-triton`?** The recipe defaults to `~/glm-triton`, which is fine when every node has the same user. Our early attempt used a home-directory path from the head node in the launch config — and that path simply didn't exist on the workers, whose home directories differed. Any absolute path that exists identically on all four nodes works; we standardized on `/var/tmp/glm-triton` and never thought about it again. (This is also why we told you to use one username everywhere.)

The 10 kernel files you should now have in `/var/tmp/glm-triton/` *(list from the QuantTrio recipe; the kernels are CosmicRaisins' work, Apache-2.0)*:

```
b12x_sparse_helpers.py
deepseek_v2.py
flashmla_sparse.py
patch_flashmla_ops.py
sm12x_deep_gemm_fallbacks.py
sm12x_mqa.py
sm12x_sparse_mla_attn.py
sparse_attn_indexer.py
sparse_mla_env.py
sparse_mla_kernels.py
```

These implement the sparse-MLA attention path for the GB10's `sm_121` GPU architecture — the reason GLM 5.2 runs on this hardware at all.

### Download the Weights — Once

Download the checkpoint **once**, on the node with the best internet, then fan it out over the fabric later (Phase 7). The recipe's one-liner *(from the QuantTrio recipe)*:

```bash
hf download QuantTrio/GLM-5.2-Int4-Int8Mix \
  --local-dir /var/tmp/models/glm52-int4-int8mix
```

Run it inside `tmux` (`tmux new -s glm-download`) so it survives your SSH session. Your Time Budget calculation from the prerequisites tells you whether this is a lunch break or an overnight job — at gigabit speeds it's about an hour; on our unusually thin pipe it was days. We used the equivalent `huggingface_hub.snapshot_download()` Python call with `resume_download=True` — either works, and the resume behaviour is what matters, as we're about to explain.

### The Overnight Stall — a Small Story About Trust

Night one, we checked the download before bed: 40-ish shards done, process healthy, ETA acceptable. Morning check: **122 of 124 shards present**, process still running. Wonderful — nearly done!

Except it wasn't nearly done. It wasn't *doing anything*. It had stalled at around 2 a.m. — and spent six hours looking busy.

Diagnosis went like this:

```bash
# When were files last actually written?
ls -lt /var/tmp/models/glm52-int4-int8mix/ | head -5
# → newest file: six hours old. Suspicious.

# What are the TCP connections doing?
ss -t state close-wait
# → a socket to the CDN in CLOSE-WAIT. The far end hung up long ago.

# Is anything being written right now?
watch -n 5 'ls -lh /var/tmp/models/glm52-int4-int8mix/*.safetensors | tail -3'
# → sizes frozen. Nothing.
```

Root cause: the remote end had dropped the connection mid-shard, and the HTTP client held the dead socket open forever rather than timing out. A process that is *alive* is not a process that is *working* — file counts and `ps` output measure the former; **file timestamps measure the latter**.

The fix is to make restarts free and automatic. Since the downloader resumes from completed shards, wrap it in a retry loop:

```bash
#!/bin/bash
# retry-download.sh — restart the downloader until it exits clean
while true; do
    hf download QuantTrio/GLM-5.2-Int4-Int8Mix \
        --local-dir /var/tmp/models/glm52-int4-int8mix
    if [ $? -eq 0 ]; then
        echo "Download complete!"
        break
    fi
    echo "Downloader exited non-zero; retrying in 30s..."
    sleep 30
done
```

(This handles the downloader *failing*. For the hung-but-alive case, kill the stuck process — the retry loop or a rerun resumes from where it left off. If you want full automation, add a watchdog that kills the downloader when the newest file's mtime exceeds some age.)

> **Lesson:** when checking any long-running job, verify *evidence of progress* (timestamps, sizes, counters moving), not *evidence of existence* (process alive, files present). This lesson comes back with interest during the build phase.

When it finishes you should have ~124 safetensors shards plus config and tokenizer files — roughly 391 GiB on disk. The MTP drafter needs no separate download; it lives inside the checkpoint at layer 78 *(from the QuantTrio recipe)*.

### The Hub-Layout Symlink

The serve configuration expects the model at a Hugging Face hub-style path inside the container. Create this symlink now (and later on every node) *(from the QuantTrio recipe)*:

```bash
mkdir -p /var/tmp/models/hub
ln -sfn ../glm52-int4-int8mix /var/tmp/models/hub/glm52-int4-int8mix
```

---

## Phase 6: Building the Custom vLLM Image

Stock vLLM does not serve GLM 5.2 on GB10 hardware. The recipe builds a custom image in layers:

1. **Base build** — vLLM at a pinned commit, compiled for GB10, via eugr's harness
2. **sm12x sparse-MLA mod** — installs CosmicRaisins' Triton kernels and a DeepGEMM bypass
3. **b12x mod** — a library making sparse-MLA decode safe under CUDA graph capture (without it, graph capture crashes)
4. **Indexer MTP-overhang patch** — Tony's fix for a vLLM bug that crashes the engine at ≥3 concurrent requests

All four steps run on one build machine (any Spark works — we used the head node); the finished image is then distributed.

### Step 1: The Base Build

*(Commands from the QuantTrio recipe.)* Build against the recipe's pinned vLLM commit — the exact commit matters, as Bug #1 will demonstrate at length:

```bash
cd /var/tmp/spark-vllm-docker
./build-and-copy.sh --vllm-ref ab666069935c1f23e8ef56038b4659ac9e8f19f8 \
  -t vllm-node-tf5-glm52-b12x:probe --tf5
```

Expect 35–60 minutes. Or, if you're us: expect the first attempt to die at ~60% with `ml-dtypes download failed — network unreachable`, discarding eight minutes of work because a transient network blip hit mid-`pip install`. Long builds with mid-flight network fetches are fragile; if your network is flaky, pre-fetch the known trouble packages so pip can resolve them locally:

```bash
pip3 download ml-dtypes --dest /tmp/pip-cache/
```

(Then point pip at it with `--find-links /tmp/pip-cache` in the relevant build step, or simply retry the build — Docker's layer cache makes the second pass much faster. Ours took 9 minutes. That same layer cache is also about to cause the single most educational bug of the deployment. Foreshadowing.)

### Steps 2–4: Bake the Mods and the Patch

*(Procedure and commands from the QuantTrio recipe.)* The pattern: start a container from the base image, mount in the kernels and the mod/patch scripts, run them, and `docker commit` the result as the final image.

One critical detail first: the sm12x mod script expects the kernels mounted at exactly `/root/models/models15/glm-triton` — that's the `KERNELS=` path hard-coded at the top of `mods/glm52-sm12x-sparse/run.sh`. Mount them there regardless of where they live on the host:

```bash
cd /var/tmp/glm52-recipe

docker run -d --name glm52-modding \
  -v /var/tmp/glm-triton:/root/models/models15/glm-triton:ro \
  -v $(pwd)/mods/glm52-sm12x-sparse:/mods/glm52-sm12x-sparse:ro \
  -v $(pwd)/mods/glm52-b12x-sparse:/mods/glm52-b12x-sparse:ro \
  -v $(pwd)/patches:/patches:ro \
  vllm-node-tf5-glm52-b12x:probe sleep infinity

# Mod 1: Triton sparse-MLA kernels + DeepGEMM bypass (ciprianveg's script, replicating CosmicRaisins' mods)
docker exec glm52-modding bash /mods/glm52-sm12x-sparse/run.sh

# Mod 2: b12x, for CUDA-graph-safe sparse-MLA decode (ditto)
docker exec glm52-modding bash /mods/glm52-b12x-sparse/run.sh

# Patch: the indexer MTP-overhang fix (Tony's contribution — see below)
docker exec glm52-modding python3 /patches/fix-indexer-mtp-overhang.py

# Commit the result — the ENTRYPOINT/CMD changes are NOT optional (see warning)
docker commit \
  --change 'ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]' \
  --change 'CMD []' \
  glm52-modding vllm-node-tf5-glm52-b12x:probe-modded

docker rm -f glm52-modding
```

**Verify before moving on** *(success criteria from the QuantTrio recipe)*: both mod scripts print `✓` lines; the sm12x script must end with `=== glm52-sm12x-sparse complete ===`; the b12x script must show a successful `import b12x`; the patch prints `patched: .../indexer.py` (it's idempotent — safe to re-run).

> ⚠️ **Two Docker traps, verbatim warnings from the QuantTrio recipe (both bit Tony; the second nearly bit us):**
>
> 1. **`docker commit` inherits entrypoint state from the patch container.** If the container you're committing was started with an entrypoint override or a bare command like `sleep infinity`, the committed image carries that forward — and will never boot vLLM. Always commit with `--change 'ENTRYPOINT [...]' --change 'CMD []'` exactly as shown.
> 2. **Piping stdin scripts into containers requires `docker exec -i`.** Without `-i`, the script **silently no-ops** — no error, no output, nothing runs — and you proceed to commit an unpatched image that fails much later, far from the actual mistake.

**About that indexer patch (step 4):** `patches/fix-indexer-mtp-overhang.py` fixes an upstream vLLM bug that Tony found, fixed, and contributed back — a perfect example of the compounding community effort this whole stack runs on. The short version *(full story in the patch's docstring in the recipe repo)*: vLLM's DSA indexer sizes an internal block-table buffer from `max_model_len` alone, but MTP's speculative tokens can extend a request one block *past* that. When `max_model_len` is an exact multiple of the block size — and 200,000 ÷ 64 is exact — the buffer comes up one block short, and at ≥3 concurrent requests the engine crashes with `RuntimeError: The expanded size of the tensor (3125) must match the existing size (3126)`. With the patch baked in, Tony's full concurrency sweep (1 through 6 simultaneous streams) runs with zero crashes. If you ever intend to serve more than two concurrent requests — bake this patch.

### Stage NCCL 2.30.4 on Every Node

*(From the QuantTrio recipe.)* The recipe pins NCCL 2.30.4, swapped in over the image's bundled version at runtime via `LD_PRELOAD` — which means the library file must exist on every node's disk. On **each** node:

```bash
pip download nvidia-nccl-cu13==2.30.4 -d /tmp/nccl --no-deps
mkdir -p /var/tmp/models/hub/nccl-2.30.4
cd /tmp/nccl && unzip -o nvidia_nccl_cu13-2.30.4*.whl 'nvidia/nccl/lib/libnccl.so.2'
cp nvidia/nccl/lib/libnccl.so.2 /var/tmp/models/hub/nccl-2.30.4/
```

Why pin a specific NCCL at all? Because NCCL is the layer doing all cross-node communication, its behaviour differs across versions in ways that matter enormously on unusual fabrics like a 4-Spark RoCE cluster — 2.30.4 is the version this recipe was validated on. Treat it like part of the model.

---

## Phase 7: Distribution

Everything — image, weights, kernels, NCCL — must exist on all four nodes. This is where the 200 Gbps fabric earns its keep: you downloaded 405 GB from the internet once; you will never download it again.

### Cluster-Internal SSH

Give the head node its own key for reaching workers over the fabric:

```bash
# On the head node
ssh-keygen -t ed25519 -f ~/.ssh/cluster_key -N ""
for node in ${WORKERS}; do
    ssh-copy-id -i ~/.ssh/cluster_key.pub sparkuser@${node}
done

# Smoke test
ssh -i ~/.ssh/cluster_key sparkuser@${QSFP_2} "hostname && nvidia-smi -L"
```

### Ship the Image

The modded image is ~9.3 GB compressed — minutes over the fabric:

```bash
for node in ${WORKERS}; do
    docker save vllm-node-tf5-glm52-b12x:probe-modded | gzip | \
        ssh -o Compression=no -i ~/.ssh/cluster_key sparkuser@${node} \
        "gunzip | docker load"
done
```

The `-o Compression=no` matters: we're already gzipping the stream, and stacking SSH's compression on top costs CPU for zero gain — on a link this fast, CPU is the bottleneck, not bandwidth. (On even faster setups, swap `gzip` for `pigz` — parallel gzip — or skip compression entirely, as the recipe does with a plain `docker save | ssh docker load`.)

### Ship the Weights

391 GiB × 3 workers. Run the transfers in parallel — each rsync is limited by single-stream throughput, so three at once genuinely finishes sooner:

```bash
for node in ${WORKERS}; do
    rsync -a --info=progress2 -e "ssh -i ~/.ssh/cluster_key -o Compression=no" \
        /var/tmp/models/glm52-int4-int8mix/ \
        sparkuser@${node}:/var/tmp/models/glm52-int4-int8mix/ &
done
wait
echo "All weight transfers complete"
```

Then create the hub-layout symlink on every worker (the head node already has it from Phase 5):

```bash
for node in ${WORKERS}; do
    ssh sparkuser@${node} \
      "mkdir -p /var/tmp/models/hub && ln -sfn ../glm52-int4-int8mix /var/tmp/models/hub/glm52-int4-int8mix"
done
```

### Ship the Kernels

Small but essential — the launch script bind-mounts these files into every container at runtime:

```bash
for node in ${WORKERS}; do
    ssh sparkuser@${node} "mkdir -p /var/tmp/glm-triton"
    scp /var/tmp/glm-triton/*.py sparkuser@${node}:/var/tmp/glm-triton/
done
```

And don't forget the NCCL staging step from Phase 6 on each worker, if you haven't already.

### Verify Distribution

Thirty seconds now saves an opaque failure later:

```bash
for node in ${ALL_NODES}; do
    echo "=== ${node} ==="
    ssh sparkuser@${node} '
        ls /var/tmp/glm-triton/*.py | wc -l                                # expect 10
        ls /var/tmp/models/glm52-int4-int8mix/*.safetensors | wc -l       # expect ~124
        ls -l /var/tmp/models/hub/glm52-int4-int8mix                       # symlink exists
        ls /var/tmp/models/hub/nccl-2.30.4/libnccl.so.2                    # NCCL staged
        docker images | grep glm52                                         # image loaded
    '
done
```

---

## Phase 8: The Launch Script

The recipe ships a `launch.sh` (derived from CosmicRaisins' launch harness) that orchestrates the whole cluster from the head node: one plain `docker run` per node, using **vLLM's native multi-node mode** (`--nnodes`/`--node-rank` rendezvous over the `mp` backend) — no Ray, no scheduler, no extra moving parts *(from the QuantTrio recipe)*. Workers start headless first; the head starts last and exposes the API.

You configure it by editing the `EDIT`-marked block at the top: node IPs, SSH user and key, HCA and interface names. Then:

```bash
./launch.sh --dry-run   # prints the docker commands it WOULD run — read them
./launch.sh             # actually launches
./launch.sh --stop      # docker rm -f on every node
```

Always run `--dry-run` first, every time you change the config. Reading the generated commands catches config mistakes in seconds that would otherwise take a 12-minute failed boot to surface.

### Our Modification: Per-Node NCCL Configuration

The stock config assumes homogeneous nodes — one HCA name, one interface name, one GID index, used everywhere. Phase 3 established that our cluster is *not* homogeneous. So we extended the config block into per-node lookup tables, keyed by fabric IP. This is the one structural change we made to the recipe's launch flow, and if your discovery table (Phase 3) shows any variation between nodes, you need it too:

```bash
# ---- per-node NCCL configuration (values from the Phase 3 discovery) ----
#
# S3+S4 were recabled from an earlier cluster and sit on the OTHER physical
# QSFP port: HCA mlx5_0 / interface enP2p1s0f0np0.
# S1+S2 use mlx5_1 / enP2p1s0f1np1.
# S4's GID index differs due to its interface-configuration history.
#
# Re-run discovery after ANY reboot or recabling:
#   rdma link
#   ibv_devinfo -v | grep -E 'hca_id|port|gid'
#   for i in $(seq 0 15); do cat /sys/class/infiniband/*/ports/1/gids/$i; done

declare -A NCCL_HCA=(
    ["${QSFP_1}"]="mlx5_1:1"
    ["${QSFP_2}"]="mlx5_1:1"
    ["${QSFP_3}"]="mlx5_0:1"
    ["${QSFP_4}"]="mlx5_0:1"
)
declare -A NCCL_IF=(
    ["${QSFP_1}"]="enP2p1s0f1np1"
    ["${QSFP_2}"]="enP2p1s0f1np1"
    ["${QSFP_3}"]="enP2p1s0f0np0"
    ["${QSFP_4}"]="enP2p1s0f0np0"
)
declare -A NCCL_GID=(
    ["${QSFP_1}"]="5"
    ["${QSFP_2}"]="5"
    ["${QSFP_3}"]="5"
    ["${QSFP_4}"]="3"
)
```

Each node's container then receives *its own* values:

```bash
-e NCCL_IB_HCA=${NCCL_HCA[$node]} \
-e NCCL_IB_GID_INDEX=${NCCL_GID[$node]} \
-e NCCL_SOCKET_IFNAME=${NCCL_IF[$node]} \
```

### Anatomy of a Node's Launch

So you understand what `launch.sh` generates (and can debug it when needed), here is the shape of the per-node `docker run`, annotated. Container flags and serve configuration are from the QuantTrio recipe; the per-node NCCL env is ours:

```bash
docker run -d --name vllm_slot \
    --gpus all \
    --network host \                  # NCCL + rendezvous need real node networking
    --ipc host --shm-size 16g \       # shared memory for inter-process tensors
    --device /dev/infiniband \        # ← RDMA device passthrough. NOT optional. See below.
    --cap-add IPC_LOCK \
    --ulimit memlock=-1:-1 \          # RDMA must pin (lock) memory pages
    -v /var/tmp/models:/cache/huggingface \
    # ...10 read-only bind-mounts placing each /var/tmp/glm-triton/*.py kernel
    #    file over its counterpart in the container's vLLM tree
    #    (generated by launch.sh — this is how the sparse-MLA kernels are live
    #    at runtime rather than baked in, which will matter in Bug #3)...
    -e LD_PRELOAD=/cache/huggingface/hub/nccl-2.30.4/libnccl.so.2 \   # pinned NCCL 2.30.4
    -e NCCL_IB_HCA=... -e NCCL_IB_GID_INDEX=... -e NCCL_SOCKET_IFNAME=... \  # per-node!
    -e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4 \  # ciprianveg's find: fewer channels = less contention on GB10 RoCE
    -e NCCL_DEBUG=WARN \
    -e VLLM_USE_DEEP_GEMM=0 -e VLLM_MOE_USE_DEEP_GEMM=0 -e VLLM_DEEP_GEMM_WARMUP=skip \  # Bug #4's fix
    vllm-node-tf5-glm52-b12x:probe-modded \
    vllm serve /cache/huggingface/hub/glm52-int4-int8mix \
        --served-model-name glm-5.2 \
        --host 0.0.0.0 --port 8210 \
        --nnodes 4 --node-rank ${RANK} \              # native multi-node; workers add --headless
        --distributed-executor-backend mp \
        --tensor-parallel-size 4 \
        --max-model-len 200000 \
        --kv-cache-dtype fp8_ds_mla \
        --speculative-config '{"method":"mtp","num_speculative_tokens":4,"draft_tensor_parallel_size":1,"attention_backend":"FLASHMLA_SPARSE"}' \
        --compilation-config '{"cudagraph_mode":"FULL"}' \
        --async-scheduling \
        --max-num-batched-tokens 8192 \
        --gpu-memory-utilization 0.91 \
        --kv-cache-memory-bytes 10950000000 \
        --max-num-seqs 6 \
        --reasoning-parser glm45 \
        --tool-call-parser glm47
```

### Why Each Serve Setting Is What It Is

This table is the QuantTrio recipe's accumulated wisdom — several entries represent someone's lost weekend *(rationale from the recipe, lightly expanded)*:

| Setting | Value | Why |
|---|---|---|
| `--tensor-parallel-size` | `4` | One GB10 per TP rank; 405 GB ÷ 4 ≈ 95 GiB weights per node. |
| `--kv-cache-dtype` | `fp8_ds_mla` | fp8 sparse-MLA KV cache: halves KV footprint, which is what makes 200K context fit in ~10.5 GB/node. |
| `--speculative-config` | `mtp, k=4, draft TP=1` | The drafter is in-checkpoint (layer 78). Draft TP=1 (back199640's tuning) is key: the tiny drafter gains nothing from being split four ways, and keeping it on one rank removes cross-node hops from *every* speculation step. |
| `--compilation-config` | `cudagraph_mode: FULL` | Full CUDA graphs replay the entire decode step as one pre-recorded GPU program, eliminating per-step launch overhead. **Requires the b12x mod** — without it, capture crashes (`torch.full` under capture). |
| `--async-scheduling` | on | Overlaps CPU scheduling with GPU execution (back199640) — worth real tok/s on GB10. |
| `--max-num-batched-tokens` | `8192` | Prefill chunk size: large enough for ~700+ tok/s prefill, small enough not to blow memory at depth. |
| `--gpu-memory-utilization` + `--kv-cache-memory-bytes` | `0.91` + `10950000000` | **The deterministic-boot trick.** With gmu alone, vLLM sizes the KV cache from *currently free* memory — which on unified-memory GB10 varies with page-cache state, so the same command OOMs or boots depending on what the machine did earlier. And gmu 0.90 leaves only 9.78 GiB where 200K context needs 10.19 GiB. Pinning KV to 10.95 GB at gmu 0.91 boots a 200,064-token pool every time. |
| `--max-model-len` | `200000` | The 200K headline, fitting exactly in that pinned KV budget. |
| `--max-num-seqs` | `6` | Up to 6 concurrent streams — **requires the indexer MTP-overhang patch** (Phase 6), or the engine crashes at ≥3. Drop to 1 for a pure single-stream latency build. |
| `--reasoning-parser` / `--tool-call-parser` | `glm45` / `glm47` | Correct parsers for GLM-5.2's reasoning traces and tool-call format. |
| `--distributed-executor-backend` | `mp` | Native multiprocessing with `--nnodes/--node-rank` rendezvous. No Ray. |
| `NCCL_MIN/MAX_NCHANNELS` | `4` | ciprianveg's find: narrowing NCCL's parallel channels cuts contention on GB10 RoCE — counterintuitively, *more* channels is slower here. |

### The Silent Killer: RDMA Passthrough

One more recipe warning that deserves its own heading, because the failure mode is so perfectly disguised *(from the QuantTrio recipe's troubleshooting)*:

> Without `--device /dev/infiniband` + `--cap-add IPC_LOCK` + `--ulimit memlock=-1:-1`, NCCL **silently falls back to TCP** over the socket interface. Everything works. Nothing errors. Decode is ~12 tok/s instead of ~30.

If your numbers ever look mysteriously halved, launch with `NCCL_DEBUG=INFO` and search the logs for `NET/IB` (good) versus `NET/Socket` (you're on the slow path). This one check has probably saved more community hours than any other line in the recipe.

---

## Phase 9: The Bugs

Everything above went roughly to plan. This section is what actually consumed the time — and it's the part of this document most likely to save *you* a day. Four real bugs, each with symptom, diagnosis, root cause, fix, and lesson. Plus one self-inflicted catastrophe as an interlude.

A theme to watch for: **every one of these bugs involved a system faithfully executing something other than what we believed we had told it.** The build that wasn't building what we asked. The patch that wasn't running. The check that never fired. Debugging distributed systems is mostly the art of finding the gap between belief and reality.

---

### Bug #1: Docker's Cache Served Us the Wrong vLLM

**Symptom.** The build completes and reports success. Containers start on all four nodes; weights load fully (97.95 GiB per node — correct). Then, at KV-cache initialization:

```
RuntimeError: shape '[3126, 64, 576]' is invalid for input of size 131241984
```

**Diagnosis.** We'd pinned the build to the recipe's vLLM commit (`ab666069...`) via `--vllm-ref`. Surely the container held that commit? We checked what was *actually* inside:

```bash
docker run --rm vllm-node-tf5-glm52-b12x:probe-modded python3 -c "
import vllm, subprocess
print(vllm.__version__)
print(subprocess.run(['git','log','--oneline','-1'],
      cwd='/workspace/vllm', capture_output=True, text=True).stdout)"
```

It printed a *different* commit — `158ff6f3d`, something newer.

**Root cause.** Remember our failed first build attempt (the `ml-dtypes` network error)? It left Docker layer cache behind. Docker caches build layers keyed on the Dockerfile instructions and their inputs — and **a git ref buried inside a build argument doesn't necessarily bust that cache**. On rebuild, Docker cheerfully reused a cached layer containing a vLLM checkout from a different ref. The build was fast *because it wasn't building what we asked for*. The newer vLLM's KV-cache logic was incompatible with our Triton kernel set — hence the shape error.

**Fix.** Nuke the build cache and rebuild clean:

```bash
docker builder prune -af    # freed ~50 GB
./build-and-copy.sh --vllm-ref ab666069935c1f23e8ef56038b4659ac9e8f19f8 \
  -t vllm-node-tf5-glm52-b12x:probe --tf5
```

(If your build harness has a flag to force a fresh source checkout, use that too. When in doubt: prune.)

**Lesson.** `--vllm-ref` is a *request*; the Docker layer cache decides what you *get*. Whenever a build follows a failed or different-config build, verify the artifact — actually query the commit inside the image — or prune and pay the full build time for certainty. Fast builds after failures should make you suspicious, not happy. (Note the rhyme with the download stall: liveness isn't progress, and "build succeeded" isn't "built what you asked.")

---

### Bug #2: The Patch That Wasn't Running — `__pycache__`

**Symptom.** Chasing the shape error (before Bug #1 was fully understood), we patched an attention-backend file — `indexer.py` — inside the image. We verified the patch: `cat` showed our exact new code in the file. Rebuilt the image, relaunched the cluster, waited through weight loading… identical error. Identical traceback. As if the patch didn't exist.

**Diagnosis.** The patch *file* existed. Was the patch *code running*? Python imports don't read `.py` files directly — on first import, Python compiles source to bytecode and caches it in `__pycache__/*.pyc`, then reuses the cached bytecode whenever it looks fresh. Freshness is judged by source mtime — and Docker layer operations can produce file timestamps that defeat this check, leaving a stale `.pyc` that looks up-to-date next to a newer `.py`.

You can confirm which file Python actually loads, and whether a compiled twin exists:

```bash
docker run --rm <image> python3 -c "
import importlib.util
spec = importlib.util.find_spec('vllm')   # or the specific patched module
print(spec.origin)"
# then look for a matching __pycache__/*.pyc beside the source file
```

**Root cause.** Our container was executing the cached bytecode of the *old* `indexer.py`. The source file we so carefully verified with `cat` was never being read. The `.py` was correct; the program was wrong.

**Fix.** When patching Python source inside an image, clear compiled bytecode before committing, and optionally force recompilation so any syntax error in the patch surfaces immediately:

```bash
# Inside the patch container, before docker commit:
find /workspace/vllm -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find /workspace/vllm -name "*.pyc" -delete 2>/dev/null || true
python3 -m compileall -q /workspace/vllm/vllm/
```

**Lesson.** *Verify the executing artifact, not the source artifact.* `cat` proves what's in the file; only the interpreter proves what runs. In containers — where layered filesystems play games with mtimes — assume bytecode is stale after any source patch. Clear `__pycache__` reflexively. (This is also a quiet argument for the recipe's script-driven patching in Phase 6 over hand-editing files inside containers: Tony's patch script modifies the file in a running container *before* anything imports it.)

---

### Bug #3: The KV-Cache Shape Mismatch — One Bug, Two Hiding Places

With a clean, correct build (Bug #1) executing genuinely patched code (Bug #2), we met the shape error on its own terms. This is the most technically interesting bug of the deployment, and understanding it requires one piece of real math — worth it, promise.

#### Background: 576 vs 656, or What `fp8_ds_mla` Actually Stores

Recall from the primer: MLA caches a compressed latent per token. GLM 5.2's latent has a *logical* width of **576 elements** per token per layer — a 512-element compressed KV vector plus a 64-element rotary-position component.

But `fp8_ds_mla` doesn't store 576 uniform elements. Its packed on-disk-in-memory layout per token is:

- 512 × 1-byte **fp8** values (the compressed KV latent),
- 16 bytes of **scale factors** (fp8 needs per-group scales to dequantize),
- 64 × 2-byte **bf16** values (the rotary component, kept at higher precision because position information degrades badly at fp8).

Total: 512 + 16 + 128 = **656 bytes per token per layer**. So there are two "widths" in play: logical **576**, physical **656** — and every piece of code touching this cache must agree on which one it means.

Now the arithmetic that explains the error message. The KV pool holds 3126 blocks of 64 tokens = **200,064 token slots** (there's our 200K context, plus MTP slack). The allocator — correctly — reserved physical bytes:

> 3126 × 64 × **656** = 131,241,984 — *"input of size 131241984"*

The reshape path — incorrectly — tried to view that buffer with the logical width:

> 3126 × 64 × **576** = 115,236,864 ≠ 131,241,984 → `RuntimeError: shape '[3126, 64, 576]' is invalid for input of size 131241984`

The allocation path knew about the metadata; a reshape path didn't. Two functions disagreeing about the meaning of "head size." That's the entire bug. Finding *all* the places it lived took two rounds.

#### Round 1: the Indexer Backend

**Symptom.** The shape error above, at KV-cache initialization, traceback through `vllm/v1/kv_cache_interface.py` into the DeepSeek-V3.2-style indexer backend (`indexer.py` — the same vLLM file Tony's MTP-overhang patch touches, hosting a different bug).

**Root cause.** That backend's `get_kv_cache_shape()` returned `(num_blocks, block_size, head_size)` with the logical `head_size` of 576, while the allocator had provisioned 656 bytes per slot.

**Our first fix was wrong, and the wrongness was instructive.** We hardcoded 656:

```python
def get_kv_cache_shape(num_blocks, block_size, head_size):
    return (num_blocks, block_size, 656)   # WRONG — breaks the indexer's own layers
```

Crash — from a different layer group. Because this backend also serves the *indexer layers* (part of the sparse-attention machinery), which use a small head size of 132 and **no** fp8 packing. Force-feeding them 656 broke them instead. The correct fix translates logical→physical only where the packing applies:

```python
def get_kv_cache_shape(num_blocks, block_size, head_size):
    # Main MLA layers: logical head_size 576 → physical 656 bytes/slot
    #   (512 fp8 + 16 scale bytes + 128 bf16 rope) under fp8_ds_mla.
    # Indexer layers (head_size 132) are unpacked — pass through untouched.
    actual = 656 if head_size == 576 else head_size
    return (num_blocks, block_size, actual)
```

Notably, CosmicRaisins' `flashmla_sparse.py` kernel already contained exactly this 576→656 handling in *its* shape function — the fix pattern existed in the codebase we were holding; this one code path just hadn't received it.

Rebuild (with `__pycache__` cleared! Bug #2's lesson, immediately applied), relaunch. KV-cache init passed. We celebrated for approximately fifteen minutes.

#### Round 2: the FlashMLASparse Backend

**Symptom.** Same shape error. *Different traceback* — now through `FlashMLASparseBackend.get_kv_cache_shape`, a few seconds later in startup.

**Root cause — and this one's subtle.** That backend already *had* a 656 branch:

```python
def get_kv_cache_shape(num_blocks, block_size, head_size, cache_dtype_str):
    if cache_dtype_str == "fp8_ds_mla":
        actual = 656
    else:
        actual = head_size
    return (num_blocks, block_size, actual)
```

Looks right. Never fires. We traced the call path and found that by the time vLLM invokes this function, the dtype string it passes has been **resolved to `"auto"`** — the KV dtype's journey through vLLM's config plumbing normalizes it before this call site, so the string comparison compares `"auto"` to `"fp8_ds_mla"` and takes the wrong branch, always. A correct-looking check, testing a value that never arrives in the tested form.

**Fix.** Key off the invariant that actually survives the plumbing — the logical head size — exactly as in Round 1:

```python
def get_kv_cache_shape(num_blocks, block_size, head_size, cache_dtype_str=None):
    # Decide by head_size, NOT cache_dtype_str: the caller may pass "auto"
    # even when fp8_ds_mla is active, so the string check never fires.
    actual = 656 if head_size == 576 else head_size
    return (num_blocks, block_size, actual)
```

And here's a practical mercy: this function lives in `flashmla_sparse.py` — one of the ten kernel files that launch.sh **bind-mounts into the container at runtime** rather than baking into the image. No rebuild needed. Edit the file on all four nodes and relaunch:

```bash
# The fix must land on EVERY node — each mounts its own local copy
for node in ${ALL_NODES}; do
    ssh sparkuser@${node} "vim /var/tmp/glm-triton/flashmla_sparse.py"  # or push the edited file
done
```

**Lessons — three of them, all earned:**

1. **When you fix a bug, immediately hunt its siblings.** `indexer.py` and `flashmla_sparse.py` implement the same interface; the same conceptual bug lived in both, expressed differently. Fifteen minutes of "grep for every `get_kv_cache_shape` in the tree" after Round 1 would have prevented Round 2 entirely.
2. **A guard that never fires is worse than no guard** — it reads as handled. Verify not only that the check is correct, but that the value being checked can actually arrive in the form the check expects.
3. **Know your patch surfaces.** This stack has *baked* code (needs image rebuild + redistribution — Bugs #1/#2 territory) and *mounted* code (edit on nodes, relaunch — seconds). Knowing which file lives where turns an hour-long fix cycle into a minute-long one.

---

### Bug #4: DeepGEMM Demands to Exist

**Symptom.** KV cache initializes (progress!). Weights load. Then, during engine warmup:

```
RuntimeError: DeepGEMM is not available. Please install DeepGEMM or disable it.
```

**Root cause.** DeepGEMM is an optimized GEMM (matrix-multiply) library that vLLM likes to use for MoE dispatch. Our container doesn't have it — by design: the sm12x mod installs fallback kernels *bypassing* DeepGEMM, because it isn't the right tool on this GPU architecture. But a warmup code path in this vLLM revision still probed for it unconditionally and treated absence as fatal.

**Fix.** Tell every code path, firmly, to stop asking. In the launch script's environment:

```bash
-e VLLM_USE_DEEP_GEMM=0 \
-e VLLM_MOE_USE_DEEP_GEMM=0 \
-e VLLM_DEEP_GEMM_WARMUP=skip \
```

All three — they gate different call sites (general use, the MoE path, and the warmup probe that actually threw).

**Why not just install DeepGEMM?** Because the recipe's performance numbers were achieved without it — the Triton kernel set *is* the MoE strategy here, and disabling an unused optional dependency costs nothing on the path we actually execute. Adding it would mean another build-and-distribute cycle at 1 a.m. to enable a library our kernels bypass. Easy call.

**Lesson.** Optional-dependency probes can be load-bearing. When a stack is assembled from a recipe, expect to explicitly disable the things the recipe *doesn't* use, not just enable the things it does.

---

### Interlude: The Accidental Swap Death

A confession, offered so you don't repeat it.

Mid-debugging of Bug #3, waiting on yet another multi-minute cluster relaunch, we had a bright idea: reproduce the KV-shape error *faster* by launching a single container on the head node with `--tensor-parallel-size 1`. No cluster coordination, no waiting on four nodes — tight debug loop, right?

Think through what TP=1 means. It doesn't mean "a quarter-sized test." It means *this one node holds the entire model*. All ~400 GB of it. Into 128 GB of unified memory.

vLLM began loading weights. Memory filled. Then swap filled. Then S1 stopped being a computer in any meaningful sense — SSH dead, console dead, the machine grinding its SSD in a swap storm for fifteen-plus minutes until it finally rebooted itself. On a unified-memory machine there is no polite `torch.cuda.OutOfMemoryError` backstop: GPU memory *is* system memory, so instead of a clean crash you get a system-wide death spiral where the OOM killer can't act fast enough to matter.

The honest accounting: we lost about twenty minutes to the reboot — and gained a rule.

> **Rule:** on unified-memory machines, memory mistakes are *availability* incidents, not error messages. Never launch a model at a parallelism degree that can't hold it. To debug one node's behaviour, launch the full TP=4 cluster and read that node's logs.

Silver lining: by the time S1 finished rebooting, we'd pushed the Bug #3 Round 2 fix to the other three nodes. S1 came back, took its copy of the fix, and the third full launch attempt began.

---

## Phase 10: First Inference

05:03 BST, July 13th. Third launch. Four containers up. We watched the head node's logs the way you watch a rocket on a pad:

```
Loading model weights... [97.95 GiB/node]
Elapsed: 372.1s
```

Six minutes of weight loading. Then Triton kernel compilation (~58 s), CUDA graph capture (~24 s — piecewise then full captures, ~1.5 GiB), and:

```
INFO:     Started server process [1]
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8210
```

*(For planning purposes, the recipe budgets ~12 min of weight load plus ~10 min of cudagraph warmup before the API answers — our boot came up quicker, but expect the longer figure on a cold first boot and don't panic while nothing seems to be happening.)*

05:12 BST:

```bash
curl -s http://${HEAD_NODE_IP}:8210/v1/models
```

```json
{"object": "list", "data": [{"id": "glm-5.2", "object": "model", ...}]}
```

It's *listed*. But listed isn't thinking. The real test:

```bash
curl http://${HEAD_NODE_IP}:8210/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "glm-5.2", "messages": [{"role": "user", "content": "What are you?"}]}'
```

> *"I am a large language model trained by Z.ai."*

— complete with a full reasoning trace ahead of the answer. Six hundred and seventy-one billion parameters, sliced across four small machines on a desk, speaking in one voice at five in the morning.

The first *creative* request went out about a minute later, because of course it did: *write a haiku about what you are.* Four nodes deliberated over RoCE, MTP drafted, the big model verified, and out came:

> *Four sparks, one quiet mind —*
> *thoughts crossing a copper sea,*
> *dawn finds us thinking.*

Straight into the team chat. Deployment complete.

### What Success Looks Like in the Logs

Watch for these lines on the head node (`docker logs -f vllm_slot`) — each confirms a subsystem you built in an earlier phase:

```
Loading 256 experts across 4 workers...          ← unpruned MoE, TP=4 (Phases 5–6)
KV cache: fp8_ds_mla, block_size=64, 10.95 GB    ← pinned KV budget (Phase 8), Bug #3 conquered
Attention backend: FlashMLA sparse (sm12x)       ← CosmicRaisins' kernels alive
CUDAGraph: FULL mode                             ← b12x mod working
MTP: 4 speculative tokens, draft TP=1            ← in-checkpoint drafter engaged
```

And in the NCCL init output (with `NCCL_DEBUG=INFO`): `NET/IB` — not `NET/Socket`.

### Performance

What the recipe achieves on this exact configuration — measured by Tony on his cluster, with the indexer patch verified in-image; 512-token generations at temperature 0 *(benchmark table from the QuantTrio recipe)*:

| Concurrency | Aggregate tok/s | Per-stream avg | MTP accept length |
|---|---|---|---|
| 1 (warm median) | **28.8** | 28.8 | 3.3–3.6 |
| 2 | 37.6 | 20.2 | 3.50 |
| 3 | 39.3 | 13.6 | 3.22 |
| 4 | 53.5 | 14.1 | 3.28 |
| 5 | 59.1 | 12.5 | 3.22 |
| 6 | **60.5** | 10.6 | 3.23 |

Two notes worth carrying over: the single-stream figure is a *warm* median — the cold first request after boot reads lower (16–22 tok/s), so warm up before you benchmark. And the concurrency-3-and-above rows exist *only because of the indexer MTP-overhang patch*; unpatched engines crash right there.

Our own initial numbers: ~2 s to first token, single-stream throughput approaching the recipe's 28.8 tok/s target, 97.95 GiB of weights per node with the 10.95 GB KV reservation. We haven't run the formal sweep yet — but reasoning traces, 200K context, and speculative decoding all behave exactly as advertised.

---

## Quick Reference

### Pre-Launch Checklist

```bash
# 0. All nodes reachable on the fabric
for node in ${ALL_NODES}; do
    echo -n "${node}: "; ping -c 1 -W 1 ${node} >/dev/null 2>&1 && echo OK || echo UNREACHABLE
done

# 1. Ten kernel files on every node
for node in ${ALL_NODES}; do ssh sparkuser@${node} 'ls /var/tmp/glm-triton/*.py | wc -l'; done

# 2. Weights + hub symlink + NCCL staged on every node
for node in ${ALL_NODES}; do ssh sparkuser@${node} \
    'ls /var/tmp/models/hub/glm52-int4-int8mix >/dev/null && ls /var/tmp/models/hub/nccl-2.30.4/libnccl.so.2'; done

# 3. Drop page caches — the unified-memory ritual (from the QuantTrio recipe; genuinely required)
for node in ${ALL_NODES}; do
    ssh sparkuser@${node} 'sync && echo 3 | sudo tee /proc/sys/vm/drop_caches' >/dev/null
done

# 4. Dry-run, read the output, then launch
./launch.sh --dry-run
./launch.sh

# 5. Patience (budget up to ~20 min on a cold boot), then:
curl -s http://${HEAD_NODE_IP}:8210/v1/models
```

Why the page-cache ritual matters *(from the QuantTrio recipe)*: loading ~98 GiB of weights fills the page cache on a machine where CPU and GPU share the same memory pool. Two distinct symptoms if you skip it: nondeterministic boot OOMs (solved jointly with the pinned `--kv-cache-memory-bytes`), and mid-load stalls where shard progress freezes at 100% CPU in kernel reclaim even with double-digit GB "free" — the recipe's cure is an unconditional `sync; echo 3 > /proc/sys/vm/drop_caches` every 60 s on every node during the load phase; a manual drop unsticks a frozen load within seconds.

### Environment Variables

| Variable | Value | Purpose |
|---|---|---|
| `NCCL_IB_HCA` | `mlx5_0:1` / `mlx5_1:1` | Which RDMA device — **per node**, from your discovery table |
| `NCCL_IB_GID_INDEX` | e.g. `3` or `5` | Which GID table entry (RoCE v2 + your fabric IP) — **per node** |
| `NCCL_SOCKET_IFNAME` | `enP2p1s0f0np0` / `enP2p1s0f1np1` | Fabric interface for NCCL's bootstrap/socket traffic — **per node** |
| `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` | `4` | Fewer channels, less contention on GB10 RoCE *(ciprianveg, via the recipe)* |
| `NCCL_DEBUG` | `WARN` (or `INFO` when debugging) | Log level; `INFO` reveals `NET/IB` vs `NET/Socket` |
| `LD_PRELOAD` | path to staged `libnccl.so.2` | Pin NCCL 2.30.4 over the image's bundled version *(recipe)* |
| `VLLM_USE_DEEP_GEMM` | `0` | Bug #4: disable DeepGEMM |
| `VLLM_MOE_USE_DEEP_GEMM` | `0` | Bug #4: disable it on the MoE path |
| `VLLM_DEEP_GEMM_WARMUP` | `skip` | Bug #4: skip the warmup probe that throws |

*(If you configure PFC/ECN traffic classes on your switch per the companion switch guide, add `NCCL_IB_TC=106` and `NCCL_IB_SL=3` so NCCL's traffic actually lands in the lossless class — see [MikroTik CRS812 RoCE Switch Setup, Step 5](what-is-fabric.md#step-5-tell-nccl-to-label-its-traffic).)*

### File Layout (Every Node)

```
/var/tmp/
├── models/
│   ├── glm52-int4-int8mix/          # ~391 GiB: ~124 safetensors shards + config
│   └── hub/
│       ├── glm52-int4-int8mix -> ../glm52-int4-int8mix    # hub-layout symlink
│       └── nccl-2.30.4/libnccl.so.2                       # pinned NCCL for LD_PRELOAD
├── glm-triton/                      # the 10 kernel .py files (bind-mounted at launch)
└── (head node only)
    ├── glm52-recipe/                # tonyd2wild's repo: mods/, patches/, launch.sh
    ├── glm-5.2-gb10/                # CosmicRaisins' kernels repo
    └── spark-vllm-docker/           # eugr's build harness
```

### If You Only Remember Five Things

1. **WiFi setup before ethernet** on first boot, or the OOBE update hangs at 0%.
2. **Verify what's actually in your build** — Docker's cache can hand you a different vLLM commit than the one you pinned. `docker builder prune -af` when in doubt.
3. **Clear `__pycache__`** whenever you patch Python inside an image; verify the *executing* artifact, not the source file.
4. **The 656 rule:** every `get_kv_cache_shape` in the attention path must return 656 when `head_size == 576` (fp8_ds_mla packing) — keyed on `head_size`, *not* on the dtype string, which arrives as `"auto"`. Check *both* the indexer and FlashMLASparse backends.
5. **Per-node NCCL config from measured values.** `rdma link` + the GID table on every node; never assume homogeneity, and re-check after reboots.

---

## Credits & Acknowledgements

To borrow the QuantTrio recipe's own words: this deployment stands entirely on the shoulders of the people below. If you follow this guide, their work is what you are using.

First and foremost:

- **[tonyd2wild](https://github.com/tonyd2wild/GLM-5.2-QuantTrio-200K-4x-DGX-Spark)** — the QuantTrio recipe this entire deployment follows: the build procedure, the baked-mod workflow, the serve configuration and its rationale, the deterministic-boot memory budgeting, the benchmarks, the indexer MTP-overhang patch, the load-phase page-cache procedure, and the troubleshooting wisdom quoted throughout. This guide is our unboxing-to-inference journey layered on his foundation.

And the community his recipe credits, whose work flows through everything here:

- **[CosmicRaisins](https://github.com/CosmicRaisins/glm-5.2-gb10)** — the entire sm_121 sparse-MLA port: the `glm-5.2-gb10` repo, the 10 Triton kernels, the DeepGEMM bypass, and the launch harness the recipe's `launch.sh` derives from (Apache-2.0). As Tony puts it: nothing here works without this.
- **Zatz** — proved the unpruned 256-expert Int4-Int8Mix checkpoint fits and flies on 4× GB10 (NVIDIA forum thread [374125](https://forums.developer.nvidia.com/t/glm-5-2-on-a-4x-gb10-cluster-22-tok-s-decode-256k-ctx-recipe/374125), posts #57 and #84).
- **back199640** — the tuning that closed the performance gap (thread 374125, posts #80 and #89): `--async-scheduling`, MTP k=4 with draft TP=1, and explicit `--kv-cache-memory-bytes` for deterministic boot.
- **ciprianveg** — the baked-mod scripts replicating CosmicRaisins' mods (thread 374125, post #34) and the NCCL channel-narrowing discovery (`NCCL_MIN/MAX_NCHANNELS=4`, post #107).
- **[eugr](https://github.com/eugr/spark-vllm-docker)** — the `spark-vllm-docker` build harness used to build the vLLM container for GB10.
- **[QuantTrio](https://huggingface.co/QuantTrio)** — the `GLM-5.2-Int4-Int8Mix` checkpoint itself.
- **p33zy** — explored the alternative NVFP4 quantization path and GB10 hardware-acceleration trade-offs (thread 374125).
- **aidendle94** — shared container/image resources (originally for DeepSeek on GB10) that partially carried over to the GLM-5.2 bring-up (thread 374125).
- **Claude Code** — technical clarifications on the forum thread: sm_121 capability detection, cudagraph capture safety, b12x install requirements, and the sparse-MLA indexer path (thread 374125).
- **Z.ai** — GLM 5.2 itself.
- **AEON-7** — the vLLM Ultimate DGX Spark container, prior art and reference for the sm12x work.
- **The DGX Spark community** at large, particularly the NVIDIA developer forums — the primary sources for this whole lineage are threads [374125](https://forums.developer.nvidia.com/t/glm-5-2-on-a-4x-gb10-cluster-22-tok-s-decode-256k-ctx-recipe/374125) and [375416](https://forums.developer.nvidia.com/t/followup-mystery-solved-4x-spark-glm-5-2-nfp4-24tp-s-128k-ctx-no-reap/375416); read both.

**Licensing note:** the recipe repo is Apache-2.0 — required and deliberate, since its `launch.sh` derives from CosmicRaisins' Apache-2.0 harness and its mods replicate his Apache-2.0 mod scripts. Respect the license and the NOTICE attributions if you redistribute any of it.

**What we contributed on top:** the unboxing-to-cluster provisioning path and its gotchas (the OOBE ethernet hang, the mixed-cabling/heterogeneous-HCA discovery, the per-node NCCL configuration pattern), the download-stall detection-and-retry procedure, the switch-side RoCE configuration walkthrough in the [companion switch guide](what-is-fabric.md) (distilled from an earlier cluster project), and the original diagnosis and fixes for the Docker build-cache/wrong-commit trap, the `__pycache__` staleness trap, the two-site 576→656 KV-cache shape bug (including the `"auto"` dtype-string pitfall), and the DeepGEMM disable set. Offered back to the community in the same spirit everything above was offered to us.

---

## Closing Notes

Three days from cardboard to conversation. In hindsight, the striking thing is the shape of the time: every one of the four bugs took under thirty minutes to *fix* — and hours to *find*. The commands in this guide are the easy 90%. The value, we hope, is in the other 10%: knowing that file counts lie and timestamps don't; that a successful build isn't necessarily the build you asked for; that the file you patched isn't necessarily the code that runs; that a fixed bug probably has a sibling; and that four identical machines almost never are.

If you hit something this guide doesn't cover, our debugging checklist, distilled:

1. **What's actually in the container?** Query the vLLM commit inside the image; don't trust the build flags you passed.
2. **Is your patched code actually executing?** Check for stale `__pycache__`; verify with the interpreter, not `cat`.
3. **Which shape function is on the stack?** At least two backends need the 576→656 treatment.
4. **Is NCCL on IB or Socket?** `NCCL_DEBUG=INFO`, grep for `NET/`. Halved performance = TCP fallback.
5. **Are the nodes really identical?** `rdma link`, GID tables, kernel versions, interface names. Measure; don't assume.

The model itself, once running, is everything the numbers promise — 200K context, real reasoning traces, speculative decoding that makes a 671B model feel snappy, on hardware that sits on a desk and hums politely. The journey there is just rougher than any recipe can fully smooth. That's not a flaw in the recipes; it's the nature of the frontier. Tony's recipe took days off our journey. Maybe this guide takes a day off yours.

Good luck. Check your timestamps.

---

*Deployed July 11–13, 2026. First inference 05:12 BST, July 13. Written from logs, shell history, and memory — in that order of trustworthiness.*
