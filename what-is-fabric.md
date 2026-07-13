# What Is Fabric?

## How to Build a Lossless RoCE Fabric with a MikroTik CRS812 — for DGX Spark Clusters and Beyond

*An Ethernet switch fresh from the box will pass your pings, light its link LEDs, and quietly sabotage RDMA. This guide starts from first principles — what a fabric actually is, why you need one, what RDMA asks of a switch — then turns a MikroTik CRS812 into a proper, lossless, battle-tested fabric for a multi-node DGX Spark cluster. About ten RouterOS commands, but the principles travel to any switch you own.*

---

## Why You're Here

If you have four or more DGX Sparks, you need a switch. For two nodes, a single back-to-back QSFP cable is all you need. For three nodes, a ring topology using both QSFP ports per node often works — details below. Four or more nodes need a crossbar switch for any-to-any connectivity.

The switch we use — and the one this guide configures — is the **MikroTik CRS812**. It is not the only switch that works, but it's one of the most popular in the DGX Spark community, and it's the one we've battle-tested. Other QSFP switches with RoCE, PFC, and ECN support should work just as well: the principles here (correct port speed, jumbo frames, priority flow control, congestion notification, DSCP classification) are universal — adapt the vendor-specific commands to your platform.

A word on provenance, because it's why this guide exists: on an earlier deployment (a multi-node Qwen 397B experiment on the same family of switch), a misconfigured switch cost us weeks of believing "RDMA just doesn't work through the switch." It did work. The switch had simply never been told what RDMA needs. This guide is the distillation of that lesson.

**Who this is for:** someone with a handful of RDMA-capable nodes, a QSFP switch, and no assumed background in switch configuration or RDMA networking. Every concept is explained before it's used.

## The Vocabulary You'll Need (Two Minutes)

Four terms carry this whole guide:

- **RDMA (Remote Direct Memory Access):** network transfers that go NIC-to-memory without the CPU touching each packet. It's what makes multi-node GPU workloads fast — and it has opinions about the network it runs on.
- **RoCE (RDMA over Converged Ethernet, v2):** RDMA carried over ordinary Ethernet hardware. The DGX Spark's ConnectX-7 NIC speaks RoCE v2 over its QSFP port at 200 Gbps.
- **NCCL (NVIDIA Collective Communications Library, "nickel"):** the library GPU frameworks use to synchronize data between nodes. NCCL rides on RDMA when it can — and *silently* falls back to plain TCP when it can't, which typically halves your performance without a single error message. Much of this guide exists to keep NCCL on the fast path.
- **The GID table:** RoCE addresses endpoints by entries in a per-port GID table — roughly, the RDMA-layer address book, each entry binding a protocol version to an interface address. NCCL must be told *which entry* to use, per node, and the right answer genuinely differs between machines. We'll show you how to look it up.

## Naming and Placeholder Conventions

Because this guide is public, all IP addresses are placeholders — the only real address in it is `192.168.88.1`, RouterOS's factory default. Substitute your own values:

| Placeholder | Meaning |
|---|---|
| `${QSFP_1}` … `${QSFP_4}` | Each node's static IP on the QSFP cluster fabric (pick any private subnet) |
| `<port>` | A switch fabric port name (e.g. `qsfp-dd-1` — run `/interface ethernet print` to see yours) |
| `<fabric-interface>` | A node's fabric NIC name in Linux (e.g. `enP2p1s0f1np1` on a Spark) |

Our worked example is a four-node DGX Spark cluster — nodes **S1 through S4** — but nothing here depends on that count.

## Why a Switch at All?

With **two** nodes, you don't need one: a single QSFP cable back-to-back gives you a direct 200 Gbps RDMA link, and it's genuinely the simplest possible fabric. If you're building a two-node cluster, stop reading and cable the machines together.

With **three** nodes, you can probably get away without a switch too — each DGX Spark has two QSFP ports, so a ring (A→B→C→A) provides a path between every pair. This is a configuration people in the community use successfully, though we haven't tested it ourselves. Ring topology does introduce a hop for traffic between nodes that aren't directly cabled, which adds a few microseconds of latency, but NCCL generally handles this fine at this scale.

With **four or more**, the geometry changes. Distributed inference splits a model across nodes, and after almost every layer the nodes must recombine their partial results (an "all-reduce") — so the traffic is *any-to-any*: every node talks to every other node, constantly. Each Spark has effectively one QSFP uplink's worth of fabric, so you can't wire four machines into a full mesh; pairs of back-to-back links would leave some node pairs with no direct path at all. A switch solves this the boring, correct way: every node gets one cable to the switch, and the switch provides a full-bandwidth path between any pair. That's its entire job — a crossbar with a seat for every node.

The catch is that a switch is also a new place for things to go wrong *invisibly*. An Ethernet switch fresh from the box will happily pass your pings, show green link lights, and still sabotage RDMA — because out of the box it's configured for ordinary Ethernet, and RDMA is not ordinary Ethernet.

## What RDMA Asks of a Switch

Normal Ethernet handles congestion by dropping packets and letting TCP retransmit. Web traffic barely notices. RDMA *hates* drops: a lost packet doesn't retransmit gracefully — the transfer stalls, and NCCL sits waiting for data that will never arrive. The symptom is a **hang**, not an error message, which is the worst kind of symptom to debug. So the fabric needs to be as close to lossless as you can make it. Concretely, the switch must provide:

1. **Correct port speed.** Actually negotiated to what the NICs expect — don't assume auto-negotiation got it right (ours didn't; see pitfalls).
2. **Jumbo frames (MTU 9000)** on every fabric port — matching the NICs.
3. **PFC (Priority Flow Control).** When a buffer fills, instead of dropping packets, the switch sends the transmitter a "pause" for that traffic class. A brief pause is recoverable; a drop can hang NCCL.
4. **ECN (Explicit Congestion Notification).** The early-warning system: as queues *start* to build, the switch marks passing packets, and the endpoints slow down before PFC's emergency brake is ever needed.
5. **DSCP classification.** How the switch knows *which* packets are RDMA. Packets carry a DSCP label (a 6-bit priority tag in the IP header); the switch maps labels to traffic classes and applies the protections above to the right class.

If you want a mental model: **DSCP is the lane marking** that says which lane a packet drives in, **ECN is the "congestion ahead, slow down" sign**, and **PFC is the emergency brake**. You're about to paint the lanes, install the sign, and wire up the brake.

Do you *strictly* need PFC and ECN for a small inference cluster? On a lightly loaded fabric you might get away with speed + MTU alone — some clusters do. But when the lossless config is missing, it bites as an *intermittent hang under load*: everything verifies fine, light testing passes, and then a real inference workload wedges the cluster twenty minutes in. We chased exactly that ghost for weeks on the earlier project. The full configuration is about ten commands. Do it once.

## Our Hardware

| Component | Details |
|---|---|
| Switch | MikroTik CRS812 (QSFP-DD ports, 200 Gbps per port) |
| Switch OS | RouterOS 7.19.x |
| Node NICs | NVIDIA ConnectX-7 (in 4 × DGX Spark) |
| Measured RDMA bandwidth through the switch | ~109 Gb/s per link (`ib_write_bw`) |
| Measured RDMA latency through the switch | ~3 μs (`ib_write_lat`) |

The commands below are RouterOS. Port names vary by CRS812 variant (`qsfp-dd-1`, `qsfp56-dd-1-1`, and similar) — run `/interface ethernet print` on your switch and substitute your own names wherever you see `<port>`. Everything else translates directly; the QoS subsystem is the same across the CRS8xx family.

## Step 0: Safety First

Three facts that make this a low-drama procedure:

- **Steps 1–4 below are inert.** They only *define* profiles, maps, and rules. Nothing changes on the wire until the final `qos-hw-offloading=yes`.
- **The rollback is one command:** `/interface/ethernet/switch set switch1 qos-hw-offloading=no` returns the switch to its previous behaviour.
- **Back up first anyway:** `/export file=before-roce` saves the full config as a file you can download from the switch's Files menu.

Getting a console: a factory RouterOS device answers on `192.168.88.1` (its default management subnet) via SSH, the web UI, or WinBox. One useful trick if the switch's management address isn't reachable from your normal network: RouterOS bridges management across its ports by default, so you can reach it *through the fabric itself* — give one node a temporary second address on the management subnet (`sudo ip addr add 192.168.88.10/24 dev <fabric-interface>`) and SSH to the switch from there. RouterOS also has a REST API, so everything below can be driven with `curl` if you prefer scripts to consoles.

## Step 1: Check the Firmware

The QoS features below (PFC, ECN, per-class queues) need **RouterOS 7.17 or later**; we ran 7.19.x. If you're on 7.23+, RouterOS adds automatic DSCP mapping and "lossless traffic class" shortcuts that simplify Step 4 — the manual version below still works everywhere. Check with `/system resource print`, and upgrade via `/system package update` if you're below 7.17.

## Step 2: Verify Port Speed — Don't Trust Auto-Negotiation

The pitfall that headlined our earlier debugging marathon: **the switch's QSFP ports came up at a lower speed than the NICs expected**, and nothing complained — traffic flowed, just far below expectations. Depending on the CRS812 variant and cabling, a physical port may also present as *multiple* virtual interfaces (breakout lanes), each of which needs its speed set individually.

Check what each fabric port actually negotiated:

```routeros
/interface ethernet monitor <port> once
```

The rate should match your cabling (200G, or 2×100G lanes on breakout configurations). If it doesn't, set the speed explicitly on the port (`/interface ethernet set <port> speed=...`) rather than leaving it to auto-negotiation, and re-check. Cross-check from the node side with `ethtool <fabric-interface> | grep Speed` — on a Spark it should report `200000Mb/s`.

## Step 3: MTU 9000 on Every Fabric Port

MikroTik ships QSFP ports at an MTU of 1584. That's the single most common silent killer: the nodes send 9000-byte jumbo frames, the switch fragments or drops them, and everything "works" at a crawl. Set both the L2 and L3 MTU generously on every fabric port:

```routeros
/interface ethernet set <port> mtu=9000 l2mtu=9216
```

(The node side needs MTU 9000 on its fabric interface too — both ends of every link.)

Verify end-to-end with a don't-fragment jumbo ping from one node to another (`-M do` forbids fragmentation; 8972 is 9000 minus IP/ICMP headers):

```bash
ping -c 3 -M do -s 8972 ${QSFP_2}
```

If the normal ping passes and this one fails, some port in the path is still at the default MTU.

## Step 4: The Lossless Configuration — QoS, PFC, ECN

Now the lane markings, the sign, and the brake. The standard RoCE convention uses two DSCP labels: **26** for the RDMA data itself, and **48** for **CNPs** (Congestion Notification Packets — the tiny "please slow down" messages that ECN generates back toward the sender; they must never queue behind the very congestion they're reporting, so they get strict priority).

```routeros
# 1. QoS profiles: label DSCP 26 as "roce" (traffic class 3),
#    DSCP 48 as "cnp" (traffic class 6)
/interface/ethernet/switch/qos/profile
add dscp=26 name=roce traffic-class=3
add dscp=48 name=cnp traffic-class=6

# 2. Map incoming packets to those profiles by DSCP
#    (manual on 7.19; automatic on 7.23+)
/interface/ethernet/switch/qos/map/ip
add dscp=26 profile=roce
add dscp=48 profile=cnp

# 3. Queue scheduling: ECN marking on the RoCE queue,
#    strict priority for CNPs
/interface/ethernet/switch/qos/tx-manager/queue
set 1 schedule=high-priority-group weight=1
set 3 schedule=high-priority-group weight=1 ecn=yes
set 6 schedule=strict-priority

# 4. A PFC profile for the RoCE traffic class (pause instead of drop,
#    in both directions)
/interface/ethernet/switch/qos/priority-flow-control
add name=pfc-roce traffic-class=3 rx=yes tx=yes

# 5. Apply to every fabric port: trust the DSCP labels packets arrive
#    with, attach the PFC profile, and give the RoCE queue an egress
#    rate (RouterOS requires one alongside PFC — set it to line rate)
/interface/ethernet/switch/qos/port
set <port-1> trust-l3=trust pfc=pfc-roce egress-rate-queue3=200G
set <port-2> trust-l3=trust pfc=pfc-roce egress-rate-queue3=200G
set <port-3> trust-l3=trust pfc=pfc-roce egress-rate-queue3=200G
set <port-4> trust-l3=trust pfc=pfc-roce egress-rate-queue3=200G

# 6. THE BIG ONE — nothing above takes effect until this:
/interface/ethernet/switch
set switch1 qos-hw-offloading=yes
```

Confirm the profiles took effect (they should show as hardware-offloaded and active):

```routeros
/interface/ethernet/switch/qos/profile print detail
```

## Step 5: Tell NCCL to Label Its Traffic

The switch now protects packets labelled DSCP 26 — but it can only protect what's labelled, and by default NCCL sends its RDMA traffic *unlabelled* (DSCP 0, best-effort lane, no PFC, no ECN). Everything will still work under light load, then hang under heavy load, which is precisely the trap. Close the loop by adding two variables to the environment of every NCCL-using process on every node — exported in the shell, or, if your workload runs in containers (most do), passed as `docker run` flags:

```bash
-e NCCL_IB_TC=106 \   # the traffic-class byte: DSCP 26, shifted left 2 bits,
                      # plus the ECN-capable flag → (26 << 2) | 2 = 106
-e NCCL_IB_SL=3 \     # service level, matching the switch's traffic class 3
```

(If you're following our [GLM 5.2 Quad-Spark Deployment](glm-5.2-quad-spark-deployment.md), note that its launch configuration doesn't include these by default, because the upstream recipe it follows doesn't use them — add them to the per-node NCCL environment block once you've enabled this guide's PFC/ECN configuration.)

## Interlude: Discover Each Node's RDMA Identity

Before the verification step, you need two facts *per node*: which **HCA** (RDMA device — `mlx5_0` or `mlx5_1` on a Spark, depending on which physical QSFP port the cable is in) it uses, and which **GID index** addresses its fabric IP. Do not assume the answers match across nodes — on our own "identical" machines they didn't, in both respects. On each node:

```bash
# Which RDMA devices exist and which are up
rdma link

# Dump the GID table with types (substitute your HCA: mlx5_0 or mlx5_1)
for i in $(seq 0 15); do
    gid=$(cat /sys/class/infiniband/mlx5_1/ports/1/gids/$i 2>/dev/null || echo "none")
    type=$(cat /sys/class/infiniband/mlx5_1/ports/1/gid_attrs/types/$i 2>/dev/null || echo "none")
    echo "GID $i: $gid  ($type)"
done
```

You're looking for the entry that is **type RoCE v2** and **contains your fabric IP** — IPv4 addresses appear embedded in the GID's IPv6-mapped form; the last four bytes of the GID are your IP in hex. That entry's index is that node's `NCCL_IB_GID_INDEX`. (If `show_gids` is available on your system, it prints this table in one friendly shot.)

Two habits that keep this table stable: configure fabric IPs *persistently* (a NetworkManager connection with autoconnect, not an ad-hoc `ip addr add`) — otherwise a link-local `169.254.x.x` address can squat the port after a reboot and shift the GID table out from under your configuration — and re-run this discovery after any reboot or recabling.

## Step 6: Verify the Fabric, Layer by Layer

Verify in order, so a failure points at exactly one layer:

**1. Plain reachability** — IP works at all:

```bash
ping -c 3 ${QSFP_2}
```

**2. Jumbo frames** — MTU 9000 passes end to end:

```bash
ping -c 3 -M do -s 8972 ${QSFP_2}
```

**3. Link speed** — on each node:

```bash
ethtool <fabric-interface> | grep Speed    # expect 200000Mb/s
```

**4. Raw RDMA bandwidth and latency** — the `perftest` tools (`ib_write_bw`, `ib_write_lat`) exercise the actual RDMA path, no NCCL involved. Run each test as a pair — a server on one node, a client on another. Substitute each node's HCA from the discovery interlude above:

```bash
# On S2 (server side):
ib_write_bw -d mlx5_1 --report_gbits -R -D 5

# On S1 (client side):
ib_write_bw -d mlx5_1 --report_gbits -R -D 5 ${QSFP_2}
```

(The `-R` flag uses the RDMA connection manager to resolve addressing — the same mechanism NCCL uses, so it's a faithful rehearsal.) Through this switch we measured **~109 Gb/s**; anything in that neighbourhood is healthy. Single-digit Gb/s means an MTU or port-speed problem — go back to Steps 2–3. Then the same dance with `ib_write_lat`: expect single-digit **microseconds** (~3 μs for us) through the switch.

**5. NCCL itself** — the definitive test is your first real multi-node launch with `NCCL_DEBUG=INFO` in the environment. In the logs, this is victory:

```
NCCL INFO NET/IB : Using [0]mlx5_1:1/RoCE
```

and this is defeat: any mention of `NET/Socket` — NCCL couldn't use RDMA and fell back to plain TCP. Everything will still *work*, at roughly half speed, with no error anywhere; the pitfall table below is your suspect list, starting with the container flags RDMA needs (`--device /dev/infiniband`, `--cap-add IPC_LOCK`, `--ulimit memlock=-1:-1`) and the per-node HCA/GID settings. (If you'd rather test NCCL before committing to a full workload launch, NVIDIA's `nccl-tests` suite — `all_reduce_perf` in particular — runs the same collectives standalone.)

## Common Pitfalls

Every row of this table is a scar, from one cluster or the other:

| Pitfall | Symptom | Fix |
|---|---|---|
| Switch port MTU left at default (1584) | Plain ping works, jumbo ping fails; transfers crawl; RDMA connects intermittently | Step 3 — MTU 9000/L2MTU 9216 on every fabric port |
| Port speed mis-negotiated | Links up, everything "works," bandwidth a fraction of expected | Step 2 — set speeds explicitly, verify with `monitor` and `ethtool` |
| QoS defined but `qos-hw-offloading` never enabled | All of Step 4 silently inert; behaves exactly like an unconfigured switch | The last command of Step 4 — it's the master switch |
| NCCL traffic unlabelled (no `NCCL_IB_TC`) | Fine when idle, hangs under real inference load — the classic "works in testing" ghost | Step 5 — label the traffic so the lossless class applies |
| One GID index hard-coded across all nodes | NCCL works from some nodes and fails from others | Per-node GID discovery (see the interlude); the right index genuinely differs per node |
| Link-local address squatting the fabric port after reboot | A previously working node stops RDMA-ing after a reboot; GID table shifted | Persist fabric IPs properly (NetworkManager, not ad-hoc `ip addr add`) and re-verify GIDs after any reboot |
| PFC storm | Pause frames cascade and throttle the fabric (rare, but PFC's known dark side) | ECN (Step 4) prevents most congestion before PFC engages; watch the switch's pause-frame counters if throughput dips |
| `ib_write_bw` passes but NCCL still uses `NET/Socket` | RDMA layer is fine; NCCL's device/GID selection is wrong | Set per-node `NCCL_IB_HCA` / `NCCL_IB_GID_INDEX` / `NCCL_SOCKET_IFNAME` from the discovery interlude; check the RDMA container flags in Step 6 |

## The Moral

A switch fails politely. It never says "I am dropping your RDMA traffic because nobody configured PFC" — it passes your pings, lights its LEDs, and lets NCCL hang in silence. On our earlier cluster, the gap between "the switch passes traffic" and "the switch is configured for RDMA" swallowed weeks; the fix, once understood, was one methodical morning. Configure it deliberately, verify it layer by layer, and it disappears from your life — which is the highest compliment network equipment can earn.

---

*This guide was written as part of our [GLM 5.2 Quad-Spark Deployment](glm-5.2-quad-spark-deployment.md) but applies to any multi-node DGX Spark cluster — or, with the commands adapted, any RoCE fabric you're building on a switch that's never been told what RDMA needs.*
