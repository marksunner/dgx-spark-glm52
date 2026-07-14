# GLM 5.2 on 4× DGX Spark

*The complete journey from unboxing to first inference — 671B parameters, all 256 experts, 200K context, MTP speculative decoding, ~26 tok/s. Built on [tonyd2wild's QuantTrio recipe](https://github.com/tonyd2wild/GLM-5.2-QuantTrio-200K-4x-DGX-Spark). Three days, four bugs, one UPS incident, and a haiku.*

→ For all DGX Spark work (other models, benchmarks, comparisons), see **[dgx-spark](https://github.com/marksunner/dgx-spark)**.

---

## What's Here

### ✨ [GLM 5.2 Quad-Spark Deployment Guide](glm-5.2-quad-spark-deployment.md)
The main event. From shrink-wrap to serving: OOBE setup, cluster networking, custom vLLM build, the four bugs we found and fixed, and the power-loss recovery that taught us about GID table instability. Written so that someone with four DGX Sparks and enthusiasm — but no prior cluster experience — can follow it end to end.

### 🔧 [What Is Fabric?](what-is-fabric.md)
Companion guide: building a lossless RoCE fabric with a MikroTik CRS812 QSFP switch. Starts from first principles (what is RDMA? why does it hate dropped packets?), covers the RouterOS config step by step, and ends with verification and a pitfall table where every row is a scar. Standalone — usable for any multi-Spark cluster, not just GLM.

### 🔒 [GLM 5.2 Stability Fixes](glm-5.2-stability-fixes.md)
Field report: diagnosing and fixing a reproducible NCCL deadlock with MTP speculative decoding. If your cluster serves 3–5 requests then silently hangs at 96% GPU — this is the guide. Covers root cause (cudagraph rank desync + async-scheduling MTP races + NCCL NVLS), a copy-paste fix, and a step-by-step recovery ladder from 15 → 23 tok/s stable.

### 🛡️ [preflight.sh](preflight.sh)
Pre-launch GID index validator. The RoCE GID table shifts after a reboot; this script catches it before NCCL fails silently. Run `./preflight.sh --fix && ./launch-castle.sh` after any power cycle.

---

## Credits

This work stands on **[tonyd2wild's QuantTrio recipe](https://github.com/tonyd2wild/GLM-5.2-QuantTrio-200K-4x-DGX-Spark)** — the build commands, serve configuration, and performance targets are theirs. Tony's recipe credits the wider community (CosmicRaisins, Zatz, back199640, ciprianveg, eugr, QuantTrio) — full acknowledgements in the guide.

## License

Apache 2.0 — see [LICENSE](LICENSE).
