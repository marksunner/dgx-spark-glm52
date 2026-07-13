# DGX Spark Guides

Practical, honest deployment guides for NVIDIA DGX Spark clusters — written from real hardware, real mistakes, and real recoveries. Everything here was tested on GB10 Grace Blackwell systems (128 GB unified memory, ConnectX-7 QSFP fabric). No cloud, no staging environments, no sanitised happy paths.

→ **[dgx-spark](https://github.com/marksunner/dgx-spark)** is the wider workshop: inference engines, benchmarks, model comparisons, and things we've learned. This repo is the deployment-and-ops subset — the guides you reach for when it's time to build.

---

## Cluster Deployments

### ✨ GLM 5.2 — 4× DGX Spark
Z.ai's 671B-parameter reasoning model across four nodes. All 256 experts active, 200K context, MTP speculative decoding — the complete journey from unboxing to first inference.
→ **[Read the guide](glm-5.2-quad-spark-deployment.md)**

### 🔧 What Is Fabric? — MikroTik CRS812 RoCE Setup
Building a lossless QSFP fabric for any multi-node DGX Spark cluster. RouterOS config, PFC/ECN, jumbo frames, and the pitfalls that cost us weeks on an earlier cluster.
→ **[Read the guide](what-is-fabric.md)**

### 💪 Qwen 122B — Single-Spark Agent Stack
A complete autonomous AI agent on one DGX Spark: Qwen 3.5 122B (hybrid INT4+FP8) + Hermes + Honcho + monitoring. 41–47 tok/s.
→ **[Read the guide](https://github.com/marksunner/dgx-spark-single-stack)**

### 🕯️ Step 3.7 Flash — Single Spark
198B MoE, llama.cpp Q4_K_S GGUF, 27 tok/s, 128K context with vision. The simplest path to a big model on one box.
→ **[Read the guide](https://github.com/marksunner/dgx-spark-step37-flash)**

### ⚡ Step 3.7 Flash — Dual Spark
Same model, full 262K native context. vLLM TP=2 over RoCE RDMA, 18.5 tok/s. Docker device permissions, NCCL config, and everything that didn't work.
→ **[Read the guide](https://github.com/marksunner/dgx-spark-step37-dual)**

### 🐋 DeepSeek V4 Flash — Dual Spark Benchmark
284B MoE across two Sparks. 12.4 tok/s (this benchmark is superseded by Tony's 1M-context recipe — kept up for historical reference).
→ **[Read the guide](https://github.com/marksunner/dgx-spark-vllm-tp-benchmark)**

---

## Tools

**[`preflight.sh`](preflight.sh)** — Zero-dependency script that verifies RoCE GID indices match your launch configuration before every cluster launch. The GID table shifts after a reboot; this catches it. Run `./preflight.sh --fix && ./launch-castle.sh` and sleep easy.

---

## Credits

The GLM 5.2 guide is built on **[tonyd2wild's QuantTrio recipe](https://github.com/tonyd2wild/GLM-5.2-QuantTrio-200K-4x-DGX-Spark)**. That recipe, in turn, stands on the work of **CosmicRaisins** (Triton kernels), **Zatz** (QuantTrio checkpoint proving), **back199640** (performance tuning), **ciprianveg** (baked-mod scripts), **eugr** (container build harness), and **QuantTrio** (the checkpoint itself). Full acknowledgements in the guide.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
