# DGX Spark Guides

Practical guides for NVIDIA DGX Spark cluster deployments, written from real experience — actual shell history, actual bugs, actual 5 a.m. first tokens. Not lab-conditions documentation: these are the write-ups we wish had existed when our boxes were still full of foam.

## The Guides

### [GLM 5.2 Quad-Spark Deployment](glm-5.2-quad-spark-deployment.md)

Four DGX Sparks from shrink-wrap to serving **GLM 5.2** — all 671 billion parameters present (quantised, unpruned), 200K context, speculative decoding, on an OpenAI-compatible API. Covers the full journey: out-of-box gotchas, cluster networking, the 405 GB download, the custom vLLM build, and the four genuine bugs we hit along the way — with diagnosis, fixes, and the lessons that generalise. No prior GPU-cluster experience assumed.

### [MikroTik CRS812 RoCE Switch Setup](what-is-fabric.md)

The switch side of an RDMA fabric. If you're connecting more than two Sparks, you need a QSFP switch — and an unconfigured switch is the most convincing impostor of a broken cluster. This standalone guide covers what RDMA asks of a switch, the RouterOS commands that provide it (MTU 9000, PFC, ECN, DSCP), layer-by-layer verification, and a pitfall table where every row is a scar. Written for the CRS812, but the principles apply to any RoCE-capable switch.

## Standing on Shoulders

The GLM deployment is built directly on **[tonyd2wild's QuantTrio recipe](https://github.com/tonyd2wild/GLM-5.2-QuantTrio-200K-4x-DGX-Spark)** — a meticulous, battle-tested foundation that itself credits a whole community of DGX Spark pioneers. What these guides add is the road *around* the recipe: provisioning, networking, and the debugging journey. Full credits are in the deployment guide.

## Contributing

Feedback, corrections, and war stories are very welcome — open an issue or a PR. If one of these guides saved you a day (or cost you one because something has drifted out of date), we'd like to hear about it.
