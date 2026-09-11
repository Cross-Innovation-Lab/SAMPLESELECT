# Anonymous Sample-Conditioned Channel Selection

This repository contains the anonymized core algorithm accompanying a
double-blind submission. It intentionally excludes author names, affiliations,
private infrastructure, machine-specific paths, datasets, checkpoints, logs,
and internal experiment-management code.

## Included method

The implementation follows three stages:

1. **Source representation learning.** Train a backbone and a source-domain
   linear classifier with cross-entropy.
2. **Sample-conditioned selector training.** Freeze the backbone and source
   classifier. Global-average-pool each feature map, score its channels with an
   MLP, and construct a differentiable channel mask using sequential soft
   Gumbel Top-k. Train only the selector with
   `L_sel = L_CE + lambda * L_con`.
3. **Few-shot evaluation.** Freeze the backbone and selector, extract selected
   support/query representations, fit a fresh episodic linear head on support
   labels, and predict the query labels. Query labels are not accepted by the
   evaluation API.

During training, the selector returns a soft mask in `[0,1]^C`. During
evaluation, it uses deterministic binary Top-k selection.

The released method is the **sample-conditioned selector**. Ablation-only
selector variants are deliberately excluded because they are not part of the
proposed method.

## Frozen experiment protocol

The default configuration matches the final formal experiments:

- selector boundary: `C=640`, spatial size `4 x 5`, flattened size `12800`;
- sample descriptor: spatial global average pooling of each feature map;
- scorer: `640 -> 2560 -> 2560 -> 640`, with dropout `0.5`;
- keep ratio `0.7`, hence `k=448`, and fixed Gumbel temperature `0.3`;
- Stage 2 uses sequential soft Gumbel Top-k without replacement and
  `hard=False`;
- Stage 3 uses deterministic hard Top-448 and performs no Gumbel sampling;
- `L_sel = L_CE + 0.02 L_con`, with contrastive temperature `0.07` and
  grouped batches of shape `16 x 8`;
- Stage 2 freezes the backbone, source head, and selector BatchNorm affine
  parameters/statistics; only scorer `Linear` parameters are optimized;
- Stage 3 freezes all representation modules and fits a fresh 5-way linear
  head using support labels only.
- Data-order schedules are fixed across model-initialization repetitions;
  method comparisons reuse identical episode definitions and head seeds.
- Checkpoints contain state dictionaries and inert metadata only and are
  loaded through `weights_only=True`.

The formal runs use training seeds `44`, `16`, and `21`. These are independent
training repetitions; selector hyperparameters are fixed across seeds and are
not re-selected.

## Repository layout

```text
configs/default.json                 Method-level default configuration
src/channel_selection/selector.py   MLP scorer and sequential Gumbel Top-k
src/channel_selection/losses.py     Grouped supervised contrastive loss
src/channel_selection/model.py      Three-stage model wrapper and objectives
src/channel_selection/training.py   Minimal Stage 1/Stage 2 optimization steps
src/channel_selection/episodic.py   Fresh-head few-shot evaluation
src/channel_selection/checkpoint.py Safe state-dict checkpoint I/O
tests/test_core.py                   Shape, freezing, and leakage checks
PROTOCOL.md                          Frozen method and evaluation contract
```

## Installation and smoke test

```bash
python -m pip install -e ".[test]"
pytest -q
python tools/smoke_test.py
```

The included `ReferenceBackbone` is intentionally small and is used only by
the tests. Replace it with the experiment backbone; the required interface is
a feature map of shape `[B,C,H,W]`. The default configuration records the
paper-scale selector boundary (`C=640`, spatial size `4 x 5`) without embedding
any machine-specific location.

## Data interface

The release does not redistribute datasets. A training loader should provide:

- an input tensor;
- a source-domain class label for cross-entropy;
- a foreground/class identity for forming grouped contrastive positives.

Each contrastive anchor must have at least one same-class positive within its
group. Positive pairs may differ in background. At evaluation time, construct
an episode from labeled support examples and unlabeled query inputs, then call
`evaluate_episode`. Its API intentionally has no query-label argument.

For Stage 2, construct the optimizer from the explicit whitelist:

```python
model.set_stage2_trainable()
optimizer = torch.optim.Adam(
    model.stage2_parameters(), lr=1e-3, weight_decay=0.0
)
```

Do not pass all selector parameters blindly: the two selector BatchNorm layers
are part of the forward pass but remain frozen by the formal SelectorOnly
protocol. At paper scale the selector contains `9,846,400` persistent
parameters, of which `9,836,160` Linear parameters are optimized; the remaining
`10,240` BatchNorm affine parameters are frozen.

During Stage 3, `selected_components(...)["indices"]` returns the actual
deterministic selected channel indices with shape `[batch,k]`. It is `None`
during soft-mask Stage 2 training. This supports compact mask export without
introducing any sampling or query-label dependency.

For paired method comparisons, pass the same `head_seed` for the same episode
to every method. This controls fresh-head initialization in addition to using
the same support and query examples.

Use the state-dict-only checkpoint helpers instead of serializing live Python
objects:

```python
from channel_selection import load_weights_checkpoint, save_weights_checkpoint

save_weights_checkpoint("model.pt", model, optimizer=optimizer, scheduler=scheduler)
load_weights_checkpoint("model.pt", model)
```

## Paths and checkpoints

No absolute paths are stored. Configure external locations with environment
variables such as `DATA_ROOT` and `OUTPUT_ROOT`, or supply paths from your own
launcher. Checkpoints and datasets are excluded from version control by
default.

## Release scope and verification

This compact package contains the proposed sample-conditioned method, its
frozen method-level protocol, unit tests, and a CPU smoke test. Ablation-only
selectors, datasets, pretrained weights, experiment outputs, and private
launch infrastructure are not included.

Before distribution, the package was assembled from an explicit file
allowlist. Its tests, smoke test, manifest verification, and anonymity scan
were run again after extraction from the release archive. The scanner can be
re-run with:

```bash
python tools/anonymity_scan.py .
```

## Double-blind notice

Do not add author-identifying repository URLs, badges, acknowledgements,
machine names, usernames, absolute paths, experiment-tracking links, or commit
history before the review process ends.
