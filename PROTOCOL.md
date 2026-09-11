# Frozen Protocol

This file records the method-level settings required to reproduce the released
sample-conditioned channel selector. Dataset paths, private manifests, and
machine-specific launch infrastructure are intentionally not distributed.

## Stage 1: source representation learning

- Backbone output: `[B,640,4,5]`
- Flattened source representation: `12800`
- Source classifier: `Linear(12800,25)`
- Optimizer: Adam, learning rate `0.001`, weight decay `0`
- Batch size: `128`
- Schedule: `30` epochs, `124` steps per epoch, `3720` total steps

Each training seed (`44`, `16`, or `21`) produces its own Stage 1 checkpoint.
Within one seed, the full-representation evaluation and selector training must
use that same checkpoint. The data-order schedule is held fixed across these
model-initialization repetitions.

## Stage 2: selector-only training

- Backbone and source classifier: frozen and in evaluation mode
- Selector BatchNorm affine parameters and running statistics: frozen
- Trainable modules: the three scorer Linear layers only
- Descriptor: `GAP(F_i)` with shape `[B,640]`; it is detached from the backbone
- Scorer: `640 -> 2560 -> 2560 -> 640`
- Dropout: `0.5`
- Keep ratio: `0.7`; selected channels: `448`
- Sequential soft Gumbel Top-k: `hard=False`, without replacement
- Temperature: fixed `0.3`, with no annealing
- Loss: `L_CE + 0.02 L_con`
- Contrastive temperature: `0.07`
- Group shape: `[16,8]`
- Optimizer: Adam, learning rate `0.001`, weight decay `0`
- Schedule: `30` epochs, `124` steps per epoch, `3720` total steps

Positive pairs share the foreground class and may have different backgrounds.

## Stage 3: few-shot evaluation

- Backbone and selector: frozen
- Mask: deterministic per-sample hard Top-448; no Gumbel sampling
- Head: a fresh `Linear(12800,5)` for every episode
- Head fitting: support labels only, `100` steps, inner batch size `4`
- Head optimizer: SGD, learning rate `0.01`, momentum `0.9`, dampening `0.9`,
  weight decay `0.001`
- Evaluation: 5-way, 1-shot and 5-shot, 10 queries per class
- Query labels are used only for scoring predictions and are not accepted by
  the prediction API.
- A paired comparison must use the same episode definition and fresh-head seed
  for every compared method.

The public implementation exposes deterministic hard-mask indices for compact
analysis/export. Dataset-specific episode construction and private manifests
are outside the scope of this anonymous core release.

## Scope

This package contains only the proposed sample-conditioned method.
Ablation-only selection baselines are not included.

## Checkpoints

Release checkpoints contain model/optimizer/scheduler state dictionaries and
plain metadata only. Live Python objects are never serialized, and loading is
performed with `weights_only=True`.
