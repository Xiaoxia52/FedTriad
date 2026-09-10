# Frozen method and ablation contract

## C1: Parallel-Serial-Local complementary federated learning

For each communication round, active hospitals train three complementary paths:

- **P (population consensus):** all active clients start from the same server
  encoder/classifier, train independently, and return parameters for
  sample-size-weighted FedAvg.
- **S (cross-hospital transfer):** one persistent shared model visits active
  hospitals serially.  A deterministic rotating start position prevents one
  hospital from always being first or last.
- **L (hospital specificity):** every hospital retains one persistent private
  model.  L is trained locally and is never uploaded or averaged.

All enabled branches receive ordinary classification gradients through both
encoder and head. BatchNorm is trainable. There is no feature detachment and no
train-time routing network. Personalized prediction is the probability average
of the available P/S/L branches; global prediction excludes private L.

## C2: Global Class-Risk Calibration

At the validation-selected PSL checkpoint, hospital k computes, from training
examples only:

- one P/S/L negative-log-likelihood sum for every class: `A_k in R^(3 x C)`;
- one sample count per class: `n_k in R^C`.

The server computes `A = sum_k A_k`, `n = sum_k n_k`, and
`R_bc = A_bc / n_c`. Thus the upload is only `4C` scalars per hospital and raw
images/features never leave the hospital. For a new sample, the neutral P/S/L
mean supplies a soft class belief. Expected historical class risk plus current
branch entropy is converted by softmax into three weights with a small floor.
Their weighted probability sum is the final prediction.

The class-agnostic control collapses every branch's table to one overall risk
and repeats it across classes. It therefore uses the same checkpoint, data
boundary, uncertainty term, hyperparameters, and communication form. Any gain
of Global Class-Risk over both Uniform and this control isolates the value of
class-conditioned reliability rather than generic branch reweighting.

## Locked experimental matrix

| Ablation | P | S | L | class-aware risk | New training |
|---|:---:|:---:|:---:|:---:|---:|
| P | yes | no | no | no | reuse |
| P+S | yes | yes | no | no | reuse |
| P+L | yes | no | yes | no | 18 |
| P+S+L Uniform | yes | yes | yes | no | 6 PathMNIST; 12 reuse |
| PSL + class-agnostic risk | yes | yes | yes | no | offline only |
| FedTriad full | yes | yes | yes | yes | offline only |

Training additions total 24 runs: 18 P+L plus 6 missing PathMNIST PSL. The full
method has 18 checkpoints (3 datasets x 2 alphas x 3 seeds). The risk ablations
reuse identical weights and differ only in inference, eliminating optimization
noise from that comparison.

## Reporting contract

Primary personalized result: client-local-test Macro-F1 selected by aggregated
client validation Macro-F1. Report accuracy, balanced accuracy, AUROC, NLL,
worst-client value, between-client standard deviation, and communication/time
cost as supporting evidence. Separately report official-full-test metrics for
the collaborative ensemble, selected without test-label tuning. Never compare
a personalized row directly with a deployable-global baseline row.

The synthetic Dirichlet clients simulate hospitals; the paper must not claim
that these partitions are verified real institutions. The class-risk table is
not differentially private or securely aggregated unless such machinery is
added in future work.
