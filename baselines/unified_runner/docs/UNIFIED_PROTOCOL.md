# Unified comparison protocol

## Main comparison

- Datasets: BloodMNIST, OrganAMNIST, and PathMNIST, using the official 28 x 28 arrays.
- Methods: Local, FedAvg, FedProx, FedProto, FedPAC, FedTGP, FedSOL, FedSA,
  cwFedAvg, FedSimSup, and FedRCA.
- Five simulated hospital clients; three clients participate in every round (60%).
- The selected-client sequence is a deterministic function of seed and round, so every
  method sees the same sequence.
- Dirichlet label splits: alpha 0.5 and 0.1. Every method reuses the same saved
  partition identity for a given dataset, alpha, and seed.
- Seeds: 0 first; 1 and 2 after the seed-0 engineering and convergence audit.
- Shared MedicalCNN, 300 communication rounds, one local epoch, batch size 64.
- SGD, learning rate 0.01, weight decay 5e-4, and multi-step decay by 0.1 at
  rounds 180 and 255.
- Personalized checkpoint selection uses mean client validation Macro-F1. FedRCA's
  deployable global checkpoint uses official validation Macro-F1. Test data is never
  used for tuning or checkpoint selection.
- Report Accuracy, Macro-F1, Balanced Accuracy, AUROC, worst-client performance,
  client standard deviation, communication bytes, and training time.

## FedRCA topology and one-time partition policy

- Every hospital independently fits class-wise pixel PCA+K-means once from all of
  its official local training images. No image or sample is subsampled.
- Cluster assignments are mapped back to original training-row identities and then
  remain fixed. Learned network features are never re-clustered during training.
- Images, PCA models, centers, assignments, and region supports remain client-local;
  only the same model parameter payload as ordinary model-aggregation FL is sent.
- A frozen pre-round global model and the homogeneous client model process the same
  local image. RBF topology KL is computed from the `pool2` and `relu3` spatial maps.
  Classification heads are outside this alignment.
- Fixed class-region supports supply capped, client-normalized sample weights to the
  topology KL. This connects the one-time partition to optimization while preserving
  every sample's complete intermediate feature map.
- K-means adds only one setup cost. Training adds one teacher forward pass and two
  small 49-by-49 relation graphs per batch; inference uses the normal classifier only.

FedSimSup requires partial participation: under 100% participation, its paper-defined
similarity aggregation applies to no client and therefore degenerates. The 60% setting
is consequently part of the shared protocol for all methods.

## Implementation provenance

- FedTGP: official AAAI 2024 repository, `TsingZ0/FedTGP`, commit
  `c77cbbb31eb30d13066cd11f7f4a2e732aeaae24`.
- FedSOL: official CVPR 2024 repository, `Lee-Gihun/FedSOL`, commit
  `ebc3f640f929e59616e07c036c24f906892b510f`.
- FedSA: official AAAI 2025 repository, `lokinko/FedSA`, commit
  `75bf1067fb96d759d9fa6e9d2c6f1b5bd53ee2a6`.
- cwFedAvg: official ICCV 2025 repository, `regulationLee/cwFedAvg`, commit
  `59ed9c1a0e8809520011ec1e1797cd40b2b84e2e`.
- FedSimSup: implemented from Eqs. (3)-(12) of the ICCV 2025 paper. The repository
  URL printed in the paper (`jqLi1626/FedSimSup`) returned HTTP 404 on 2026-09-05.

The upstream repositories are references only and are intentionally not vendored in
this repository. Published numbers are not copied into the result table because this
project uses a new common medical-data protocol.
