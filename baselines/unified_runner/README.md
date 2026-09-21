# FedRCA：统一医学联邦学习实验项目

这个目录只保留一套正式协议和两个 GPU 入口，用于在三个 28 x 28
MedMNIST 数据集上公平比较 11 个方法。

## 项目在做什么

普通集中训练会把所有医院的图像放在一起；联邦学习不移动图像，而是让每家
医院在本地训练，再交换模型或压缩统计量。困难在于不同医院拥有的疾病类别和
样本比例不同，直接平均模型常常会互相干扰。

FedRCA 在每家医院内部、每个类别的全部 28 x 28 训练图像上只做一次像素
PCA+K-means，并把固定区域编号映射回全部原始样本。每轮训练以冻结的轮前全局
编码器为教师、同构客户端编码器为学生，对齐两层中间特征图的 RBF 空间拓扑；
小类别—区域得到温和增权。分类头不参与 RBF。V2.1 的完整全局分类头和不上传的
类别平衡私有头继续保留，分别支持新医院部署和参与医院个性化。

## 正式比较

- 数据集：BloodMNIST、OrganAMNIST、PathMNIST，均为官方 28 x 28 数据。
- 方法：Local、FedAvg、FedProx、FedProto、FedPAC、FedTGP、FedSOL、FedSA、
  cwFedAvg、FedSimSup、FedRCA。
- 每个任务 5 个模拟医院，每轮按统一序列抽取 3 个参加。
- Dirichlet alpha：0.5 与 0.1。
- 300 轮、1 个本地 epoch、batch size 64、同一个 MedicalCNN。
- SGD 初始学习率 0.01，在第 180、255 轮衰减为原来的 0.1。
- FedRCA 的医院内像素 K-means 只初始化一次、使用全部训练样本；训练中不重复
  聚类，也不上传聚类统计。RBF 只作为编码器训练约束，推理不增加计算分支。
- 先跑 seed 0，工程检查通过后再跑 seed 1 和 2。

完整协议、指标和实现来源见 [统一协议](docs/UNIFIED_PROTOCOL.md)。

## 目录

```text
FedRCA/
├─ configs/       唯一正式主配置与消融配置
├─ data/          三个 MedMNIST 原始 npz
├─ docs/          方法和实验协议
├─ fedrca/        训练、聚合、评估代码
├─ runs/          正式实验输出
├─ tests/         单元测试与断点续跑测试
├─ validation/    smoke 配置与验证结果（不进入论文表格）
├─ main.py        单实验/通用命令入口
└─ RUN_MAIN_GPU.py  正式批量 GPU 入口
```

## 安装与检查

```powershell
cd <repository-root>\baselines\unified_runner
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -u .\RUN_MAIN_GPU.py --preview --seeds 0 --job 4 --data-dir <path-to-medmnist>
```

## 正式训练

先跑 seed 0：

```powershell
python -u .\RUN_MAIN_GPU.py --seeds 0 --job 4 --data-dir <path-to-medmnist>
```

明天跑 seed 1 和 2：

```powershell
python -u .\RUN_MAIN_GPU.py --seeds 1 2 --job 4 --data-dir <path-to-medmnist>
```

如果运行被中断，使用原命令加 `--resume`。已完成的输出目录不会被覆盖；续跑时
会核对配置、数据划分和实现摘要，避免把不同版本拼在一起。

`--job 4` 表示同时运行四个独立实验，不会改变任何方法设置。若显存不够，改成
`--job 2` 或 `--job 1` 即可。

项目没有实现差分隐私或安全聚合，不能声称具有形式化隐私保证。Smoke 结果只
用于证明代码链路可运行，不能作为论文性能结果。

`RUN_SERIAL_BASELINES_GPU.py` runs the protocol-matched CWT and FedSeq
controls with the same `--data-dir` option. Results and caches are ignored
experiment artifacts; this release contains no baseline results, checkpoints,
datasets, or upstream source trees. The required runtime dependencies are
listed separately in `requirements.txt` (including SciPy for the FedPAC
solver).
