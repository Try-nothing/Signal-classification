# 🏆 2025 iFLYTEK AI 开发者大赛  
## 通信信号自动调制识别挑战赛  
### 我的参赛手记 & 最终成绩 **Top-4**  
> 最终得分：0.74627 | 排名：No.4  

---

## 📌 赛事概览
| 项目 | 内容 |
|---|---|
| **主办方** | 科大讯飞 × 中国信息协会 |
| **赛道** | 通信信号自动调制识别 |
| **任务** | 基于 IQ 信号的多类别调制方式分类 |
| **数据规模** | 22 万样本 × 11 类 × 39 种 SNR |
| **评测指标** | 准确率 Accuracy |
| **赛程** | 07-01 ~ 08-23（单轮赛制） |

---

## 🎯 任务拆解
本次赛题分为两个数据集，要求 **一套模型** 同时解决：

| 数据集 | 样本量 | 调制类别 | 信噪比 | 数据格式 | 挑战点 |
|---|---|---|---|---|---|
| **Dataset-1** | 220 000 | 11 类（8 数字 + 3 模拟） | ‑20 dB ~ 18 dB | `2×128` IQ | 大类别、宽 SNR 跨度 |
| **Dataset-2** | 1 500 | 3 类（QPSK/16QAM/64QAM） | 30 dB | `1×101` 复数 | 跨域泛化 |

> 统一模型得分权重 **70 % Dataset-1 + 30 % Dataset-2**，  
> 若分别建模则权重降至 70 % + 20 %，因此 **“一套模型通用”** 是拿分关键。

---

## 🛠️ 我的解决方案
### 1. 核心思路
- **时序建模**：将 IQ 两路 128 点信号视为多元时间序列  
- **注意力机制**：用 **LSTM + Attention** 捕捉长程依赖  
- **五折交叉验证**：充分利用 17.6 万训练样本，抑制过拟合  

### 2. 技术栈
| 模块 | 选型 |
|---|---|
| 深度学习框架 | [tsai](https://github.com/timeseriesAI/tsai) + PyTorch |
| 模型 | `LSTMAttention` |
| 数据增强 | `TSStandardize` |
| 优化器 | Ranger + One-Cycle Policy |
| 训练策略 | 5-Fold CV，每折 25 epochs，lr_max = 5e-3 |

### 3. 关键代码片段
```python
from tsai.all import *

# 五折交叉验证
for fold, (trn_idx, val_idx) in enumerate(KFold(5).split(X)):
    X_trn, y_trn = X[trn_idx],  y[trn_idx]-1
    X_val, y_val = X[val_idx],  y[val_idx]-1

    tfms  = [None, [Categorize()]]
    dsets = TSDatasets([X_trn, X_val], [y_trn, y_val], tfms=tfms, inplace=True)
    dls   = TSDataLoaders.from_dsets(dsets.train, dsets.valid,
                                     bs=[64,64], batch_tfms=[TSStandardize()])

    model = LSTMAttention(c_in=2, c_out=11, seq_len=128)
    learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
    learn.fit_one_cycle(25, lr_max=5e-3)
    learn.save_all(path=f"./fold_{fold}")
```

---

## 📊 结果与复盘
| 阶段 | 得分 | 备注 |
|---|---|---|
| 公榜 | 0.74627 | 单模即 Top-4 |
| 私榜 | 未公布 | 官方统一 8-29 公布 |

> **Gap to Top-3 ≈ 0.8 %**，遗憾止步领奖台。  
> 复盘原因：  
> 1. 未对 Dataset-2 做特定 Finetune，泛化略弱；  
> 2. 未引入额外数据增强（频域 Mask、SNR Mixup）；  
> 3. 单模型上限，可尝试多模型融合 + 蒸馏。

---
