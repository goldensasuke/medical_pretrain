# Yucca Pretrain: Mask-based Self-Supervised Pretraining for Image Anomaly Detection

這個專案提供一個完整框架，使用 **Mask-based Self-Supervised Learning (MAE-style)** 對影像資料進行預訓練，方便後續用於異常偵測任務。專案整合了 **資料預處理、模型預訓練**，並可透過 **WandB** 追蹤訓練過程。

---

## 功能特色

- 自動判斷影像副檔名 (`.png`, `.jpg`, `.jpeg`, `.tiff`)
- 將影像 resize 成指定大小並存成 `.npy` 和 `.pkl`
- 支援多種 backbone 模型，預設為 `EfficientNet`
- Mask-based self-supervised pretraining，無需標籤資料
- 預訓練後可直接用於異常偵測（重建誤差方法）
- 訓練過程可上傳到 **WandB** 進行可視化監控

---

## 專案結構

```
yucca_pretrain/
│
├─ pyproject.toml           # 專案依賴管理
├─ README.md
├─ src/
│   └─ yucca_pretrain/
│       ├─ __init__.py
│       ├─ data_preprocess.py
│       ├─ models.py
│       └─ pretrain_utils.py
└─ scripts/
    └─ pretrain.py          # CLI 執行入口
```

---

## 安裝

1. 建立虛擬環境

```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

2. 安裝專案及依賴

```bash
pip install -e .
```

---

## 使用方式

```bash
python scripts/pretrain.py \
    --data_dir "./raw_images" \
    --output_dir "./processed" \
    --model_name "efficientnet_b0" \
    --epochs 10 \
    --lr 1e-4 \
    --project_name "my_mae_project"
```

---

## WandB 整合

- 訓練過程自動上傳 loss 和 epoch 資訊
- 可即時監控訓練曲線與模型表現
- 開始前先登入 WandB：

```bash
wandb login
```

---

## 預訓練模型輸出

預訓練完成後會產生：

- `{model_name}_pretrained_mae.pth`：模型權重
- `images.npy`：處理後的影像資料
- `metadata.pkl`：影像對應 metadata

