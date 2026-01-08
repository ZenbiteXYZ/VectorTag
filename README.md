# 🏷️ VectorTag: Automatic Image Tagging System

A deep learning system for image tags generation. Uses ResNET-18 backbone.

## 🎯 Overview

**VectorTag** automatically tags images with semantic labels (e.g., "building", "food", "person", "nature"). The system:

- **Multi-label classification:** Each image can have multiple tags simultaneously.
- **Interpretable predictions:** Uses Grad-CAM to visualize which image regions influenced each tag.
- **Interactive UI:** Streamlit-based web interface for inference and exploration.
- **Production-ready:** Docker containerization for easy deployment.

### Key Features

- **ResNet-18** backbone pretrained on ImageNet
- **BCEWithLogitsLoss** with class weighting to handle imbalanced datasets
- **Grad-CAM visualization** for model interpretability
- **Data augmentation** (rotation, flips, color jitter)
- **LR Scheduler** with early stopping to prevent overfitting
- **Streamlit UI** for interactive inference
- **Docker deployment** ready

---

## 📊 Project Structure

```
VectorTag/
├── src/
│   ├── core/
│   │   └── config.py                 # Central configuration (paths, hyperparams)
│   ├── data/
│   │   ├── tagged_dataset.py         # PyTorch Dataset class
│   │   ├── loaders.py                # DataLoader creation with transforms
│   │   └── taxonomy.py               # Tag synonyms and hierarchies
│   ├── models/
│   │   └── baseline.py               # ResNet-18 model definition
│   ├── scripts/
│   │   └── train.py                  # Training loop with auto-plotting
│   ├── ui/
│   │   ├── app.py                    # Main Streamlit app
│   │   ├── modes/
│   │   │   ├── base.py               # Abstract inference mode
│   │   │   └── standard.py           # Standard inference mode
│   │   └── components/               # Reusable UI components
│   └── utils/
│       ├── gradcam.py                # Grad-CAM heatmap generation
│       └── plotting.py               # Training visualization
├── models/
│   └── standard/
│       ├── weights/                  # Saved model weights (.pth)
│       └── classes/                  # Class definitions (.json)
├── assets/
│   ├── exp_00X_*.png                 # Training curves from experiments
│   └── comparison_*.png              # Grad-CAM comparisons
├── data/
│   └── raw/
│       └── various_tagged_images/    # Dataset images + metadata.csv
├── experiments.md                    # Detailed experiment logs
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Container definition
└── README.md                         # This file
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository (choose one method below)

# HTTPS
git clone https://github.com/ZenbiteXYZ/VectorTag.git

# Or SSH
git clone git@github.com:ZenbiteXYZ/VectorTag.git

# Navigate to directory
cd VectorTag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

Download the [Kaggle Various Tagged Images dataset](https://www.kaggle.com/datasets/greg115/various-tagged-images) and extract to:

```
data/raw/various_tagged_images/
├── metadata.csv
├── image_1.jpg
├── image_2.jpg
└── ...
```

### 3. Launch UI

```bash
streamlit run src/ui/app.py
```

Then navigate to `http://localhost:8501` in your browser.

> **Note:** Pre-trained model weights are already included in `models/standard/weights/`. For retraining with custom settings, see [Training Your Own Model](#-training-your-own-model) section below.

---

## 📈 Experimental Results

### Experiment 005: **BCEWithLogitsLoss + Class Weights** ⭐ Current Best

**Configuration:**
- **Loss:** BCEWithLogitsLoss with `pos_weight` (balanced classes)
- **Data:** 200K samples, Top-150 tags, stratified split
- **Training:** 12 epochs, LR Scheduler, Weight Decay=1e-5

**Results:**

| Epoch | Train Loss | Val Loss | Notes |
|-------|------------|----------|-------|
| 1     | 0.2569     | 0.2028   | High loss due to pos_weight |
| 2     | 0.2107     | 0.1925   | Quick descent |
| 7     | 0.1726     | **0.1835** | **Best validation point** |
| 12    | 0.1372     | 0.1932   | Training continues |

![Learning Curve Exp 005](assets/exp_005_weighted.png)

**Key Insights:**
1. ✅ **Sharp Grad-CAM:** Model focuses on relevant image regions without noise.
2. ✅ **High Confidence:** Predictions reach 70%+ for confident tags.
3. ⚠️ **Overfitting starts:** After epoch 7, validation loss increases (expected with imbalanced data).
4. ✅ **Best generalization:** Compared to other loss functions (Focal Loss performed worse).

### Visual Analysis

**Building Tag (Grad-CAM Comparison):**
- **Exp 002 (BCE):** Clear boundary, but low confidence (37%).
- **Exp 004 (Focal):** Blurry boundary, more noise (43% confidence).
- **Exp 005 (Weighted BCE):** ⭐ **Best:** Sharp boundary, high confidence (70%), captures building outline without sky.

![GradCAM Building Comparison](assets/comparison_building_2x4x5.png)

**Food Tag (Grad-CAM Comparison):**

![GradCAM Food Comparison](assets/comparison_food_2x4x5.png)

---

## 🔧 Configuration

Edit `src/core/config.py` to customize:

```python
# Model
BATCH_SIZE = 16              # Reduce for high-res images
LEARNING_RATE = 1e-4         # Baseline learning rate
EPOCHS = 12                  # Total epochs
WEIGHT_DECAY = 1e-5          # L2 regularization

# Data
TOP_K = 150                  # Use top-150 most frequent tags
MAX_SAMPLES = 200_000        # Limit dataset size (for speed)
```

---

## 🐳 Docker Deployment

```bash
# Build image
docker build -t vectortag-ui .

# Run container
docker run --rm -p 8501:8501 \
  -v $(pwd)/models:/app/models \
  vectortag-ui
```

Access UI at `http://localhost:8501`

---

## 🎓 Training Your Own Model

To train or retrain the model with custom settings:

```bash
python src/scripts/train.py
```

**What happens:**
- Loads data with augmentation (crop, flip, rotation, color jitter)
- Computes class weights for imbalanced tags
- Trains ResNet-18 for N epochs
- Saves best model to `models/standard/weights/`
- Auto-generates learning curve plot

**All settings are configurable in `src/core/config.py`:**
- `TOP_K`: Number of tags (default: 150)
- `BATCH_SIZE`: Batch size (default: 32)
- `EPOCHS`: Training epochs (default: 12)
- `LEARNING_RATE`: Base LR (default: 1e-4)
- `WEIGHT_DECAY`: Regularization (default: 1e-5)

---

## 📚 Key Technical Components

### Data Processing (`src/data/`)

- **TaggedImagesDataset:** Multi-label dataset with tag synonyms and hierarchies.
- **Stratified split:** Ensures rare classes are equally distributed in train/val.
- **Smart subsampling:** Weights samples by class rarity for balanced mini-batches.

### Model Architecture (`src/models/baseline.py`)

```python
ResNet-18 (ImageNet pretrained)
    ↓
Feature Extractor → 512D
    ↓
Linear(512 → 256)
    ↓
  ReLU()
    ↓
Dropout(0.4)
    ↓
Linear(256 → num_classes)
    ↓
BCEWithLogitsLoss (per-class)
```

### Grad-CAM Visualization (`src/utils/gradcam.py`)

- Computes class activation maps using gradients.

### Training Pipeline (`src/scripts/train.py`)

1. **Class weight computation:** `pos_weight = (N_neg / N_pos)` clamped to [1.0, 20.0]
2. **LR Scheduler:** ReduceLROnPlateau reduces learning rate on validation plateau
3. **Early stopping:** Saves only best model based on validation loss
4. **Auto-plotting:** Generates learning curve after training

---

## 🔮 Future Improvements

- [ ] **Dynamic tag addition:** Add new tags without full model retraining
- [ ] **Vision Transformer (ViT):** Replace ResNet-18 with ViT for better accuracy

---

## 📖 Experiment Log

Detailed experiments are documented in [experiments.md](experiments.md):

- **Exp 001:** Baseline (Overfitting issue discovered)
- **Exp 002:** Synonyms + Dropout + Augmentation (Overfitting solved)
- **Exp 003:** LR Scheduler + Weight Decay (Better convergence)
- **Exp 004:** FocalLoss (Poor Grad-CAM quality)
- **Exp 005:** **Weighted BCE** ⭐ (Current best)

---

## 📦 Dependencies

- **torch, torchvision:** Deep learning
- **pillow, pandas:** Image & data processing
- **scikit-learn:** Stratified split
- **streamlit:** Web UI
- **pydantic:** Configuration management

---

## 🤝 Contributing

Contributions welcome! Areas of interest:

- Better loss functions for imbalanced multi-label classification
- Improved data augmentation strategies
- Alternative backbone architectures
- Performance optimizations

---

## 📄 License

See [LICENSE](LICENSE) file.

---
