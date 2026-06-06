# 🚗 Vehicle Type Classification System

A deep learning web app that classifies **21 vehicle types** from images using ResNet-50 transfer learning, served via a Flask web interface with rich data visualizations.

---

## 📁 Project Structure

```
vehicle_classifier/
├── train.py              ← Model training script
├── app.py                ← Flask web application
├── _classes.csv          ← Dataset labels (CSV)
├── requirements.txt      ← Python dependencies
├── README.md
│
├── images/               ← ⚠️  PUT YOUR IMAGES HERE
│
├── models/               ← Auto-created after training
│   ├── vehicle_classifier.pth
│   └── model_meta.json
│
├── static/
│   ├── css/style.css
│   ├── js/app.js
│   └── images/           ← Auto-created training plots
│
└── templates/
    └── index.html
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> Python 3.9+ recommended. CUDA GPU optional but speeds up training.

### 2. Add Your Images

Place all your training images inside the `images/` folder:

```
images/
├── 44_jpg.rf.218740cf...jpg
├── IMG_3611_JPG_jpg.rf...jpg
└── ...
```

The filenames must match the `filename` column in `_classes.csv`.

### 3. Train the Model

```bash
python train.py --images_dir ./images --epochs 20
```

**Optional arguments:**
| Argument | Default | Description |
|---|---|---|
| `--images_dir` | `./images` | Path to your images folder |
| `--csv` | `./_classes.csv` | Path to labels CSV |
| `--epochs` | `20` | Training epochs (more = better but slower) |
| `--model_dir` | `./models` | Where to save the trained model |
| `--plot_dir` | `./static/images` | Where to save training plots |

Training automatically:
- Saves the best model checkpoint
- Generates loss/accuracy curves
- Generates class distribution chart
- Generates confusion matrix

### 4. Run the Web App

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 🎯 Features

| Feature | Description |
|---|---|
| **Image Classification** | Upload any vehicle image → instant prediction |
| **Confidence Scores** | Top-10 class probabilities with visual bars |
| **Bar Chart** | Horizontal bar chart of top predictions |
| **Radar Chart** | Category-level confidence radar |
| **Pie Chart** | Confidence distribution donut chart |
| **Dataset Explorer** | Class distribution & category breakdown charts |
| **Training Plots** | Loss curves, accuracy, confusion matrix |

---

## 🏷️ Vehicle Classes (21)

| Class | Category |
|---|---|
| Ambulance | Emergency |
| Fire Truck | Emergency |
| Bus | Passenger |
| Bus- Small | Passenger |
| Hatchback | Passenger |
| Sedan | Passenger |
| SUV | Passenger |
| Box Truck | Commercial |
| Pickup | Commercial |
| Pickup- Utility | Commercial |
| Tow Truck | Commercial |
| Van | Commercial |
| Concrete Mixer | Construction |
| Construction Equipment | Construction |
| Garbage Truck | Municipal |
| Motorbike | Non-Motor |
| Cyclist | Non-Motor |
| Tractor Trailer | Heavy |
| Trailer | Heavy |
| Truck- 2-Axle | Heavy |
| Truck- Multi-Axle | Heavy |

---

## 🧠 Model Architecture

- **Backbone**: ResNet-50 (ImageNet pre-trained)
- **Fine-tuned layers**: `layer4` + custom head
- **Custom head**: Dropout(0.4) → Linear(2048→512) → ReLU → Dropout(0.3) → Linear(512→21) → Sigmoid
- **Loss**: Binary Cross-Entropy (multi-label)
- **Optimizer**: Adam (lr=1e-3)
- **Scheduler**: StepLR (step=7, gamma=0.1)

---

## 💡 Tips

- **More epochs** (30–50) generally improves accuracy
- **GPU training** is much faster: `torch` auto-detects CUDA
- For **imbalanced classes**, consider weighted sampling
- The model supports **multi-label** prediction (one image can be multiple classes)
