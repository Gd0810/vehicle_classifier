"""
Vehicle Type Classification - Flask Web Application
====================================================
Run: python app.py
Then open http://localhost:5000 in your browser.

Make sure you have trained the model first:
    python train.py --images_dir ./images --epochs 20
"""

import os
import json
import io
import base64
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "vehicle_classifier.pth")
META_PATH  = os.path.join(BASE_DIR, "models", "model_meta.json")
CSV_PATH   = os.path.join(BASE_DIR, "_classes.csv")
IMG_SIZE   = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = [
    "Ambulance", "Box Truck", "Bus", "Bus- Small", "Concrete Mixer",
    "Construction Equipment", "Cyclist", "Fire Truck", "Garbage Truck",
    "Hatchback", "Motorbike", "Pickup", "Pickup- Utility", "SUV",
    "Sedan", "Tow Truck", "Tractor Trailer", "Trailer",
    "Truck- 2-Axle", "Truck- Multi-Axle", "Van"
]

# ── Vehicle metadata ───────────────────────────────────────────────────────────
VEHICLE_INFO = {
    "Ambulance":             {"emoji": "🚑", "category": "Emergency",    "color": "#ef4444"},
    "Box Truck":             {"emoji": "🚚", "category": "Commercial",   "color": "#f97316"},
    "Bus":                   {"emoji": "🚌", "category": "Passenger",    "color": "#3b82f6"},
    "Bus- Small":            {"emoji": "🚐", "category": "Passenger",    "color": "#60a5fa"},
    "Concrete Mixer":        {"emoji": "🏗️", "category": "Construction", "color": "#a78bfa"},
    "Construction Equipment":{"emoji": "🚧", "category": "Construction", "color": "#c084fc"},
    "Cyclist":               {"emoji": "🚴", "category": "Non-Motor",    "color": "#34d399"},
    "Fire Truck":            {"emoji": "🚒", "category": "Emergency",    "color": "#f43f5e"},
    "Garbage Truck":         {"emoji": "🗑️", "category": "Municipal",    "color": "#6b7280"},
    "Hatchback":             {"emoji": "🚗", "category": "Passenger",    "color": "#38bdf8"},
    "Motorbike":             {"emoji": "🏍️", "category": "Non-Motor",    "color": "#4ade80"},
    "Pickup":                {"emoji": "🛻", "category": "Commercial",   "color": "#fb923c"},
    "Pickup- Utility":       {"emoji": "🛻", "category": "Commercial",   "color": "#fdba74"},
    "SUV":                   {"emoji": "🚙", "category": "Passenger",    "color": "#2dd4bf"},
    "Sedan":                 {"emoji": "🚗", "category": "Passenger",    "color": "#818cf8"},
    "Tow Truck":             {"emoji": "🏎️", "category": "Commercial",   "color": "#f59e0b"},
    "Tractor Trailer":       {"emoji": "🚛", "category": "Heavy",        "color": "#dc2626"},
    "Trailer":               {"emoji": "🚜", "category": "Heavy",        "color": "#b45309"},
    "Truck- 2-Axle":         {"emoji": "🚚", "category": "Heavy",        "color": "#92400e"},
    "Truck- Multi-Axle":     {"emoji": "🚛", "category": "Heavy",        "color": "#78350f"},
    "Van":                   {"emoji": "🚐", "category": "Commercial",   "color": "#0ea5e9"},
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB


# ── Model loader ───────────────────────────────────────────────────────────────
def build_model(num_classes: int) -> nn.Module:
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(m.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
        nn.Sigmoid(),
    )
    return m


model       = None
model_meta  = {}

def load_model():
    global model, model_meta
    if not os.path.exists(MODEL_PATH):
        return False
    try:
        with open(META_PATH) as f:
            model_meta = json.load(f)
        num_classes = model_meta.get("num_classes", len(CLASSES))
        model = build_model(num_classes).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        return True
    except Exception as e:
        print(f"[WARN] Could not load model: {e}")
        return False


# ── Inference helper ───────────────────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def predict(pil_image: Image.Image):
    img_tensor = TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img_tensor)[0].cpu().numpy()
    top_idx  = int(np.argmax(output))
    top_conf = float(output[top_idx]) * 100
    results  = sorted(
        [{"class": CLASSES[i], "confidence": round(float(output[i]) * 100, 2)}
         for i in range(len(CLASSES))],
        key=lambda x: -x["confidence"]
    )
    return results, CLASSES[top_idx], top_conf


# ── Chart generators ───────────────────────────────────────────────────────────
def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def make_bar_chart(results: list) -> str:
    top = results[:10]
    labels = [r["class"] for r in top]
    values = [r["confidence"] for r in top]
    colors = [VEHICLE_INFO.get(l, {}).get("color", "#58a6ff") for l in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1],
                   edgecolor="none", height=0.65)
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", color="white", fontsize=9)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Confidence (%)", color="#8b949e")
    ax.set_title("Top-10 Predicted Classes", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axvline(50, color="#30363d", linestyle="--", linewidth=1)
    plt.tight_layout()
    result = fig_to_b64(fig)
    plt.close(fig)
    return result


def make_radar_chart(results: list) -> str:
    # Category-level confidence
    cat_scores: dict = {}
    cat_counts: dict = {}
    for r in results:
        info = VEHICLE_INFO.get(r["class"], {})
        cat  = info.get("category", "Other")
        cat_scores[cat] = cat_scores.get(cat, 0) + r["confidence"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    cats   = list(cat_scores.keys())
    scores = [cat_scores[c] / cat_counts[c] for c in cats]

    N      = len(cats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    scores_plot = scores + scores[:1]
    angles      = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    ax.plot(angles, scores_plot, color="#58a6ff", linewidth=2)
    ax.fill(angles, scores_plot, color="#58a6ff", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, color="white", fontsize=9)
    ax.tick_params(colors="white")
    ax.set_yticklabels([])
    ax.spines["polar"].set_color("#30363d")
    ax.grid(color="#30363d", linewidth=0.5)
    ax.set_title("Category Confidence Radar", color="white", fontsize=12,
                 fontweight="bold", pad=20)
    plt.tight_layout()
    result = fig_to_b64(fig)
    plt.close(fig)
    return result


def make_pie_chart(results: list) -> str:
    top = [r for r in results if r["confidence"] > 5][:6]
    if not top:
        top = results[:3]
    labels = [r["class"] for r in top]
    values = [r["confidence"] for r in top]
    colors = [VEHICLE_INFO.get(l, {}).get("color", "#58a6ff") for l in labels]

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    wedges, texts, autotexts = ax.pie(
        values, labels=None, colors=colors, autopct="%1.1f%%",
        startangle=140, pctdistance=0.8,
        wedgeprops={"edgecolor": "#0d1117", "linewidth": 2}
    )
    for t in autotexts:
        t.set_color("white")
        t.set_fontsize(9)
    ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.12),
              facecolor="#21262d", labelcolor="white", fontsize=9, ncol=2)
    ax.set_title("Confidence Distribution", color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()
    result = fig_to_b64(fig)
    plt.close(fig)
    return result


# ── Dataset stats (cached) ─────────────────────────────────────────────────────
_stats_cache = None

def get_dataset_stats():
    global _stats_cache
    if _stats_cache:
        return _stats_cache
    try:
        df = pd.read_csv(CSV_PATH)
        df.columns = df.columns.str.strip()
        counts = {cls: int(df[cls].sum()) for cls in CLASSES if cls in df.columns}
        total  = sum(counts.values())

        # Category breakdown
        cat_counts: dict = {}
        for cls, cnt in counts.items():
            cat = VEHICLE_INFO.get(cls, {}).get("category", "Other")
            cat_counts[cat] = cat_counts.get(cat, 0) + cnt

        _stats_cache = {
            "total_images"   : len(df),
            "total_labels"   : total,
            "class_counts"   : counts,
            "category_counts": cat_counts,
            "top_class"      : max(counts, key=counts.get) if counts else "N/A",
        }
    except Exception as e:
        _stats_cache = {"error": str(e)}
    return _stats_cache


# ── Dataset chart generators ───────────────────────────────────────────────────
def make_dataset_bar() -> str:
    stats = get_dataset_stats()
    counts = stats.get("class_counts", {})
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
    labels = [c[0] for c in sorted_counts]
    values = [c[1] for c in sorted_counts]
    colors_list = [VEHICLE_INFO.get(l, {}).get("color", "#58a6ff") for l in labels]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    bars = ax.bar(labels, values, color=colors_list, edgecolor="#0d1117", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
                str(val), ha="center", va="bottom", color="white", fontsize=8)
    ax.set_title("Dataset — Samples per Class", color="white", fontsize=13, fontweight="bold")
    ax.set_xlabel("Vehicle Type", color="#8b949e")
    ax.set_ylabel("Count", color="#8b949e")
    ax.tick_params(colors="white", labelsize=8)
    plt.xticks(rotation=45, ha="right")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    plt.tight_layout()
    result = fig_to_b64(fig)
    plt.close(fig)
    return result


def make_category_donut() -> str:
    stats = get_dataset_stats()
    cat_counts = stats.get("category_counts", {})
    labels = list(cat_counts.keys())
    values = list(cat_counts.values())
    palette = ["#ef4444","#f97316","#3b82f6","#8b5cf6","#10b981","#06b6d4","#f59e0b"]
    colors_list = palette[:len(labels)]

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    wedges, texts, autotexts = ax.pie(
        values, labels=None, colors=colors_list, autopct="%1.1f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops={"edgecolor": "#0d1117", "linewidth": 3, "width": 0.55}
    )
    for t in autotexts:
        t.set_color("white")
        t.set_fontsize(9)
    ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.12),
              facecolor="#21262d", labelcolor="white", fontsize=9, ncol=2)
    ax.set_title("Dataset by Category", color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    result = fig_to_b64(fig)
    plt.close(fig)
    return result


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    model_ready = model is not None
    stats = get_dataset_stats()
    return render_template("index.html",
                           model_ready=model_ready,
                           model_meta=model_meta,
                           stats=stats,
                           classes=CLASSES,
                           vehicle_info=VEHICLE_INFO)


@app.route("/predict", methods=["POST"])
def predict_route():
    if model is None:
        return jsonify({"error": "Model not loaded. Please train first."}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        img_bytes = file.read()
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    results, top_class, top_conf = predict(pil_img)
    info = VEHICLE_INFO.get(top_class, {"emoji": "🚗", "category": "Unknown", "color": "#58a6ff"})

    bar_chart   = make_bar_chart(results)
    radar_chart = make_radar_chart(results)
    pie_chart   = make_pie_chart(results)

    # Return base64 of uploaded image for display
    img_b64 = "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode()

    return jsonify({
        "prediction" : top_class,
        "confidence" : round(top_conf, 2),
        "emoji"      : info["emoji"],
        "category"   : info["category"],
        "color"      : info["color"],
        "results"    : results[:10],
        "bar_chart"  : bar_chart,
        "radar_chart": radar_chart,
        "pie_chart"  : pie_chart,
        "image_b64"  : img_b64,
    })


@app.route("/dataset_stats")
def dataset_stats():
    bar   = make_dataset_bar()
    donut = make_category_donut()
    return jsonify({
        "stats"         : get_dataset_stats(),
        "bar_chart"     : bar,
        "category_donut": donut,
    })


@app.route("/training_plots")
def training_plots():
    plots = {}
    plot_dir = os.path.join(BASE_DIR, "static", "images")
    for name in ["training_curves", "class_distribution", "confusion_matrix"]:
        path = os.path.join(plot_dir, f"{name}.png")
        if os.path.exists(path):
            with open(path, "rb") as f:
                plots[name] = "data:image/png;base64," + base64.b64encode(f.read()).decode()
    return jsonify(plots)


if __name__ == "__main__":
    print("\n🚗  Vehicle Type Classifier")
    print("=" * 40)
    if load_model():
        print(f"✅  Model loaded  |  Best Val Acc: {model_meta.get('best_val_acc','?')}%")
    else:
        print("⚠️   No trained model found.")
        print("    Run:  python train.py --images_dir ./images")
    print(f"🌐  Starting Flask → http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
