"""
CROPSPECTRA — Autonomous Crop Inspection Rover
Inference & Disease Advisory Engine
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


# Disease descriptions and field recommendations for the autonomous rover
DISEASE_ADVISORY = {
    "bacterial_leaf_blight": {
        "common_name": "Bacterial Leaf Blight (Xanthomonas oryzae)",
        "symptoms": "Water-soaked lesions that turn yellow-to-white along leaf margins with wavy borders.",
        "severity": "High",
        "action": "Ensure balanced nitrogen fertilizer application; drain field temporarily if flooded; apply copper-based bactericides if severe."
    },
    "brown_spot": {
        "common_name": "Brown Spot (Bipolaris oryzae)",
        "symptoms": "Oval or round brown lesions with grey or whitish centers and yellowish halos.",
        "severity": "Moderate",
        "action": "Correct soil potassium and silicon deficiencies; treat seeds with fungicides; avoid water stress during tillering."
    },
    "healthy": {
        "common_name": "Healthy Rice Leaf",
        "symptoms": "Vibrant green uniform pigmentation without lesion spots or necrosis.",
        "severity": "None",
        "action": "No treatment required. Maintain current irrigation and nutrient management schedule."
    },
    "leaf_blast": {
        "common_name": "Rice Leaf Blast (Magnaporthe oryzae)",
        "symptoms": "Spindle-shaped or diamond-shaped lesions with grey centers and brown borders.",
        "severity": "Critical",
        "action": "Avoid excessive nitrogen; spray systemic fungicides (e.g., Tricyclazole or Isoprothiolane); maintain proper water depth."
    },
    "leaf_scald": {
        "common_name": "Leaf Scald (Microdochium oryzae)",
        "symptoms": "Zonate patterns with alternating light and dark bands starting from the leaf tips.",
        "severity": "Moderate to High",
        "action": "Avoid high nitrogen levels; spray recommended foliar fungicides (e.g., Benomyl or Carbendazim); improve field aeration."
    },
    "narrow_brown_spot": {
        "common_name": "Narrow Brown Leaf Spot (Cercospora janseana)",
        "symptoms": "Short, narrow, linear brown spots parallel to the leaf veins.",
        "severity": "Low to Moderate",
        "action": "Apply balanced potassium fertilizer; utilize resistant rice cultivars; spray propiconazole if prevalent in late heading."
    }
}


def load_model(checkpoint_path="models/best_cropspectra_mobilenetv3.pth", mapping_path="models/class_mapping.json", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Class mapping not found at: {mapping_path}")

    with open(mapping_path, "r") as f:
        class_mapping = json.load(f)

    classes = class_mapping["classes"]
    num_classes = len(classes)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = checkpoint.get("arch", "mobilenet_v3_small")

    if arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, classes, device


def predict_image(image_path, model, classes, device, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0)
        confidence, pred_idx = torch.max(probs, dim=0)

    pred_class = classes[pred_idx.item()]
    conf_score = confidence.item()

    advisory = DISEASE_ADVISORY.get(pred_class, {
        "common_name": pred_class,
        "symptoms": "N/A",
        "severity": "Unknown",
        "action": "Consult agricultural extension specialist."
    })

    top_probs = {classes[i]: round(probs[i].item() * 100, 2) for i in range(len(classes))}

    return {
        "image_path": str(image_path),
        "predicted_class": pred_class,
        "confidence": round(conf_score * 100, 2),
        "advisory": advisory,
        "all_probabilities": top_probs
    }


def main():
    parser = argparse.ArgumentParser(description="CROPSPECTRA Image Inference & Disease Advisory")
    parser.add_argument("--image", type=str, required=True, help="Path to leaf image file")
    parser.add_argument("--model", type=str, default="models/best_cropspectra_mobilenetv3.pth", help="Path to .pth checkpoint")
    parser.add_argument("--mapping", type=str, default="models/class_mapping.json", help="Path to class mapping json")

    args = parser.parse_args()

    model, classes, device = load_model(args.model, args.mapping)
    result = predict_image(args.image, model, classes, device)

    print("\n" + "=" * 60)
    print("[CROPSPECTRA ROVER] - LEAF INSPECTION REPORT")
    print("=" * 60)
    print(f"Leaf Image:         {result['image_path']}")
    print(f"Diagnosis:          {result['advisory']['common_name']}")
    print(f"Confidence:         {result['confidence']}%")
    print(f"Severity Level:     {result['advisory']['severity']}")
    print(f"Symptoms:           {result['advisory']['symptoms']}")
    print(f"Recommended Action: {result['advisory']['action']}")
    print("-" * 60)
    print("Class Probabilities:")
    for cls_name, prob in sorted(result["all_probabilities"].items(), key=lambda x: x[1], reverse=True):
        print(f"  * {cls_name:25s}: {prob:6.2f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
