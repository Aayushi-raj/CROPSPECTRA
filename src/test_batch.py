"""
CROPSPECTRA — Batch Model Tester
Runs inference on random validation samples across all 6 classes and displays results.
"""

import os
import random
from pathlib import Path
from predict import load_model, predict_image

def main():
    val_dir = Path("DATASET/archive1/RiceLeafsDisease/validation")
    model, classes, device = load_model("models/best_cropspectra_mobilenetv3.pth", "models/class_mapping.json")

    print("\n" + "=" * 75)
    print("[CROPSPECTRA] - VALIDATION SAMPLES TESTING")
    print("=" * 75)
    print(f"{'True Class':<22} | {'Predicted Class':<22} | {'Confidence':<10} | {'Match'}")
    print("-" * 75)

    correct = 0
    total = 0

    for cls in classes:
        cls_dir = val_dir / cls
        images = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.JPG"))
        if not images:
            continue
        # Pick 2 random samples per class
        samples = random.sample(images, min(2, len(images)))
        for img_path in samples:
            res = predict_image(str(img_path), model, classes, device)
            is_match = (res["predicted_class"] == cls)
            if is_match:
                correct += 1
            total += 1
            status = "[MATCH]" if is_match else "[MISMATCH]"
            print(f"{cls:<22} | {res['predicted_class']:<22} | {res['confidence']:>6.2f}%    | {status}")

    print("=" * 75)
    print(f"Sample Accuracy: {correct}/{total} ({correct/total*100:.1f}%)\n")

if __name__ == "__main__":
    main()
