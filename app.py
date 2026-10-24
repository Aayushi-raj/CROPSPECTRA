"""
CROPSPECTRA — Interactive Web Dashboard for Rice Leaf Disease Inspection
Built using Python's standard library http.server (no extra pip dependencies required).
"""

import os
import io
import sys
import json
import random
import mimetypes
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import base64

import torch
from torchvision import transforms
from PIL import Image

# Import model inference components
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from predict import load_model, DISEASE_ADVISORY

PORT = 5000
MODEL_PATH = "models/best_cropspectra_mobilenetv3.pth"
MAPPING_PATH = "models/class_mapping.json"
VAL_DIR = Path("DATASET/archive1/RiceLeafsDisease/validation")

print("[*] Loading CROPSPECTRA MobileNetV3 model...")
model, classes, device = load_model(MODEL_PATH, MAPPING_PATH)
print(f"[OK] Model loaded on {device} with {len(classes)} classes.")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CROPSPECTRA — Autonomous Rover Leaf Disease Classifier</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0b111a;
            --card-bg: rgba(22, 32, 48, 0.75);
            --card-border: rgba(255, 255, 255, 0.08);
            --primary: #10b981;
            --primary-glow: rgba(16, 185, 129, 0.3);
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --accent: #38bdf8;
            --danger: #ef4444;
            --warning: #f59e0b;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background-color: var(--bg);
            background-image: 
                radial-gradient(at 0% 0%, rgba(16, 185, 129, 0.12) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(56, 189, 248, 0.1) 0px, transparent 50%);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        header {
            padding: 1.5rem 2.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--card-border);
            backdrop-filter: blur(12px);
            background: rgba(11, 17, 26, 0.8);
            position: sticky;
            top: 0;
            z-index: 50;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .badge-rover {
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid var(--primary);
            color: var(--primary);
            padding: 0.25rem 0.65rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        .brand h1 {
            font-size: 1.35rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, #ffffff 30%, #a7f3d0 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .model-tag {
            font-size: 0.85rem;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background: var(--primary);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--primary);
        }

        main {
            flex: 1;
            max-width: 1200px;
            width: 100%;
            margin: 0 auto;
            padding: 2.5rem 1.5rem;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }

        @media (max-width: 900px) {
            main {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 1.25rem;
            padding: 1.75rem;
            backdrop-filter: blur(16px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        }

        .card-title {
            font-size: 1.15rem;
            font-weight: 600;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        /* Upload Area */
        .dropzone {
            border: 2px dashed rgba(255, 255, 255, 0.15);
            border-radius: 1rem;
            padding: 2.5rem 1.5rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.25s ease;
            background: rgba(255, 255, 255, 0.02);
            position: relative;
        }

        .dropzone:hover, .dropzone.dragover {
            border-color: var(--primary);
            background: rgba(16, 185, 129, 0.05);
            box-shadow: 0 0 25px var(--primary-glow);
        }

        .dropzone svg {
            width: 48px;
            height: 48px;
            stroke: var(--primary);
            margin-bottom: 0.75rem;
        }

        .dropzone p {
            color: var(--text-muted);
            font-size: 0.95rem;
            margin-bottom: 0.5rem;
        }

        .btn-upload {
            background: var(--primary);
            color: #042f24;
            font-weight: 600;
            padding: 0.6rem 1.25rem;
            border-radius: 0.5rem;
            border: none;
            cursor: pointer;
            display: inline-block;
            margin-top: 0.5rem;
            transition: transform 0.15s ease, filter 0.15s ease;
        }

        .btn-upload:hover {
            filter: brightness(1.1);
            transform: translateY(-1px);
        }

        .preview-container {
            margin-top: 1.25rem;
            position: relative;
            border-radius: 0.75rem;
            overflow: hidden;
            display: none;
            max-height: 280px;
            border: 1px solid var(--card-border);
        }

        .preview-container img {
            width: 100%;
            height: 260px;
            object-fit: cover;
            display: block;
        }

        /* Quick Samples */
        .samples-section {
            margin-top: 1.5rem;
        }

        .samples-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.75rem;
            font-weight: 600;
        }

        .sample-buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .sample-btn {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--card-border);
            color: var(--text-main);
            padding: 0.45rem 0.85rem;
            border-radius: 0.5rem;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .sample-btn:hover {
            background: rgba(16, 185, 129, 0.2);
            border-color: var(--primary);
            color: #fff;
        }

        /* Results Panel */
        .empty-state {
            text-align: center;
            padding: 4rem 1.5rem;
            color: var(--text-muted);
        }

        .empty-state svg {
            width: 56px;
            height: 56px;
            stroke: rgba(255, 255, 255, 0.2);
            margin-bottom: 1rem;
        }

        .results-panel {
            display: none;
        }

        .diagnosis-card {
            background: rgba(16, 185, 129, 0.08);
            border: 1px solid rgba(16, 185, 129, 0.25);
            border-radius: 1rem;
            padding: 1.25rem;
            margin-bottom: 1.5rem;
        }

        .diagnosis-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 0.5rem;
        }

        .diagnosis-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #fff;
        }

        .severity-badge {
            padding: 0.25rem 0.65rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
        }

        .severity-Critical { background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid #ef4444; }
        .severity-High { background: rgba(249, 115, 22, 0.2); color: #fdba74; border: 1px solid #f97316; }
        .severity-Moderate { background: rgba(245, 158, 11, 0.2); color: #fde68a; border: 1px solid #f59e0b; }
        .severity-Low { background: rgba(59, 130, 246, 0.2); color: #93c5fd; border: 1px solid #3b82f6; }
        .severity-None { background: rgba(16, 185, 129, 0.2); color: #86efac; border: 1px solid #10b981; }

        .confidence-row {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin: 1rem 0;
        }

        .confidence-val {
            font-size: 2rem;
            font-weight: 800;
            color: var(--primary);
        }

        .confidence-bar-bg {
            flex: 1;
            height: 10px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 999px;
            overflow: hidden;
        }

        .confidence-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #10b981, #38bdf8);
            border-radius: 999px;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .info-group {
            margin-top: 0.85rem;
        }

        .info-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            font-weight: 600;
            margin-bottom: 0.25rem;
        }

        .info-text {
            font-size: 0.9rem;
            line-height: 1.45;
            color: #cbd5e1;
        }

        /* Probabilities list */
        .prob-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.65rem;
            font-size: 0.85rem;
        }

        .prob-label {
            width: 140px;
            color: var(--text-muted);
            text-transform: capitalize;
        }

        .prob-bar-container {
            flex: 1;
            height: 6px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 999px;
            margin: 0 0.85rem;
            overflow: hidden;
        }

        .prob-bar-inner {
            height: 100%;
            background: var(--accent);
            border-radius: 999px;
            transition: width 0.4s ease;
        }

        .prob-val {
            width: 45px;
            text-align: right;
            font-weight: 600;
        }

        .loading-spinner {
            display: none;
            text-align: center;
            padding: 3rem;
            color: var(--primary);
            font-weight: 600;
        }

        .spinner {
            width: 36px;
            height: 36px;
            border: 3px solid rgba(16, 185, 129, 0.2);
            border-top-color: var(--primary);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin: 0 auto 1rem;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>

    <header>
        <div class="brand">
            <span class="badge-rover">Rover Inspection AI</span>
            <h1>CROPSPECTRA</h1>
        </div>
        <div class="model-tag">
            <span class="status-dot"></span>
            <span>MobileNetV3 (93.37% Val Acc)</span>
        </div>
    </header>

    <main>
        <!-- LEFT: Upload & Sample Panel -->
        <section class="card">
            <div class="card-title">
                <span>Upload Leaf Image</span>
                <span style="font-size: 0.8rem; color: var(--text-muted);">RGB JPEG/PNG</span>
            </div>

            <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
                <svg fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 001.5-1.5V6a1.5 1.5 0 00-1.5-1.5H3.75A1.5 1.5 0 002.25 6v12a1.5 1.5 0 001.5 1.5zm10.5-11.25h.008v.008h-.008V8.25zm.375 0a.375.375 0 11-.75 0 .375.375 0 01.75 0z" />
                </svg>
                <p>Drag and drop your rice leaf image here</p>
                <button type="button" class="btn-upload">Browse File</button>
                <input type="file" id="fileInput" accept="image/*" style="display:none" onchange="handleFileSelect(event)">
            </div>

            <div class="preview-container" id="previewContainer">
                <img id="previewImg" src="" alt="Selected Leaf">
            </div>

            <div class="samples-section">
                <div class="samples-title">Or Test with Validation Dataset Samples:</div>
                <div class="sample-buttons">
                    <button class="sample-btn" onclick="loadSample('bacterial_leaf_blight')">Bacterial Leaf Blight</button>
                    <button class="sample-btn" onclick="loadSample('brown_spot')">Brown Spot</button>
                    <button class="sample-btn" onclick="loadSample('healthy')">Healthy Leaf</button>
                    <button class="sample-btn" onclick="loadSample('leaf_blast')">Leaf Blast</button>
                    <button class="sample-btn" onclick="loadSample('leaf_scald')">Leaf Scald</button>
                    <button class="sample-btn" onclick="loadSample('narrow_brown_spot')">Narrow Brown Spot</button>
                </div>
            </div>
        </section>

        <!-- RIGHT: Results & Rover Advisory -->
        <section class="card">
            <div class="card-title">
                <span>Inspection & Diagnosis</span>
                <span id="scanStatus" style="font-size: 0.8rem; color: var(--primary);">System Ready</span>
            </div>

            <div class="loading-spinner" id="loadingSpinner">
                <div class="spinner"></div>
                <p>Analyzing leaf morphology via MobileNetV3...</p>
            </div>

            <div class="empty-state" id="emptyState">
                <svg fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
                </svg>
                <p>Select or upload a rice leaf photo to begin automated inspection.</p>
            </div>

            <div class="results-panel" id="resultsPanel">
                <div class="diagnosis-card">
                    <div class="diagnosis-header">
                        <div>
                            <span class="info-label">Primary Diagnosis</span>
                            <div class="diagnosis-title" id="diagTitle">--</div>
                        </div>
                        <span class="severity-badge" id="diagSeverity">--</span>
                    </div>

                    <div class="confidence-row">
                        <div>
                            <div class="confidence-val" id="diagConfidence">--%</div>
                            <span class="info-label">Confidence</span>
                        </div>
                        <div class="confidence-bar-bg">
                            <div class="confidence-bar-fill" id="diagConfBar" style="width: 0%"></div>
                        </div>
                    </div>

                    <div class="info-group">
                        <div class="info-label">Visual Symptoms</div>
                        <div class="info-text" id="diagSymptoms">--</div>
                    </div>

                    <div class="info-group" style="margin-top: 0.85rem;">
                        <div class="info-label">Rover Recommended Action</div>
                        <div class="info-text" id="diagAction" style="color: #6ee7b7; font-weight: 500;">--</div>
                    </div>
                </div>

                <div class="samples-title">Full Probability Distribution</div>
                <div id="probList"></div>
            </div>
        </section>
    </main>

    <script>
        const dropzone = document.getElementById('dropzone');

        ['dragenter', 'dragover'].forEach(name => {
            dropzone.addEventListener(name, (e) => { e.preventDefault(); dropzone.classList.add('dragover'); }, false);
        });
        ['dragleave', 'drop'].forEach(name => {
            dropzone.addEventListener(name, (e) => { e.preventDefault(); dropzone.classList.remove('dragover'); }, false);
        });

        dropzone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) uploadAndPredict(files[0]);
        });

        function handleFileSelect(e) {
            if (e.target.files.length > 0) uploadAndPredict(e.target.files[0]);
        }

        function uploadAndPredict(file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                showPreview(e.target.result);
                sendPrediction(e.target.result);
            };
            reader.readAsDataURL(file);
        }

        function showPreview(dataUrl) {
            const preview = document.getElementById('previewImg');
            preview.src = dataUrl;
            document.getElementById('previewContainer').style.display = 'block';
        }

        async function loadSample(clsName) {
            setLoading(true);
            try {
                const res = await fetch(`/api/sample?class=${clsName}`);
                const data = await res.json();
                if (data.data_url) {
                    showPreview(data.data_url);
                    renderResults(data.prediction);
                }
            } catch (err) {
                alert('Error loading sample: ' + err);
            } finally {
                setLoading(false);
            }
        }

        async function sendPrediction(base64Data) {
            setLoading(true);
            try {
                const res = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: base64Data })
                });
                const data = await res.json();
                renderResults(data);
            } catch (err) {
                alert('Prediction failed: ' + err);
            } finally {
                setLoading(false);
            }
        }

        function setLoading(isLoading) {
            document.getElementById('loadingSpinner').style.display = isLoading ? 'block' : 'none';
            if (isLoading) {
                document.getElementById('emptyState').style.display = 'none';
                document.getElementById('resultsPanel').style.display = 'none';
                document.getElementById('scanStatus').innerText = 'Analyzing...';
            }
        }

        function renderResults(data) {
            document.getElementById('emptyState').style.display = 'none';
            document.getElementById('resultsPanel').style.display = 'block';
            document.getElementById('scanStatus').innerText = 'Analysis Complete';

            document.getElementById('diagTitle').innerText = data.advisory.common_name;
            document.getElementById('diagConfidence').innerText = data.confidence + '%';
            document.getElementById('diagConfBar').style.width = data.confidence + '%';

            const severityEl = document.getElementById('diagSeverity');
            severityEl.innerText = data.advisory.severity;
            severityEl.className = 'severity-badge severity-' + data.advisory.severity;

            document.getElementById('diagSymptoms').innerText = data.advisory.symptoms;
            document.getElementById('diagAction').innerText = data.advisory.action;

            const probList = document.getElementById('probList');
            probList.innerHTML = '';

            const sorted = Object.entries(data.all_probabilities).sort((a, b) => b[1] - a[1]);
            for (const [cls, val] of sorted) {
                const row = document.createElement('div');
                row.className = 'prob-row';
                row.innerHTML = `
                    <span class="prob-label">${cls.replace(/_/g, ' ')}</span>
                    <div class="prob-bar-container">
                        <div class="prob-bar-inner" style="width: ${val}%"></div>
                    </div>
                    <span class="prob-val">${val}%</span>
                `;
                probList.appendChild(row);
            }
        }
    </script>
</body>
</html>
"""

def predict_pil(img):
    tensor = transform(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0)
        confidence, pred_idx = torch.max(probs, dim=0)

    pred_class = classes[pred_idx.item()]
    conf_score = round(confidence.item() * 100, 2)

    advisory = DISEASE_ADVISORY.get(pred_class, {
        "common_name": pred_class,
        "symptoms": "N/A",
        "severity": "Unknown",
        "action": "Consult agricultural extension specialist."
    })

    top_probs = {classes[i]: round(probs[i].item() * 100, 2) for i in range(len(classes))}

    return {
        "predicted_class": pred_class,
        "confidence": conf_score,
        "advisory": advisory,
        "all_probabilities": top_probs
    }


class CropspectraHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Clean logging
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif parsed.path == "/api/sample":
            params = urllib.parse.parse_qs(parsed.query)
            cls_name = params.get("class", ["healthy"])[0]

            cls_dir = VAL_DIR / cls_name
            if not cls_dir.exists():
                cls_dir = VAL_DIR / "healthy"

            images = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.JPG"))
            if not images:
                self.send_error(404, "No sample images found")
                return

            chosen_img_path = random.choice(images)
            img = Image.open(chosen_img_path)
            prediction = predict_pil(img)

            # Convert to data URL
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            res_payload = {
                "sample_class": cls_name,
                "data_url": f"data:image/jpeg;base64,{img_b64}",
                "prediction": prediction
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res_payload).encode("utf-8"))

        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path == "/api/predict":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body.decode("utf-8"))
                b64_str = data.get("image", "")
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]

                img_bytes = base64.b64decode(b64_str)
                img = Image.open(io.BytesIO(img_bytes))
                res = predict_pil(img)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(res).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_error(404, "Not Found")


def run_server():
    server = HTTPServer(("127.0.0.1", PORT), CropspectraHandler)
    print("\n" + "=" * 60)
    print(f"[CROPSPECTRA] Web Dashboard running at:")
    print(f"URL: http://127.0.0.1:{PORT}")
    print("=" * 60)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        server.server_close()


if __name__ == "__main__":
    run_server()
