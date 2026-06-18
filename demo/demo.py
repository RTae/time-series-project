import os
import sys
import json
import base64
import io
import urllib.parse
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import webbrowser

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to sys.path
demo_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(demo_dir)
sys.path.insert(0, project_root)

from src.dataset import ThingsEEGDataset
# from src.models.supaeeg import SUPAEEG
from src.encoders.vision_encoder import InternViTFeatureLookup
from src.utilities import Config, make_model

# ---------------------------------------------------------------------------
# DYNAMIC SELECTIONS DEFAULT CONFIGURATION
# ---------------------------------------------------------------------------
DEFAULT_SUBJECT = 1
DEFAULT_PROTOCOL = "intra"
DEFAULT_CONCEPT = "00197_wheelchair"
DEFAULT_AVERAGE = "true"
# ---------------------------------------------------------------------------

_dataset_cache = {}

def load_config() -> Config:
    from omegaconf import OmegaConf
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(demo_dir)
    cfg = OmegaConf.load(os.path.join(project_root, "conf", "config.yaml"))
    config = Config()
    for field_name in config.__dataclass_fields__:
        if hasattr(cfg, field_name):
            setattr(config, field_name, getattr(cfg, field_name))
    # Make relative dataset/internvit paths absolute based on project_root
    if not os.path.isabs(config.dataset_dir):
        config.dataset_dir = os.path.abspath(os.path.join(project_root, config.dataset_dir))
    if not os.path.isabs(config.internvit_dir):
        config.internvit_dir = os.path.abspath(os.path.join(project_root, config.internvit_dir))
    return config


def get_dataset(subject: int, config: Config) -> ThingsEEGDataset:
    if subject not in _dataset_cache:
        _dataset_cache[subject] = ThingsEEGDataset(
            dataset_dir=config.dataset_dir,
            data_type="test",
            subject=subject,
            load_images=False,
            data_average=config.data_average_test
        )
    return _dataset_cache[subject]


def get_checkpoint_path(protocol: str, subject: int) -> str:
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    if protocol == "intra":
        path = os.path.join(demo_dir, "intra_full", "intra", f"supaeeg_intra_sub{subject:02d}.pt")
    elif protocol == "inter":
        path = os.path.join(demo_dir, "inter_full", "outputs", "2026-06-06", "inter", f"supaeeg_loso_sub{subject:02d}.pt")
    else:
        raise ValueError(f"Unknown protocol: {protocol}")
    
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    return path


def plot_eeg(eeg_tensor: torch.Tensor, is_averaged: bool) -> str:
    fig, ax = plt.subplots(figsize=(6, 3))
    data = eeg_tensor.numpy()
    for i in range(data.shape[0]):
        ax.plot(data[i] + i * 3.0, linewidth=1)
    
    title = "EEG Signal (Averaged ERP)" if is_averaged else "EEG Signal (Single Trial - Noisy)"
    ax.set_title(title)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Channels")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def run_inference(subject: int, target_concept: str, checkpoint_path: str) -> dict:
    config = load_config()
    
    # Load cached or new test dataset
    dataset = get_dataset(subject, config)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(config, device)
    model.eval()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    
    # Find the correct concept index in the 200 test set concepts
    concept_idx = -1
    for i, c in enumerate(dataset.image_meta_data['test_img_concepts']):
        if c == target_concept:
            concept_idx = i
            break
    if concept_idx == -1:
        raise ValueError(f"Concept '{target_concept}' not found in the test dataset split")
        
    # Get the 80 trials for this concept (using the repetitions factor)
    indices = [concept_idx * dataset.number_of_repetitions + r for r in range(dataset.number_of_repetitions)]
    
    # Load all EEG trials for this concept (always averaged for decoder)
    eeg_tensors = []
    target_file = None
    for idx in indices:
        eeg_tensor, _, _, _, _, _, img_file = dataset[idx]
        eeg_tensors.append(eeg_tensor)
        if target_file is None:
            target_file = img_file
            
    eeg_batch = torch.stack(eeg_tensors).to(device)  # (N_trials, 17, 100)
    
    # Compute average EEG embedding
    with torch.no_grad():
        zE_trials = model.embed(eeg_batch)  # (N_trials, 512)
        zE = torch.nn.functional.normalize(zE_trials.mean(dim=0, keepdim=True), dim=1).cpu().numpy()  # (1, 512)
        
    # Get test concept gallery
    concepts = sorted(list(set(dataset.image_meta_data['test_img_concepts'])))
    concept_to_file = {}
    for i in range(len(dataset.image_meta_data['test_img_concepts'])):
        c = dataset.image_meta_data['test_img_concepts'][i]
        f = dataset.image_meta_data['test_img_files'][i]
        if c not in concept_to_file:
            concept_to_file[c] = f
            
    # Retrieve & encode gallery image features
    feature_path = os.path.join(config.internvit_dir, "internvit_features.npy")
    lookup = InternViTFeatureLookup(feature_path=feature_path)
    files = [concept_to_file[c] for c in concepts]
    gallery_features = lookup.retrieve_batch(concepts, files)  # (200, 5, 3200)
    
    with torch.no_grad():
        zI = model.encode_image(gallery_features.to(device), subject_ids=None).cpu().numpy()  # (200, 512)
        
    # Compute cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(zE, zI)[0]
    
    top_indices = np.argsort(-sim)[:5]
    
    results = []
    for rank, idx in enumerate(top_indices, 1):
        results.append({
            "rank": rank,
            "concept": concepts[idx],
            "image_file": concept_to_file[concepts[idx]],
            "similarity": float(sim[idx])
        })
        
    return {
        "results": results,
        "target_file": target_file
    }

# ---------------------------------------------------------------------------
# Load test concepts once on server start
# ---------------------------------------------------------------------------
try:
    _startup_config = load_config()
    _startup_dataset = get_dataset(DEFAULT_SUBJECT, _startup_config)
    ALL_CONCEPTS = sorted(list(set(_startup_dataset.image_meta_data['test_img_concepts'])))
except Exception as e:
    print(f"Warning: Failed to load dataset concepts: {e}")
    ALL_CONCEPTS = []

# ---------------------------------------------------------------------------
# HTTP Web Server
# ---------------------------------------------------------------------------

class DemoHTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        url = urllib.parse.urlparse(self.path)
        path = url.path
        query = urllib.parse.parse_qs(url.query)
        
        if path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode("utf-8"))
            
        elif path == "/api/meta":
            self.send_json({
                "default_subject": DEFAULT_SUBJECT,
                "default_concept": DEFAULT_CONCEPT,
                "default_protocol": DEFAULT_PROTOCOL,
                "default_average": DEFAULT_AVERAGE,
                "concepts": ALL_CONCEPTS
            })
            
        elif path == "/api/sample":
            try:
                subject = int(query.get("subject", [str(DEFAULT_SUBJECT)])[0])
                concept = query.get("concept", [DEFAULT_CONCEPT])[0]
                average = query.get("average", ["true"])[0].lower() == "true"
                
                config = load_config()
                dataset = get_dataset(subject, config)
                
                # Find the correct concept index in the 200 test set concepts
                concept_idx = -1
                for i, c in enumerate(dataset.image_meta_data['test_img_concepts']):
                    if c == concept:
                        concept_idx = i
                        break
                if concept_idx == -1:
                    raise ValueError(f"Concept '{concept}' not found in the test dataset split")
                
                # Get the 80 trials for this concept (using the repetitions factor)
                indices = [concept_idx * dataset.number_of_repetitions + r for r in range(dataset.number_of_repetitions)]
                
                if average:
                    # Average the EEG traces across all trials for a clean ERP visualization
                    eeg_traces = [dataset[idx][0] for idx in indices]
                    eeg_display = torch.stack(eeg_traces).mean(dim=0)
                else:
                    # Load only the first trial (repetition index 0)
                    eeg_display, _, _, _, _, _, _ = dataset[indices[0]]
                
                # Use the target image file from the first trial sample
                _, _, _, _, _, _, image_file = dataset[indices[0]]
                
                eeg_plot = plot_eeg(eeg_display, average)
                
                self.send_json({
                    "image_file": image_file,
                    "eeg_plot": eeg_plot
                })
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
                
        elif path == "/api/decode":
            try:
                subject = int(query.get("subject", [str(DEFAULT_SUBJECT)])[0])
                concept = query.get("concept", [DEFAULT_CONCEPT])[0]
                protocol = query.get("protocol", [DEFAULT_PROTOCOL])[0]
                
                checkpoint_path = get_checkpoint_path(protocol, subject)
                res = run_inference(subject, concept, checkpoint_path)
                self.send_json(res)
            except Exception as e:
                self.send_json({"error": str(e)})
                
        elif path == "/api/image":
            concept = query.get("concept", [""])[0]
            image_file = query.get("file", [""])[0]

            config = load_config()
            img_dir = os.path.abspath(os.path.join(config.dataset_dir, "test_images"))
            requested = os.path.abspath(os.path.join(img_dir, concept, image_file))

            # Prevent path traversal: requested path must stay within img_dir
            if not requested.startswith(img_dir + os.sep):
                self.send_response(400)
                self.end_headers()
                return

            if os.path.isfile(requested):
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                with open(requested, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
            self.send_response(404)
            self.end_headers()
            
    def send_json(self, data: dict) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))


# HTML UI Dashboard Template
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SUPAEEG Visual Decoding Demo</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 30px;
            background-color: #f8f9fa;
            color: #333;
        }
        .control-panel {
            background-color: #fff;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px 20px;
            margin-bottom: 20px;
            display: flex;
            gap: 20px;
            align-items: center;
        }
        .control-group {
            display: flex;
            flex-direction: column;
            gap: 5px;
            flex: 1;
        }
        .control-group label {
            font-weight: bold;
            font-size: 14px;
            color: #555;
        }
        .control-group select {
            padding: 8px 12px;
            border: 1px solid #ccc;
            border-radius: 4px;
            background-color: #fff;
            font-size: 14px;
            cursor: pointer;
        }
        .container {
            display: flex;
            gap: 20px;
        }
        .column {
            flex: 1;
            background: #fff;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button {
            padding: 12px;
            background-color: #007bff;
            color: #fff;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
            margin-bottom: 20px;
        }
        button:hover {
            background-color: #0056b3;
        }
        .image-box {
            border: 1px solid #ccc;
            padding: 10px;
            text-align: center;
            background: #fafafa;
            margin-bottom: 20px;
        }
        .image-box img {
            max-width: 100%;
            max-height: 300px;
            display: block;
            margin: 10px auto;
        }
        .result-item {
            display: flex;
            align-items: center;
            padding: 12px;
            border-bottom: 1px solid #eee;
        }
        .result-item img {
            width: 70px;
            height: 70px;
            object-fit: cover;
            margin-right: 15px;
            border: 1px solid #ccc;
        }
        .result-details {
            flex: 1;
        }
    </style>
</head>
<body>
    <div class="control-panel">
        <div class="control-group">
            <label for="select-subject">Subject</label>
            <select id="select-subject">
                <option value="1">Subject 1</option>
                <option value="2">Subject 2</option>
                <option value="3">Subject 3</option>
                <option value="4">Subject 4</option>
                <option value="5">Subject 5</option>
                <option value="6">Subject 6</option>
                <option value="7">Subject 7</option>
                <option value="8">Subject 8</option>
                <option value="9">Subject 9</option>
                <option value="10">Subject 10</option>
            </select>
        </div>
        
        <div class="control-group">
            <label for="select-protocol">Alignment Protocol</label>
            <select id="select-protocol">
                <option value="intra">Intra-Subject</option>
                <option value="inter">Cross-Subject / LOSO</option>
            </select>
        </div>
        
        <div class="control-group">
            <label for="select-concept">Target Concept</label>
            <select id="select-concept">
                <!-- Populated dynamically -->
            </select>
        </div>
        
        <div class="control-group">
            <label for="select-average">EEG Trial Mode</label>
            <select id="select-average">
                <option value="true">Averaged ERP (80 Trials)</option>
                <option value="false">Single Trial (Noisy Run)</option>
            </select>
        </div>
    </div>
    
    <div class="container">
        <!-- Input Column -->
        <div class="column">
            <h2>Presented Stimulus & EEG</h2>
            <hr><br>
            
            <div class="image-box">
                <strong>EEG Signal Plot</strong>
                <div id="eeg-plot-container">Loading EEG...</div>
            </div>
            
            <div class="image-box">
                <strong>Target Image (Presented Stimulus)</strong>
                <div id="target-image-container">Loading Target Image...</div>
            </div>
        </div>
        
        <!-- Action & Output Column -->
        <div class="column">
            <h2>Decoder Inference</h2>
            <hr><br>
            
            <button id="btn-decode">Run Decoding Model</button>
            
            <h3>Top 5 Retrieved Images</h3>
            <div id="results-container">Click button to decode EEG signal.</div>
        </div>
    </div>

    <script>
        const selectSubject = document.getElementById('select-subject');
        const selectProtocol = document.getElementById('select-protocol');
        const selectConcept = document.getElementById('select-concept');
        const selectAverage = document.getElementById('select-average');
        
        const eegPlotContainer = document.getElementById('eeg-plot-container');
        const targetImageContainer = document.getElementById('target-image-container');
        const btnDecode = document.getElementById('btn-decode');
        const resultsContainer = document.getElementById('results-container');
        
        // Initial setup
        window.addEventListener('DOMContentLoaded', async () => {
            try {
                const configRes = await fetch('/api/meta');
                const configData = await configRes.json();
                
                // Populate unique concept selector
                selectConcept.innerHTML = '';
                configData.concepts.forEach(c => {
                    const option = document.createElement('option');
                    option.value = c;
                    option.innerText = c;
                    selectConcept.appendChild(option);
                });
                
                // Set default form values
                selectSubject.value = configData.default_subject;
                selectProtocol.value = configData.default_protocol;
                selectConcept.value = configData.default_concept;
                selectAverage.value = configData.default_average;
                
                // Set change listeners
                selectSubject.addEventListener('change', loadSample);
                selectProtocol.addEventListener('change', loadSample);
                selectConcept.addEventListener('change', loadSample);
                selectAverage.addEventListener('change', loadSample);
                
                await loadSample();
            } catch (e) {
                console.error("Failed to load server configuration:", e);
            }
        });
        
        async function loadSample() {
            const subject = selectSubject.value;
            const protocol = selectProtocol.value;
            const concept = selectConcept.value;
            const average = selectAverage.value;
            
            // Clear prior results/images
            resultsContainer.innerHTML = 'Click button to decode EEG signal.';
            eegPlotContainer.innerHTML = 'Loading EEG...';
            targetImageContainer.innerHTML = 'Loading Target Image...';
            
            try {
                const sampleRes = await fetch(`/api/sample?subject=${subject}&concept=${concept}&average=${average}`);
                const sampleData = await sampleRes.json();
                
                if (sampleData.error) {
                    eegPlotContainer.innerHTML = `<span style="color:red;">Error: ${sampleData.error}</span>`;
                    targetImageContainer.innerHTML = `<span style="color:red;">Error: ${sampleData.error}</span>`;
                    return;
                }
                
                eegPlotContainer.innerHTML = `<img src="data:image/png;base64,${sampleData.eeg_plot}" alt="EEG">`;
                
                const imgUrl = `/api/image?concept=${concept}&file=${sampleData.image_file}`;
                targetImageContainer.innerHTML = `<img src="${imgUrl}" alt="${concept}"><br><strong>${concept}</strong>`;
            } catch (e) {
                eegPlotContainer.innerHTML = "Error loading EEG";
                targetImageContainer.innerHTML = "Error loading Image";
                console.error(e);
            }
        }
        
        btnDecode.addEventListener('click', async () => {
            const subject = selectSubject.value;
            const protocol = selectProtocol.value;
            const concept = selectConcept.value;
            
            btnDecode.disabled = true;
            btnDecode.innerText = "Decoding...";
            resultsContainer.innerHTML = "Running model inference...";
            
            try {
                const res = await fetch(`/api/decode?subject=${subject}&concept=${concept}&protocol=${protocol}`);
                const data = await res.json();
                
                if (data.error) {
                    resultsContainer.innerHTML = `<span style="color:red;">Error: ${data.error}</span>`;
                    return;
                }
                
                resultsContainer.innerHTML = '';
                data.results.forEach(item => {
                    const imgUrl = `/api/image?concept=${item.concept}&file=${item.image_file}`;
                    const isCorrect = item.concept === concept;
                    const style = isCorrect ? 'background-color: #d4edda; border: 1px solid #c3e6cb;' : '';
                    
                    const div = document.createElement('div');
                    div.className = 'result-item';
                    div.style = style;
                    div.innerHTML = `
                        <img src="${imgUrl}" alt="${item.concept}">
                        <div class="result-details">
                            <strong>Rank ${item.rank}: ${item.concept}</strong><br>
                            Similarity: ${item.similarity.toFixed(4)}
                        </div>
                    `;
                    resultsContainer.appendChild(div);
                });
            } catch (e) {
                resultsContainer.innerHTML = '<span style="color:red;">Error running decoding</span>';
                console.error(e);
            } finally {
                btnDecode.disabled = false;
                btnDecode.innerText = "Run Decoding Model";
            }
        });
    </script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Runner setup
# ---------------------------------------------------------------------------

def main() -> None:
    port = 8080
    while port < 8100:
        try:
            server_address = ('127.0.0.1', port)
            httpd = HTTPServer(server_address, DemoHTTPHandler)
            break
        except OSError:
            port += 1
            
    print(f"SUPAEEG visual decoding demo dashboard is ready at http://localhost:{port}/")
    print("Press Ctrl+C to terminate.")
    
    # Auto-open browser in background thread
    def open_browser():
        try:
            webbrowser.open(f"http://localhost:{port}/")
        except Exception:
            pass
    threading.Timer(1.0, open_browser).start()
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down demo server.")
        httpd.server_close()

if __name__ == "__main__":
    main()
