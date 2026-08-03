"""
06_deploy_flask.py
Web deployment menggunakan Flask untuk klasifikasi penyakit daun multi-crop

Jalankan:
    python scripts/06_deploy_flask.py
    python scripts/06_deploy_flask.py --port 8080
    python scripts/06_deploy_flask.py --checkpoint checkpoints/baseline_9class.pth

Akses di browser: http://localhost:5000
"""

import sys
import os
import io
import json
import argparse
import torch
from pathlib import Path
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CFG
from src.dataset import get_transforms
from src.model import load_checkpoint, count_parameters

try:
    from flask import Flask, request, render_template_string, jsonify
except ImportError:
    print("[ERROR] Flask belum terinstall. Jalankan: pip install flask")
    sys.exit(1)


# Checkpoint default untuk deployment -- rasio 20% dengan bobot val-derived,
# klaim utama tesis: akurasi setara baseline (96,00%) dengan parameter,
# FLOPs, dan waktu inferensi lebih kecil. Override dengan --checkpoint.
DEFAULT_CHECKPOINT = "multicriteria_20pct_30ep_valweights.pth"


# ==================== TEMPLATE HTML ====================
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Klasifikasi Penyakit Daun - Famili Poaceae</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: Arial, sans-serif;
            background: #f5f5f5;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: flex-start;
            padding: 40px 20px;
        }
        .container {
            background: white;
            border-radius: 12px;
            padding: 40px;
            max-width: 700px;
            width: 100%;
            box-shadow: 0 2px 16px rgba(0,0,0,0.08);
        }
        h1 {
            font-size: 24px;
            color: #1a1a2e;
            margin-bottom: 8px;
            text-align: center;
        }
        .subtitle {
            color: #666;
            text-align: center;
            font-size: 14px;
            margin-bottom: 30px;
        }
        .upload-area {
            border: 2px dashed #ccc;
            border-radius: 8px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: border-color 0.3s;
            margin-bottom: 20px;
        }
        .upload-area:hover { border-color: #1a6b3c; }
        .upload-area img {
            max-width: 300px;
            max-height: 300px;
            margin-bottom: 10px;
            border-radius: 6px;
        }
        input[type="file"] { display: none; }
        .btn {
            background: #1a6b3c;
            color: white;
            border: none;
            padding: 14px 32px;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            width: 100%;
            transition: background 0.3s;
        }
        .btn:hover { background: #145530; }
        .btn:disabled { background: #ccc; cursor: not-allowed; }
        .result {
            margin-top: 30px;
            padding: 24px;
            border-radius: 8px;
            background: #f0faf4;
            border: 1px solid #1a6b3c;
        }
        .result h3 { color: #1a6b3c; margin-bottom: 12px; }
        .result-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #e0e0e0;
        }
        .result-item:last-child { border-bottom: none; }
        .confidence {
            font-weight: bold;
            color: #1a6b3c;
        }
        .plant-tag {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            color: white;
            margin-left: 8px;
        }
        .plant-padi { background: #2e7d32; }
        .plant-tebu { background: #e65100; }
        .plant-jagung { background: #1565c0; }
        .footer {
            margin-top: 30px;
            text-align: center;
            color: #999;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Klasifikasi Penyakit Daun</h1>
        <p class="subtitle">Tanaman Famili Poaceae (Padi, Tebu, Jagung)<br>
            MobileNetV2 + Pruning Terstruktur Multi-Kriteria</p>

        <form id="uploadForm" enctype="multipart/form-data">
            <div class="upload-area" id="dropArea" onclick="document.getElementById('fileInput').click()">
                <div id="preview">
                    <p>Klik atau seret gambar daun ke sini</p>
                    <p style="color:#999; font-size:13px; margin-top:8px">Format: JPG, JPEG, PNG (maks. 10 MB)</p>
                </div>
            </div>
            <input type="file" id="fileInput" name="image" accept=".jpg,.jpeg,.png" onchange="previewImage(this)">
            <button type="submit" class="btn" id="submitBtn" disabled>Klasifikasi</button>
        </form>

        <div id="resultArea" style="display:none"></div>

        <div class="result">
            <h3>Informasi Model</h3>
            <div class="result-item"><span>Checkpoint</span><span>{{ model_info.checkpoint }}</span></div>
            <div class="result-item"><span>Jumlah Parameter</span><span>{{ model_info.num_params_display }}</span></div>
            <div class="result-item"><span>Rasio Pemangkasan</span><span>{{ model_info.pruning_ratio_display }}</span></div>
            <div class="result-item"><span>Akurasi Uji Tercatat</span><span>{{ model_info.test_accuracy_display }}</span></div>
        </div>

        <div class="footer">
            Tesis Magister Teknik Informatika | Nurul Umayah Hafilda | Universitas Hasanuddin | 2026
        </div>
    </div>

    <script>
        function previewImage(input) {
            if (input.files && input.files[0]) {
                const validExt = /\.(jpg|jpeg|png)$/i.test(input.files[0].name);
                if (!validExt) {
                    alert('Format tidak didukung. Hanya menerima berkas JPG, JPEG, atau PNG.');
                    input.value = '';
                    document.getElementById('submitBtn').disabled = true;
                    return;
                }
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('preview').innerHTML =
                        '<img src="' + e.target.result + '" alt="Preview">' +
                        '<p style="color:#666; font-size:13px; margin-top:8px">' +
                        input.files[0].name + '</p>';
                    document.getElementById('submitBtn').disabled = false;
                };
                reader.readAsDataURL(input.files[0]);
            }
        }

        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const formData = new FormData();
            const fileInput = document.getElementById('fileInput');
            if (!fileInput.files[0]) return;

            formData.append('image', fileInput.files[0]);
            document.getElementById('submitBtn').disabled = true;
            document.getElementById('submitBtn').textContent = 'Memproses...';

            try {
                const response = await fetch('/predict', { method: 'POST', body: formData });
                const data = await response.json();

                if (!response.ok) {
                    alert('Error: ' + (data.error || 'Terjadi kesalahan'));
                    document.getElementById('submitBtn').disabled = false;
                    document.getElementById('submitBtn').textContent = 'Klasifikasi';
                    return;
                }

                const plantColors = {'Padi': 'plant-padi', 'Tebu': 'plant-tebu', 'Jagung': 'plant-jagung'};

                let html = '<div class="result"><h3>Hasil Diagnosis</h3>';
                html += '<div class="result-item"><span>Tanaman</span>' +
                    '<span><span class="plant-tag ' + (plantColors[data.plant] || '') + '">' +
                    data.plant + '</span></span></div>';
                html += '<div class="result-item"><span>Penyakit</span>' +
                    '<span>' + data.disease + '</span></div>';
                html += '<div class="result-item"><span>Confidence</span>' +
                    '<span class="confidence">' + data.confidence + '%</span></div>';
                html += '<div class="result-item"><span>Waktu Inferensi</span>' +
                    '<span>' + data.inference_ms + ' ms</span></div>';
                html += '</div>';

                html += '<div class="result"><h3>Seluruh Probabilitas Kelas</h3>';
                data.all_classes.forEach(function(item, idx) {
                    const tagClass = plantColors[item.plant] || '';
                    const rowStyle = idx === 0 ? ' style="font-weight:bold"' : '';
                    html += '<div class="result-item"' + rowStyle + '><span>' + item.class +
                        ' <span class="plant-tag ' + tagClass + '">' + item.plant + '</span></span>' +
                        '<span>' + item.score + '%</span></div>';
                });
                html += '</div>';

                document.getElementById('resultArea').innerHTML = html;
                document.getElementById('resultArea').style.display = 'block';
            } catch (err) {
                alert('Error: ' + err.message);
            }

            document.getElementById('submitBtn').disabled = false;
            document.getElementById('submitBtn').textContent = 'Klasifikasi';
        });
    </script>
</body>
</html>
"""


def load_model(checkpoint_path):
    """Muat model dari checkpoint untuk inferensi. Verifikasi jumlah
    parameter SELALU dijalankan -- dihitung langsung dari tensor
    state_dict checkpoint, bukan bergantung pada key metadata "num_params"
    yang memang tidak pernah disimpan train_model(). Tidak ada fallback ke
    model pretrained -- deployment wajib memakai checkpoint terlatih,
    supaya tidak diam-diam menyajikan prediksi acak dari classifier head
    yang belum pernah dilatih pada dataset Poaceae."""
    device = torch.device("cpu")  # Inferensi di CPU untuk deployment

    if not checkpoint_path or not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan: {checkpoint_path}. "
            "Deployment wajib memakai checkpoint terlatih -- tentukan path yang "
            "valid lewat --checkpoint, atau jalankan 12_multicriteria_valweights.py "
            "untuk menghasilkan checkpoint default."
        )

    model, checkpoint = load_checkpoint(checkpoint_path, device)

    state_dict = checkpoint["model_state_dict"]
    param_names = {name for name, _ in model.named_parameters()}
    checkpoint_param_count = sum(
        state_dict[name].numel() for name in param_names if name in state_dict
    )
    model_param_count = count_parameters(model)
    if checkpoint_param_count != model_param_count:
        raise RuntimeError(
            f"Verifikasi arsitektur gagal saat memuat {checkpoint_path}: "
            f"jumlah parameter checkpoint ({checkpoint_param_count:,}) tidak "
            f"cocok dengan model yang direkonstruksi ({model_param_count:,})."
        )
    print(f"[INFO] Verifikasi parameter OK: {model_param_count:,} parameter")

    model.eval()
    return model


def verify_class_consistency(model):
    """Verifikasi jumlah keluaran model dan urutan CFG.CLASS_NAMES konsisten
    dengan dataset_split/test, supaya prediksi tidak salah label diam-diam."""
    out_features = model.classifier[-1].out_features
    if out_features != len(CFG.CLASS_NAMES):
        raise RuntimeError(
            f"Jumlah keluaran model ({out_features}) tidak sama dengan "
            f"panjang CFG.CLASS_NAMES ({len(CFG.CLASS_NAMES)})."
        )

    test_dir = CFG.DATASET_DIR.parent / "dataset_split" / "test"
    if not test_dir.exists():
        print(f"[PERINGATAN] {test_dir} tidak ditemukan, verifikasi urutan kelas dilewati.")
        return

    folder_classes = sorted(p.name for p in test_dir.iterdir() if p.is_dir())
    if folder_classes != CFG.CLASS_NAMES:
        raise RuntimeError(
            "CFG.CLASS_NAMES tidak sama persis dengan urutan alfabetis folder "
            f"dataset_split/test.\n  CFG.CLASS_NAMES : {CFG.CLASS_NAMES}\n"
            f"  Folder (sorted) : {folder_classes}"
        )

    print(f"[INFO] Verifikasi kelas OK: {len(CFG.CLASS_NAMES)} kelas, "
          f"urutan cocok dengan dataset_split/test.")


def load_model_info(checkpoint_path):
    """Cari entri model yang sedang dimuat di outputs/tabel_hasil_lengkap.json
    untuk panel informasi model -- nilainya TIDAK ditulis sebagai konstanta
    di kode, murni hasil pencarian di berkas hasil eksperimen. Kalau berkas
    atau entrinya tidak ada, cetak PERINGATAN saat startup (aplikasi tetap
    boleh berjalan, tapi diam-diam menampilkan "Tidak tercatat" tanpa
    penjelasan itu membingungkan)."""
    ckpt_name = Path(checkpoint_path).name
    info = {
        "checkpoint": ckpt_name,
        "num_params_display": "Tidak tercatat",
        "pruning_ratio_display": "Tidak tercatat",
        "test_accuracy_display": "Tidak tercatat",
    }

    table_path = CFG.OUTPUT_DIR / "tabel_hasil_lengkap.json"
    if not table_path.exists():
        print(f"[PERINGATAN] {table_path} tidak ditemukan -- panel info model untuk "
              f"'{ckpt_name}' akan menampilkan 'Tidak tercatat' untuk parameter, "
              f"rasio, dan akurasi. Jalankan 17_generate_final_table.py untuk mengisi ini.")
        return info

    with open(table_path) as f:
        table = json.load(f)
    results = table.get("results", {})

    entry = None
    ratio_label = None

    base = results.get("baseline")
    if base and base.get("checkpoint") == ckpt_name:
        entry = base
        ratio_label = "0% (baseline, tidak dipangkas)"
    else:
        scenario_labels = {
            "l1": "L1 saja", "bn": "BN saja",
            "entropy": "Entropi saja", "multicriteria": "Multi-Kriteria",
        }
        for scenario_key, label in scenario_labels.items():
            for ratio_key, candidate in results.get(scenario_key, {}).items():
                if candidate.get("checkpoint") == ckpt_name:
                    entry = candidate
                    ratio_label = f"{ratio_key} ({label})"
                    break
            if entry:
                break

    if entry:
        info["num_params_display"] = f"{entry['num_params']:,}"
        info["pruning_ratio_display"] = ratio_label
        info["test_accuracy_display"] = f"{entry['accuracy']*100:.2f}%"
    else:
        print(f"[PERINGATAN] Checkpoint '{ckpt_name}' tidak ditemukan di "
              f"{table_path.name} -- panel info model akan menampilkan "
              f"'Tidak tercatat' untuk parameter, rasio, dan akurasi.")

    return info


def predict(model, image_bytes):
    """Jalankan inferensi pada satu gambar, kembalikan seluruh probabilitas kelas."""
    import time

    transform = get_transforms("test")
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    end = time.perf_counter()

    inference_ms = (end - start) * 1000

    probs = probabilities[0].tolist()
    all_classes = []
    for idx, cls_name in enumerate(CFG.CLASS_NAMES):
        all_classes.append({
            "class": cls_name.replace("_", " "),
            "plant": CFG.PLANT_MAP.get(cls_name, "Unknown"),
            "score": probs[idx] * 100,
        })
    all_classes.sort(key=lambda c: c["score"], reverse=True)
    top = all_classes[0]

    return {
        "disease": top["class"],
        "plant": top["plant"],
        "confidence": f"{top['score']:.2f}",
        "inference_ms": f"{inference_ms:.2f}",
        "all_classes": [
            {"class": c["class"], "plant": c["plant"], "score": f"{c['score']:.2f}"}
            for c in all_classes
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Web Deployment Flask")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path ke checkpoint model pruned")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Checkpoint default eksplisit jika tidak dispesifikasi -- HARUS ada,
    # tidak jatuh ke berkas lain.
    if args.checkpoint is None:
        default_path = CFG.CHECKPOINT_DIR / DEFAULT_CHECKPOINT
        if not default_path.exists():
            print(f"[ERROR] Checkpoint default tidak ditemukan: {default_path}")
            print("Jalankan 12_multicriteria_valweights.py terlebih dahulu, "
                  "atau tentukan checkpoint lain lewat --checkpoint.")
            sys.exit(1)
        args.checkpoint = str(default_path)
        print(f"[INFO] Memakai checkpoint default: {args.checkpoint}")

    # Muat model
    model = load_model(args.checkpoint)
    verify_class_consistency(model)
    model_info = load_model_info(args.checkpoint)

    # Buat Flask app
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

    @app.errorhandler(413)
    def file_too_large(e):
        return jsonify({"error": "Ukuran berkas melebihi batas maksimum 10 MB."}), 413

    @app.route("/")
    def index():
        return render_template_string(HTML_TEMPLATE, model_info=model_info)

    @app.route("/predict", methods=["POST"])
    def predict_endpoint():
        if "image" not in request.files:
            return jsonify({"error": "Tidak ada file gambar"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "File kosong"}), 400

        ext = Path(file.filename).suffix.lower()
        if ext not in (".jpg", ".jpeg", ".png"):
            return jsonify({
                "error": f"Format '{ext or '(tanpa ekstensi)'}' tidak didukung. "
                         "Hanya menerima berkas JPG, JPEG, atau PNG."
            }), 400

        image_bytes = file.read()

        # Verifikasi isi benar-benar citra (bukan sekadar ekstensi), pada
        # salinan BytesIO terpisah -- Image.verify() tidak boleh dipakai
        # ulang untuk decode sesungguhnya.
        try:
            Image.open(io.BytesIO(image_bytes)).verify()
        except Exception:
            return jsonify({"error": "Berkas bukan citra yang valid."}), 400

        try:
            result = predict(model, image_bytes)
        except Exception:
            return jsonify({"error": "Berkas tidak bisa dibaca sebagai gambar."}), 400
        return jsonify(result)

    print(f"\n{'=' * 60}")
    print(f"WEB SERVER KLASIFIKASI PENYAKIT DAUN POACEAE")
    print(f"{'=' * 60}")
    print(f"  URL     : http://localhost:{args.port}")
    print(f"  Model   : {args.checkpoint}")
    print(f"  Kelas   : {CFG.NUM_CLASSES}")
    print(f"{'=' * 60}")
    print(f"\nBuka browser dan akses http://localhost:{args.port}")
    print(f"Tekan Ctrl+C untuk menghentikan server.\n")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
