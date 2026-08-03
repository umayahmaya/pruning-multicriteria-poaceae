"""
06_deploy_flask.py
Web deployment menggunakan Flask untuk klasifikasi penyakit daun multi-crop

Jalankan:
    python scripts/06_deploy_flask.py
    python scripts/06_deploy_flask.py --port 8080

Akses di browser: http://localhost:5000
Pilih model lewat dropdown "Rasio Pemangkasan" di halaman -- seluruh 6
model (baseline + rasio 10/20/30/40/50 persen, bobot val-derived) dimuat
sekali saat startup dan disimpan di memori.
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


# Model yang dimuat saat startup: (kunci rasio, label tampilan, nama berkas
# checkpoint). Kunci "baseline" dan bobot val-derived (_valweights) dipakai
# konsisten dengan klaim tesis -- lihat outputs/tabel_hasil_lengkap.json.
MODEL_CHOICES = [
    ("baseline", "Baseline (tanpa pemangkasan)", "baseline_9class.pth"),
    ("10", "Rasio 10%", "multicriteria_10pct_30ep_valweights.pth"),
    ("20", "Rasio 20% (bawaan)", "multicriteria_20pct_30ep_valweights.pth"),
    ("30", "Rasio 30%", "multicriteria_30pct_30ep_valweights.pth"),
    ("40", "Rasio 40%", "multicriteria_40pct_30ep_valweights.pth"),
    ("50", "Rasio 50%", "multicriteria_50pct_30ep_valweights.pth"),
]
MODEL_LABELS = {key: label for key, label, _ in MODEL_CHOICES}
DEFAULT_RATIO_KEY = "20"


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
        .field { margin-bottom: 16px; }
        .field label {
            display: block;
            margin-bottom: 6px;
            font-size: 14px;
            color: #333;
            font-weight: bold;
        }
        .field select {
            width: 100%;
            padding: 10px;
            border-radius: 6px;
            border: 1px solid #ccc;
            font-size: 14px;
            background: white;
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
        .disease-panel.tipe-cendawan { background: #fff8e1; border: 1px solid #f9a825; }
        .disease-panel.tipe-virus { background: #ffebee; border: 1px solid #c62828; }
        .disease-panel.tipe-tidak-ada { background: #e8f5e9; border: 1px solid #2e7d32; }
        .patogen-badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            color: white;
        }
        .patogen-badge.cendawan { background: #f9a825; }
        .patogen-badge.virus { background: #c62828; }
        .patogen-badge.tidak-ada { background: #2e7d32; }
        .virus-warning {
            background: #c62828;
            color: white;
            padding: 10px 14px;
            border-radius: 6px;
            font-weight: bold;
            margin: 12px 0;
        }
        .disease-list { margin: 6px 0 10px 20px; }
        .disease-list li { margin-bottom: 4px; }
        .disease-note {
            margin-top: 16px;
            font-size: 11px;
            color: #888;
            line-height: 1.5;
        }
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
            <div class="field">
                <label for="ratioSelect">Pilih Model (Rasio Pemangkasan)</label>
                <select id="ratioSelect" name="ratio">
                    {% for key, label, _ in model_choices %}
                    <option value="{{ key }}"{% if key == default_ratio %} selected{% endif %}>{{ label }}</option>
                    {% endfor %}
                </select>
            </div>
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

        function renderDiseaseInfo(info, catatan) {
            if (!info) {
                return '<div class="result"><h3>Informasi Penanganan</h3>' +
                    '<p style="color:#888">Informasi penanganan belum tersedia untuk kelas ini.</p></div>';
            }
            const tipeSlug = info.tipe_patogen === 'Cendawan' ? 'cendawan' :
                (info.tipe_patogen === 'Virus' ? 'virus' : 'tidak-ada');
            let html = '<div class="result disease-panel tipe-' + tipeSlug + '">';
            html += '<h3>Informasi Penanganan: ' + info.nama_penyakit + '</h3>';
            html += '<p><span class="patogen-badge ' + tipeSlug + '">' + info.tipe_patogen +
                '</span> &nbsp;' + info.patogen + '</p>';
            if (info.tipe_patogen === 'Virus') {
                html += '<div class="virus-warning">PERINGATAN: Penyakit virus ini tidak dapat ' +
                    'diobati. Penanganan berupa eradikasi (pemusnahan tanaman terinfeksi).</div>';
            }
            html += '<p style="margin-top:10px">' + info.ringkasan + '</p>';
            html += '<h4 style="margin-top:14px">Penanganan</h4><ul class="disease-list">' +
                info.penanganan.map(function (x) { return '<li>' + x + '</li>'; }).join('') + '</ul>';
            html += '<h4>Pencegahan</h4><ul class="disease-list">' +
                info.pencegahan.map(function (x) { return '<li>' + x + '</li>'; }).join('') + '</ul>';
            if (info.sumber && info.sumber.length) {
                html += '<h4>Sumber</h4><ul class="disease-list">' +
                    info.sumber.map(function (x) { return '<li>' + x + '</li>'; }).join('') + '</ul>';
            }
            if (catatan) {
                html += '<p class="disease-note">' + catatan + '</p>';
            }
            html += '</div>';
            return html;
        }

        function renderEfficiencyPanel(eff, inferenceMs) {
            if (!eff) return '';
            let html = '<div class="result"><h3>Panel Efisiensi Model (' + eff.model_label + ')</h3>';
            html += '<div class="result-item"><span>Jumlah Parameter</span><span>' +
                eff.num_params + ' <span style="color:#666">(' + eff.num_params_change + ')</span></span></div>';
            html += '<div class="result-item"><span>Ukuran Model</span><span>' +
                eff.model_size_mb + ' <span style="color:#666">(' + eff.model_size_mb_change + ')</span></span></div>';
            html += '<div class="result-item"><span>FLOPs</span><span>' +
                eff.flops + ' <span style="color:#666">(' + eff.flops_change + ')</span></span></div>';
            html += '<div class="result-item"><span>Akurasi Uji Tercatat</span><span>' +
                eff.accuracy + ' <span style="color:#666">(' + eff.accuracy_change + ')</span></span></div>';
            html += '<div class="result-item"><span>Waktu Inferensi (barusan)</span><span>' +
                inferenceMs + ' ms</span></div>';
            html += '<p class="disease-note">Akurasi uji yang ditampilkan merupakan hasil satu ' +
                'kali pelatihan dengan seed 42. Validasi multi-seed menunjukkan variasi akurasi ' +
                'antar seed sebesar 0,81 sampai 1,40 poin persentase.</p>';
            html += '</div>';
            return html;
        }

        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const formData = new FormData();
            const fileInput = document.getElementById('fileInput');
            if (!fileInput.files[0]) return;

            formData.append('image', fileInput.files[0]);
            formData.append('ratio', document.getElementById('ratioSelect').value);
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

                html += renderDiseaseInfo(data.disease_info, data.catatan_penanganan);
                html += renderEfficiencyPanel(data.efficiency, data.inference_ms);

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
            "Deployment wajib memakai checkpoint terlatih -- jalankan "
            "12_multicriteria_valweights.py untuk menghasilkan checkpoint yang hilang."
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


def load_hasil_table():
    """Muat outputs/tabel_hasil_lengkap.json sekali saat startup -- sumber
    tunggal nilai acuan Panel Efisiensi Model (jumlah parameter, ukuran MB,
    FLOPs, akurasi uji). Tidak ada nilai ini yang ditulis sebagai konstanta
    di kode. Mengembalikan None kalau berkas tidak ada (soft-warning, bukan
    hard-fail, karena aplikasi tetap bisa melayani prediksi tanpa panel ini)."""
    table_path = CFG.OUTPUT_DIR / "tabel_hasil_lengkap.json"
    if not table_path.exists():
        print(f"[PERINGATAN] {table_path} tidak ditemukan -- Panel Efisiensi "
              "Model akan menampilkan 'Tidak tercatat' untuk seluruh metrik acuan. "
              "Jalankan 17_generate_final_table.py untuk mengisi ini.")
        return None
    with open(table_path) as f:
        return json.load(f)


def load_efficiency_metrics(checkpoint_name, table):
    """Cari entri checkpoint_name di tabel_hasil_lengkap.json, kembalikan
    nilai MENTAH (num_params, model_size_mb, flops, accuracy) -- bukan
    string tampilan -- supaya persentase perubahan terhadap baseline bisa
    dihitung ulang setiap request. None kalau tidak ditemukan."""
    if table is None:
        return None
    results = table.get("results", {})
    base = results.get("baseline")
    if base and base.get("checkpoint") == checkpoint_name:
        return {k: base[k] for k in ("num_params", "model_size_mb", "flops", "accuracy")}
    for scenario_key in ("l1", "bn", "entropy", "multicriteria"):
        for candidate in results.get(scenario_key, {}).values():
            if candidate.get("checkpoint") == checkpoint_name:
                return {k: candidate[k] for k in ("num_params", "model_size_mb", "flops", "accuracy")}
    return None


def load_penanganan_data():
    """Muat outputs/penanganan_penyakit.json sekali saat startup. Panel
    Informasi Penanganan adalah fitur inti (bukan pelengkap opsional),
    jadi berkas yang hilang membuat aplikasi berhenti, bukan diam-diam
    menonaktifkan panelnya."""
    data_path = CFG.OUTPUT_DIR / "penanganan_penyakit.json"
    if not data_path.exists():
        print(f"[ERROR] {data_path} tidak ditemukan. Panel Informasi Penanganan "
              "membutuhkan berkas ini untuk berjalan.")
        sys.exit(1)
    with open(data_path, encoding="utf-8") as f:
        return json.load(f)


def format_probability(score_percent):
    """Nilai di bawah 0,01 persen ditampilkan 4 desimal supaya tidak
    membulat jadi 0,00; selebihnya 2 desimal seperti biasa."""
    if score_percent < 0.01:
        return f"{score_percent:.4f}"
    return f"{score_percent:.2f}"


def build_efficiency_panel(ratio_key, efficiency_by_ratio):
    """Susun Panel Efisiensi Model untuk rasio yang dipakai pada satu
    prediksi: nilai mentah dari tabel_hasil_lengkap.json plus perubahan
    terhadap baseline, dihitung saat itu juga. Akurasi dilaporkan sebagai
    selisih POIN PERSENTASE (bukan persentase relatif) karena akurasi
    sendiri sudah berupa persentase -- num_params/model_size_mb/flops
    tetap persentase relatif seperti biasa."""
    panel = {"model_label": MODEL_LABELS.get(ratio_key, ratio_key)}
    current = efficiency_by_ratio.get(ratio_key)
    baseline = efficiency_by_ratio.get("baseline")

    metrics = (
        ("num_params", lambda v: f"{v:,}", "percent"),
        ("model_size_mb", lambda v: f"{v:.2f} MB", "percent"),
        ("flops", lambda v: f"{v:,.0f}", "percent"),
        ("accuracy", lambda v: f"{v * 100:.2f}%", "points"),
    )
    for name, fmt, change_type in metrics:
        if not current or current.get(name) is None:
            panel[name] = "Tidak tercatat"
            panel[f"{name}_change"] = "Tidak tercatat"
            continue
        value = current[name]
        panel[name] = fmt(value)
        if ratio_key == "baseline":
            panel[f"{name}_change"] = "Referensi"
            continue
        base_value = baseline.get(name) if baseline else None
        if base_value is None or (change_type == "percent" and base_value == 0):
            panel[f"{name}_change"] = "Tidak tercatat"
        elif change_type == "points":
            diff_points = (value - base_value) * 100
            panel[f"{name}_change"] = f"{diff_points:+.2f} poin"
        else:
            pct = (value - base_value) / base_value * 100
            panel[f"{name}_change"] = f"{pct:+.2f}%"
    return panel


def predict(model, image_bytes, penanganan_data):
    """Jalankan inferensi pada satu gambar, kembalikan seluruh probabilitas
    kelas beserta informasi penanganan penyakit untuk kelas teratas."""
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
            "raw_class": cls_name,
            "plant": CFG.PLANT_MAP.get(cls_name, "Unknown"),
            "score": probs[idx] * 100,
        })
    all_classes.sort(key=lambda c: c["score"], reverse=True)
    top = all_classes[0]

    disease_info = penanganan_data.get("kelas", {}).get(top["raw_class"])
    catatan = penanganan_data.get("_catatan")

    return {
        "disease": top["class"],
        "plant": top["plant"],
        "confidence": format_probability(top["score"]),
        "inference_ms": f"{inference_ms:.2f}",
        "all_classes": [
            {"class": c["class"], "plant": c["plant"], "score": format_probability(c["score"])}
            for c in all_classes
        ],
        "disease_info": disease_info,
        "catatan_penanganan": catatan,
    }


def main():
    parser = argparse.ArgumentParser(description="Web Deployment Flask")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    hasil_table = load_hasil_table()
    penanganan_data = load_penanganan_data()

    # Muat keenam model sekali saat startup. Checkpoint hilang atau gagal
    # verifikasi (parameter/urutan kelas) membuat aplikasi berhenti dengan
    # pesan yang menyebut checkpoint mana -- tidak ada fallback diam-diam.
    models = {}
    efficiency_by_ratio = {}
    for ratio_key, label, ckpt_filename in MODEL_CHOICES:
        ckpt_path = CFG.CHECKPOINT_DIR / ckpt_filename
        try:
            model = load_model(ckpt_path)
            verify_class_consistency(model)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[ERROR] Gagal memuat model '{label}' ({ckpt_filename}): {e}")
            sys.exit(1)
        models[ratio_key] = model
        efficiency_by_ratio[ratio_key] = load_efficiency_metrics(ckpt_filename, hasil_table)
        if efficiency_by_ratio[ratio_key] is None:
            print(f"[PERINGATAN] Checkpoint '{ckpt_filename}' ({label}) tidak ditemukan "
                  "di tabel_hasil_lengkap.json -- Panel Efisiensi Model akan menampilkan "
                  "'Tidak tercatat' untuk rasio ini.")
        print(f"[INFO] Model dimuat: {label} ({ckpt_filename})")

    # Buat Flask app
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

    @app.errorhandler(413)
    def file_too_large(e):
        return jsonify({"error": "Ukuran berkas melebihi batas maksimum 10 MB."}), 413

    @app.route("/")
    def index():
        return render_template_string(
            HTML_TEMPLATE, model_choices=MODEL_CHOICES, default_ratio=DEFAULT_RATIO_KEY
        )

    @app.route("/predict", methods=["POST"])
    def predict_endpoint():
        if "image" not in request.files:
            return jsonify({"error": "Tidak ada file gambar"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "File kosong"}), 400

        ratio_key = request.form.get("ratio", DEFAULT_RATIO_KEY)
        if ratio_key not in models:
            return jsonify({"error": f"Rasio model '{ratio_key}' tidak dikenal."}), 400

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
            result = predict(models[ratio_key], image_bytes, penanganan_data)
        except Exception:
            return jsonify({"error": "Berkas tidak bisa dibaca sebagai gambar."}), 400

        result["efficiency"] = build_efficiency_panel(ratio_key, efficiency_by_ratio)
        return jsonify(result)

    print(f"\n{'=' * 60}")
    print(f"WEB SERVER KLASIFIKASI PENYAKIT DAUN POACEAE")
    print(f"{'=' * 60}")
    print(f"  URL     : http://localhost:{args.port}")
    print(f"  Model   : {len(models)} rasio dimuat ({', '.join(MODEL_LABELS.values())})")
    print(f"  Kelas   : {CFG.NUM_CLASSES}")
    print(f"{'=' * 60}")
    print(f"\nBuka browser dan akses http://localhost:{args.port}")
    print(f"Tekan Ctrl+C untuk menghentikan server.\n")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
