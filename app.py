"""
Flask Backend - ACNet Liver/Tumor Segmentation
Uses acnet_best.pth (Tumor Dice: 0.8692, epoch 28)

KEY FIXES vs old app.py:
  1. ACNet class (not ACNetViT) — matches training notebook exactly
  2. acnet_best.pth with 'model_state' key — 234/234 weights loaded
  3. Preprocessing: [0,1] only — matches notebook LiverSliceDataset exactly
     (notebook saves HU-windowed CT as [0,1] float in .npz, no extra norm)
  4. Result image = CT grayscale + semi-transparent color overlay (visible always)
  5. Screenshot detection: warns user to upload raw CT PNG, not a screenshot
"""

import os
import uuid
import base64
import torch
import torch.nn.functional as F
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
from model import ACNet

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

IMG_SIZE = 256

# ── Model Load ───────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

model = ACNet(in_ch=1, n_classes=3, base=32)
checkpoint = torch.load('acnet_best.pth', map_location=DEVICE, weights_only=False)
state_dict = checkpoint['model_state']

if any(k.startswith('module.') for k in state_dict.keys()):
    state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

model.load_state_dict(state_dict, strict=True)
print(f"✅ Model: epoch {checkpoint['epoch']} | Tumor Dice {checkpoint['tumor_dice']:.4f} | 234/234 weights")

model.to(DEVICE)
model.eval()


# ── Screenshot Detection ─────────────────────────────────────────────────────
def is_screenshot(arr_gray):
    """
    A screenshot is typically:
      - Very wide (aspect ratio > 1.5)
      - High mean brightness (lots of white UI background)
    A real CT scan image is:
      - Roughly square or portrait
      - Mostly dark (lots of black background around the scan)
    """
    h, w = arr_gray.shape
    aspect = w / h
    mean_brightness = arr_gray.mean()

    # Screenshot heuristics: wide AND bright
    if aspect > 1.4 and mean_brightness > 150:
        return True
    # Very large image (likely a screenshot from a 1080p or 1440p screen)
    if w > 900 and h > 500 and mean_brightness > 130:
        return True
    return False


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(image_path):
    """
    Exact normalization matching notebook LiverSliceDataset.__getitem__:
      - .npz stores HU-windowed CT as float32 in [0, 1]
      - Dataset loads it raw: img.astype(np.float32) — no extra normalization
      - So model expects [0, 1] input

    Returns: (tensor, ct_pil_256, is_screenshot_flag)
    """
    img   = Image.open(image_path).convert('L')
    arr   = np.array(img, dtype=np.float32)

    screenshot_flag = is_screenshot(arr)

    # Resize directly to 256x256
    img_256 = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr_256 = np.array(img_256, dtype=np.float32) / 255.0  # [0, 1]

    tensor = torch.tensor(arr_256, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor.to(DEVICE), img_256, screenshot_flag


# ── Result: CT + Color Overlay ────────────────────────────────────────────────
def make_overlay(ct_gray_pil, pred_mask):
    """
    Blend grayscale CT with semi-transparent liver (green) and tumor (red) overlay.
    Result is always visible — no more pure black images.
    """
    ct_rgb  = np.array(ct_gray_pil.convert('RGB'), dtype=np.float32)
    overlay = ct_rgb.copy()

    liver_mask = pred_mask == 1
    tumor_mask = pred_mask == 2

    if liver_mask.any():
        overlay[liver_mask] = overlay[liver_mask] * 0.45 + np.array([0, 210, 0]) * 0.55
    if tumor_mask.any():
        overlay[tumor_mask] = overlay[tumor_mask] * 0.35 + np.array([255, 30, 30]) * 0.65

    return overlay.clip(0, 255).astype(np.uint8)


# ── Probability Stats ─────────────────────────────────────────────────────────
def compute_probabilities(prob_map, pred_mask):
    total_pixels = pred_mask.size
    stats = {}
    for cls_idx, cls_name in enumerate(['background', 'liver', 'tumor']):
        cls_prob = prob_map[cls_idx]
        cls_mask = pred_mask == cls_idx
        count    = int(cls_mask.sum())
        stats[cls_name] = {
            'pixel_count'   : count,
            'pixel_pct'     : round(count / total_pixels * 100, 2),
            'mean_conf'     : round(float(cls_prob.mean()) * 100, 2),
            'regional_conf' : round(float(cls_prob[cls_mask].mean()) * 100, 2)
                              if count > 0 else 0.0,
        }
    return stats


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file found in request!'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected!'}), 400

    ext      = os.path.splitext(file.filename)[1] or '.png'
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    return jsonify({'filename': filename})


@app.route('/process', methods=['POST'])
def process():
    data     = request.get_json()
    filename = data.get('filename')
    if not filename:
        return jsonify({'error': 'Filename missing!'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found on server!'}), 404

    try:
        tensor, ct_pil, screenshot_flag = preprocess(filepath)

        # Warn if screenshot detected — model works on raw CT images only
        if screenshot_flag:
            return jsonify({
                'error': (
                    'Screenshot detected! Please upload a raw CT scan image '
                    '(grayscale PNG/JPG of a single CT slice), not a screenshot of the app. '
                    'Export the CT slice directly from your imaging software.'
                )
            }), 400

        with torch.no_grad():
            logits   = model(tensor)
            probs    = F.softmax(logits, dim=1)
            prob_map = probs.squeeze(0).cpu().numpy()
            pred     = logits.argmax(dim=1).squeeze().cpu().numpy()

        # CT + color overlay result (always visible)
        overlay_rgb = make_overlay(ct_pil, pred)
        result_filename = f"result_{filename.rsplit('.', 1)[0]}.png"
        result_path     = os.path.join(RESULT_FOLDER, result_filename)
        Image.fromarray(overlay_rgb).save(result_path)

        prob_stats     = compute_probabilities(prob_map, pred)
        tumor_detected = prob_stats['tumor']['pixel_count'] > 0

        with open(result_path, 'rb') as f:
            result_b64 = base64.b64encode(f.read()).decode('utf-8')

        liver_conf = prob_stats['liver']['regional_conf']
        tumor_conf = prob_stats['tumor']['regional_conf']

        return jsonify({
            'result_image'    : result_b64,
            'liver_prob'      : liver_conf,
            'tumor_prob'      : tumor_conf,
            'liver_dsc'       : "{:.2f}".format(liver_conf / 100),
            'tumor_dsc'       : "{:.2f}".format(tumor_conf / 100),
            'result_filename' : result_filename,
            'tumor_detected'  : tumor_detected,
            'probabilities'   : {
                'liver'      : prob_stats['liver'],
                'tumor'      : prob_stats['tumor'],
                'background' : prob_stats['background'],
            }
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/result/<filename>')
def result(filename):
    return send_from_directory(RESULT_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)
