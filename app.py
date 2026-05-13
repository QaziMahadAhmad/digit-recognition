"""Step 3: FastAPI Backend + Frontend (Single Python File)
Then open: http://127.0.0.1:8000 """

import io
import base64
import numpy as np
import joblib
from PIL import Image, ImageOps
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ─── App instance ─────────────────────────────────────────────────────────────
app = FastAPI(title="Digit Recognition")

# ─── Load model artifacts once at startup ─────────────────────────────────────

print("Loading its will take a few seconds")

scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
knn_model = joblib.load("knn_model.pkl")
print("KNN model is ready with K =", knn_model.n_neighbors)
print("Data is scaled and PCA is applied")
print("Reduced to", pca.n_components_, "dimensions")
print("Everything is set for training/prediction")

# ─── Request / Response schemas (Pydantic) ────────────────────────────────────
class PredictRequest(BaseModel):
    """The frontend sends a base64-encoded PNG string."""
    image: str  # "data:image/png;base64,..." or raw base64


class PredictResponse(BaseModel):
    digit:         int
    confidence:    float          # percentage e.g. 87.5
    probabilities: list[float]    # 10 values, one per digit class


# ─── Image preprocessing helper ───────────────────────────────────────────────
def preprocess(image_data: str) -> np.ndarray:

    # Remove base64 header
    if "," in image_data:
        image_data = image_data.split(",")[1]

    # Decode image
    img_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(img_bytes)).convert("L")

    # Convert to numpy
    arr = np.array(img)

    # Invert if needed
    if arr.mean() > 127:
        arr = 255 - arr

    # Thresholding (remove gray noise)
    arr[arr < 50] = 0

    # Find bounding box of digit
    coords = np.argwhere(arr > 0)

    if len(coords) == 0:
        raise ValueError("Empty image")

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    # Crop digit
    arr = arr[y0:y1, x0:x1]

    # Convert back to PIL
    img = Image.fromarray(arr)

    # Resize while keeping aspect ratio
    img.thumbnail((20, 20), Image.LANCZOS)

    # Create black 28x28 background
    new_img = Image.new("L", (28, 28), 0)

    # Center digit
    paste_x = (28 - img.width) // 2
    paste_y = (28 - img.height) // 2

    new_img.paste(img, (paste_x, paste_y))

    # Convert to array
    arr = np.array(new_img).astype(np.float32)

    # Normalize like MNIST
    arr = arr.reshape(1, -1)

    # Scale + PCA
    arr_scaled = scaler.transform(arr)
    arr_pca = pca.transform(arr_scaled)

    return arr_pca

# ─── HTML page — one big Python string ────────────────────────────────────────
# HTMLResponse sends this directly; no Jinja2 or templates folder needed.
PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Digit Recognition — PCA + KNN</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Syne:wght@400;700;800&display=swap" rel="stylesheet"/>
  <style>
    :root {
      --bg:      #080b10;
      --surface: #0f1520;
      --border:  #1e2d45;
      --accent:  #00e5ff;
      --accent2: #ff4081;
      --text:    #e8f0f8;
      --muted:   #5a7a99;
      --mono:    'Share Tech Mono', monospace;
      --sans:    'Syne', sans-serif;
    }
    *, *::before, *::after { box-sizing:border-box; margin:0; padding:0; }

    body {
      background:var(--bg); color:var(--text); font-family:var(--sans);
      min-height:100vh; display:flex; align-items:center;
      justify-content:center; padding:2rem 1rem;
    }

    /* animated grid overlay */
    body::before {
      content:''; position:fixed; inset:0; pointer-events:none;
      background-image:
        linear-gradient(rgba(0,229,255,.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,229,255,.04) 1px, transparent 1px);
      background-size:40px 40px;
      animation:drift 20s linear infinite;
    }
    @keyframes drift { to { background-position:40px 40px; } }
    @keyframes up    { from { opacity:0; transform:translateY(20px); } to { opacity:1; transform:none; } }

    .shell {
      position:relative; z-index:1; width:100%; max-width:500px;
      display:flex; flex-direction:column; gap:1.5rem;
      animation:up .5s ease both;
    }

    /* ── header ── */
    header { text-align:center; }
    .badge {
      display:inline-block; font-family:var(--mono);
      font-size:.7rem; letter-spacing:.12em;
      color:var(--accent); border:1px solid var(--accent);
      border-radius:99px; padding:.22rem .85rem; margin-bottom:.7rem;
    }
    h1 { font-size:clamp(2.4rem,8vw,3.4rem); font-weight:800; letter-spacing:-.02em; }
    h1 span { color:var(--accent); }
    .sub { margin-top:.4rem; color:var(--muted); font-size:.9rem; }

    /* ── card ── */
    .card {
      background:var(--surface); border:1px solid var(--border);
      border-radius:14px; padding:1.6rem;
      display:flex; flex-direction:column; gap:1.2rem;
      box-shadow:0 0 28px rgba(0,229,255,.12), 0 8px 40px rgba(0,0,0,.5);
    }

    /* ── tabs ── */
    .tabs { display:flex; gap:.4rem; background:rgba(0,0,0,.3); border-radius:10px; padding:.3rem; }
    .tab {
      flex:1; padding:.5rem; border:none; background:transparent;
      color:var(--muted); border-radius:8px;
      font-family:var(--sans); font-size:.88rem; font-weight:700; cursor:pointer;
      transition:background .2s, color .2s;
    }
    .tab.on { background:var(--border); color:var(--accent); }
    .tab:hover:not(.on) { color:var(--text); }

    /* ── canvas ── */
    .cvs-wrap { display:flex; flex-direction:column; align-items:center; gap:.8rem; }
    #drawCanvas {
      border:2px solid var(--border); border-radius:10px;
      cursor:crosshair; background:#000;
      width:280px; height:280px; display:block; touch-action:none;
      transition:border-color .2s, box-shadow .2s;
    }
    #drawCanvas:hover { border-color:var(--accent); box-shadow:0 0 20px rgba(0,229,255,.2); }
    .cvs-row { display:flex; align-items:center; justify-content:space-between; width:280px; }
    .brush { display:flex; align-items:center; gap:.5rem; font-size:.82rem; color:var(--muted); }
    input[type=range] { accent-color:var(--accent); width:80px; }

    /* ── upload ── */
    .drop {
      display:flex; flex-direction:column; align-items:center;
      justify-content:center; gap:.4rem;
      border:2px dashed var(--border); border-radius:10px;
      padding:2rem 1rem; cursor:pointer; min-height:200px; text-align:center;
      transition:border-color .2s, background .2s;
    }
    .drop:hover { border-color:var(--accent); background:rgba(0,229,255,.04); }
    .drop-ico  { font-size:2rem; color:var(--accent); }
    .drop-main { font-weight:700; }
    .drop-sub  { font-size:.8rem; color:var(--muted); }
    #preview {
      max-width:110px; max-height:110px; border-radius:8px;
      display:none; border:1px solid var(--border); margin-top:.4rem;
    }

    /* ── utils ── */
    .hidden { display:none !important; }

    /* ── buttons ── */
    .btn { border:none; border-radius:10px; cursor:pointer; font-family:var(--sans); font-weight:700; transition:opacity .2s, transform .1s; }
    .btn:active { transform:scale(.97); }
    .btn-primary { background:var(--accent); color:#000; padding:.85rem; font-size:1rem; width:100%; }
    .btn-primary:hover { box-shadow:0 0 20px rgba(0,229,255,.4); }
    .btn-primary:disabled { opacity:.5; cursor:not-allowed; }
    .btn-sm { background:transparent; color:var(--muted); border:1px solid var(--border); padding:.38rem .85rem; font-size:.83rem; }
    .btn-sm:hover { color:var(--accent); border-color:var(--accent); }

    /* ── result ── */
    .result { text-align:center; border-top:1px solid var(--border); padding-top:1rem; animation:up .35s ease; }
    .big {
      font-family:var(--mono); font-size:5rem; color:var(--accent);
      line-height:1; text-shadow:0 0 40px rgba(0,229,255,.5);
    }
    .conf { font-size:.88rem; color:var(--muted); margin-top:.3rem; font-family:var(--mono); }
    .conf b { color:var(--accent2); }

    /* ── probability bars ── */
    .bars { display:flex; align-items:flex-end; justify-content:center; gap:5px; height:80px; margin-top:1rem; }
    .bw   { display:flex; flex-direction:column; align-items:center; gap:3px; flex:1; }
    .bar  { width:100%; border-radius:4px 4px 0 0; background:var(--border); min-height:2px; transition:height .5s cubic-bezier(.34,1.56,.64,1); }
    .bar.top { background:var(--accent); box-shadow:0 0 10px rgba(0,229,255,.4); }
    .blbl { font-family:var(--mono); font-size:.65rem; color:var(--muted); }

    /* ── error ── */
    .err {
      background:rgba(255,64,129,.1); border:1px solid var(--accent2);
      color:var(--accent2); border-radius:8px;
      padding:.7rem 1rem; font-size:.85rem; font-family:var(--mono); text-align:center;
    }

    /* ── pipeline footer ── */
    .pipe { display:flex; align-items:center; justify-content:center; gap:.3rem; flex-wrap:wrap; }
    .ps   { text-align:center; font-size:.72rem; color:var(--muted); line-height:1.6; }
    .pa   { color:var(--border); font-size:.85rem; }
  </style>
</head>
<body>
<div class="shell">

  <header>
    <h1>Digit<span>.</span>AI</h1>
    <p class="sub">Draw a digit or upload an image — the model will recognise it.</p>
  </header>

  <div class="card">

    <!-- Tabs -->
    <div class="tabs">
      <button class="tab on" id="t-draw"   onclick="switchTab('draw')">✏️ Draw</button>
      <button class="tab"    id="t-upload" onclick="switchTab('upload')">📂 Upload</button>
    </div>

    <!-- Draw panel -->
    <div id="p-draw" class="cvs-wrap">
      <canvas id="drawCanvas" width="280" height="280"></canvas>
      <div class="cvs-row">
        <button class="btn btn-sm" onclick="clearCanvas()">Clear</button>
        <label class="brush">Brush <input type="range" id="sz" min="8" max="32" value="18"/></label>
      </div>
    </div>

    <!-- Upload panel -->
    <div id="p-upload" class="hidden">
      <label class="drop" id="dz">
        <input type="file" id="fi" accept="image/*" hidden/>
        <div class="drop-ico">⬆</div>
        <p class="drop-main">Click or drag an image here</p>
        <p class="drop-sub">PNG · JPG · BMP — any digit image</p>
        <img id="preview" alt="preview"/>
      </label>
    </div>

    <!-- Submit -->
    <button class="btn btn-primary" id="pb" onclick="predict()">Recognise Digit</button>

    <!-- Result -->
    <div class="result hidden" id="res">
      <div class="big" id="rd">—</div>
      <div class="conf">Confidence: <b id="rc">—</b></div>
      <div class="bars" id="bars"></div>
    </div>

    <!-- Error -->
    <div class="err hidden" id="err"></div>

  </div>

  <!-- Pipeline steps -->
  
  

</div>

<script>
  /* ── Canvas drawing ── */
  const canvas = document.getElementById('drawCanvas');
  const ctx    = canvas.getContext('2d');
  let   drawing = false, hasStrokes = false;
  ctx.strokeStyle = '#fff';
  ctx.lineJoin = ctx.lineCap = 'round';

  function pos(e) {
    const r  = canvas.getBoundingClientRect();
    const sx = canvas.width / r.width, sy = canvas.height / r.height;
    const s  = e.touches ? e.touches[0] : e;
    return { x:(s.clientX - r.left)*sx, y:(s.clientY - r.top)*sy };
  }
  function brushSize() { return +document.getElementById('sz').value; }

  canvas.onmousedown  = e => { drawing=true; ctx.beginPath(); const p=pos(e); ctx.moveTo(p.x,p.y); };
  canvas.onmousemove  = e => { if(!drawing) return; ctx.lineWidth=brushSize(); const p=pos(e); ctx.lineTo(p.x,p.y); ctx.stroke(); hasStrokes=true; };
  canvas.onmouseup    = canvas.onmouseleave = () => drawing=false;
  canvas.ontouchstart = e => { e.preventDefault(); drawing=true; ctx.beginPath(); const p=pos(e); ctx.moveTo(p.x,p.y); };
  canvas.ontouchmove  = e => { e.preventDefault(); if(!drawing) return; ctx.lineWidth=brushSize(); const p=pos(e); ctx.lineTo(p.x,p.y); ctx.stroke(); hasStrokes=true; };
  canvas.ontouchend   = () => drawing=false;

  function clearCanvas() { ctx.clearRect(0,0,canvas.width,canvas.height); hasStrokes=false; hide('res'); hide('err'); }

  /* ── Tab switching ── */
  function switchTab(t) {
    ['draw','upload'].forEach(x => {
      document.getElementById('p-'+x).classList.toggle('hidden', x!==t);
      document.getElementById('t-'+x).classList.toggle('on', x===t);
    });
    hide('res'); hide('err');
  }

  /* ── File upload / drag-drop ── */
  let uploadedB64 = null;
  document.getElementById('fi').onchange = e => loadFile(e.target.files[0]);
  const dz = document.getElementById('dz');
  dz.ondragover  = e => { e.preventDefault(); dz.style.borderColor='var(--accent)'; };
  dz.ondragleave = ()  => dz.style.borderColor='';
  dz.ondrop = e => { e.preventDefault(); dz.style.borderColor=''; loadFile(e.dataTransfer.files[0]); };
  function loadFile(f) {
    if (!f) return;
    const r = new FileReader();
    r.onload = ev => {
      uploadedB64 = ev.target.result;
      const p = document.getElementById('preview');
      p.src = uploadedB64; p.style.display='block';
    };
    r.readAsDataURL(f);
  }

  /* ── Predict — calls POST /predict ── */
  async function predict() {
    const isDrawTab = !document.getElementById('p-draw').classList.contains('hidden');
    const imageData = isDrawTab
      ? (hasStrokes ? canvas.toDataURL('image/png') : null)
      : uploadedB64;

    if (!imageData) {
      showErr(isDrawTab ? 'Please draw a digit first.' : 'Please upload an image first.');
      return;
    }

    const btn = document.getElementById('pb');
    btn.disabled=true; btn.textContent='Thinking…';
    hide('res'); hide('err');

    try {
      const resp = await fetch('/predict', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ image: imageData }),
      });
      if (!resp.ok) {
        const e = await resp.json();
        showErr(e.detail || 'Server error'); return;
      }
      const data = await resp.json();
      showResult(data.digit, data.confidence, data.probabilities);
    } catch(e) {
      showErr('Network error — is the server running?');
    } finally {
      btn.disabled=false; btn.textContent='Recognise Digit';
    }
  }

  /* ── Render result ── */
  function showResult(digit, conf, probs) {
    document.getElementById('rd').textContent = digit;
    document.getElementById('rc').textContent = conf + '%';
    const max = Math.max(...probs);
    document.getElementById('bars').innerHTML = probs.map((p,i) => `
      <div class="bw">
        <div class="bar ${p===max?'top':''}" style="height:${Math.max(2,(p/100)*72)}px" title="Digit ${i}: ${p}%"></div>
        <span class="blbl">${i}</span>
      </div>`).join('');
    show('res');
  }

  function showErr(msg) { document.getElementById('err').textContent=msg; show('err'); hide('res'); }
  function show(id) { document.getElementById(id).classList.remove('hidden'); }
  function hide(id) { document.getElementById(id).classList.add('hidden'); }
</script>
</body>
</html>"""


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the UI — the entire page is the PAGE string defined above."""
    return PAGE


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    """
    POST /predict
    Body  : { "image": "<base64 PNG string>" }
    Returns: { "digit": int, "confidence": float, "probabilities": [float x 10] }

    FastAPI automatically validates the request body against PredictRequest
    and serialises the response against PredictResponse using Pydantic.
    """
    if not body.image:
        raise HTTPException(status_code=400, detail="No image data received.")

    try:
        arr_pca = preprocess(body.image)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Image processing failed: {e}")

    # KNN prediction
    digit = int(knn_model.predict(arr_pca)[0])

    # Build per-class confidence from neighbour votes
    _, indices       = knn_model.kneighbors(arr_pca)
    neighbour_labels = knn_model._y[indices[0]]
    votes            = np.bincount(neighbour_labels, minlength=10)
    probabilities    = (votes / votes.sum()).tolist()
    confidence       = round(probabilities[digit] * 100, 1)

    return PredictResponse(
        digit         = digit,
        confidence    = confidence,
        probabilities = [round(p * 100, 1) for p in probabilities],
    )


# ─── Entry point ──────────────────────────────────────────────────────────────
# Run with:  uvicorn app:app --reload
# Or directly:  python app.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)