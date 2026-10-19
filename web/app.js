// Bird Species Classifier – runs the exported ConvNeXt model in the browser with ONNX Runtime Web.
// model.onnx takes one 224x224 image and returns calibrated probabilities for the 200 species
// plus a Grad-CAM map of the top class (see export_web.py).

const UNSURE = 0.5; // below this top-1 confidence the app says it isn't sure
const ORT_DIST = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

const $ = (id) => document.getElementById(id);
let meta, session, current = null, truth = null, lastCam = null, lastView = null, busy = false;

ort.env.wasm.wasmPaths = ORT_DIST;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

// ---------- model loading ----------
async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = +res.headers.get("Content-Length") || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const buf = new Uint8Array(total);
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf.set(value, got);
    got += value.length;
    onProgress(got / total, got, total);
  }
  return buf;
}

async function init() {
  const status = $("modelStatus");
  try {
    meta = await (await fetch("meta.json")).json();
    renderExamples();
    renderAbout();
    const bytes = await fetchWithProgress("model.onnx", (f, got, total) => {
      $("modelBar").style.width = `${(f * 100).toFixed(0)}%`;
      $("modelText").textContent = `Loading model… ${(got / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`;
    });
    $("modelText").textContent = "Starting model…";
    session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
    status.classList.add("ready");
    $("modelText").textContent = "Model ready · runs on your device";
    if (current) predict();
  } catch (err) {
    console.error(err);
    status.classList.add("error");
    $("modelText").textContent = `Couldn't load the model (${err.message}). Try refreshing the page.`;
  }
}

// ---------- image helpers ----------
// High-quality downscale by halving steps (plain drawImage aliases badly on big photos).
function resizeTo(src, w, h) {
  let cur = src, cw = src.naturalWidth || src.width, ch = src.naturalHeight || src.height;
  while (cw / 2 >= w && ch / 2 >= h) {
    const c = document.createElement("canvas");
    c.width = Math.round(cw / 2); c.height = Math.round(ch / 2);
    const ctx = c.getContext("2d");
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(cur, 0, 0, c.width, c.height);
    cur = c; cw = c.width; ch = c.height;
  }
  const out = document.createElement("canvas");
  out.width = w; out.height = h;
  const ctx = out.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(cur, 0, 0, w, h);
  return out;
}

// Same as torchvision Resize(short side) + CenterCrop(size) used in evaluation.
function resizeCenterCrop(img, short, size) {
  const w = img.naturalWidth, h = img.naturalHeight;
  const s = short / Math.min(w, h);
  const rw = w <= h ? short : Math.floor(w * s);
  const rh = h <= w ? short : Math.floor(h * s);
  const resized = resizeTo(img, rw, rh);
  const out = document.createElement("canvas");
  out.width = out.height = size;
  out.getContext("2d").drawImage(resized, Math.round((rw - size) / 2), Math.round((rh - size) / 2), size, size, 0, 0, size, size);
  return out;
}

function toTensor(canvas) {
  const S = meta.img_size;
  const px = canvas.getContext("2d").getImageData(0, 0, S, S).data;
  const data = new Float32Array(3 * S * S);
  for (let i = 0; i < S * S; i++) {
    for (let c = 0; c < 3; c++) data[c * S * S + i] = (px[i * 4 + c] / 255 - meta.mean[c]) / meta.std[c];
  }
  return new ort.Tensor("float32", data, [1, 3, S, S]);
}

// ---------- Grad-CAM overlay ----------
const STOPS = [[40, 11, 84], [101, 21, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]];

function upsample(cam, ch, cw, size) { // bilinear, align_corners=False (as in PyTorch)
  const out = new Float32Array(size * size);
  for (let y = 0; y < size; y++) {
    const sy = Math.min(Math.max((y + 0.5) * ch / size - 0.5, 0), ch - 1);
    const y0 = Math.floor(sy), y1 = Math.min(y0 + 1, ch - 1), fy = sy - y0;
    for (let x = 0; x < size; x++) {
      const sx = Math.min(Math.max((x + 0.5) * cw / size - 0.5, 0), cw - 1);
      const x0 = Math.floor(sx), x1 = Math.min(x0 + 1, cw - 1), fx = sx - x0;
      const top = cam[y0 * cw + x0] * (1 - fx) + cam[y0 * cw + x1] * fx;
      const bot = cam[y1 * cw + x0] * (1 - fx) + cam[y1 * cw + x1] * fx;
      out[y * size + x] = top * (1 - fy) + bot * fy;
    }
  }
  return out;
}

function drawCam() {
  if (!lastCam || !lastView) return;
  const canvas = $("cam"), size = canvas.width, ctx = canvas.getContext("2d");
  const strength = +$("strength").value;
  ctx.drawImage(lastView, 0, 0);
  const img = ctx.getImageData(0, 0, size, size), px = img.data;
  for (let i = 0; i < size * size; i++) {
    const v = lastCam[i], pos = v * (STOPS.length - 1);
    const lo = Math.min(Math.floor(pos), STOPS.length - 2), f = pos - lo;
    const a = 0.65 * v * strength;
    for (let c = 0; c < 3; c++) {
      const heat = STOPS[lo][c] * (1 - f) + STOPS[lo + 1][c] * f;
      px[i * 4 + c] = px[i * 4 + c] * (1 - a) + heat * a;
    }
  }
  ctx.putImageData(img, 0, 0);
}

// ---------- prediction ----------
const pct = (p) => (p >= 0.01 ? `${(p * 100).toFixed(0)}%` : "<1%");

async function predict() {
  if (!current || !session || busy) return;
  busy = true;
  const img = current;
  $("verdict").className = "verdict busy";
  $("verdict").innerHTML = "<p>Identifying…</p>";
  await new Promise((r) => setTimeout(r, 20)); // let the UI update first
  try {
    const t0 = performance.now();
    const S = meta.img_size;
    const input = toTensor(resizeCenterCrop(img, meta.resize, S));
    const out = await session.run({ image: input });
    const ms = performance.now() - t0;
    if (img !== current) return; // a newer photo was chosen meanwhile
    const probs = out.probs.data, cam = out.cam.data, [, ch, cw] = out.cam.dims;
    const top = [...probs.keys()].sort((a, b) => probs[b] - probs[a]).slice(0, 5);
    renderResult(top, probs);
    renderTruth(top[0]);
    lastView = resizeCenterCrop(img, Math.round(S * 2 * 1.14), S * 2);
    lastCam = upsample(cam, ch, cw, S * 2);
    $("camPlaceholder").hidden = true;
    drawCam();
    $("timing").textContent = `Ran on your device in ${(ms / 1000).toFixed(2)} s`;
  } catch (err) {
    console.error(err);
    $("verdict").className = "verdict";
    $("verdict").innerHTML = `<p class="note">Something went wrong: ${err.message}</p>`;
  } finally {
    busy = false;
    if (img !== current) predict();
  }
}

function renderResult(top, probs) {
  const best = meta.classes[top[0]], conf = probs[top[0]];
  const v = $("verdict");
  if (conf >= UNSURE) {
    v.className = "verdict";
    v.innerHTML = `<p class="name"></p><p class="conf">${pct(conf)} confident</p>`;
    v.querySelector(".name").textContent = best;
  } else {
    v.className = "verdict unsure";
    v.innerHTML = `<p class="name"></p><p class="conf">Only ${pct(conf)} confident</p>
      <p class="note">The bird may be small, partly hidden or not one of the 200 species this model knows. Check the other guesses.</p>`;
    v.querySelector(".name").textContent = `Not sure – maybe ${best}?`;
  }
  $("top5").innerHTML = top.map((i) => `
    <li><span>${meta.classes[i]}</span><span class="pct">${pct(probs[i])}</span>
    <span class="track"><span class="fill" style="width:${Math.max(probs[i] * 100, 0.5)}%"></span></span></li>`).join("");
  $("refs").innerHTML = top.slice(0, 3).map((i) => `
    <figure><img src="reference/${i}.jpg" alt="Reference photo of ${meta.classes[i]}" loading="lazy">
    <figcaption>${meta.classes[i]} <span>${pct(probs[i])}</span></figcaption></figure>`).join("");
}

// For example photos: show the true species and whether the model got it right
function renderTruth(predicted) {
  const box = $("truth");
  if (truth === null) { box.hidden = true; return; }
  const ok = meta.classes[predicted] === truth;
  box.className = `truth ${ok ? "ok" : "bad"}`;
  box.innerHTML = `<span class="mark">${ok ? "✓" : "✗"}</span><span>True species: <b></b> · ${ok ? "the model is correct" : "the model got this one wrong"}</span>`;
  box.querySelector("b").textContent = truth;
  box.hidden = false;
}

function markExample(i) {
  document.querySelectorAll("#examples button").forEach((b) => b.classList.toggle("active", +b.dataset.i === i));
}

// ---------- input ----------
// `label` is the true species for example photos, null for the user's own photos
function loadImage(src, label = null) {
  const img = new Image();
  img.onload = () => {
    current = img;
    truth = label;
    $("truth").hidden = true;
    if (label === null) markExample(-1);
    $("preview").src = img.src;
    $("preview").hidden = false;
    $("dropHint").hidden = true;
    if (session) predict();
    else { $("verdict").className = "verdict busy"; $("verdict").innerHTML = "<p>Waiting for the model to load…</p>"; }
  };
  img.onerror = () => {
    $("verdict").className = "verdict";
    $("verdict").innerHTML = `<p class="note">That file couldn't be opened as an image. Try a JPG or PNG.</p>`;
  };
  img.src = src;
}

function loadFile(file) {
  if (file && file.type.startsWith("image/")) loadImage(URL.createObjectURL(file));
}

function clearAll() {
  current = lastCam = lastView = truth = null;
  $("truth").hidden = true;
  markExample(-1);
  $("file").value = "";
  $("preview").hidden = true;
  $("preview").removeAttribute("src");
  $("dropHint").hidden = false;
  $("verdict").className = "verdict";
  $("verdict").innerHTML = '<p class="muted">Upload a photo to get started.</p>';
  $("top5").innerHTML = "";
  $("timing").textContent = "";
  $("refs").innerHTML = '<p class="placeholder muted">Compare the photo against each guess here.</p>';
  const c = $("cam");
  c.getContext("2d").clearRect(0, 0, c.width, c.height);
  $("camPlaceholder").hidden = false;
}

function renderExamples() {
  if (!meta.examples?.length) return;
  $("examplesWrap").hidden = false;
  $("examples").innerHTML = meta.examples.map((e, i) =>
    `<button type="button" data-i="${i}" title="${e.species}"><img src="${e.file}" alt="Example: ${e.species}" loading="lazy"><span>${e.species}</span></button>`).join("");
  $("examples").addEventListener("click", (ev) => {
    const b = ev.target.closest("button");
    if (!b) return;
    const ex = meta.examples[b.dataset.i];
    markExample(+b.dataset.i);
    loadImage(ex.file, ex.species);
  });
}

function renderAbout() {
  const c = meta.config || {}, t = meta.test;
  const lines = [
    `<b>Backbone:</b> <code>${meta.model_name}</code> (ImageNet-pretrained), fine-tuned on CUB-200 (${meta.classes.length} species, ${meta.n_train.toLocaleString()} training photos).`,
  ];
  if (t) lines.push(`<b>Test set (${t.n.toLocaleString()} unseen photos):</b> top-1 accuracy <b>${(t.top1 * 100).toFixed(1)}%</b>, average per-class accuracy <b>${(t.avg_class * 100).toFixed(1)}%</b>.`);
  if (c.reg) lines.push(`<b>Training:</b> checkpoint from epoch ${meta.epoch} of ${c.epochs} (chosen by validation accuracy), AdamW, cosine learning rate, regularization preset <code>${c.reg}</code> (augmentation ${c.aug}, label smoothing ${c.label_smoothing}, weight decay ${c.weight_decay}, Mixup ${c.mixup}, CutMix ${c.cutmix}, random erasing ${c.random_erasing}, stochastic depth ${c.drop_path}, dropout ${c.drop_rate}).`);
  if (meta.temperature !== 1) lines.push(`<b>Confidence:</b> calibrated with temperature scaling (T = ${meta.temperature.toFixed(2)}, fitted on the validation split), so a shown confidence of 80% means the model is right about 80% of the time.`);
  lines.push("<b>Heatmap:</b> Grad-CAM highlights the regions that pushed the model towards its top answer.");
  lines.push("<b>Privacy:</b> the model runs in your browser with ONNX Runtime Web; photos never leave your device.");
  lines.push("<b>Limits:</b> only knows these 200 species; photos with several birds, or tiny or hidden birds, are harder. Some photos may overlap with the model's ImageNet pretraining data.");
  $("about").innerHTML = lines.map((l) => `<p>${l}</p>`).join("");
}

const drop = $("drop");
$("file").addEventListener("change", (e) => loadFile(e.target.files[0]));
$("browseBtn").addEventListener("click", () => $("file").click());
$("clearBtn").addEventListener("click", clearAll);
$("strength").addEventListener("input", drawCam);
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file").click(); } });
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => loadFile(e.dataTransfer.files[0]));
window.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item) loadFile(item.getAsFile());
});

init();
