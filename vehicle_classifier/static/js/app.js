/* ── Tab Navigation ──────────────────────────────────────────────── */
const navLinks = document.querySelectorAll('.nav-link');
const sections = document.querySelectorAll('.tab-section');

navLinks.forEach(link => {
  link.addEventListener('click', e => {
    e.preventDefault();
    const tab = link.dataset.tab;

    navLinks.forEach(l => l.classList.remove('active'));
    sections.forEach(s => s.classList.remove('active'));

    link.classList.add('active');
    document.getElementById(tab)?.classList.add('active');

    if (tab === 'dataset')  loadDatasetStats();
    if (tab === 'training') loadTrainingPlots();
  });
});

/* ── Drop Zone ───────────────────────────────────────────────────── */
const dropZone   = document.getElementById('dropZone');
const fileInput  = document.getElementById('fileInput');
const dropInner  = document.getElementById('dropInner');
const previewImg = document.getElementById('previewImg');
const classifyBtn= document.getElementById('classifyBtn');

let selectedFile = null;

function showPreview(file) {
  if (!file || !file.type.startsWith('image/')) return;
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.hidden = false;
  dropInner.style.display = 'none';
  classifyBtn.disabled = false;
}

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) showPreview(file);
});

dropZone.addEventListener('click', (e) => {
  // If the user clicked the label/button itself, let the label trigger the file
  // input. Prevent double-opening by not calling `fileInput.click()` again.
  if (e.target && (e.target.tagName && e.target.tagName.toLowerCase() === 'label')) return;
  if (e.target && e.target.classList && e.target.classList.contains('btn-upload')) return;
  fileInput.click();
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) showPreview(fileInput.files[0]);
});

/* ── Classify ────────────────────────────────────────────────────── */
classifyBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  const overlay = document.getElementById('loadingOverlay');
  overlay.hidden = false;

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const resp = await fetch('/predict', { method: 'POST', body: formData });
    const data = await resp.json();
    overlay.hidden = true;

    if (data.error) {
      alert('Error: ' + data.error);
      return;
    }

    renderResult(data);
    renderCharts(data);

  } catch (err) {
    overlay.hidden = true;
    alert('Network error: ' + err.message);
  }
});

/* ── Render Result ───────────────────────────────────────────────── */
function renderResult(data) {
  document.getElementById('resultPlaceholder').hidden = true;
  document.getElementById('resultContent').hidden = false;

  document.getElementById('predEmoji').textContent    = data.emoji;
  document.getElementById('predClass').textContent    = data.prediction;
  document.getElementById('predCategory').textContent = data.category;
  document.getElementById('predConf').textContent     = data.confidence + '%';

  const predCard = document.getElementById('predCard');
  predCard.style.borderColor = data.color;
  predCard.style.background  = hexToRgba(data.color, 0.07);

  // Confidence bar
  setTimeout(() => {
    document.getElementById('confFill').style.width = data.confidence + '%';
    document.getElementById('confVal').textContent  = data.confidence + '%';
  }, 50);

  // Top results list
  const topResultsEl = document.getElementById('topResults');
  topResultsEl.innerHTML = '';
  data.results.forEach(r => {
    const row = document.createElement('div');
    row.className = 'top-result-row';
    row.innerHTML = `
      <span class="tr-label">${r.class}</span>
      <div class="tr-bar-wrap">
        <div class="tr-bar-fill" style="width:${r.confidence}%"></div>
      </div>
      <span class="tr-pct">${r.confidence.toFixed(1)}%</span>
    `;
    topResultsEl.appendChild(row);
  });
}

/* ── Render Charts ───────────────────────────────────────────────── */
function renderCharts(data) {
  const row = document.getElementById('chartsRow');
  row.hidden = false;
  document.getElementById('barChartImg').src   = data.bar_chart;
  document.getElementById('radarChartImg').src = data.radar_chart;
  document.getElementById('pieChartImg').src   = data.pie_chart;
}

/* ── Dataset Stats ───────────────────────────────────────────────── */
let datasetLoaded = false;

async function loadDatasetStats() {
  if (datasetLoaded) return;
  try {
    const resp = await fetch('/dataset_stats');
    const data = await resp.json();
    const stats = data.stats;

    if (stats.total_images) document.getElementById('dsTotal').textContent  = stats.total_images.toLocaleString();
    if (stats.total_labels) document.getElementById('dsLabels').textContent = stats.total_labels.toLocaleString();
    if (stats.top_class)    document.getElementById('dsTopClass').textContent = stats.top_class;

    // Bar chart
    const barChart = document.getElementById('dsBarChart');
    const barLoader = document.getElementById('dsBarLoader');
    barChart.src = data.bar_chart;
    barChart.hidden = false;
    barLoader.hidden = true;

    // Donut
    const pieChart = document.getElementById('dsPieChart');
    const pieLoader = document.getElementById('dsPieLoader');
    pieChart.src = data.category_donut;
    pieChart.hidden = false;
    pieLoader.hidden = true;

    datasetLoaded = true;
  } catch (e) {
    console.error('Dataset stats error:', e);
  }
}

/* ── Training Plots ──────────────────────────────────────────────── */
let trainingLoaded = false;

async function loadTrainingPlots() {
  if (trainingLoaded) return;
  try {
    const resp = await fetch('/training_plots');
    const plots = await resp.json();

    if (plots.training_curves) {
      document.getElementById('trainCurves').src    = plots.training_curves;
      document.getElementById('trainCurves').hidden = false;
      document.getElementById('trainLoader1').hidden = true;
    } else {
      document.getElementById('trainLoader1').textContent = 'Not available yet. Train the model first.';
    }

    if (plots.class_distribution) {
      document.getElementById('trainDist').src    = plots.class_distribution;
      document.getElementById('trainDist').hidden = false;
      document.getElementById('trainLoader2').hidden = true;
    } else {
      document.getElementById('trainLoader2').textContent = 'Not available yet.';
    }

    if (plots.confusion_matrix) {
      document.getElementById('trainCM').src    = plots.confusion_matrix;
      document.getElementById('trainCM').hidden = false;
      document.getElementById('trainLoader3').hidden = true;
    } else {
      document.getElementById('trainLoader3').textContent = 'Not available yet.';
    }

    trainingLoaded = true;
  } catch (e) {
    console.error('Training plots error:', e);
  }
}

/* ── Helpers ─────────────────────────────────────────────────────── */
function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}
