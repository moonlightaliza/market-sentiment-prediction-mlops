/* ============================================================
   main.js — MarketPulse Sentiment Predictor
   Sections:
     1. Config
     2. State
     3. Tab Switching
     4. Sample Prompts
     5. Input Helpers
     6. Main Analysis Flow
     7. API Call
     8. Render: Result Card
     9. Render: Error
    10. Keyboard Shortcut
   ============================================================ */


/* ── 1. Config ────────────────────────────────────────────── */
/*
   BACKEND_URL is injected by index.html via window.BACKEND_URL.

   Development (FastAPI serves the frontend — same origin):
     window.BACKEND_URL = '';          ← no CORS needed

   After deploying to Railway / Render / etc., set in index.html:
     window.BACKEND_URL = 'https://marketpulse.up.railway.app';
*/
const BACKEND_URL = window.BACKEND_URL ?? '';


/* ── 2. State ─────────────────────────────────────────────── */

let activeTab = 'text';


/* ── 3. Tab Switching ─────────────────────────────────────── */

function switchTab(tab, el) {
  activeTab = tab;

  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  el.classList.add('active');

  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-' + tab).classList.add('active');

  clearResult();
}


/* ── 4. Sample Prompts ────────────────────────────────────── */

function useSample(el) {
  switchTab('text', document.querySelectorAll('.tab')[0]);
  document.getElementById('news-input').value = el.textContent.trim();
  clearResult();
}


/* ── 5. Input Helpers ─────────────────────────────────────── */

function getInput() {
  if (activeTab === 'text') {
    return document.getElementById('news-input').value.trim();
  }
  return document.getElementById('ticker-input').value.trim();
}

function clearResult() {
  const r = document.getElementById('result');
  r.classList.add('hidden');
  r.innerHTML = '';
}

function setButtonLoading(btn) {
  btn.disabled = true;
  btn.innerHTML = '<div class="spinner"></div> Analyzing…';
}

function resetButton(btn) {
  btn.disabled = false;
  btn.innerHTML = `
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"
         stroke-linecap="round" stroke-linejoin="round">
      <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
    </svg>
    Analyze Sentiment
  `;
}


/* ── 6. Main Analysis Flow ────────────────────────────────── */

async function runAnalysis() {
  const input = getInput();
  if (!input) return;

  const btn = document.getElementById('analyze-btn');
  setButtonLoading(btn);
  clearResult();

  try {
    const data = await fetchPrediction(input);
    renderResult(data, input);
  } catch (err) {
    renderError(err.message);
  } finally {
    resetButton(btn);
  }
}


/* ── 7. API Call ──────────────────────────────────────────── */
/*
   Calls POST /api/predict on the FastAPI backend.

   The backend returns:
     {
       "sentiment":  "bullish" | "bearish" | "neutral",
       "score":      0.87,
       "confidence": 0.91,
       "breakdown":  { "bullish": 0.87, "neutral": 0.09, "bearish": 0.04 },
       "summary":    "HTML string"
     }
*/

async function fetchPrediction(input) {
  const payload = activeTab === 'text'
    ? { text: input }
    : { ticker: input.toUpperCase() };

  const response = await fetch(`${BACKEND_URL}/api/predict`, {
    method  : 'POST',
    headers : { 'Content-Type': 'application/json' },
    body    : JSON.stringify(payload),
  });

  if (!response.ok) {
    let detail = `Server error ${response.status}`;
    try {
      const err = await response.json();
      // FastAPI validation errors come as { detail: [...] }
      if (Array.isArray(err.detail)) {
        detail = err.detail.map(e => e.msg).join(', ');
      } else if (typeof err.detail === 'string') {
        detail = err.detail;
      }
    } catch (_) { /* ignore JSON parse errors */ }
    throw new Error(detail);
  }

  return response.json();
}


/* ── 8. Render: Result Card ───────────────────────────────── */

function renderResult(data, input) {
  const { sentiment, score, breakdown, summary, confidence } = data;

  const pct        = n => (n * 100).toFixed(1) + '%';
  const shortInput = input.length > 60 ? input.slice(0, 57) + '…' : input;
  const timestamp  = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const bKeys      = Object.entries(breakdown).sort((a, b) => b[1] - a[1]);

  const r = document.getElementById('result');
  r.classList.remove('hidden');
  r.innerHTML = `
    <div class="result-card">

      <div class="result-header">
        <div class="result-label-group">
          <span class="sentiment-badge badge-${sentiment}">${sentiment}</span>
          <span class="result-title">${shortInput}</span>
        </div>
        <span class="result-score-text score-${sentiment}">${pct(score)}</span>
      </div>

      <div class="result-body">

        <!-- Confidence bar -->
        <div>
          <div class="gauge-label">
            <span>CONFIDENCE</span>
            <span>${pct(confidence)}</span>
          </div>
          <div class="gauge-track">
            <div class="gauge-fill fill-${sentiment}" style="width:0%" id="gauge-fill"></div>
          </div>
        </div>

        <!-- Class breakdown -->
        <div>
          <div class="gauge-label" style="margin-bottom:10px">
            <span>CLASS BREAKDOWN</span>
          </div>
          <div class="breakdown">
            ${bKeys.map(([key, val]) => `
              <div class="breakdown-row">
                <span class="breakdown-key">${key.toUpperCase()}</span>
                <div class="mini-track">
                  <div class="mini-fill fill-${key}"
                       style="width:0%"
                       data-target="${(val * 100).toFixed(1)}%"
                       id="bar-${key}">
                  </div>
                </div>
                <span class="breakdown-val score-${key}">${pct(val)}</span>
              </div>
            `).join('')}
          </div>
        </div>

        <!-- Model summary -->
        <div class="result-summary">${summary}</div>

      </div>

      <div class="result-meta">
        <span>MODEL · market-sentiment-v1</span>
        <span>${timestamp}</span>
      </div>

    </div>
  `;

  // Animate bars after the DOM has painted
  requestAnimationFrame(() => {
    setTimeout(() => {
      document.getElementById('gauge-fill').style.width = pct(confidence);
      bKeys.forEach(([key]) => {
        const bar = document.getElementById('bar-' + key);
        if (bar) bar.style.width = bar.dataset.target;
      });
    }, 60);
  });
}


/* ── 9. Render: Error ─────────────────────────────────────── */

function renderError(message) {
  const r = document.getElementById('result');
  r.classList.remove('hidden');
  r.innerHTML = `
    <div style="
      background: var(--card);
      border: 1px solid var(--red-dim);
      border-radius: 16px;
      padding: 1.5rem 1.75rem;
      color: var(--red);
      font-size: 0.85rem;
      font-family: var(--mono);
    ">
      ERROR · ${message || 'Failed to reach prediction API'}
    </div>
  `;
}


/* ── 10. Keyboard Shortcut ────────────────────────────────── */

// Ctrl + Enter (or Cmd + Enter on Mac) triggers analysis
document.addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    runAnalysis();
  }
});