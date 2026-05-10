/* ============================================================
   main.js — MarketPulse Sentiment Predictor
   Sections:
     1. State
     2. Tab Switching
     3. Sample Prompts
     4. Input Helpers
     5. Main Analysis Flow
     6. API Call  ← replace mock with real endpoint here
     7. Mock Predictor (remove after API is connected)
     8. Render: Result Card
     9. Render: Error
    10. Keyboard Shortcut
   ============================================================ */


/* ── 1. State ─────────────────────────────────────────────── */

let activeTab = 'text';


/* ── 2. Tab Switching ─────────────────────────────────────── */

function switchTab(tab, el) {
  activeTab = tab;

  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  el.classList.add('active');

  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-' + tab).classList.add('active');

  clearResult();
}


/* ── 3. Sample Prompts ────────────────────────────────────── */

function useSample(el) {
  // Always switch to the text tab when a sample is clicked
  switchTab('text', document.querySelectorAll('.tab')[0]);
  document.getElementById('news-input').value = el.textContent.trim();
  clearResult();
}


/* ── 4. Input Helpers ─────────────────────────────────────── */

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


/* ── 5. Main Analysis Flow ────────────────────────────────── */

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


/* ── 6. API Call ──────────────────────────────────────────── */
/*
   Once your FastAPI / Flask backend is running, delete the
   mockPredict() call below and uncomment this real fetch.

   Your endpoint should accept:
     POST /api/predict
     Body: { "text": "..." }   or   { "ticker": "AAPL" }

   And return:
     {
       "sentiment":  "bullish" | "bearish" | "neutral",
       "score":      0.87,          // top-class probability
       "confidence": 0.91,          // model confidence
       "breakdown":  { "bullish": 0.87, "neutral": 0.09, "bearish": 0.04 },
       "summary":    "Optional explanation string."
     }
*/

async function fetchPrediction(input) {

  // ── MOCK (remove after connecting API) ──
  await new Promise(resolve => setTimeout(resolve, 1400));
  return mockPredict(input);
  // ── END MOCK ────────────────────────────

  /*
  const payload = activeTab === 'text'
    ? { text: input }
    : { ticker: input };

  const response = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Server error ${response.status}`);
  }

  return response.json();
  */
}


/* ── 7. Mock Predictor (remove after API is connected) ───── */

function mockPredict(input) {
  const text = input.toLowerCase();

  const bullishWords = ['beat', 'record', 'growth', 'strong', 'rally', 'surge', 'profit', 'gain', 'cuts'];
  const bearishWords = ['decline', 'fall', 'loss', 'crisis', 'default', 'uncertainty', 'risk', 'recession'];

  const bScore = bullishWords.filter(w => text.includes(w)).length;
  const rScore = bearishWords.filter(w => text.includes(w)).length;

  let sentiment, score, breakdown;

  if (bScore > rScore) {
    sentiment = 'bullish';
    score     = 0.72 + Math.random() * 0.2;
    breakdown = { bullish: score, neutral: (1 - score) * 0.6, bearish: (1 - score) * 0.4 };
  } else if (rScore > bScore) {
    sentiment = 'bearish';
    score     = 0.65 + Math.random() * 0.22;
    breakdown = { bearish: score, neutral: (1 - score) * 0.55, bullish: (1 - score) * 0.45 };
  } else {
    sentiment = 'neutral';
    score     = 0.48 + Math.random() * 0.14;
    breakdown = { neutral: score, bullish: (1 - score) * 0.52, bearish: (1 - score) * 0.48 };
  }

  const summaries = {
    bullish: 'The model detects <span>positive market signals</span> in this input. Key indicators suggest optimism around near-term price movement.',
    bearish: 'The model identifies <span>negative market pressure</span>. Indicators point toward caution or downside risk in the short term.',
    neutral: 'The model finds <span>mixed or inconclusive signals</span>. Market direction appears uncertain based on available indicators.'
  };

  return {
    sentiment,
    score,
    breakdown,
    summary:    summaries[sentiment],
    confidence: 0.84 + Math.random() * 0.12
  };
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