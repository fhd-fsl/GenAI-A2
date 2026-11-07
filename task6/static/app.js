/**
 * TrialMatch AI — Web Frontend Logic (app.js)
 * Implements:
 * - Dynamic tab switching
 * - Hardware diagnostics polling
 * - Preset clinical cases loading
 * - Patient note sentence parsing & interactive evidence highlighting
 * - Dynamic criteria builder
 * - Asynchronous screening with rule-based verdict visualization
 * - HTML/PDF and JSON screening report export
 * - Tab 2 Live decoding comparison runner
 * - Tab 3 Live tiled attention runner & interactive Chart.js benchmark visualizations
 */

// Global State
let currentPresets = [];
let currentScreeningResult = null;
let benchmarkDataCache = null;
let charts = {};

// Sample Prompts for Tab 2 Comparison Lab
const SAMPLE_COMPARE_PROMPTS = {
  json: `<|im_start|>system\nYou are a clinical NLP extractor. Extract the patient's condition, prescribed medication, and dosage into a valid JSON object.<|im_end|>\n<|im_start|>user\nPatient is a 54-year-old male diagnosed with severe hypertension. Commencing Lisinopril 20mg orally once daily and Metoprolol Succinate 50mg.<|im_end|>\n<|im_start|>assistant\n\`\`\`json\n{`,
  narrative: `<|im_start|>system\nYou are an expert medical scribe. Summarize the following case.<|im_end|>\n<|im_start|>user\nPatient presents with acute chest tightness radiating to the left jaw for 2 hours. ECG shows ST-segment elevation in leads V1-V4. Troponin I is elevated at 4.2 ng/mL. Write the Assessment and Plan.<|im_end|>\n<|im_start|>assistant\n`,
  reasoning: `<|im_start|>system\nYou are an AI clinical specialist.<|im_end|>\n<|im_start|>user\nExplain the mechanism of action of SGLT-2 inhibitors in heart failure with preserved ejection fraction (HFpEF).<|im_end|>\n<|im_start|>assistant\n`,
};

document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  initHardwareStatus();
  initPresets();
  initCriteriaBuilder();
  initScreenerEvents();
  initDecodingLabEvents();
  initAttentionLabEvents();
  initBenchmarkDashboards();
});

/* ==========================================================================
   Navigation
   ========================================================================== */
function initNavigation() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const targetId = btn.dataset.tab;
      document.getElementById(targetId).classList.add("active");
    });
  });

  document.querySelectorAll(".subtab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".subtab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".subtab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const targetId = btn.dataset.subtab;
      document.getElementById(targetId).classList.add("active");
    });
  });
}

/* ==========================================================================
   Hardware Diagnostics
   ========================================================================== */
async function initHardwareStatus() {
  try {
    const res = await fetch("/api/status");
    if (!res.ok) throw new Error("HTTP error " + res.status);
    const data = await res.json();
    const textEl = document.getElementById("hardwareText");
    if (data.cuda_available) {
      textEl.textContent = `${data.gpu_name} | VRAM: ${data.vram_allocated_mb} MB / ${data.vram_total_mb} MB (${data.vram_usage_percent}%)`;
    } else {
      textEl.textContent = "CPU Mode | CUDA Unavailable";
    }
  } catch (e) {
    document.getElementById("hardwareText").textContent = "Backend Offline";
  }
}

/* ==========================================================================
   Presets Loading
   ========================================================================== */
async function initPresets() {
  try {
    const res = await fetch("/api/presets");
    if (!res.ok) return;
    const data = await res.json();
    currentPresets = data.presets || [];
    const select = document.getElementById("presetCaseSelect");
    select.innerHTML = '<option value="">-- Choose a Clinical Case (Held-Out Test or Unseen) --</option>';

    currentPresets.forEach((p, idx) => {
      const opt = document.createElement("option");
      opt.value = idx;
      opt.textContent = p.title;
      select.appendChild(opt);
    });

    select.addEventListener("change", () => {
      const idx = select.value;
      if (idx === "") return;
      loadPresetByIndex(parseInt(idx, 10));
    });

    // Auto-select first preset case
    if (currentPresets.length > 0) {
      select.value = "0";
      loadPresetByIndex(0);
    }
  } catch (e) {
    console.error("Failed to load presets:", e);
  }
}

function loadPresetByIndex(idx) {
  const p = currentPresets[idx];
  if (!p) return;

  document.getElementById("patientNoteInput").value = p.note;
  document.getElementById("metaPatientId").textContent = p.patient_id;
  document.getElementById("metaTrialId").textContent = p.trial_id;
  document.getElementById("presetMeta").style.display = "flex";

  updateSentenceViewer();

  // Populate criteria
  const container = document.getElementById("criteriaListContainer");
  container.innerHTML = "";
  if (p.criteria && p.criteria.length > 0) {
    p.criteria.forEach((c) => {
      addCriterionRow(c.criterion_id, c.criterion_type, c.criterion_text, c.gold_eligibility, c.gold_sentences);
    });
  } else {
    addCriterionRow("CRIT_1", "inclusion", "");
  }
}

/* ==========================================================================
   Criteria Builder
   ========================================================================== */
function initCriteriaBuilder() {
  document.getElementById("btnAddCriterion").addEventListener("click", () => {
    const nextIdx = document.querySelectorAll(".criterion-item-builder").length + 1;
    addCriterionRow(`CRIT_${nextIdx}`, "inclusion", "");
  });

  document.getElementById("btnClearInputs").addEventListener("click", () => {
    document.getElementById("presetCaseSelect").value = "";
    document.getElementById("presetMeta").style.display = "none";
    document.getElementById("patientNoteInput").value = "";
    document.getElementById("criteriaListContainer").innerHTML = "";
    updateSentenceViewer();
    addCriterionRow("CRIT_1", "inclusion", "");
  });

  document.getElementById("patientNoteInput").addEventListener("input", updateSentenceViewer);
  document.getElementById("btnToggleSentenceView").addEventListener("click", () => {
    const viewer = document.getElementById("sentenceBreakdownViewer");
    viewer.style.display = viewer.style.display === "none" ? "flex" : "none";
  });
}

function addCriterionRow(id, type, text, goldElig = "", goldSent = "") {
  const container = document.getElementById("criteriaListContainer");
  const div = document.createElement("div");
  div.className = "criterion-item-builder";
  div.dataset.goldElig = goldElig;
  div.dataset.goldSent = goldSent;

  div.innerHTML = `
    <div class="builder-top">
      <input type="text" class="form-input crit-id-input" value="${id}" style="width: 130px; font-size: 0.75rem;" placeholder="ID">
      <select class="type-select ${type}">
        <option value="inclusion" ${type === "inclusion" ? "selected" : ""}>Inclusion</option>
        <option value="exclusion" ${type === "exclusion" ? "selected" : ""}>Exclusion</option>
      </select>
      <button class="btn-remove-crit" title="Remove">&times;</button>
    </div>
    <textarea class="form-textarea crit-text-input" rows="2" placeholder="Criterion statement (e.g. ECOG performance status 0 or 1)">${text}</textarea>
  `;

  const typeSelect = div.querySelector(".type-select");
  typeSelect.addEventListener("change", () => {
    typeSelect.className = `type-select ${typeSelect.value}`;
  });

  div.querySelector(".btn-remove-crit").addEventListener("click", () => {
    div.remove();
  });

  container.appendChild(div);
}

function updateSentenceViewer() {
  const note = document.getElementById("patientNoteInput").value.trim();
  const viewer = document.getElementById("sentenceBreakdownViewer");
  viewer.innerHTML = "";

  if (!note) {
    document.getElementById("sentenceCountBadge").textContent = "0";
    return;
  }

  // Parse lines or split sentences
  const lines = note.split("\n");
  let sentences = [];
  if (lines.length > 1 && /^\d+\.\s+/.test(lines[0])) {
    sentences = lines.map((l) => l.replace(/^\d+\.\s+/, "").trim()).filter(Boolean);
  } else {
    sentences = note.match(/[^.!?]+[.!?]+(\s|$)/g) || [note];
    sentences = sentences.map((s) => s.trim()).filter(Boolean);
  }

  document.getElementById("sentenceCountBadge").textContent = sentences.length;

  sentences.forEach((s, idx) => {
    const row = document.createElement("div");
    row.className = "sentence-row";
    row.id = `sentence-row-${idx}`;
    row.innerHTML = `<span class="s-idx">[${idx}]</span> ${s}`;
    viewer.appendChild(row);
  });
}

/* ==========================================================================
   Tab 1: Screener Execution & Rendering
   ========================================================================== */
function initScreenerEvents() {
  const modeSelect = document.getElementById("decodingModeSelect");
  const gammaGroup = document.getElementById("gammaControlGroup");
  const gammaRange = document.getElementById("gammaRange");
  const gammaLabel = document.getElementById("gammaValueLabel");

  modeSelect.addEventListener("change", () => {
    gammaGroup.style.display = modeSelect.value === "speculative" ? "block" : "none";
  });

  gammaRange.addEventListener("input", () => {
    gammaLabel.textContent = gammaRange.value;
  });

  document.getElementById("btnRunScreening").addEventListener("click", runScreening);

  // Filters
  document.querySelectorAll(".pill-filter").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".pill-filter").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      filterResults(btn.dataset.filter);
    });
  });

  // Export Buttons
  document.getElementById("btnExportHtml").addEventListener("click", exportHtmlReport);
  document.getElementById("btnExportJson").addEventListener("click", exportJsonReport);
}

async function runScreening() {
  const note = document.getElementById("patientNoteInput").value.trim();
  if (!note) {
    alert("Please enter or select a patient clinical note.");
    return;
  }

  const critItems = document.querySelectorAll(".criterion-item-builder");
  if (critItems.length === 0) {
    alert("Please add at least one eligibility criterion.");
    return;
  }

  const criteria = [];
  critItems.forEach((div) => {
    const id = div.querySelector(".crit-id-input").value.trim() || "CRIT";
    const type = div.querySelector(".type-select").value;
    const text = div.querySelector(".crit-text-input").value.trim();
    const goldElig = div.dataset.goldElig || "";
    const goldSent = div.dataset.goldSent || "";
    if (text) {
      criteria.push({
        criterion_id: id,
        criterion_type: type,
        criterion_text: text,
        gold_eligibility: goldElig,
        gold_sentences: goldSent,
      });
    }
  });

  if (criteria.length === 0) {
    alert("Please enter criterion text for evaluation.");
    return;
  }

  const decodingMode = document.getElementById("decodingModeSelect").value;
  const gamma = parseInt(document.getElementById("gammaRange").value, 10);

  // UI Loading State
  const btn = document.getElementById("btnRunScreening");
  const spinner = document.getElementById("screenBtnSpinner");
  btn.disabled = true;
  spinner.style.display = "inline-block";

  try {
    const res = await fetch("/api/screen", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        note: note,
        criteria: criteria,
        decoding_mode: decodingMode,
        gamma: gamma,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || "Screening failed");
    }

    const data = await res.json();
    currentScreeningResult = data;
    renderScreeningResults(data);
    initHardwareStatus(); // Refresh VRAM usage
  } catch (e) {
    alert(`Screening Error: ${e.message}`);
  } finally {
    btn.disabled = false;
    spinner.style.display = "none";
  }
}

function renderScreeningResults(data) {
  const verdict = data.overall_verdict;
  const vCard = document.getElementById("overallVerdictCard");
  vCard.className = `verdict-banner ${verdict.status_color}`;
  vCard.style.display = "block";

  document.getElementById("verdictTitle").textContent = verdict.verdict_title;
  document.getElementById("verdictSummary").textContent = verdict.summary;
  document.getElementById("verdictRule").textContent = verdict.rule_applied;

  const vStats = document.getElementById("verdictStats");
  vStats.innerHTML = `
    <span class="v-stat-pill">&#9989; Included: <strong>${verdict.counts.included}</strong></span>
    <span class="v-stat-pill">&#10006; Not Included: <strong>${verdict.counts.not_included}</strong></span>
    <span class="v-stat-pill">&#9989; Not Excluded: <strong>${verdict.counts.not_excluded}</strong></span>
    <span class="v-stat-pill">&#10006; Excluded: <strong>${verdict.counts.excluded}</strong></span>
    <span class="v-stat-pill">&#9888; Missing Info: <strong>${verdict.counts.not_enough_info}</strong></span>
    <span class="v-stat-pill">&#9866; Not Applicable: <strong>${verdict.counts.not_applicable}</strong></span>
  `;

  // Toolbar counts
  document.getElementById("resultsToolbar").style.display = "flex";
  document.getElementById("countAll").textContent = data.criteria_results.length;
  document.getElementById("countInc").textContent = data.criteria_results.filter((c) => c.criterion_type === "inclusion").length;
  document.getElementById("countExc").textContent = data.criteria_results.filter((c) => c.criterion_type === "exclusion").length;
  document.getElementById("countMiss").textContent = verdict.counts.not_enough_info;

  // Render cards
  const container = document.getElementById("criteriaResultsContainer");
  container.innerHTML = "";

  data.criteria_results.forEach((c) => {
    const card = document.createElement("div");
    card.className = "decision-card";
    card.dataset.type = c.criterion_type;
    card.dataset.prediction = c.prediction;

    const predClassMap = {
      included: "included",
      "not included": "not-included",
      "not excluded": "not-excluded",
      excluded: "excluded",
      "not enough information": "not-enough-info",
      "not applicable": "not-applicable",
    };
    const pClass = predClassMap[c.prediction] || "not-enough-info";

    // Evidence sentences pills
    let evidencePillsHtml = "";
    if (c.evidence_sentence_ids && c.evidence_sentence_ids.length > 0) {
      c.evidence_sentence_ids.forEach((sid) => {
        evidencePillsHtml += `<span class="sentence-click-pill" onclick="highlightSentence(${sid})">Sentence [${sid}]</span>`;
      });
    } else {
      evidencePillsHtml = '<span style="color: var(--text-dim); font-size: 0.75rem;">None Cited (Implicit or Missing)</span>';
    }

    let evidenceQuotesHtml = "";
    if (c.evidence_text && c.evidence_text.length > 0) {
      evidenceQuotesHtml = `<div class="evidence-quote">"${c.evidence_text.join('" "')}"</div>`;
    }

    let goldHtml = "";
    if (c.gold_eligibility) {
      goldHtml = `<span class="gold-comparison">Gold Ground Truth: <strong>${c.gold_eligibility}</strong></span>`;
    }

    card.innerHTML = `
      <div class="decision-top">
        <div class="crit-badge-group">
          <span class="crit-id-badge">${c.criterion_id}</span>
          <span class="crit-type-badge ${c.criterion_type}">${c.criterion_type}</span>
        </div>
        <span class="pred-badge ${pClass}">${c.prediction}</span>
      </div>
      <div class="crit-body-text">${c.criterion_text}</div>
      <div class="evidence-block">
        <div class="evidence-header">
          <span>Patient Record Evidence:</span>
          <div>${evidencePillsHtml}</div>
        </div>
        ${evidenceQuotesHtml}
      </div>
      <div class="explanation-text"><strong>Justification:</strong> ${c.explanation}</div>
      <div class="crit-footer">
        <div>${goldHtml}</div>
        <div>Latency: ${c.latency_ms} ms (${c.tokens_per_sec} tok/s)</div>
      </div>
    `;

    container.appendChild(card);
  });

  // Telemetry
  const telem = data.telemetry;
  document.getElementById("screeningTelemetryCard").style.display = "block";
  document.getElementById("telemetryModeBadge").textContent = data.decoding_mode.toUpperCase();
  document.getElementById("telemTime").textContent = `${telem.total_time_ms} ms`;
  document.getElementById("telemTokens").textContent = telem.total_tokens_generated;
  document.getElementById("telemThroughput").textContent = `${telem.overall_tokens_per_sec} tok/s`;
  document.getElementById("telemPassesSaved").textContent = telem.total_target_passes_saved;
  document.getElementById("telemVram").textContent = `${telem.peak_vram_mb} MB`;
}

function highlightSentence(idx) {
  // Ensure viewer is open
  const viewer = document.getElementById("sentenceBreakdownViewer");
  viewer.style.display = "flex";

  // Clear previous highlights
  document.querySelectorAll(".sentence-row").forEach((r) => r.classList.remove("highlight-evidence"));

  const target = document.getElementById(`sentence-row-${idx}`);
  if (target) {
    target.classList.add("highlight-evidence");
    target.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

function filterResults(filter) {
  document.querySelectorAll(".decision-card").forEach((card) => {
    if (filter === "all") {
      card.style.display = "block";
    } else if (filter === "inclusion") {
      card.style.display = card.dataset.type === "inclusion" ? "block" : "none";
    } else if (filter === "exclusion") {
      card.style.display = card.dataset.type === "exclusion" ? "block" : "none";
    } else if (filter === "missing") {
      card.style.display = card.dataset.prediction === "not enough information" ? "block" : "none";
    }
  });
}

/* ==========================================================================
   Report Export
   ========================================================================== */
function exportJsonReport() {
  if (!currentScreeningResult) return;
  const blob = new Blob([JSON.stringify(currentScreeningResult, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `trial_screening_provenance_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

function exportHtmlReport() {
  if (!currentScreeningResult) return;
  const res = currentScreeningResult;
  const verdict = res.overall_verdict;

  let criteriaTableRows = "";
  res.criteria_results.forEach((c) => {
    criteriaTableRows += `
      <tr>
        <td><strong>${c.criterion_id}</strong></td>
        <td><span style="text-transform:uppercase;">${c.criterion_type}</span></td>
        <td>${c.criterion_text}</td>
        <td><strong>${c.prediction.toUpperCase()}</strong></td>
        <td>${c.evidence_sentence_ids.join(", ") || "None"}</td>
        <td>${c.explanation}</td>
      </tr>
    `;
  });

  const html = `
    <!DOCTYPE html>
    <html>
    <head>
      <title>Clinical Trial Eligibility Screening Assessment</title>
      <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 2rem; color: #24292f; }
        .banner { background-color: #fff8c5; border: 1px solid rgba(210,153,34,0.4); padding: 1rem; border-radius: 6px; margin-bottom: 1.5rem; font-size: 0.85rem; }
        .header { border-bottom: 2px solid #0969da; padding-bottom: 0.5rem; margin-bottom: 1.5rem; }
        .verdict-box { padding: 1rem; border-radius: 6px; background-color: #f6f8fa; border: 1px solid #d0d7de; margin-bottom: 1.5rem; }
        table { width: 100%; border-collapse: collapse; margin-top: 1rem; font-size: 0.85rem; }
        th, td { border: 1px solid #d0d7de; padding: 0.5rem 0.75rem; text-align: left; }
        th { background-color: #f6f8fa; }
        pre { background: #f6f8fa; padding: 0.75rem; border-radius: 4px; font-size: 0.8rem; white-space: pre-wrap; }
        @media print { button { display: none; } }
      </style>
    </head>
    <body>
      <div class="banner">
        <strong>EDUCATIONAL RESEARCH PROTOTYPE ONLY:</strong> Generated by Generative AI Coursework System. Not for official clinical trial enrollment or medical decisions.
      </div>
      <div class="header">
        <h2>Clinical Trial Eligibility Screening Report</h2>
        <p>Target Model: Qwen/Qwen2.5-3B-Instruct (4-bit NF4 + LoRA) | Decoding: ${res.decoding_mode.toUpperCase()}</p>
        <p>Timestamp: ${new Date().toISOString()}</p>
      </div>

      <div class="verdict-box">
        <h3>Preliminary Verdict: ${verdict.verdict_title}</h3>
        <p>${verdict.summary}</p>
        <p><small>Rule Trace: ${verdict.rule_applied}</small></p>
      </div>

      <h3>Patient Record (Normalized & Indexed)</h3>
      <pre>${res.patient_note_formatted}</pre>

      <h3>Criterion-Level Decisions & Evidence Localization</h3>
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Type</th>
            <th>Criterion Text</th>
            <th>Decision</th>
            <th>Evidence Sentences</th>
            <th>Justification</th>
          </tr>
        </thead>
        <tbody>
          ${criteriaTableRows}
        </tbody>
      </table>

      <br><hr>
      <button onclick="window.print()" style="padding: 0.5rem 1rem; font-size: 1rem; cursor: pointer;">Print / Save as PDF</button>
    </body>
    </html>
  `;

  const win = window.open("", "_blank");
  win.document.write(html);
  win.document.close();
}

/* ==========================================================================
   Tab 2: Live Decoding Comparison Lab
   ========================================================================== */
function initDecodingLabEvents() {
  const presetSelect = document.getElementById("comparePromptPreset");
  const promptInput = document.getElementById("comparePromptInput");
  const budgetRange = document.getElementById("rngCompareBudget");
  const budgetLabel = document.getElementById("lblCompareBudget");
  const gammaRange = document.getElementById("rngCompareGamma");
  const gammaLabel = document.getElementById("lblCompareGamma");

  promptInput.value = SAMPLE_COMPARE_PROMPTS.json;

  presetSelect.addEventListener("change", () => {
    if (presetSelect.value !== "custom") {
      promptInput.value = SAMPLE_COMPARE_PROMPTS[presetSelect.value] || "";
    }
  });

  budgetRange.addEventListener("input", () => {
    budgetLabel.textContent = budgetRange.value;
  });

  gammaRange.addEventListener("input", () => {
    gammaLabel.textContent = gammaRange.value;
    document.getElementById("mSpecTag").textContent = `Task 5 (\u03b3=${gammaRange.value})`;
  });

  document.getElementById("btnRunCompareDecoding").addEventListener("click", runCompareDecoding);
}

async function runCompareDecoding() {
  const prompt = document.getElementById("comparePromptInput").value.trim();
  if (!prompt) {
    alert("Please provide a prompt for comparison.");
    return;
  }

  const budget = parseInt(document.getElementById("rngCompareBudget").value, 10);
  const gamma = parseInt(document.getElementById("rngCompareGamma").value, 10);

  const btn = document.getElementById("btnRunCompareDecoding");
  const spinner = document.getElementById("compareBtnSpinner");
  btn.disabled = true;
  spinner.style.display = "inline-block";

  try {
    const res = await fetch("/api/compare_decoding", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: prompt,
        gamma: gamma,
        max_new_tokens: budget,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || "Comparison failed");
    }

    const data = await res.json();
    renderCompareResults(data);
    initHardwareStatus();
  } catch (e) {
    alert(`Comparison Execution Error: ${e.message}`);
  } finally {
    btn.disabled = false;
    spinner.style.display = "none";
  }
}

function renderCompareResults(data) {
  const eq = data.token_equality;
  const eqBanner = document.getElementById("tokenEquivalenceBanner");
  eqBanner.style.display = "flex";

  if (eq.all_modes_exact_match) {
    document.getElementById("eqTitle").textContent = "100.00% Exact Greedy Token Equivalence Verified";
    document.getElementById("eqDetails").textContent = `All 3 modes (Uncached, KV-Cached, and Speculative \u03b3=${data.gamma}) produced identical ${eq.num_tokens_cached} tokens. Zero drift or hallucination.`;
  } else {
    document.getElementById("eqTitle").textContent = "Token Discrepancy Detected";
    document.getElementById("eqDetails").textContent = "Check generation outputs for non-deterministic variances.";
  }

  document.getElementById("compareResultsGrid").style.display = "grid";

  // Uncached
  const u = data.modes.uncached;
  document.getElementById("mUncTime").textContent = `${u.latency_ms} ms`;
  document.getElementById("mUncTps").textContent = `${u.tokens_per_sec} tok/s`;
  document.getElementById("mUncVram").textContent = `${u.peak_vram_mb} MB`;
  document.getElementById("mUncText").textContent = u.text;

  // KV-Cached
  const k = data.modes.kv_cached;
  document.getElementById("mKvTime").textContent = `${k.latency_ms} ms`;
  document.getElementById("mKvTps").textContent = `${k.tokens_per_sec} tok/s`;
  document.getElementById("mKvSpeedup").textContent = `${k.speedup_vs_uncached}x`;
  document.getElementById("mKvVram").textContent = `${k.peak_vram_mb} MB`;
  document.getElementById("mKvText").textContent = k.text;

  // Speculative
  const s = data.modes.speculative;
  document.getElementById("mSpecTime").textContent = `${s.latency_ms} ms`;
  document.getElementById("mSpecTps").textContent = `${s.tokens_per_sec} tok/s`;
  document.getElementById("mSpecSpeedupCached").textContent = `${s.speedup_vs_cached}x`;
  document.getElementById("mSpecAlpha").textContent = `${s.acceptance_rate_percent}%`;
  document.getElementById("mSpecPasses").textContent = s.target_passes_saved;
  document.getElementById("mSpecVram").textContent = `${s.peak_vram_mb} MB`;
  document.getElementById("mSpecText").textContent = s.text;
}

/* ==========================================================================
   Tab 3: Live Attention Runner
   ========================================================================== */
function initAttentionLabEvents() {
  document.getElementById("btnRunLiveAttention").addEventListener("click", runLiveAttention);
}

async function runLiveAttention() {
  const seqLen = parseInt(document.getElementById("attSeqLenSelect").value, 10);
  const headDim = parseInt(document.getElementById("attHeadDimSelect").value, 10);
  const blockR = parseInt(document.getElementById("attBlockRSelect").value, 10);
  const blockC = parseInt(document.getElementById("attBlockCSelect").value, 10);
  const isCausal = document.getElementById("attCausalSelect").value === "true";

  const btn = document.getElementById("btnRunLiveAttention");
  const spinner = document.getElementById("attBtnSpinner");
  btn.disabled = true;
  spinner.style.display = "inline-block";

  try {
    const res = await fetch("/api/run_attention", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        seq_len: seqLen,
        head_dim: headDim,
        block_r: blockR,
        block_c: blockC,
        is_causal: isCausal,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || "Attention execution failed");
    }

    const data = await res.json();
    renderLiveAttentionResults(data);
    initHardwareStatus();
  } catch (e) {
    alert(`Attention Kernel Error: ${e.message}`);
  } finally {
    btn.disabled = false;
    spinner.style.display = "none";
  }
}

function renderLiveAttentionResults(data) {
  const container = document.getElementById("attentionLiveResults");
  container.style.display = "grid";

  const r = data.results;
  document.getElementById("attRefTime").textContent = `${r.reference_sdpa.latency_ms} ms`;
  document.getElementById("attRefMem").textContent = `${r.reference_sdpa.peak_vram_mb} MB`;

  document.getElementById("attTiledTime").textContent = `${r.tiled_online_sdpa.latency_ms} ms`;
  document.getElementById("attTiledMem").textContent = `${r.tiled_online_sdpa.peak_vram_mb} MB`;
  document.getElementById("attMemSavedBadge").textContent = `${r.memory_reduction_percent}% Memory Saved`;

  document.getElementById("attFusedTime").textContent = `${r.pytorch_fused_sdpa.latency_ms} ms`;
  document.getElementById("attFusedMem").textContent = `${r.pytorch_fused_sdpa.peak_vram_mb} MB`;

  const num = data.numerical_agreement;
  document.getElementById("attMaxDiff").textContent = num.max_absolute_error.toExponential(3);
  document.getElementById("attMeanDiff").textContent = num.mean_absolute_error.toExponential(3);
}

/* ==========================================================================
   Tab 3: Interactive Benchmark Dashboards
   ========================================================================== */
async function initBenchmarkDashboards() {
  try {
    const res = await fetch("/api/benchmarks");
    if (!res.ok) return;
    benchmarkDataCache = await res.json();
    renderTask2Dashboard(benchmarkDataCache.task2_metrics);
    renderTask3Dashboard(benchmarkDataCache.task3_benchmark);
    renderTask4Dashboard(benchmarkDataCache.task4_benchmark);
    renderTask5Dashboard(benchmarkDataCache.task5_benchmark);
  } catch (e) {
    console.error("Failed to load benchmark dashboards:", e);
  }
}

function renderTask2Dashboard(t2) {
  if (!t2) return;
  // Confusion Matrix Table
  const cm = t2.confusion_matrix;
  let tableHtml = '<table class="matrix-table"><thead><tr><th>Gold \\ Pred</th>';
  const shortLabels = ["incl", "not-inc", "not-exc", "excl", "missing", "n/a"];
  shortLabels.forEach((l) => (tableHtml += `<th>${l}</th>`));
  tableHtml += "</tr></thead><tbody>";

  cm.matrix.forEach((row, rIdx) => {
    tableHtml += `<tr><th>${shortLabels[rIdx]}</th>`;
    row.forEach((val, cIdx) => {
      const cls = rIdx === cIdx && val > 0 ? "diag-cell" : val > 0 ? "error-cell" : "";
      tableHtml += `<td class="${cls}">${val}</td>`;
    });
    tableHtml += "</tr>";
  });
  tableHtml += "</tbody></table>";
  document.getElementById("task2ConfusionTableContainer").innerHTML = tableHtml;

  // Split Chart (Inclusion vs Exclusion)
  const ctx = document.getElementById("task2SplitChart").getContext("2d");
  charts.task2 = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Accuracy", "Macro F1", "Evidence F1 (Macro)", "Evidence F1 (Micro)"],
      datasets: [
        { label: "Inclusion Criteria (n=75)", data: [73.3, 39.3, 69.7, 61.9], backgroundColor: "#3fb950" },
        { label: "Exclusion Criteria (n=131)", data: [73.3, 49.8, 90.0, 49.2], backgroundColor: "#58a6ff" },
      ],
    },
    options: { responsive: true, scales: { y: { beginAtZero: true, max: 100 } } },
  });
}

function renderTask3Dashboard(t3) {
  if (!t3) return;
  const ctx1 = document.getElementById("task3ThroughputChart").getContext("2d");
  charts.task3Tps = new Chart(ctx1, {
    type: "bar",
    data: {
      labels: ["Short (N=32)", "Short (N=128)", "Medium (N=32)", "Medium (N=128)", "Long (N=32)", "Long (N=128)"],
      datasets: [
        { label: "Uncached Baseline (tok/s)", data: [6.8, 5.8, 5.3, 4.5, 4.1, 2.9], backgroundColor: "#8b949e" },
        { label: "Explicit KV-Cached (tok/s)", data: [6.7, 6.8, 7.2, 7.5, 7.3, 7.4], backgroundColor: "#388bfd" },
      ],
    },
    options: { responsive: true, scales: { y: { beginAtZero: true } } },
  });

  const ctx2 = document.getElementById("task3MemoryChart").getContext("2d");
  charts.task3Mem = new Chart(ctx2, {
    type: "line",
    data: {
      labels: ["32 tokens", "64 tokens", "128 tokens"],
      datasets: [
        { label: "Dedicated KV Cache (MB)", data: [3.0, 4.1, 6.4], borderColor: "#3fb950", tension: 0.2 },
        { label: "Memory Saved vs Uncached (MB)", data: [25.8, 53.3, 108.9], borderColor: "#58a6ff", tension: 0.2 },
      ],
    },
    options: { responsive: true, scales: { y: { beginAtZero: true } } },
  });
}

function renderTask4Dashboard(t4) {
  if (!t4) return;
  const ctx1 = document.getElementById("task4MemoryChart").getContext("2d");
  charts.task4Mem = new Chart(ctx1, {
    type: "line",
    data: {
      labels: ["128", "256", "512", "1024", "2048", "4096"],
      datasets: [
        { label: "Reference SDPA Peak MB (O(N^2))", data: [9.9, 13.2, 24.4, 65.2, 220.2, 824.3], borderColor: "#f85149", tension: 0.2 },
        { label: "Tiled Online Softmax Peak MB (O(N))", data: [9.9, 10.9, 12.9, 16.9, 24.9, 41.0], borderColor: "#3fb950", tension: 0.2 },
        { label: "PyTorch Fused Peak MB", data: [9.1, 10.1, 12.1, 16.1, 24.1, 40.1], borderColor: "#58a6ff", tension: 0.2 },
      ],
    },
    options: { responsive: true, scales: { y: { beginAtZero: true } } },
  });

  const ctx2 = document.getElementById("task4LatencyChart").getContext("2d");
  charts.task4Lat = new Chart(ctx2, {
    type: "bar",
    data: {
      labels: ["128", "256", "512", "1024", "2048", "4096"],
      datasets: [
        { label: "Reference SDPA (ms)", data: [0.59, 0.34, 0.41, 1.58, 6.96, 26.69], backgroundColor: "#f85149" },
        { label: "PyTorch Fused Flash-2 (ms)", data: [0.09, 0.12, 0.21, 0.32, 1.01, 3.63], backgroundColor: "#3fb950" },
      ],
    },
    options: { responsive: true, scales: { y: { beginAtZero: true } } },
  });
}

function renderTask5Dashboard(t5) {
  if (!t5) return;
  const ctx1 = document.getElementById("task5SpeedupChart").getContext("2d");
  charts.task5Speedup = new Chart(ctx1, {
    type: "line",
    data: {
      labels: ["gamma=1", "gamma=3", "gamma=5", "gamma=7"],
      datasets: [
        { label: "Structured JSON Speedup", data: [1.07, 1.56, 1.75, 1.46], borderColor: "#3fb950", tension: 0.2 },
        { label: "Medical QA Speedup", data: [1.06, 1.43, 1.34, 1.10], borderColor: "#58a6ff", tension: 0.2 },
        { label: "Clinical Narrative Speedup", data: [0.92, 1.28, 1.24, 1.15], borderColor: "#d29922", tension: 0.2 },
      ],
    },
    options: {
      responsive: true,
      scales: {
        y: {
          min: 0.8,
          ticks: {
            callback: (v) => `${v}x`,
          },
        },
      },
    },
  });

  const ctx2 = document.getElementById("task5AlphaChart").getContext("2d");
  charts.task5Alpha = new Chart(ctx2, {
    type: "bar",
    data: {
      labels: ["gamma=1", "gamma=3", "gamma=5", "gamma=7"],
      datasets: [
        { label: "Structured JSON Acceptance (alpha %)", data: [88.9, 70.0, 62.9, 50.6], backgroundColor: "#3fb950" },
        { label: "Clinical Narrative Acceptance (alpha %)", data: [73.0, 58.0, 44.3, 42.2], backgroundColor: "#d29922" },
      ],
    },
    options: { responsive: true, scales: { y: { beginAtZero: true, max: 100 } } },
  });
}

