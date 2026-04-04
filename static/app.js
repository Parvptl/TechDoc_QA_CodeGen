/* DS Mentor Pro -- Deployable website logic */
(function () {
  "use strict";

  const STAGE_NAMES = {
    1: "Problem Understanding",
    2: "Data Loading",
    3: "Exploratory Data Analysis",
    4: "Preprocessing",
    5: "Feature Engineering",
    6: "Modeling",
    7: "Evaluation",
  };

  let sessionId = "web_" + Math.random().toString(36).slice(2, 10);
  let completedStages = new Set();
  let skills = Object.fromEntries([...Array(7).keys()].map((i) => [i + 1, 0.5]));
  let datasetProfile = null;
  let projectPlan = null;
  let currentQuiz = null;

  const dom = {
    chatMessages: document.getElementById("chat-messages"),
    chatInput: document.getElementById("chat-input"),
    sendBtn: document.getElementById("send-btn"),
    fileInput: document.getElementById("file-input"),
    uploadArea: document.getElementById("upload-area"),
    datasetInfo: document.getElementById("dataset-info"),
    resetBtn: document.getElementById("reset-btn"),
    sidebarToggle: document.getElementById("sidebar-toggle"),
    sidebar: document.getElementById("sidebar"),
    sidebarBackdrop: document.getElementById("sidebar-backdrop"),
    nextStepHint: document.getElementById("next-step-hint"),
    analyticsCards: document.getElementById("analytics-cards"),
    trendConfidence: document.getElementById("trend-confidence"),
    trendQuiz: document.getElementById("trend-quiz"),
    trendLatency: document.getElementById("trend-latency"),
    exportMdBtn: document.getElementById("export-md-btn"),
    exportJsonBtn: document.getElementById("export-json-btn"),
    reportPath: document.getElementById("report-path"),
    projectNextStage: document.getElementById("project-next-stage"),
    checkpointEvidence: document.getElementById("checkpoint-evidence"),
    submitCheckpointBtn: document.getElementById("submit-checkpoint-btn"),
    checkpointFeedback: document.getElementById("checkpoint-feedback"),
    generateQuizBtn: document.getElementById("generate-quiz-btn"),
    quizContainer: document.getElementById("quiz-container"),
    quizMeta: document.getElementById("quiz-meta"),
    quizQuestions: document.getElementById("quiz-questions"),
    submitQuizBtn: document.getElementById("submit-quiz-btn"),
    quizResult: document.getElementById("quiz-result"),
    quizStatusChip: document.getElementById("quiz-status-chip"),
    quizStatusTime: document.getElementById("quiz-status-time"),
    checkpointStatusChip: document.getElementById("checkpoint-status-chip"),
    checkpointStatusTime: document.getElementById("checkpoint-status-time"),
    reportStatusChip: document.getElementById("report-status-chip"),
    reportStatusTime: document.getElementById("report-status-time"),
    toastContainer: document.getElementById("toast-container"),
  };
  let lastFocusEl = null;

  init();

  function init() {
    renderPipeline();
    renderSkills();
    bindInputHandlers();
    bindQuickPromptButtons();
    bindSidebarActions();
    refreshProgressAndDashboard();
  }

  function bindInputHandlers() {
    dom.chatInput.addEventListener("input", () => {
      dom.chatInput.style.height = "auto";
      dom.chatInput.style.height = dom.chatInput.scrollHeight + "px";
      dom.sendBtn.disabled = !dom.chatInput.value.trim();
    });
    dom.chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        if (dom.chatInput.value.trim()) sendMessage();
      }
    });
    dom.chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && dom.sidebar.classList.contains("open")) {
        setSidebarOpen(false);
      }
    });
    dom.sendBtn.addEventListener("click", () => {
      if (dom.chatInput.value.trim()) sendMessage();
    });
  }

  function bindQuickPromptButtons() {
    dom.chatMessages.querySelectorAll(".quick-btn").forEach((btn) => {
      btn.onclick = () => {
        dom.chatInput.value = btn.dataset.q;
        dom.sendBtn.disabled = false;
        sendMessage();
      };
    });
  }

  function bindSidebarActions() {
    dom.sidebarToggle.addEventListener("click", () =>
      setSidebarOpen(!dom.sidebar.classList.contains("open"))
    );
    if (dom.sidebarBackdrop) {
      dom.sidebarBackdrop.addEventListener("click", () => setSidebarOpen(false));
    }
    window.addEventListener("resize", () => {
      if (window.innerWidth > 768) {
        dom.sidebar.classList.remove("open");
        dom.sidebar.setAttribute("aria-hidden", "false");
        if (dom.sidebarBackdrop) dom.sidebarBackdrop.classList.remove("show");
        dom.sidebarToggle.setAttribute("aria-expanded", "false");
      }
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && dom.sidebar.classList.contains("open")) {
        setSidebarOpen(false);
      }
    });

    dom.resetBtn.addEventListener("click", () => {
      if (!confirm("Reset the current session?")) return;
      sessionId = "web_" + Math.random().toString(36).slice(2, 10);
      completedStages = new Set();
      skills = Object.fromEntries([...Array(7).keys()].map((i) => [i + 1, 0.5]));
      datasetProfile = null;
      projectPlan = null;
      currentQuiz = null;
      dom.chatMessages.innerHTML = document.querySelector(".welcome-msg")
        ? dom.chatMessages.innerHTML
        : `<div class="welcome-msg"><h3>Welcome to DS Mentor Pro</h3><p>Ask a question about any stage of the data science pipeline. Upload a dataset to get context-aware guidance.</p></div>`;
      dom.datasetInfo.style.display = "none";
      dom.projectNextStage.textContent = "Upload dataset to enable.";
      dom.checkpointFeedback.textContent = "";
      dom.quizContainer.style.display = "none";
      dom.quizResult.textContent = "";
      setStatusChip(dom.quizStatusChip, "No quiz yet", "neutral");
      setStatusChip(dom.checkpointStatusChip, "Not submitted", "neutral");
      setStatusChip(dom.reportStatusChip, "Idle", "neutral");
      setStatusTime(dom.quizStatusTime);
      setStatusTime(dom.checkpointStatusTime);
      setStatusTime(dom.reportStatusTime);
      renderPipeline();
      renderSkills();
      refreshProgressAndDashboard();
      bindQuickPromptButtons();
    });

    dom.fileInput.addEventListener("change", handleUpload);
    dom.submitCheckpointBtn.addEventListener("click", submitCheckpoint);
    dom.generateQuizBtn.addEventListener("click", generateQuiz);
    dom.submitQuizBtn.addEventListener("click", submitQuiz);
    dom.exportMdBtn.addEventListener("click", () => exportReport("markdown"));
    dom.exportJsonBtn.addEventListener("click", () => exportReport("json"));
  }

  async function handleUpload() {
    const file = dom.fileInput.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);

    dom.uploadArea.querySelector(".upload-text").textContent = "Uploading...";
    try {
      const res = await fetch(`/api/v1/upload?session_id=${encodeURIComponent(sessionId)}`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (data.profile_summary || data.profile) {
        datasetProfile = data.profile || data.profile_summary;
        projectPlan = data.project_plan || null;
        renderDatasetInfo(datasetProfile);
        renderProjectPlanHint();
        showToast("Dataset uploaded successfully.", "success");
        setStatusChip(dom.checkpointStatusChip, "Ready to submit", "info");
        setStatusTime(dom.checkpointStatusTime);
      } else if (data.profile_error) {
        showToast("Could not parse file: " + data.profile_error, "error");
        setStatusChip(dom.checkpointStatusChip, "Upload parse error", "error");
        setStatusTime(dom.checkpointStatusTime);
      }
    } catch (err) {
      showToast("Upload failed: " + err.message, "error");
      setStatusChip(dom.checkpointStatusChip, "Upload failed", "error");
      setStatusTime(dom.checkpointStatusTime);
    } finally {
      dom.uploadArea.querySelector(".upload-text").textContent = "Upload CSV or Excel";
      dom.fileInput.value = "";
      await refreshProgressAndDashboard();
    }
  }

  async function sendMessage() {
    const query = dom.chatInput.value.trim();
    if (!query) return;
    removeWelcome();
    appendUserMsg(query);
    dom.chatInput.value = "";
    dom.chatInput.style.height = "auto";
    dom.sendBtn.disabled = true;
    const typingEl = showTyping();

    try {
      const res = await fetch("/api/v1/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, code: "", session_id: sessionId }),
      });
      const data = await res.json();
      typingEl.remove();
      appendAssistantMsg(data);
      if (data.next_step) dom.nextStepHint.textContent = data.next_step;
      await refreshProgressAndDashboard();
      if (data.mode === "project_gate") renderProjectPlanHint();
      if (window.innerWidth <= 768) setSidebarOpen(false);
    } catch (err) {
      typingEl.remove();
      appendError("Request failed: " + err.message);
      showToast("Request failed: " + err.message, "error");
    }
  }

  async function refreshProgressAndDashboard() {
    try {
      const progressRes = await fetch(`/api/v1/session/${encodeURIComponent(sessionId)}/progress`);
      const progress = await progressRes.json();
      completedStages = new Set(progress.completed_stages || []);
      const mastery = progress.stage_mastery || {};
      for (let i = 1; i <= 7; i++) skills[i] = Number(mastery[i] ?? mastery[String(i)] ?? 0.5);
      renderPipeline();
      renderSkills();
      renderAnalyticsCards(progress);
    } catch (_err) {
      // noop
    }

    try {
      const dashRes = await fetch(`/api/v1/session/${encodeURIComponent(sessionId)}/dashboard`);
      const dashboard = await dashRes.json();
      const trends = dashboard.trends || {};
      renderSpark(dom.trendConfidence, (trends.confidence_series || []).map((x) => x.value), "");
      renderSpark(dom.trendQuiz, (trends.quiz_series || []).map((x) => x.value * 100), "alt");
      renderSpark(dom.trendLatency, (trends.latency_series || []).map((x) => x.value), "warn");
    } catch (_err) {
      // noop
    }
  }

  async function generateQuiz() {
    try {
      const res = await fetch("/api/v1/quiz/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });
      currentQuiz = await res.json();
      renderQuiz();
      showToast("Quiz generated.", "info");
      setStatusChip(dom.quizStatusChip, "Generated", "info");
      setStatusTime(dom.quizStatusTime);
    } catch (err) {
      dom.quizResult.textContent = "Quiz generation failed: " + err.message;
      showToast("Quiz generation failed: " + err.message, "error");
      setStatusChip(dom.quizStatusChip, "Generation failed", "error");
      setStatusTime(dom.quizStatusTime);
    }
  }

  function renderQuiz() {
    if (!currentQuiz || !currentQuiz.questions) return;
    dom.quizContainer.style.display = "block";
    dom.quizMeta.textContent = `Stage ${currentQuiz.stage} | Difficulty: ${currentQuiz.difficulty}`;
    let html = "";
    currentQuiz.questions.forEach((q) => {
      html += `<div class="quiz-q"><strong>${escapeHtml(q.id)}.</strong> ${escapeHtml(q.question)}`;
      if (q.type === "mcq") {
        (q.options || []).forEach((opt) => {
          html += `<label><input type="radio" name="quiz_${escapeAttr(q.id)}" value="${escapeAttr(opt)}"> ${escapeHtml(opt)}</label>`;
        });
      } else {
        html += `<input type="text" name="quiz_${escapeAttr(q.id)}" placeholder="Your answer">`;
      }
      html += "</div>";
    });
    dom.quizQuestions.innerHTML = html;
    dom.quizResult.textContent = "";
  }

  async function submitQuiz() {
    if (!currentQuiz || !currentQuiz.questions) return;
    const answers = currentQuiz.questions.map((q) => {
      const sel = escapeSelector(q.id);
      if (q.type === "mcq") {
        const selected = dom.quizQuestions.querySelector(`input[name="quiz_${sel}"]:checked`);
        return { id: q.id, answer: selected ? selected.value : "" };
      }
      const inp = dom.quizQuestions.querySelector(`input[name="quiz_${sel}"]`);
      return { id: q.id, answer: inp ? inp.value : "" };
    });

    try {
      const res = await fetch("/api/v1/quiz/grade", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, stage: currentQuiz.stage, answers }),
      });
      const result = await res.json();
      dom.quizResult.textContent = `Quiz score: ${Math.round((result.score || 0) * 100)}% (${result.correct}/${result.total})`;
      showToast(`Quiz submitted: ${Math.round((result.score || 0) * 100)}%.`, "success");
      const pct = Math.round((result.score || 0) * 100);
      setStatusChip(dom.quizStatusChip, `${pct}%`, pct >= 67 ? "success" : "error");
      setStatusTime(dom.quizStatusTime);
      await refreshProgressAndDashboard();
    } catch (err) {
      dom.quizResult.textContent = "Quiz submit failed: " + err.message;
      showToast("Quiz submit failed: " + err.message, "error");
      setStatusChip(dom.quizStatusChip, "Submit failed", "error");
      setStatusTime(dom.quizStatusTime);
    }
  }

  async function submitCheckpoint() {
    const nextStage = (projectPlan && projectPlan.next_required_stage) || 1;
    const evidence = dom.checkpointEvidence.value.trim();
    if (!evidence) {
      dom.checkpointFeedback.textContent = "Add evidence before submitting.";
      return;
    }
    try {
      const res = await fetch("/api/v1/project/checkpoint", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, stage_num: nextStage, evidence }),
      });
      const out = await res.json();
      dom.checkpointFeedback.textContent = out.message || "";
      showToast(out.message || "Checkpoint updated.", out.accepted ? "success" : "info");
      if (out.accepted) {
        setStatusChip(dom.checkpointStatusChip, "Accepted", "success");
      } else if (out.assessment && typeof out.assessment.score === "number") {
        setStatusChip(dom.checkpointStatusChip, `Score ${Math.round(out.assessment.score * 100)}%`, "error");
      } else {
        setStatusChip(dom.checkpointStatusChip, "Pending", "info");
      }
      setStatusTime(dom.checkpointStatusTime);
      if (out.project_plan) projectPlan = out.project_plan;
      renderProjectPlanHint();
      await refreshProgressAndDashboard();
    } catch (err) {
      dom.checkpointFeedback.textContent = "Checkpoint submit failed: " + err.message;
      showToast("Checkpoint submit failed: " + err.message, "error");
      setStatusChip(dom.checkpointStatusChip, "Submit failed", "error");
      setStatusTime(dom.checkpointStatusTime);
    }
  }

  async function exportReport(format) {
    try {
      const res = await fetch(
        `/api/v1/session/${encodeURIComponent(sessionId)}/report?format=${encodeURIComponent(format)}`
      );
      const out = await res.json();
      dom.reportPath.textContent = out.path ? `Saved: ${out.path}` : "Report export failed";
      if (out.path) {
        showToast(`Report exported (${out.format}).`, "success");
        setStatusChip(dom.reportStatusChip, `Exported ${out.format}`, "success");
        setStatusTime(dom.reportStatusTime);
      } else {
        showToast("Report export failed.", "error");
        setStatusChip(dom.reportStatusChip, "Export failed", "error");
        setStatusTime(dom.reportStatusTime);
      }
    } catch (err) {
      dom.reportPath.textContent = "Report export failed: " + err.message;
      showToast("Report export failed: " + err.message, "error");
      setStatusChip(dom.reportStatusChip, "Export failed", "error");
      setStatusTime(dom.reportStatusTime);
    }
  }

  function renderProjectPlanHint() {
    if (!projectPlan) {
      dom.projectNextStage.textContent = "Upload dataset to enable.";
      return;
    }
    dom.projectNextStage.textContent = `Next required stage: ${projectPlan.next_required_stage}`;
  }

  function setSidebarOpen(isOpen) {
    const opening = Boolean(isOpen);
    if (opening) {
      lastFocusEl = document.activeElement;
    }
    dom.sidebar.classList.toggle("open", opening);
    const mobile = window.innerWidth <= 768;
    dom.sidebar.setAttribute("aria-hidden", mobile ? (opening ? "false" : "true") : "false");
    if (dom.sidebarBackdrop) dom.sidebarBackdrop.classList.toggle("show", Boolean(isOpen));
    dom.sidebarToggle.setAttribute("aria-expanded", opening ? "true" : "false");

    if (opening) {
      const firstFocusable = dom.sidebar.querySelector(
        'button, [href], input, textarea, select, [tabindex]:not([tabindex="-1"])'
      );
      if (firstFocusable) firstFocusable.focus();
    } else if (lastFocusEl && typeof lastFocusEl.focus === "function") {
      lastFocusEl.focus();
    }
  }

  function appendUserMsg(text) {
    const div = document.createElement("div");
    div.className = "msg msg-user";
    div.textContent = text;
    dom.chatMessages.appendChild(div);
    scrollBottom();
  }

  function appendAssistantMsg(data) {
    const div = document.createElement("div");
    div.className = "msg msg-assistant";
    let html = renderMarkdown(data.text || "");

    if (data.code) html += `<pre><code>${escapeHtml(data.code)}</code></pre>`;
    if (data.code_explained) {
      html += `<details class="code-details"><summary>Line-by-line explanation</summary><pre><code>${escapeHtml(
        data.code_explained
      )}</code></pre></details>`;
    }

    const conf = Number(data.confidence || 0);
    if (conf > 0) {
      const cls = conf >= 70 ? "conf-high" : conf >= 40 ? "conf-mid" : "conf-low";
      html += `<span class="confidence-badge ${cls}">${conf.toFixed(0)}% confidence</span>`;
    }
    if (data.confidence_label) html += `<div class="meta-line">${escapeHtml(data.confidence_label)}</div>`;
    if (data.critic && data.critic.feedback) html += `<div class="quality-note">${escapeHtml(data.critic.feedback)}</div>`;

    (data.pipeline_warnings || []).forEach((w) => {
      html += `<div class="warning-box warning-pipeline">${escapeHtml(w)}</div>`;
    });
    (data.antipattern_warnings || []).forEach((w) => {
      html += `<div class="warning-box warning-antipattern">${escapeHtml(w)}</div>`;
    });
    (data.misconception_alerts || []).forEach((m) => {
      html += `<div class="warning-box warning-misconception">${escapeHtml(
        m.correction || m.misconception || "Misconception alert"
      )}</div>`;
    });

    if (data.suggested_questions && data.suggested_questions.length) {
      html += '<div class="suggestions">';
      data.suggested_questions.forEach((sq) => {
        const q = sq.question || "";
        const label = q.length > 70 ? `${q.slice(0, 67)}...` : q;
        html += `<button class="suggestion-btn" data-q="${escapeAttr(q)}">${escapeHtml(label)}</button>`;
      });
      html += "</div>";
    }

    div.innerHTML = html;
    div.querySelectorAll(".suggestion-btn").forEach((btn) => {
      btn.onclick = () => {
        dom.chatInput.value = btn.dataset.q;
        dom.sendBtn.disabled = false;
        sendMessage();
      };
    });
    dom.chatMessages.appendChild(div);
    scrollBottom();
  }

  function appendError(text) {
    const div = document.createElement("div");
    div.className = "msg msg-assistant";
    div.style.borderColor = "#f44336";
    div.textContent = text;
    dom.chatMessages.appendChild(div);
    scrollBottom();
  }

  function showTyping() {
    const div = document.createElement("div");
    div.className = "typing-indicator";
    div.innerHTML =
      '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';
    dom.chatMessages.appendChild(div);
    scrollBottom();
    return div;
  }

  function removeWelcome() {
    const w = dom.chatMessages.querySelector(".welcome-msg");
    if (w) w.remove();
  }

  function scrollBottom() {
    dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
  }

  function renderPipeline() {
    const container = document.getElementById("pipeline-stages");
    let html = "";
    for (let s = 1; s <= 7; s++) {
      const done = completedStages.has(s);
      html += `<div class="stage-row ${done ? "done" : ""}">
        <div class="stage-marker ${done ? "marker-done" : "marker-pending"}">${done ? "&#10003;" : s}</div>
        ${STAGE_NAMES[s]}
      </div>`;
    }
    container.innerHTML = html;
    const pct = (completedStages.size / 7) * 100;
    document.getElementById("progress-bar").style.width = `${pct}%`;
    document.getElementById("progress-text").textContent = `${completedStages.size}/7 completed`;
  }

  function renderSkills() {
    const container = document.getElementById("skill-levels");
    let html = "";
    for (let s = 1; s <= 7; s++) {
      const val = Number(skills[s] || 0.5);
      const label = val < 0.3 ? "Beginner" : val <= 0.7 ? "Intermediate" : "Advanced";
      html += `<div class="skill-row">
        <div class="skill-label">${STAGE_NAMES[s]} <span>${label}</span></div>
        <div class="skill-bar-bg"><div class="skill-bar" style="width:${val * 100}%"></div></div>
      </div>`;
    }
    container.innerHTML = html;
  }

  function renderDatasetInfo(p) {
    if (!p) return;
    dom.datasetInfo.style.display = "block";
    dom.datasetInfo.innerHTML = `
      <strong>${escapeHtml(p.filename || "dataset")}</strong><br>
      ${escapeHtml(String(p.rows || 0))} rows &middot; ${escapeHtml(String(p.columns || 0))} columns<br>
      ${p.total_missing_pct !== undefined ? `Missing: ${escapeHtml(String(p.total_missing_pct))}%` : ""}
      ${p.target_guess ? `<br>Likely target: <strong>${escapeHtml(p.target_guess)}</strong>` : ""}
    `;
  }

  function renderAnalyticsCards(progress) {
    const cards = [
      ["Questions", progress.questions_asked || 0],
      ["Misconceptions", progress.misconceptions || 0],
      ["Avg conf", `${Number(progress.avg_confidence || 0).toFixed(1)}%`],
      ["Avg latency", `${progress.avg_response_ms || 0} ms`],
      ["Quizzes", progress.quiz_count || 0],
      ["Avg quiz", `${Math.round((progress.avg_quiz_score || 0) * 100)}%`],
      ["CP passed", progress.checkpoint_passed || 0],
      ["CP total", progress.checkpoint_total || 0],
    ];
    dom.analyticsCards.innerHTML = cards
      .map(([k, v]) => `<div class="metric-card"><span class="k">${escapeHtml(String(k))}</span><span class="v">${escapeHtml(String(v))}</span></div>`)
      .join("");
    if (progress.next_step) dom.nextStepHint.textContent = progress.next_step;
  }

  function renderSpark(container, values, cls) {
    if (!container) return;
    const vals = values.slice(-20);
    if (!vals.length) {
      container.innerHTML = `<span class="hint">No data yet</span>`;
      return;
    }
    const max = Math.max(...vals, 1);
    container.innerHTML = vals
      .map((v) => {
        const h = Math.max(4, Math.round((v / max) * 28));
        return `<div class="spark-bar ${cls || ""}" style="height:${h}px"></div>`;
      })
      .join("");
  }

  function renderMarkdown(text) {
    let html = escapeHtml(text || "");
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_m, _lang, code) => `<pre><code>${code.trim()}</code></pre>`);
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");
    html = html.replace(/\n/g, "<br>");
    return html;
  }

  function escapeHtml(str) {
    const d = document.createElement("div");
    d.textContent = str == null ? "" : String(str);
    return d.innerHTML;
  }

  function escapeAttr(str) {
    return String(str || "")
      .replace(/&/g, "&amp;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function escapeSelector(str) {
    if (window.CSS && typeof window.CSS.escape === "function") return window.CSS.escape(String(str || ""));
    return String(str || "").replace(/[^a-zA-Z0-9_-]/g, "_");
  }

  function showToast(message, type) {
    if (!dom.toastContainer || !message) return;
    const toast = document.createElement("div");
    toast.className = `toast ${type || "info"}`;
    toast.textContent = message;
    dom.toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.classList.add("hide");
      setTimeout(() => {
        if (toast.parentNode) toast.parentNode.removeChild(toast);
      }, 220);
    }, 2800);
  }

  function setStatusChip(el, text, type) {
    if (!el) return;
    el.textContent = text || "";
    el.classList.remove("success", "error", "info", "neutral");
    el.classList.add(type || "neutral");
  }

  function setStatusTime(el) {
    if (!el) return;
    const now = new Date();
    const hh = String(now.getHours()).padStart(2, "0");
    const mm = String(now.getMinutes()).padStart(2, "0");
    el.textContent = `updated ${hh}:${mm}`;
  }
})();
