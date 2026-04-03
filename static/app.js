/* DS Mentor Pro -- Frontend logic */
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

  /* ---- State ---- */
  let completedStages = new Set();
  let skills = {};
  for (let i = 1; i <= 7; i++) skills[i] = 0.5;
  let datasetProfile = null;
  let sessionId = "web_" + Math.random().toString(36).slice(2, 10);

  /* ---- DOM refs ---- */
  const chatMessages = document.getElementById("chat-messages");
  const chatInput = document.getElementById("chat-input");
  const sendBtn = document.getElementById("send-btn");
  const fileInput = document.getElementById("file-input");
  const uploadArea = document.getElementById("upload-area");
  const datasetInfo = document.getElementById("dataset-info");
  const resetBtn = document.getElementById("reset-btn");
  const sidebarToggle = document.getElementById("sidebar-toggle");
  const sidebar = document.getElementById("sidebar");

  /* ---- Init ---- */
  renderPipeline();
  renderSkills();

  /* ---- Sidebar toggle (mobile) ---- */
  sidebarToggle.addEventListener("click", () =>
    sidebar.classList.toggle("open")
  );

  /* ---- Input handling ---- */
  chatInput.addEventListener("input", () => {
    chatInput.style.height = "auto";
    chatInput.style.height = chatInput.scrollHeight + "px";
    sendBtn.disabled = !chatInput.value.trim();
  });

  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (chatInput.value.trim()) sendMessage();
    }
  });

  sendBtn.addEventListener("click", () => {
    if (chatInput.value.trim()) sendMessage();
  });

  /* ---- Quick prompt buttons ---- */
  document.querySelectorAll(".quick-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      chatInput.value = btn.dataset.q;
      sendBtn.disabled = false;
      sendMessage();
    });
  });

  /* ---- Reset ---- */
  resetBtn.addEventListener("click", () => {
    if (!confirm("Reset the current session?")) return;
    completedStages.clear();
    for (let i = 1; i <= 7; i++) skills[i] = 0.5;
    datasetProfile = null;
    sessionId = "web_" + Math.random().toString(36).slice(2, 10);
    chatMessages.innerHTML = "";
    renderPipeline();
    renderSkills();
    datasetInfo.style.display = "none";
    addWelcome();
  });

  /* ---- File upload ---- */
  fileInput.addEventListener("change", async () => {
    const file = fileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    uploadArea.querySelector(".upload-text").textContent = "Uploading...";
    try {
      const res = await fetch("/upload", { method: "POST", body: formData });
      const data = await res.json();
      if (data.profile) {
        datasetProfile = data.profile;
        renderDatasetInfo(data.profile);
      } else if (data.profile_error) {
        alert("Could not parse file: " + data.profile_error);
      }
    } catch (err) {
      alert("Upload failed: " + err.message);
    }
    uploadArea.querySelector(".upload-text").textContent = "Upload CSV or Excel";
    fileInput.value = "";
  });

  /* ---- Send message ---- */
  async function sendMessage() {
    const query = chatInput.value.trim();
    if (!query) return;

    removeWelcome();
    appendUserMsg(query);
    chatInput.value = "";
    chatInput.style.height = "auto";
    sendBtn.disabled = true;

    const typingEl = showTyping();

    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, code: "", session_id: sessionId }),
      });
      const data = await res.json();
      typingEl.remove();
      appendAssistantMsg(data);
      updateState(data);
    } catch (err) {
      typingEl.remove();
      appendError("Request failed: " + err.message);
    }
  }

  /* ---- Render helpers ---- */

  function appendUserMsg(text) {
    const div = document.createElement("div");
    div.className = "msg msg-user";
    div.textContent = text;
    chatMessages.appendChild(div);
    scrollBottom();
  }

  function appendAssistantMsg(data) {
    const div = document.createElement("div");
    div.className = "msg msg-assistant";

    let html = renderMarkdown(data.text || "");

    if (data.code) {
      html += `<pre><code>${escapeHtml(data.code)}</code></pre>`;
    }

    /* confidence badge */
    const conf = data.confidence || 0;
    if (conf > 0) {
      const cls =
        conf >= 70 ? "conf-high" : conf >= 40 ? "conf-mid" : "conf-low";
      html += `<span class="confidence-badge ${cls}">${conf.toFixed(
        0
      )}% confidence</span>`;
    }

    /* pipeline warnings */
    if (data.pipeline_warnings && data.pipeline_warnings.length) {
      data.pipeline_warnings.forEach((w) => {
        html += `<div class="warning-box warning-pipeline">${escapeHtml(
          w
        )}</div>`;
      });
    }

    /* antipattern warnings */
    if (data.antipattern_warnings && data.antipattern_warnings.length) {
      data.antipattern_warnings.forEach((w) => {
        html += `<div class="warning-box warning-antipattern">${escapeHtml(
          w
        )}</div>`;
      });
    }

    /* suggested questions */
    if (data.suggested_questions && data.suggested_questions.length) {
      html += '<div class="suggestions">';
      data.suggested_questions.forEach((sq) => {
        const label =
          sq.question.length > 70
            ? sq.question.slice(0, 67) + "..."
            : sq.question;
        html += `<button class="suggestion-btn" data-q="${escapeAttr(
          sq.question
        )}">${escapeHtml(label)}</button>`;
      });
      html += "</div>";
    }

    div.innerHTML = html;

    /* wire suggestion buttons */
    div.querySelectorAll(".suggestion-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        chatInput.value = btn.dataset.q;
        sendBtn.disabled = false;
        sendMessage();
      });
    });

    chatMessages.appendChild(div);
    scrollBottom();
  }

  function appendError(text) {
    const div = document.createElement("div");
    div.className = "msg msg-assistant";
    div.style.borderColor = "#f44336";
    div.textContent = text;
    chatMessages.appendChild(div);
    scrollBottom();
  }

  function showTyping() {
    const div = document.createElement("div");
    div.className = "typing-indicator";
    div.innerHTML =
      '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';
    chatMessages.appendChild(div);
    scrollBottom();
    return div;
  }

  function removeWelcome() {
    const w = chatMessages.querySelector(".welcome-msg");
    if (w) w.remove();
  }

  function addWelcome() {
    chatMessages.innerHTML = `
      <div class="welcome-msg">
        <h3>Welcome to DS Mentor Pro</h3>
        <p>Ask a question about any stage of the data science pipeline. Upload a dataset to get context-aware guidance.</p>
        <div class="quick-prompts">
          <button class="quick-btn" data-q="What is the goal of a Titanic survival prediction task?">Problem Understanding</button>
          <button class="quick-btn" data-q="How do I load a CSV file with pandas?">Data Loading</button>
          <button class="quick-btn" data-q="How do I create a correlation heatmap?">EDA</button>
          <button class="quick-btn" data-q="How do I fill missing values in a numeric column?">Preprocessing</button>
          <button class="quick-btn" data-q="How do I apply one-hot encoding?">Feature Engineering</button>
          <button class="quick-btn" data-q="Train a Random Forest classifier">Modeling</button>
          <button class="quick-btn" data-q="How do I calculate AUC-ROC score?">Evaluation</button>
        </div>
      </div>`;
    chatMessages.querySelectorAll(".quick-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        chatInput.value = btn.dataset.q;
        sendBtn.disabled = false;
        sendMessage();
      });
    });
  }

  function scrollBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  /* ---- Sidebar renders ---- */

  function renderPipeline() {
    const container = document.getElementById("pipeline-stages");
    let html = "";
    for (let s = 1; s <= 7; s++) {
      const done = completedStages.has(s);
      html += `<div class="stage-row ${done ? "done" : ""}">
        <div class="stage-marker ${done ? "marker-done" : "marker-pending"}">${
        done ? "&#10003;" : s
      }</div>
        ${STAGE_NAMES[s]}
      </div>`;
    }
    container.innerHTML = html;

    const pct = (completedStages.size / 7) * 100;
    document.getElementById("progress-bar").style.width = pct + "%";
    document.getElementById("progress-text").textContent =
      completedStages.size + "/7 completed";
  }

  function renderSkills() {
    const container = document.getElementById("skill-levels");
    let html = "";
    for (let s = 1; s <= 7; s++) {
      const val = skills[s] || 0.5;
      const label =
        val < 0.3 ? "Beginner" : val <= 0.7 ? "Intermediate" : "Advanced";
      html += `<div class="skill-row">
        <div class="skill-label">${STAGE_NAMES[s]} <span>${label}</span></div>
        <div class="skill-bar-bg"><div class="skill-bar" style="width:${
          val * 100
        }%"></div></div>
      </div>`;
    }
    container.innerHTML = html;
  }

  function renderDatasetInfo(p) {
    datasetInfo.style.display = "block";
    datasetInfo.innerHTML = `
      <strong>${escapeHtml(p.filename)}</strong><br>
      ${p.rows} rows &middot; ${p.columns} columns<br>
      Missing: ${p.total_missing_pct}%
      ${
        p.target_guess
          ? "<br>Likely target: <strong>" +
            escapeHtml(p.target_guess) +
            "</strong>"
          : ""
      }
    `;
  }

  /* ---- State update from response ---- */

  function updateState(data) {
    if (data.stage_num) completedStages.add(data.stage_num);
    renderPipeline();

    if (data.next_step) {
      document.getElementById("next-step-hint").textContent = data.next_step;
    }
  }

  /* ---- Markdown-lite renderer ---- */

  function renderMarkdown(text) {
    let html = escapeHtml(text);

    /* code blocks: ```python ... ``` */
    html = html.replace(
      /```(\w*)\n([\s\S]*?)```/g,
      (_, lang, code) => `<pre><code>${code.trim()}</code></pre>`
    );

    /* inline code */
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

    /* bold */
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

    /* italic */
    html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");

    /* line breaks */
    html = html.replace(/\n/g, "<br>");

    return html;
  }

  /* ---- Utils ---- */
  function escapeHtml(str) {
    const d = document.createElement("div");
    d.textContent = str;
    return d.innerHTML;
  }

  function escapeAttr(str) {
    return str
      .replace(/&/g, "&amp;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }
})();
