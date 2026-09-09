const askButton = document.querySelector("#ask");
const evalButton = document.querySelector("#eval");
const questionBox = document.querySelector("#question");
const topKBox = document.querySelector("#topK");
const scoreFloorBox = document.querySelector("#scoreFloor");
const filterBox = document.querySelector("#filter");
const answerBox = document.querySelector("#answer");
const citationsBox = document.querySelector("#citations");
const metricsBox = document.querySelector("#metrics");

askButton.addEventListener("click", async () => {
  try {
    setLoading("Retrieving evidence...");
    const response = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildQueryPayload())
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "The query failed.");
    }
    answerBox.classList.remove("muted");
    answerBox.textContent = payload.text;
    citationsBox.innerHTML = "";
    for (const citation of payload.citations) {
      const item = document.createElement("li");
      item.innerHTML = `<strong>${citation.document_id}</strong><br><span>${citation.quote}</span>`;
      citationsBox.appendChild(item);
    }
    metricsBox.textContent = JSON.stringify(payload.metrics, null, 2);
  } catch (error) {
    showError(error.message);
  }
});

evalButton.addEventListener("click", async () => {
  setLoading("Running evaluation suite...");
  const response = await fetch("/api/evaluate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ top_k: 5 })
  });
  const payload = await response.json();
  answerBox.textContent = "Evaluation complete. Summary metrics are shown on the right.";
  citationsBox.innerHTML = "";
  metricsBox.textContent = JSON.stringify(payload.summary, null, 2);
});

function setLoading(message) {
  answerBox.classList.add("muted");
  answerBox.textContent = message;
  citationsBox.innerHTML = "";
  metricsBox.textContent = "{}";
}

function showError(message) {
  answerBox.classList.add("muted");
  answerBox.textContent = message;
  citationsBox.innerHTML = "";
  metricsBox.textContent = "{}";
}

function buildQueryPayload() {
  const payload = {
    question: questionBox.value,
    top_k: Number.parseInt(topKBox.value || "5", 10),
    filters: parseFilters(filterBox.value),
  };
  if (scoreFloorBox.value.trim()) {
    payload.score_floor = Number.parseFloat(scoreFloorBox.value);
  }
  return payload;
}

function parseFilters(raw) {
  const filters = {};
  for (const item of raw.split(",")) {
    const part = item.trim();
    if (!part) {
      continue;
    }
    const [key, ...rest] = part.split("=");
    if (!key.trim() || rest.length === 0) {
      throw new Error(`Invalid filter "${part}". Use key=value.`);
    }
    filters[key.trim()] = rest.join("=").trim();
  }
  return filters;
}
