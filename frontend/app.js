const askButton = document.querySelector("#ask");
const evalButton = document.querySelector("#eval");
const questionBox = document.querySelector("#question");
const answerBox = document.querySelector("#answer");
const citationsBox = document.querySelector("#citations");
const metricsBox = document.querySelector("#metrics");

askButton.addEventListener("click", async () => {
  setLoading("Retrieving evidence...");
  const response = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: questionBox.value, top_k: 5 })
  });
  const payload = await response.json();
  answerBox.classList.remove("muted");
  answerBox.textContent = payload.text;
  citationsBox.innerHTML = "";
  for (const citation of payload.citations) {
    const item = document.createElement("li");
    item.innerHTML = `<strong>${citation.document_id}</strong><br><span>${citation.quote}</span>`;
    citationsBox.appendChild(item);
  }
  metricsBox.textContent = JSON.stringify(payload.metrics, null, 2);
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

