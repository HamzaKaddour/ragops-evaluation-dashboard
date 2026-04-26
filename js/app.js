const state = {
  summary: null,
  runs: [],
  selectedPipeline: "All pipelines"
};

const plotLayout = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#e5e7eb", family: "Inter, system-ui, sans-serif" },
  margin: { t: 20, r: 20, b: 50, l: 55 },
  xaxis: { gridcolor: "rgba(148,163,184,0.16)", zerolinecolor: "rgba(148,163,184,0.2)" },
  yaxis: { gridcolor: "rgba(148,163,184,0.16)", zerolinecolor: "rgba(148,163,184,0.2)", range: [0, 1] },
  legend: { orientation: "h", y: -0.22 }
};

async function loadData() {
  const [summaryRes, runsRes] = await Promise.all([
    fetch("data/evaluation_summary.json"),
    fetch("data/rag_runs.json")
  ]);

  state.summary = await summaryRes.json();
  state.runs = await runsRes.json();

  renderSummaryCards();
  renderPipelineSelect();
  renderCharts();
  renderTraces();
}

function pct(value) {
  return `${Math.round(value * 100)}%`;
}

function money(value) {
  return `$${Number(value).toFixed(2)}`;
}

function renderSummaryCards() {
  const metrics = state.summary.summary_metrics;
  const cards = [
    ["Best pipeline", metrics.best_pipeline, "Highest combined retrieval and answer reliability"],
    ["Recall@5", pct(metrics.avg_recall_at_5), "Average across evaluated queries"],
    ["Groundedness", pct(metrics.avg_groundedness), "Evidence-supported answer rate"],
    ["Cost / 1K queries", money(metrics.estimated_cost_per_1000_queries_usd), "Estimated operational cost"]
  ];

  document.getElementById("summaryGrid").innerHTML = cards.map(([label, value, note]) => `
    <article class="metric-card">
      <div class="metric-label">${label}</div>
      <div class="metric-value">${value}</div>
      <div class="metric-note">${note}</div>
    </article>
  `).join("");
}

function renderPipelineSelect() {
  const pipelines = ["All pipelines", ...new Set(state.runs.map(run => run.pipeline))];
  const select = document.getElementById("pipelineSelect");
  select.innerHTML = pipelines.map(name => `<option value="${name}">${name}</option>`).join("");
  select.addEventListener("change", event => {
    state.selectedPipeline = event.target.value;
    renderTraces();
  });
}

function renderCharts() {
  const pipelines = state.summary.pipelines;
  const names = pipelines.map(item => item.name);

  Plotly.newPlot("retrievalChart", [
    { x: names, y: pipelines.map(p => p.recall_at_5), name: "Recall@5", type: "bar" },
    { x: names, y: pipelines.map(p => p.mrr), name: "MRR", type: "bar" },
    { x: names, y: pipelines.map(p => p.ndcg_at_5), name: "nDCG@5", type: "bar" }
  ], plotLayout, { responsive: true, displayModeBar: false });

  Plotly.newPlot("riskChart", [
    { x: names, y: pipelines.map(p => p.groundedness), name: "Groundedness", type: "scatter", mode: "lines+markers", line: { width: 4 } },
    { x: names, y: pipelines.map(p => p.hallucination_risk), name: "Hallucination risk", type: "scatter", mode: "lines+markers", line: { width: 4 } },
    { x: names, y: pipelines.map(p => p.citation_coverage), name: "Citation coverage", type: "scatter", mode: "lines+markers", line: { width: 4 } }
  ], plotLayout, { responsive: true, displayModeBar: false });

  Plotly.newPlot("opsChart", [
    { x: names, y: pipelines.map(p => p.avg_latency_ms), name: "Latency (ms)", type: "bar", yaxis: "y" },
    { x: names, y: pipelines.map(p => p.cost_per_1000_queries_usd), name: "Cost / 1K queries ($)", type: "scatter", mode: "lines+markers", yaxis: "y2", line: { width: 4 } }
  ], {
    ...plotLayout,
    yaxis: { title: "Latency (ms)", gridcolor: "rgba(148,163,184,0.16)" },
    yaxis2: { title: "Cost ($)", overlaying: "y", side: "right", gridcolor: "rgba(0,0,0,0)" }
  }, { responsive: true, displayModeBar: false });

  const failureTotals = {};
  pipelines.forEach(p => {
    Object.entries(p.failure_modes).forEach(([key, value]) => {
      failureTotals[key] = (failureTotals[key] || 0) + value;
    });
  });

  Plotly.newPlot("failureChart", [{
    labels: Object.keys(failureTotals).map(formatLabel),
    values: Object.values(failureTotals),
    type: "pie",
    hole: 0.45,
    textinfo: "label+percent"
  }], {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#e5e7eb", family: "Inter, system-ui, sans-serif" },
    margin: { t: 10, r: 10, b: 10, l: 10 },
    legend: { orientation: "h", y: -0.1 }
  }, { responsive: true, displayModeBar: false });
}

function renderTraces() {
  const filtered = state.selectedPipeline === "All pipelines"
    ? state.runs
    : state.runs.filter(run => run.pipeline === state.selectedPipeline);

  document.getElementById("traceList").innerHTML = filtered.map(run => `
    <article class="trace-card">
      <div class="trace-top">
        <div>
          <div class="trace-question">${run.query}</div>
          <p class="trace-answer">${run.answer}</p>
        </div>
        <span class="chip">${run.pipeline}</span>
      </div>

      <div class="trace-metrics">
        <div class="trace-metric"><span>Recall@5</span><strong>${pct(run.metrics.recall_at_5)}</strong></div>
        <div class="trace-metric"><span>Precision@5</span><strong>${pct(run.metrics.precision_at_5)}</strong></div>
        <div class="trace-metric"><span>Groundedness</span><strong>${pct(run.metrics.groundedness)}</strong></div>
        <div class="trace-metric"><span>Risk</span><strong>${pct(run.metrics.hallucination_risk)}</strong></div>
        <div class="trace-metric"><span>Latency</span><strong>${run.metrics.latency_ms} ms</strong></div>
      </div>

      <div class="sources">
        ${run.retrieved_sources.map(source => `
          <div class="source"><strong>${source.title}</strong> · score ${source.score}<br>${source.snippet}</div>
        `).join("")}
      </div>
    </article>
  `).join("");
}

function formatLabel(value) {
  return value.replaceAll("_", " ").replace(/\b\w/g, char => char.toUpperCase());
}

loadData().catch(error => {
  console.error(error);
  document.body.insertAdjacentHTML("afterbegin", `<div style="padding:1rem;background:#7f1d1d;color:white">Failed to load dashboard data. Use a local server instead of opening index.html directly.</div>`);
});
