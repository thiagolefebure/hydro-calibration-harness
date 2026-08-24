/* Presentation-only. Loads demo/assets/demo_data.json. No calibration logic. */

function formatInt(value) {
  return Number(value).toLocaleString("en-US");
}

function formatKgeArrow(from, to) {
  return `${Number(from).toFixed(3)} → ${Number(to).toFixed(3)}`;
}

function formatBiasArrow(from, to) {
  const asPct = (v) => {
    const pct = 100 * Number(v);
    return `${pct >= 0 ? "+" : ""}${pct.toFixed(1)}%`;
  };
  return `${asPct(from)} → ${asPct(to)}`;
}

function yearRange(period) {
  return `${period[0].slice(0, 4)}–${period[1].slice(0, 4)}`;
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function setHref(id, value) {
  const el = document.getElementById(id);
  if (el) el.setAttribute("href", value);
}

function setSrc(id, value) {
  const el = document.getElementById(id);
  if (el) el.setAttribute("src", value);
}

function activateTab(name) {
  document.querySelectorAll(".tab").forEach((btn) => {
    const active = btn.dataset.tab === name;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-selected", active ? "true" : "false");
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    const active = panel.id === `panel-${name}`;
    panel.classList.toggle("active", active);
    if (active) {
      panel.removeAttribute("hidden");
    } else {
      panel.setAttribute("hidden", "");
    }
  });
}

function render(data) {
  setText("prototype-badge", data.prototype_badge);
  setText("page-title", data.title);
  setText("page-subtitle", data.subtitle);
  setText("secondary-line", data.secondary_line);

  setText("basin-name", data.station.name);
  setText("station-code", data.station.code);
  setText("basin-area", `${data.station.basin_area_km2} km²`);
  setText("data-period", data.periods.analysis_label);
  setText("temporal-resolution", data.periods.temporal_resolution);
  setText("model-label", data.model.label);

  const kpis = data.kpis;
  setText("kpi-samples", formatInt(kpis.n_samples));
  setText(
    "kpi-kge",
    formatKgeArrow(kpis.validation_kge_uncalibrated, kpis.validation_kge_calibrated)
  );
  setText(
    "kpi-bias",
    formatBiasArrow(kpis.validation_bias_uncalibrated, kpis.validation_bias_calibrated)
  );
  setText("kpi-behavioral", formatInt(kpis.behavioral_members));
  setText(
    "kpi-behavioral-caption",
    `KGE_cal > ${Number(kpis.behavioral_threshold).toFixed(2)}`
  );

  setSrc("fig-calibration", data.figures.calibration);
  setSrc("fig-validation", data.figures.validation);
  setSrc("fig-uncertainty", data.figures.uncertainty);

  setText("bullet-samples", `${formatInt(kpis.n_samples)} parameter sets explored`);
  setText(
    "bullet-cal-period",
    `calibration: ${yearRange(data.periods.calibration)}`
  );
  setText(
    "bullet-val-period",
    `validation: ${yearRange(data.periods.validation)}`
  );
  setText(
    "bullet-behavioral",
    `${formatInt(kpis.behavioral_members)} behavioral parameter sets`
  );
  setText(
    "bullet-coverage",
    `empirical validation coverage: ${(100 * Number(data.uncertainty.empirical_validation_coverage)).toFixed(1)}%`
  );
  setText("bullet-envelope", data.statements.envelope_disclaimer);
  setText("validation-isolation", data.statements.validation_isolation);
  setText("parametric-limitation", data.statements.parametric_only);

  setText("config-hash", data.reproducibility.config_sha256);
  setText("git-commit", data.reproducibility.git_commit || "not available");
  setHref("report-link", data.report_html);

  const githubLink = document.getElementById("github-link");
  const githubPlaceholder = document.getElementById("github-placeholder");
  if (data.github_url) {
    githubLink.href = data.github_url;
    githubLink.hidden = false;
    githubPlaceholder.hidden = true;
  } else {
    githubLink.hidden = true;
    githubPlaceholder.hidden = false;
  }
}

function wireTabs() {
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => activateTab(btn.dataset.tab));
  });
}

async function main() {
  wireTabs();
  activateTab("calibration");
  const errorEl = document.getElementById("load-error");
  try {
    const response = await fetch("assets/demo_data.json", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(
        `Could not load assets/demo_data.json (${response.status}). Run: python run.py --export-demo`
      );
    }
    const data = await response.json();
    render(data);
  } catch (err) {
    errorEl.hidden = false;
    errorEl.textContent =
      String(err.message || err) +
      " If you opened the HTML file directly, try: python -m http.server 8000";
  }
}

main();
