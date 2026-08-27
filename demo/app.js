/* Présentation uniquement. Charge demo/assets/demo_data.json. Aucune logique de calage. */

function formatInt(value) {
  return Number(value).toLocaleString("fr-FR");
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

function renderMlCorrection(ml) {
  if (!ml) return;

  setText("ml-methodology-label", ml.methodology_label);
  setText("ml-section-title", ml.section_title);
  setText("ml-section-subtitle", ml.section_subtitle);
  setText("ml-scope-note", ml.scope_note);
  setText("ml-gain-pct", ml.gain_captured_display);
  setText("ml-gain-label", ml.gain_captured_label);
  setText("ml-acf-title", ml.residual_acf.title);
  setText("ml-acf-caption", ml.residual_acf.caption);
  setText("ml-result-heading", ml.result.heading);
  setText("ml-result-body", ml.result.body);
  setText("ml-decision-heading", ml.engineering_decision.heading);
  setText("ml-decision-body", ml.engineering_decision.body);
  setText("ml-arch-caption", ml.architecture_caption);

  const layer = document.getElementById("ml-residual-layer-label");
  if (layer) {
    layer.innerHTML = `Correction pilotée par les données<br /><small>${ml.residual_layer_label}</small>`;
  }

  const pipeline = document.getElementById("ml-pipeline");
  if (pipeline) {
    pipeline.innerHTML = "";
    (ml.pipeline_steps || []).forEach((step) => {
      const li = document.createElement("li");
      li.textContent = step;
      pipeline.appendChild(li);
    });
  }

  const bars = document.getElementById("ml-kge-bars");
  if (bars) {
    bars.innerHTML = "";
    const axisMin = Number(ml.kge_axis.min);
    const axisMax = Number(ml.kge_axis.max);
    const span = axisMax - axisMin;
    (ml.models || []).forEach((model) => {
      const row = document.createElement("div");
      row.className = "ml-bar-row";
      const label = document.createElement("div");
      label.className = "ml-bar-label";
      label.textContent = model.label;
      const track = document.createElement("div");
      track.className = "ml-bar-track";
      const fill = document.createElement("div");
      fill.className = "ml-bar-fill";
      if (model.id === "physical") fill.classList.add("is-physical");
      if (model.id === "persistence") fill.classList.add("is-persistence");
      const widthPct = Math.max(
        0,
        Math.min(100, ((Number(model.kge_val) - axisMin) / span) * 100)
      );
      fill.style.width = `${widthPct}%`;
      const value = document.createElement("span");
      value.className = "ml-bar-value";
      value.textContent = Number(model.kge_val).toFixed(3);
      track.appendChild(fill);
      track.appendChild(value);
      row.appendChild(label);
      row.appendChild(track);
      bars.appendChild(row);
    });
  }

  const acfBars = document.getElementById("ml-acf-bars");
  if (acfBars) {
    acfBars.innerHTML = "";
    const lags = [
      { key: "lag_1", label: "t−1" },
      { key: "lag_2", label: "t−2" },
      { key: "lag_3", label: "t−3" },
    ];
    lags.forEach((lag) => {
      const row = document.createElement("div");
      row.className = "ml-acf-row";
      const label = document.createElement("span");
      label.textContent = lag.label;
      const track = document.createElement("div");
      track.className = "ml-acf-track";
      const fill = document.createElement("div");
      fill.className = "ml-acf-fill";
      const val = Number(ml.residual_acf[lag.key]);
      fill.style.width = `${Math.max(0, Math.min(100, val * 100))}%`;
      const num = document.createElement("strong");
      num.textContent = val.toFixed(2);
      track.appendChild(fill);
      row.appendChild(label);
      row.appendChild(track);
      row.appendChild(num);
      acfBars.appendChild(row);
    });
  }

  const exp = ml.explanation;
  setText("ml-exp-physical-title", exp.physical.title);
  setText("ml-exp-physical-body", exp.physical.body);
  setText("ml-exp-residual-title", exp.residual.title);
  setText("ml-exp-residual-eq", exp.residual.equation);
  setText("ml-exp-residual-body", exp.residual.body);
  setText("ml-exp-correction-title", exp.correction.title);
  setText("ml-exp-correction-body", exp.correction.body);
  setText("ml-exp-correction-eq", exp.correction.equation);

  const guardList = document.getElementById("ml-guardrails-list");
  if (guardList) {
    guardList.innerHTML = "";
    (ml.guardrails || []).forEach((item) => {
      const li = document.createElement("li");
      li.textContent = item;
      guardList.appendChild(li);
    });
  }
}

function renderForecastHorizons(fh) {
  if (!fh) return;

  setText("fh-oracle-title", fh.oracle_banner_title);
  setText("fh-oracle-body", fh.oracle_banner_body);
  setText("fh-oracle-emphasis", fh.oracle_banner_emphasis);
  setText("fh-section-title", fh.section_title);
  setText("fh-section-subtitle", fh.section_subtitle);
  setText("fh-scope-note", fh.scope_note);
  setText("fh-chart-title", fh.chart.title);
  setText("fh-axis-note", fh.chart.axis_note);
  setText("fh-deg-loss", fh.persistence_degradation_display);
  setText("fh-deg-caption", fh.degradation_caption);
  setText("fh-eq-hybrid", fh.equation.hybrid);
  setText("fh-eq-persistence", fh.equation.persistence);
  setText("fh-eq-residual", fh.equation.residual_def);
  setText("fh-eq-note", fh.equation.note);
  setText("fh-acf-title", fh.residual_acf.title);
  setText("fh-acf-explanation", fh.residual_acf.explanation);
  setText("fh-arch-caption", fh.architecture_caption);
  setText("fh-hf-threshold-note", fh.high_flow.threshold_note);
  setText("fh-hf-interpretation", fh.high_flow.interpretation);
  setText("fh-result-heading", fh.result.heading);
  setText("fh-result-body", fh.result.body);
  setText("fh-complexity-heading", fh.model_complexity.heading);
  setText("fh-complexity-body", fh.model_complexity.body);
  setText("fh-next-heading", fh.next_question.heading);
  setText("fh-next-body", fh.next_question.body);

  const svg = document.getElementById("fh-kge-chart");
  const legend = document.getElementById("fh-legend");
  if (svg && fh.chart) {
    svg.setAttribute("viewBox", fh.chart.view_box);
    svg.innerHTML = "";
    const ns = "http://www.w3.org/2000/svg";
    // Axes
    const axis = document.createElementNS(ns, "line");
    axis.setAttribute("x1", "28");
    axis.setAttribute("y1", String(fh.chart.y_bottom));
    axis.setAttribute("x2", "300");
    axis.setAttribute("y2", String(fh.chart.y_bottom));
    axis.setAttribute("class", "fh-axis");
    svg.appendChild(axis);
    const yAxis = document.createElementNS(ns, "line");
    yAxis.setAttribute("x1", "28");
    yAxis.setAttribute("y1", String(fh.chart.y_top));
    yAxis.setAttribute("x2", "28");
    yAxis.setAttribute("y2", String(fh.chart.y_bottom));
    yAxis.setAttribute("class", "fh-axis");
    svg.appendChild(yAxis);

    ["0.0", "0.5", "1.0"].forEach((lab, i) => {
      const y = fh.chart.y_bottom - (i / 2) * (fh.chart.y_bottom - fh.chart.y_top);
      const t = document.createElementNS(ns, "text");
      t.setAttribute("x", "24");
      t.setAttribute("y", String(y + 3));
      t.setAttribute("class", "fh-tick");
      t.setAttribute("text-anchor", "end");
      t.textContent = lab;
      svg.appendChild(t);
    });

    fh.chart.x_labels.forEach((lab, i) => {
      const t = document.createElementNS(ns, "text");
      t.setAttribute("x", String(fh.chart.x_positions[i]));
      t.setAttribute("y", String(fh.chart.y_bottom + 14));
      t.setAttribute("class", "fh-tick");
      t.setAttribute("text-anchor", "middle");
      t.textContent = lab;
      svg.appendChild(t);
    });

    const colors = {
      physical: "#64748b",
      persistence: "#1d4ed8",
      ar1: "#0ea5e9",
      ridge: "#b45309",
    };
    (fh.chart.series || []).forEach((series) => {
      const poly = document.createElementNS(ns, "polyline");
      poly.setAttribute("points", series.polyline);
      poly.setAttribute("fill", "none");
      poly.setAttribute("stroke", colors[series.id] || "#1d4ed8");
      poly.setAttribute("stroke-width", series.id === "persistence" ? "2.4" : "1.8");
      poly.setAttribute("class", `fh-series fh-series-${series.id}`);
      svg.appendChild(poly);
      series.polyline.split(" ").forEach((pt) => {
        const [x, y] = pt.split(",");
        const c = document.createElementNS(ns, "circle");
        c.setAttribute("cx", x);
        c.setAttribute("cy", y);
        c.setAttribute("r", "2.6");
        c.setAttribute("fill", colors[series.id] || "#1d4ed8");
        svg.appendChild(c);
      });
    });

    if (legend) {
      legend.innerHTML = "";
      (fh.chart.series || []).forEach((series) => {
        const li = document.createElement("li");
        li.innerHTML = `<span class="fh-swatch" style="background:${colors[series.id] || "#1d4ed8"}"></span>${series.label}`;
        legend.appendChild(li);
      });
    }
  }

  const cards = document.getElementById("fh-kpi-cards");
  if (cards) {
    cards.innerHTML = "";
    (fh.horizons || []).forEach((h) => {
      const article = document.createElement("article");
      article.className = "fh-kpi-card";
      article.innerHTML = `
        <h4>${h.label}</h4>
        <p><span>KGE physique</span><strong>${h.physical_kge_display}</strong></p>
        <p><span>KGE persistance</span><strong>${h.persistence_kge_display}</strong></p>
        <p class="fh-gain"><span>Gain de KGE</span><strong>${h.persistence_gain_display}</strong></p>
      `;
      cards.appendChild(article);
    });
  }

  const acfBars = document.getElementById("fh-acf-bars");
  if (acfBars) {
    acfBars.innerHTML = "";
    const lags = [
      { key: "lag_1", label: "t−1", display: "lag_1_display" },
      { key: "lag_2", label: "t−2", display: "lag_2_display" },
      { key: "lag_3", label: "t−3", display: "lag_3_display" },
    ];
    lags.forEach((lag) => {
      const row = document.createElement("div");
      row.className = "ml-acf-row";
      const label = document.createElement("span");
      label.textContent = lag.label;
      const track = document.createElement("div");
      track.className = "ml-acf-track";
      const fill = document.createElement("div");
      fill.className = "ml-acf-fill";
      const val = Number(fh.residual_acf[lag.key]);
      fill.style.width = `${Math.max(0, Math.min(100, val * 100))}%`;
      const num = document.createElement("strong");
      num.textContent = fh.residual_acf[lag.display];
      track.appendChild(fill);
      row.appendChild(label);
      row.appendChild(track);
      row.appendChild(num);
      acfBars.appendChild(row);
    });
  }

  const hf = document.getElementById("fh-highflow-table");
  if (hf) {
    const rows = fh.high_flow.rows || [];
    let html = `<table class="fh-table"><thead><tr><th></th>`;
    rows.forEach((r) => {
      html += `<th>${r.label}</th>`;
    });
    html += `</tr></thead><tbody><tr><td>Physique</td>`;
    rows.forEach((r) => {
      html += `<td>${r.physical_mae_display}</td>`;
    });
    html += `</tr><tr><td>Persistance</td>`;
    rows.forEach((r) => {
      html += `<td>${r.persistence_mae_display}</td>`;
    });
    html += `</tr></tbody></table>`;
    hf.innerHTML = html;
  }
}

function renderSvgKgeChart(svgId, legendId, chart, colors) {
  const svg = document.getElementById(svgId);
  const legend = document.getElementById(legendId);
  if (!svg || !chart) return;
  svg.setAttribute("viewBox", chart.view_box);
  svg.innerHTML = "";
  const ns = "http://www.w3.org/2000/svg";
  const axis = document.createElementNS(ns, "line");
  axis.setAttribute("x1", "28");
  axis.setAttribute("y1", String(chart.y_bottom));
  axis.setAttribute("x2", "300");
  axis.setAttribute("y2", String(chart.y_bottom));
  axis.setAttribute("class", "fh-axis");
  svg.appendChild(axis);
  const yAxis = document.createElementNS(ns, "line");
  yAxis.setAttribute("x1", "28");
  yAxis.setAttribute("y1", String(chart.y_top));
  yAxis.setAttribute("x2", "28");
  yAxis.setAttribute("y2", String(chart.y_bottom));
  yAxis.setAttribute("class", "fh-axis");
  svg.appendChild(yAxis);
  ["0.0", "0.5", "1.0"].forEach((lab, i) => {
    const y = chart.y_bottom - (i / 2) * (chart.y_bottom - chart.y_top);
    const t = document.createElementNS(ns, "text");
    t.setAttribute("x", "24");
    t.setAttribute("y", String(y + 3));
    t.setAttribute("class", "fh-tick");
    t.setAttribute("text-anchor", "end");
    t.textContent = lab;
    svg.appendChild(t);
  });
  chart.x_labels.forEach((lab, i) => {
    const t = document.createElementNS(ns, "text");
    t.setAttribute("x", String(chart.x_positions[i]));
    t.setAttribute("y", String(chart.y_bottom + 14));
    t.setAttribute("class", "fh-tick");
    t.setAttribute("text-anchor", "middle");
    t.textContent = lab;
    svg.appendChild(t);
  });
  (chart.series || []).forEach((series) => {
    const color = colors[series.id] || "#1d4ed8";
    const poly = document.createElementNS(ns, "polyline");
    poly.setAttribute("points", series.polyline);
    poly.setAttribute("fill", "none");
    poly.setAttribute("stroke", color);
    poly.setAttribute("stroke-width", series.id === "oracle" || series.id === "persistence" ? "2.4" : "1.8");
    svg.appendChild(poly);
    series.polyline.split(" ").forEach((pt) => {
      const [x, y] = pt.split(",");
      const c = document.createElementNS(ns, "circle");
      c.setAttribute("cx", x);
      c.setAttribute("cy", y);
      c.setAttribute("r", "2.6");
      c.setAttribute("fill", color);
      svg.appendChild(c);
    });
  });
  if (legend) {
    legend.innerHTML = "";
    (chart.series || []).forEach((series) => {
      const li = document.createElement("li");
      li.innerHTML = `<span class="fh-swatch" style="background:${colors[series.id] || "#1d4ed8"}"></span>${series.label}`;
      legend.appendChild(li);
    });
  }
}

function renderMeteoSensitivity(ms) {
  if (!ms) return;

  setText("ms-banner-title", ms.banner_title);
  setText("ms-banner-body", ms.banner_body);
  setText("ms-banner-emphasis", ms.banner_emphasis);
  setText("ms-section-title", ms.section_title);
  setText("ms-section-subtitle", ms.section_subtitle);
  setText("ms-disclaimer", ms.disclaimer);
  setText("ms-chart-title", ms.chart.title);
  setText("ms-axis-note", ms.chart.axis_note);
  setText("ms-hero-label", ms.hero_72h.label);
  setText("ms-hero-oracle", ms.hero_72h.oracle_display);
  setText("ms-hero-strong", ms.hero_72h.strong_display);
  setText("ms-hero-delta", ms.hero_72h.delta_display);
  setText("ms-hero-statement", ms.hero_72h.statement);
  setText("ms-loss-label", ms.skill_loss_label);
  setText("ms-mc-p10", ms.monte_carlo.p10_display);
  setText("ms-mc-median", ms.monte_carlo.median_display);
  setText("ms-mc-p90", ms.monte_carlo.p90_display);
  setText("ms-mc-spread", ms.monte_carlo.spread_display);
  setText("ms-mc-caption-n", ms.monte_carlo.caption_n);
  setText("ms-mc-caption-spread", ms.monte_carlo.caption_spread);
  setText("ms-persist-interp", ms.persistence_comparison.interpretation);
  setText("ms-complexity-heading", ms.model_complexity.heading);
  setText("ms-complexity-body", ms.model_complexity.body);
  setText("ms-ridge-max", ms.model_complexity.max_abs_display);
  setText("ms-complexity-footnote", ms.model_complexity.footnote);
  setText("ms-scale-title", ms.scale_comparison.title);
  setText("ms-scale-subtitle", ms.scale_comparison.subtitle);
  setText("ms-scale-meteo-label", ms.scale_comparison.meteo_label);
  setText("ms-scale-resid-label", ms.scale_comparison.residual_label);
  setText("ms-scale-meteo-val", ms.scale_comparison.meteo_display);
  setText("ms-scale-resid-val", ms.scale_comparison.residual_display);
  setText("ms-scale-note", ms.scale_comparison.note);
  setText("ms-hf-threshold", ms.high_flow.threshold_note);
  setText("ms-hf-interp", ms.high_flow.interpretation);
  setText("ms-nwp-label", ms.nwp_label);
  setText("ms-arch-caption", ms.architecture_caption);
  setText("ms-result-heading", ms.result.heading);
  setText("ms-result-body", ms.result.body);
  setText("ms-eng-heading", ms.engineering_decision.heading);
  setText("ms-eng-body", ms.engineering_decision.body);
  setText("ms-ml-heading", ms.ml_decision.heading);
  setText("ms-ml-body", ms.ml_decision.body);
  setText("ms-next-heading", ms.next_question.heading);
  setText("ms-next-body", ms.next_question.body);
  setText("ms-next-lead", ms.next_question.lead_in);

  const cards = document.getElementById("ms-scenario-cards");
  if (cards) {
    cards.innerHTML = "";
    (ms.scenario_cards || []).forEach((sc) => {
      const article = document.createElement("article");
      article.className = `ms-scenario-card is-${sc.id}`;
      article.innerHTML = `
        <h4>${sc.label}</h4>
        <p>${sc.description}</p>
        <p class="ms-purpose"><strong>Objectif :</strong> ${sc.purpose}</p>
      `;
      cards.appendChild(article);
    });
  }

  const details = document.getElementById("ms-method-details");
  if (details && ms.methodology_details) {
    const md = ms.methodology_details;
    const moderate = md.moderate || {};
    const strong = md.strong || {};
    details.textContent = [
      `Réalisations : ${md.n_realizations}`,
      `Graine : ${md.seed}`,
      `Scénario modéré : ${JSON.stringify(moderate)}`,
      `Scénario fort : ${JSON.stringify(strong)}`,
      md.note || "",
    ]
      .filter(Boolean)
      .join("\n");
  }

  renderSvgKgeChart("ms-kge-chart", "ms-legend", ms.chart, {
    oracle: "#1d4ed8",
    moderate: "#0ea5e9",
    strong: "#b45309",
  });

  const lossCards = document.getElementById("ms-loss-cards");
  if (lossCards) {
    lossCards.innerHTML = "";
    (ms.skill_loss_cards || []).forEach((card) => {
      const article = document.createElement("article");
      article.className = "fh-kpi-card";
      article.innerHTML = `
        <h4>${card.label}</h4>
        <p><span>Oracle → Fort</span></p>
        <p class="fh-gain"><span>Perte de performance</span><strong>${card.skill_loss_display}</strong></p>
      `;
      lossCards.appendChild(article);
    });
  }

  const band = document.getElementById("ms-mc-band");
  if (band) {
    const p10 = ms.monte_carlo.p10_pct;
    const p90 = ms.monte_carlo.p90_pct;
    const med = ms.monte_carlo.median_pct;
    band.innerHTML = `
      <div class="ms-mc-range" style="left:${p10}%; width:${Math.max(0, p90 - p10)}%"></div>
      <div class="ms-mc-median-mark" style="left:${med}%"></div>
      <span class="ms-mc-end" style="left:${p10}%">p10</span>
      <span class="ms-mc-end" style="left:${p90}%">p90</span>
    `;
  }

  const scenarioDisplay = { oracle: "Oracle", moderate: "Modéré", strong: "Fort" };
  const persist = document.getElementById("ms-persist-table");
  if (persist) {
    const rows = ms.persistence_comparison.rows || [];
    const hours = [...new Set(rows.map((r) => r.hours))];
    let html = `<table class="fh-table"><thead><tr><th>Scénario</th><th></th>`;
    hours.forEach((h) => {
      html += `<th>+${h} h</th>`;
    });
    html += `</tr></thead><tbody>`;
    ["moderate", "strong"].forEach((scen) => {
      html += `<tr><td rowspan="2">${scenarioDisplay[scen] || scen}</td><td>Physique</td>`;
      hours.forEach((h) => {
        const row = rows.find((r) => r.scenario === scen && r.hours === h);
        html += `<td>${row ? row.physical_display : "—"}</td>`;
      });
      html += `</tr><tr><td>Persistance</td>`;
      hours.forEach((h) => {
        const row = rows.find((r) => r.scenario === scen && r.hours === h);
        html += `<td>${row ? row.persistence_display : "—"}</td>`;
      });
      html += `</tr>`;
    });
    html += `</tbody></table>`;
    persist.innerHTML = html;
  }

  const meteoBar = document.getElementById("ms-scale-meteo-bar");
  const residBar = document.getElementById("ms-scale-resid-bar");
  if (meteoBar) meteoBar.style.width = `${ms.scale_comparison.meteo_bar_pct}%`;
  if (residBar) residBar.style.width = `${ms.scale_comparison.residual_bar_pct}%`;

  const hf = document.getElementById("ms-highflow-table");
  if (hf) {
    const rows = ms.high_flow.rows || [];
    const hours = [24, 48, 72];
    let html = `<table class="fh-table"><thead><tr><th></th><th></th>`;
    hours.forEach((h) => {
      html += `<th>+${h} h</th>`;
    });
    html += `</tr></thead><tbody>`;
    ["oracle", "strong"].forEach((scen) => {
      html += `<tr><td rowspan="2">${scenarioDisplay[scen] || scen}</td><td>Physique</td>`;
      hours.forEach((h) => {
        const row = rows.find((r) => r.scenario === scen && r.hours === h);
        html += `<td>${row ? row.physical_mae_display : "—"}</td>`;
      });
      html += `</tr><tr><td>Persistance</td>`;
      hours.forEach((h) => {
        const row = rows.find((r) => r.scenario === scen && r.hours === h);
        html += `<td>${row ? row.persistence_mae_display : "—"}</td>`;
      });
      html += `</tr>`;
    });
    html += `</tbody></table>`;
    hf.innerHTML = html;
  }
}

function renderForecastUncertainty(fu) {
  if (!fu) return;

  setText("fu-section-title", fu.section_title);
  setText("fu-section-subtitle", fu.section_subtitle);
  setText("fu-target", fu.target_display);
  setText("fu-target-inline", fu.target_display);
  setText("fu-calib-note", fu.calibration_note);
  setText("fu-central", fu.central_statement);
  setText("fu-reliability-def", fu.reliability_def);
  setText("fu-sharpness-def", fu.sharpness_def);
  setText("fu-pref-label", fu.preferred_label);
  setText("fu-hero-method", fu.hero.method);
  setText("fu-hero-nominal", fu.hero.nominal_display);
  setText("fu-hero-empirical", fu.hero.empirical_display);
  setText("fu-hero-width", fu.hero.mean_width_display);
  setText("fu-hero-error", fu.hero.coverage_error_pp_display);
  setText("fu-hero-caveat", fu.hero.caveat);
  setText("fu-ba-before", fu.before_after.behavioral_display);
  setText("fu-ba-after", fu.before_after.calibrated_display);
  setText("fu-ba-method", fu.before_after.calibrated_method);
  setText("fu-ba-caption", fu.before_after.caption);
  setText("fu-ba-note", fu.before_after.semantics_note);
  setText("fu-rel-explain", fu.reliability.explanation);
  setText("fu-sharp-note", fu.sharpness.note);
  setText("fu-regime-title", fu.regime_coverage.warning_title);
  setText("fu-regime-warning", fu.regime_coverage.warning_body);
  setText("fu-reg-overall", fu.regime_coverage.overall_display);
  setText("fu-reg-high", fu.regime_coverage.high_flow.display);
  setText("fu-reg-normal", fu.regime_coverage.normal_flow.display);
  setText("fu-reg-low", fu.regime_coverage.low_flow.display);
  setText("fu-reg-threshold", fu.regime_coverage.threshold_note);
  setText("fu-regime-why", fu.regime_coverage.why_it_matters);
  setText("fu-meteo-msg", fu.meteo_comparison.message);
  setText("fu-meteo-follow", fu.meteo_comparison.follow_up);
  setText("fu-sources-heading", fu.uncertainty_sources.heading);
  setText("fu-sources-future", fu.uncertainty_sources.future);
  setText("fu-arch-caption", fu.architecture_caption);
  setText("fu-result-heading", fu.result.heading);
  setText("fu-result-body", fu.result.body);
  setText("fu-decision-heading", fu.decision.heading);
  setText("fu-decision-body", fu.decision.body);
  setText("fu-limit-heading", fu.limit.heading);
  setText("fu-limit-body", fu.limit.body);
  setText("fu-takeaway", fu.final_takeaway);

  const bars = document.getElementById("fu-coverage-bars");
  if (bars) {
    bars.innerHTML = "";
    const methods = [
      "behavioral_parametric",
      "conditional_quantile",
      "empirical_residual",
      "split_conformal",
    ];
    const labels = {
      behavioral_parametric: "Paramétrique comportemental",
      conditional_quantile: "Quantile conditionnel",
      empirical_residual: "Résidu empirique",
      split_conformal: "Calibration conforme (split)",
    };
    methods.forEach((method) => {
      const block = document.createElement("div");
      block.className = "fu-bar-group";
      const title = document.createElement("div");
      title.className = "fu-bar-group-title";
      title.textContent = labels[method] || method;
      block.appendChild(title);
      [24, 48, 72].forEach((hours) => {
        const cell = fu.coverage_90[method][String(hours)];
        const row = document.createElement("div");
        row.className = "fu-bar-row";
        row.innerHTML = `
          <span class="fu-bar-h">+${hours} h</span>
          <div class="fu-bar-track">
            <div class="fu-bar-fill" style="width:${cell.bar_pct}%"></div>
            <div class="fu-bar-target" style="left:90%"></div>
          </div>
          <strong>${cell.empirical_pct_display}</strong>
        `;
        block.appendChild(row);
      });
      bars.appendChild(block);
    });
  }

  const cards = document.getElementById("fu-method-cards");
  if (cards) {
    cards.innerHTML = "";
    (fu.method_cards || []).forEach((card) => {
      const article = document.createElement("article");
      article.className = `fu-method-card is-${card.id}`;
      article.innerHTML = `
        <h4>${card.title}</h4>
        <p><strong>Objectif :</strong> ${card.purpose}</p>
        <p><strong>Couverture :</strong> ${card.coverage_display}</p>
        <p>${card.interpretation}</p>
      `;
      cards.appendChild(article);
    });
  }

  const relFig = document.getElementById("fu-rel-figure");
  const relWrap = document.getElementById("fu-rel-figure-wrap");
  if (relFig && relWrap && fu.reliability.figure) {
    relFig.src = fu.reliability.figure;
    relWrap.hidden = false;
  }

  const relPts = document.getElementById("fu-rel-points");
  if (relPts) {
    const pts = (fu.reliability.curves && fu.reliability.curves["24"]) || [];
    relPts.innerHTML = pts
      .map(
        (p) =>
          `<span>nominale ${p.nominal_display} → empirique ${p.empirical_display}</span>`
      )
      .join("");
  }

  const sharp = document.getElementById("fu-sharp-table");
  if (sharp) {
    let html =
      "<table class='fh-table'><thead><tr><th>Méthode</th><th>Couverture</th><th>Largeur moyenne</th></tr></thead><tbody>";
    (fu.sharpness.rows || []).forEach((r) => {
      html += `<tr><td>${r.label}</td><td>${r.coverage_display}</td><td>${r.mean_width_display}</td></tr>`;
    });
    html += "</tbody></table>";
    sharp.innerHTML = html;
  }

  const ext = document.getElementById("fu-extreme-table");
  if (ext) {
    let html =
      "<table class='fh-table'><thead><tr><th>Date</th><th>Qobs</th><th>Qponctuelle</th><th>L90</th><th>U90</th><th>Dans l'intervalle</th></tr></thead><tbody>";
    (fu.extreme_events || []).forEach((e) => {
      html += `<tr><td>${e.date}</td><td>${e.q_obs_display}</td><td>${e.q_point_display}</td><td>${e.lower_display}</td><td>${e.upper_display}</td><td>${e.inside_display}</td></tr>`;
    });
    html += "</tbody></table>";
    ext.innerHTML = html;
  }

  const meteo = document.getElementById("fu-meteo-table");
  if (meteo) {
    let html =
      "<table class='fh-table'><thead><tr><th></th><th>Oracle</th><th>Modéré</th></tr></thead><tbody>";
    (fu.meteo_comparison.rows || []).forEach((r) => {
      html += `<tr><td>${r.label}</td><td>${r.oracle_display}</td><td>${r.moderate_display}</td></tr>`;
    });
    html += "</tbody></table>";
    meteo.innerHTML = html;
  }

  const sources = document.getElementById("fu-sources-list");
  if (sources) {
    sources.innerHTML = "";
    (fu.uncertainty_sources.items || []).forEach((item) => {
      const div = document.createElement("div");
      div.className = "fu-source-item";
      div.innerHTML = `<strong>${item.title}</strong><p>${item.body}</p>`;
      sources.appendChild(div);
    });
  }
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

  setText(
    "bullet-samples",
    `${formatInt(kpis.n_samples)} jeux de paramètres explorés`
  );
  setText(
    "bullet-cal-period",
    `calage : ${yearRange(data.periods.calibration)}`
  );
  setText(
    "bullet-val-period",
    `validation : ${yearRange(data.periods.validation)}`
  );
  setText(
    "bullet-behavioral",
    `${formatInt(kpis.behavioral_members)} jeux de paramètres comportementaux`
  );
  setText(
    "bullet-coverage",
    `Couverture empirique en validation : ${(100 * Number(data.uncertainty.empirical_validation_coverage)).toFixed(1)}%`
  );
  setText("bullet-envelope", data.statements.envelope_disclaimer);
  setText("validation-isolation", data.statements.validation_isolation);
  setText("parametric-limitation", data.statements.parametric_only);
  if (data.statements.behavioral_envelope) {
    setText("behavioral-envelope", data.statements.behavioral_envelope);
  }

  const unavailable =
    (data.ui && data.ui.git_commit_unavailable) || "non disponible";
  setText("config-hash", data.reproducibility.config_sha256);
  setText("git-commit", data.reproducibility.git_commit || unavailable);
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

  renderMlCorrection(data.ml_correction);
  renderForecastHorizons(data.forecast_horizons);
  renderMeteoSensitivity(data.meteo_sensitivity);
  renderForecastUncertainty(data.forecast_uncertainty);
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
        `Impossible de charger assets/demo_data.json (${response.status}). Exécuter : python run.py --export-demo`
      );
    }
    const data = await response.json();
    render(data);
  } catch (err) {
    errorEl.hidden = false;
    errorEl.textContent =
      String(err.message || err) +
      " Si le fichier HTML a été ouvert directement, essayer : python -m http.server 8000";
  }
}

main();
