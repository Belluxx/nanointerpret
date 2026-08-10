const ui = {
  list: document.querySelector("#feature-list"),
  count: document.querySelector("#result-count"),
  search: document.querySelector("#search-input"),
  namedFilter: document.querySelector(".named-filter"),
  namedOnly: document.querySelector("#named-only-input"),
  sort: document.querySelector("#sort-select"),
  metadata: document.querySelector("#dataset-meta"),
  error: document.querySelector("#error-message"),
  sandboxForm: document.querySelector("#intervention-form"),
  prompt: document.querySelector("#prompt-input"),
  featureInput: document.querySelector("#feature-input"),
  featureOptions: document.querySelector("#feature-options"),
  interventionMode: document.querySelector("#intervention-mode"),
  amountLabel: document.querySelector("#amount-label"),
  amountInput: document.querySelector("#amount-input"),
  samplingSettings: document.querySelector("#sampling-settings-dialog"),
  samplingInputs: [...document.querySelectorAll(".sampling-setting input[type='number']")],
  generateButton: document.querySelector("#generate-button"),
  generationResults: document.querySelector("#generation-results"),
  baselineOutput: document.querySelector("#baseline-output"),
  intervenedOutput: document.querySelector("#intervened-output"),
};

let features = [];

function element(tag, className, text) {
  const result = document.createElement(tag);
  if (className) result.className = className;
  if (text !== undefined) result.textContent = text;
  return result;
}

function renderGeneration(output, prompt, continuation) {
  output.replaceChildren(
    element("span", "generation-prompt", prompt),
    element("span", "generation-continuation", continuation || "(No visible text)"),
  );
}

const compactNumber = new Intl.NumberFormat("en", {
  notation: "compact",
  maximumFractionDigits: 1,
});

function formatActivation(value) {
  if (value >= 100) return value.toFixed(0);
  if (value >= 10) return value.toFixed(1);
  if (value >= 1) return value.toFixed(2);
  return value.toPrecision(3);
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed (${response.status})`);
  }
  return payload;
}

function initializeSandbox(metadata) {
  ui.featureInput.dataset.maximum = String(metadata.d_sae - 1);

  for (const feature of features) {
    if (!feature.title) continue;
    const option = document.createElement("option");
    option.value = String(feature.id);
    option.label = `${feature.title} (#${feature.id})`;
    ui.featureOptions.append(option);
  }

  if (features.length) {
    ui.featureInput.value = String(features[0].id);
    updateAmountDefault();
  }
}

function selectedFeatureMaximum() {
  const value = ui.featureInput.value.trim();
  if (!value) return null;
  const featureId = Number(value);
  if (
    !Number.isInteger(featureId)
    || featureId < 0
    || featureId > Number(ui.featureInput.dataset.maximum)
  ) return null;

  return features.find((feature) => feature.id === featureId)?.max_activation ?? 0;
}

function updateAmountDefault() {
  if (ui.interventionMode.value === "additive") {
    ui.amountInput.value = "1";
    ui.amountInput.setAttribute("value", "1");
    return;
  }

  const maximum = selectedFeatureMaximum();
  if (maximum !== null) {
    const value = String(Number(maximum.toFixed(3)));
    ui.amountInput.value = value;
    ui.amountInput.setAttribute("value", value);
  }
}

function updateInterventionMode() {
  const clamping = ui.interventionMode.value === "clamp";
  ui.amountLabel.textContent = clamping ? "Target activation" : "Alpha";
  ui.amountInput.setAttribute(
    "aria-label",
    clamping ? "Target feature activation" : "Additive steering alpha",
  );
  updateAmountDefault();
}

function openSamplingSettings() {
  if (!ui.samplingSettings.matches(":popover-open")) {
    ui.samplingSettings.showPopover();
  }
}

function samplingSettingsAreValid() {
  for (const input of ui.samplingInputs) {
    if (input.checkValidity()) continue;
    openSamplingSettings();
    input.reportValidity();
    return false;
  }
  return true;
}

function synchronizeSetting(input) {
  const range = input.previousElementSibling;
  range.addEventListener("input", () => {
    input.value = range.value;
    updateRangeProgress(range);
  });
  input.addEventListener("input", () => {
    if (!input.checkValidity()) return;
    range.value = input.value;
    updateRangeProgress(range);
  });
  updateRangeProgress(range);
}

function updateRangeProgress(range) {
  const progress = 100 * (Number(range.value) - Number(range.min))
    / (Number(range.max) - Number(range.min));
  range.style.setProperty("--range-progress", `${progress}%`);
}

ui.interventionMode.addEventListener("change", updateInterventionMode);
ui.featureInput.addEventListener("input", () => {
  ui.featureInput.setCustomValidity("");
  if (ui.interventionMode.value === "clamp") updateAmountDefault();
});
ui.samplingInputs.forEach(synchronizeSetting);
ui.sandboxForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!samplingSettingsAreValid()) return;

  const featureId = Number(ui.featureInput.value);
  if (
    !Number.isInteger(featureId)
    || featureId < 0
    || featureId > Number(ui.featureInput.dataset.maximum)
  ) {
    ui.featureInput.setCustomValidity("Choose a feature from the suggestions or enter its numeric ID.");
    ui.featureInput.reportValidity();
    return;
  }

  const mode = ui.interventionMode.value;
  const amount = Number(ui.amountInput.value);
  const request = {
    prompt: ui.prompt.value,
    feature_id: featureId,
    mode,
    ...Object.fromEntries(ui.samplingInputs.map((input) => [input.name, Number(input.value)])),
    [mode === "clamp" ? "clamp_value" : "alpha"]: amount,
  };

  ui.generateButton.disabled = true;
  ui.generateButton.textContent = "Generating...";
  ui.sandboxForm.querySelector(".sandbox-error")?.remove();
  ui.generationResults.hidden = true;
  try {
    const payload = await fetchJson("/api/interventions/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    renderGeneration(ui.baselineOutput, request.prompt, payload.baseline);
    renderGeneration(ui.intervenedOutput, request.prompt, payload.intervened);
    ui.generationResults.hidden = false;
  } catch (error) {
    const message = element("span", "sandbox-error", `Could not generate: ${error.message}`);
    message.setAttribute("role", "alert");
    ui.generateButton.after(message);
  } finally {
    ui.generateButton.disabled = false;
    ui.generateButton.textContent = "Generate comparison";
  }
});
updateInterventionMode();

function renderMetadata(metadata) {
  const items = [
    ["Model", metadata.model_id || "-"],
    ["Tokens", metadata.processed_tokens ? compactNumber.format(metadata.processed_tokens) : "-"],
    ["Layer", metadata.layer_index ?? "-"],
    ["SAE width", metadata.d_sae ? compactNumber.format(metadata.d_sae) : "-"],
  ];

  ui.metadata.replaceChildren();
  for (const [label, value] of items) {
    const item = element("div");
    item.append(element("dt", "", label), element("dd", "", String(value)));
    ui.metadata.append(item);
  }
}

const sorters = {
  id: (a, b) => a.id - b.id,
  max: (a, b) => b.max_activation - a.max_activation || a.id - b.id,
  count: (a, b) => b.activation_count - a.activation_count || a.id - b.id,
  title: (a, b) => (a.title || `Feature ${a.id}`).localeCompare(b.title || `Feature ${b.id}`),
};

function visibleFeatures() {
  const query = ui.search.value.trim().toLocaleLowerCase();
  const visible = features.filter((feature) => {
    if (ui.namedOnly.checked && !feature.title) return false;
    return !query
      || String(feature.id).includes(query)
      || (feature.title || "").toLocaleLowerCase().includes(query);
  });
  return visible.sort(sorters[ui.sort.value]);
}

function renderFeatureList() {
  const visible = visibleFeatures();
  ui.count.textContent = visible.length === features.length
    ? `${visible.length.toLocaleString()} active features`
    : `${visible.length.toLocaleString()} of ${features.length.toLocaleString()} features`;

  const fragment = document.createDocumentFragment();
  for (const feature of visible) {
    const details = element("details", "feature");
    const summary = element("summary");
    const title = feature.title || `Feature ${feature.id}`;
    const titleElement = element("span", "feature-title", title);
    titleElement.title = title;
    summary.append(
      element("span", "feature-id", `#${feature.id}`),
      titleElement,
      element("span", "feature-stat", feature.activation_count.toLocaleString()),
      element("span", "feature-stat", formatActivation(feature.max_activation)),
      element("span", "chevron", "›"),
    );
    details.append(summary, element("div", "feature-body"));
    details.addEventListener("toggle", () => {
      if (details.open && !details.dataset.loaded) loadFeature(details, feature);
    });
    fragment.append(details);
  }
  ui.list.replaceChildren(fragment);
}

function renderContext(context, feature) {
  const card = element("article", "context-card");
  const header = element("header", "context-header");
  header.append(
    element("span", "", `Context ${context.context_id}`),
    element(
      "span",
      "context-stats",
      `Peak ${formatActivation(context.peak_activation)}, ${context.activation_count.toLocaleString()} active tokens`,
    ),
  );

  const tokens = element("pre", "tokens");
  context.tokens.forEach((text, index) => {
    const token = element("span", "token", text);
    const activation = context.activations[index];
    if (activation > 0) {
      const strength = Math.sqrt(Math.min(1, activation / feature.max_activation));
      token.classList.add("active");
      token.style.backgroundColor = `rgba(196, 79, 27, ${0.1 + 0.75 * strength})`;
      token.title = `Activation ${formatActivation(activation)}`;
    }
    tokens.append(token);
  });

  card.append(header, tokens);
  return card;
}

function renderOverview(feature, payload) {
  const overview = element("section", "feature-overview");
  const tokenSummary = element("div", "token-summary");
  tokenSummary.append(element("h3", "", "Characteristic tokens"));
  for (const group of payload.token_groups) {
    const row = element("div", "token-group");
    const tokenList = element("div", "token-list");
    for (const token of group.tokens) {
      tokenList.append(element("code", "token-name", token));
    }
    row.append(element("span", "token-percentile", `P${group.percentile}`), tokenList);
    tokenSummary.append(row);
  }

  const facts = element("dl", "feature-facts");
  for (const [label, value] of [
    ["Activating tokens", payload.activation_count.toLocaleString()],
    ["Contexts", payload.context_count.toLocaleString()],
    ["Peak activation", formatActivation(feature.max_activation)],
  ]) {
    const fact = element("div", "feature-fact");
    fact.append(
      element("dt", "", label),
      element("dd", "", value),
    );
    facts.append(fact);
  }

  overview.append(tokenSummary, facts);
  return overview;
}

function renderContextList(contexts, feature, list = element("div", "contexts")) {
  list.replaceChildren();
  for (const context of contexts) list.append(renderContext(context, feature));
  return list;
}

const RANGE_RESOLUTION = 1000;

function renderRangeView(content, feature, counts) {
  const panel = element("section");
  const plot = element("div", "distribution-plot");
  plot.setAttribute("aria-hidden", "true");
  plot.style.gridTemplateColumns = `repeat(${counts.length}, minmax(2px, 1fr))`;
  const largestBin = Math.max(...counts, 1);
  const bars = counts.map((count, index) => {
    const bar = element("span", "distribution-bar");
    bar.style.height = `${Math.max(2, 100 * Math.sqrt(count / largestBin))}%`;
    const lower = feature.max_activation * index / counts.length;
    const upper = feature.max_activation * (index + 1) / counts.length;
    bar.title = `${formatActivation(lower)}-${formatActivation(upper)}: ${count.toLocaleString()} contexts`;
    plot.append(bar);
    return bar;
  });

  const selector = element("div", "range-selector");
  const track = element("div", "range-track");
  function rangeInput(label, value) {
    const input = document.createElement("input");
    input.type = "range";
    input.min = "0";
    input.max = String(RANGE_RESOLUTION);
    input.step = "1";
    input.value = String(value);
    input.setAttribute("aria-label", label);
    return input;
  }
  const minimumInput = rangeInput("Minimum peak activation", 0);
  const maximumInput = rangeInput("Maximum peak activation", RANGE_RESOLUTION);
  const minimumLabel = element("span", "range-value");
  const maximumLabel = element("span", "range-value");
  selector.append(
    track,
    minimumInput,
    maximumInput,
    minimumLabel,
    maximumLabel,
  );

  const resultCount = element("span", "range-result-count");
  const results = element("div", "contexts");
  panel.append(
    element("h3", "range-heading", "Context peak distribution"),
    plot,
    selector,
    resultCount,
    results,
  );
  content.replaceChildren(panel);

  let loadTimer;
  let requestId = 0;

  function updateSelection(changedInput) {
    if (Number(minimumInput.value) > Number(maximumInput.value)) {
      if (changedInput === minimumInput) maximumInput.value = minimumInput.value;
      else minimumInput.value = maximumInput.value;
    }

    const minimumStep = Number(minimumInput.value);
    const maximumStep = Number(maximumInput.value);
    const start = `${100 * minimumStep / RANGE_RESOLUTION}%`;
    const end = `${100 * maximumStep / RANGE_RESOLUTION}%`;
    selector.style.setProperty("--range-start", start);
    selector.style.setProperty("--range-end", end);

    const minimum = feature.max_activation * minimumStep / RANGE_RESOLUTION;
    const maximum = feature.max_activation * maximumStep / RANGE_RESOLUTION;
    for (const [label, step, value] of [
      [minimumLabel, minimumStep, minimum],
      [maximumLabel, maximumStep, maximum],
    ]) {
      const fraction = step / RANGE_RESOLUTION;
      label.style.left = `calc(${100 * fraction}% + ${7 - 14 * fraction}px)`;
      label.textContent = formatActivation(value);
    }
    bars.forEach((bar, index) => {
      const binStart = index * RANGE_RESOLUTION / bars.length;
      const binEnd = (index + 1) * RANGE_RESOLUTION / bars.length;
      bar.classList.toggle("selected", binEnd >= minimumStep && binStart <= maximumStep);
    });

    clearTimeout(loadTimer);
    const currentRequest = ++requestId;
    loadTimer = setTimeout(
      () => loadContexts(minimum, maximum, currentRequest),
      120,
    );
  }

  async function loadContexts(minimum, maximum, currentRequest) {
    resultCount.textContent = "Loading...";
    if (!results.childElementCount) {
      results.append(element("p", "loading", "Loading contexts..."));
    }
    const query = new URLSearchParams({ min: minimum, max: maximum });
    try {
      const payload = await fetchJson(`/api/features/${feature.id}?${query}`);
      if (currentRequest !== requestId) return;
      const shown = payload.contexts.length;
      resultCount.textContent = shown === payload.matching_context_count
        ? `${shown.toLocaleString()} contexts (low to high)`
        : `Sampled ${shown.toLocaleString()} of ${payload.matching_context_count.toLocaleString()} (low to high)`;
      if (shown) renderContextList(payload.contexts, feature, results);
      else results.replaceChildren(
        element("p", "empty-state", "No contexts fall within this activation range."),
      );
    } catch (error) {
      if (currentRequest !== requestId) return;
      resultCount.textContent = "Could not load contexts";
      results.replaceChildren(
        element("p", "empty-state", `Could not load contexts: ${error.message}`),
      );
    }
  }

  minimumInput.addEventListener("input", () => updateSelection(minimumInput));
  maximumInput.addEventListener("input", () => updateSelection(maximumInput));
  updateSelection();
}

function renderContextBrowser(feature, payload) {
  const browser = element("section");
  const toolbar = element("header", "context-toolbar");
  const title = element("h3");
  const switcher = element("div", "view-switch");
  const content = element("div");
  switcher.setAttribute("role", "group");
  switcher.setAttribute("aria-label", "Context view");
  for (const [view, label] of [["strongest", "Strongest"], ["range", "Range"]]) {
    const button = element("button", "view-button", label);
    button.type = "button";
    button.addEventListener("click", () => {
      title.textContent = `${label} contexts`;
      for (const viewButton of switcher.children) {
        const selected = viewButton === button;
        viewButton.classList.toggle("selected", selected);
        viewButton.setAttribute("aria-pressed", String(selected));
      }
      if (view === "strongest") {
        content.replaceChildren(renderContextList(payload.strongest_contexts, feature));
      } else {
        renderRangeView(content, feature, payload.activation_histogram);
      }
    });
    switcher.append(button);
  }

  const controls = element("div", "context-controls");
  const legend = element("span", "legend", `0 - ${formatActivation(feature.max_activation)}`);
  legend.prepend(element("span", "legend-ramp"));
  controls.append(switcher, legend);
  toolbar.append(title, controls);
  browser.append(toolbar, content);
  switcher.firstElementChild.click();
  return browser;
}

async function loadFeature(details, feature) {
  const body = details.querySelector(".feature-body");
  body.replaceChildren(element("p", "loading", "Loading feature..."));
  try {
    const payload = await fetchJson(`/api/features/${feature.id}`);
    body.replaceChildren(renderOverview(feature, payload), renderContextBrowser(feature, payload));
    details.dataset.loaded = "true";
  } catch (error) {
    body.replaceChildren(element("p", "empty-state", `Could not load feature: ${error.message}`));
  }
}

let renderFrame;
ui.search.addEventListener("input", () => {
  cancelAnimationFrame(renderFrame);
  renderFrame = requestAnimationFrame(renderFeatureList);
});
ui.namedOnly.addEventListener("change", renderFeatureList);
ui.sort.addEventListener("change", renderFeatureList);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && ui.search.value) {
    ui.search.value = "";
    renderFeatureList();
    ui.search.focus();
  }
});

async function initialize() {
  try {
    const payload = await fetchJson("/api/summary");
    features = payload.features;
    const hasFeatureNames = features.some((feature) => feature.title);
    ui.namedFilter.hidden = !hasFeatureNames;
    ui.namedOnly.checked = hasFeatureNames;
    renderMetadata(payload.metadata);
    initializeSandbox(payload.metadata);
    renderFeatureList();
  } catch (error) {
    ui.count.textContent = "Could not load analysis";
    ui.error.textContent = error.message;
    ui.error.hidden = false;
  }
}

initialize();
