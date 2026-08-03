const PAGE_SIZE = 20;

const ui = {
  list: document.querySelector("#feature-list"),
  count: document.querySelector("#result-count"),
  search: document.querySelector("#search-input"),
  sort: document.querySelector("#sort-select"),
  metadata: document.querySelector("#dataset-meta"),
  error: document.querySelector("#error-message"),
};

let features = [];
let metadata = {};

function element(tag, className, text) {
  const result = document.createElement(tag);
  if (className) result.className = className;
  if (text !== undefined) result.textContent = text;
  return result;
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

async function fetchJson(url) {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed (${response.status})`);
  }
  return payload;
}

function showError(message) {
  ui.error.textContent = message;
  ui.error.hidden = false;
}

function renderMetadata() {
  const items = [
    ["Model", metadata.model_id || "—"],
    ["Tokens", metadata.processed_tokens ? compactNumber.format(metadata.processed_tokens) : "—"],
    ["Layer", metadata.layer_index ?? "—"],
    ["SAE width", metadata.d_sae ? compactNumber.format(metadata.d_sae) : "—"],
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
  const visible = query
    ? features.filter((feature) =>
        String(feature.id).includes(query)
        || (feature.title || "").toLocaleLowerCase().includes(query))
    : [...features];
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
  const stats = context.sample
    ? `${context.sample.bucket} · ${formatActivation(context.sample.activation)} · P${context.sample.percentile.toFixed(1)}`
    : `Peak ${formatActivation(context.peak_activation)} · ${context.activation_count.toLocaleString()} active tokens`;
  header.append(
    element("span", "", `Context ${context.context_id}`),
    element("span", "context-stats", stats),
  );

  const tokens = element("pre", "tokens");
  context.tokens.forEach((text, index) => {
    const token = element("span", "token", text);
    const activation = context.activations[index];
    if (activation > 0) {
      const strength = Math.sqrt(Math.min(1, activation / feature.max_activation));
      token.classList.add("active");
      token.style.backgroundColor = `rgba(222, 75, 47, ${0.12 + 0.72 * strength})`;
      token.title = `Activation ${formatActivation(activation)}`;
    }
    if (context.sample && index === context.sample.target_position) {
      token.classList.add("sample-target");
      token.title = `${context.sample.bucket} · activation ${formatActivation(context.sample.activation)} · P${context.sample.percentile.toFixed(1)}`;
    }
    tokens.append(token);
  });

  card.append(header, tokens);
  return card;
}

function renderOverview(feature, payload) {
  const overview = element("section", "feature-overview");
  const facts = element("ul", "feature-facts");
  for (const fact of [
    `${payload.activation_count.toLocaleString()} activating tokens`,
    `${payload.context_count.toLocaleString()} contexts`,
    `${formatActivation(feature.max_activation)} peak activation`,
  ]) {
    facts.append(element("li", "", fact));
  }

  const tokenSummary = element("div", "token-summary");
  tokenSummary.append(element("h3", "", "Characteristic tokens"));
  for (const group of payload.token_groups) {
    const row = element("div", "token-group");
    row.append(element("span", "token-percentile", `P${group.percentile}`));
    const tokenList = element("div", "token-list");
    for (const token of group.tokens) {
      const tokenName = element("code", "token-name", token.token);
      tokenName.title = `${token.activation_count.toLocaleString()} hits · mean ${formatActivation(token.mean_activation)} · peak ${formatActivation(token.max_activation)}`;
      tokenList.append(tokenName);
    }
    row.append(tokenList);
    tokenSummary.append(row);
  }

  overview.append(facts, tokenSummary);
  return overview;
}

function contextUrl(feature, view, offset = 0) {
  return `/api/features/${feature.id}/contexts?view=${view}&offset=${offset}&limit=${PAGE_SIZE}`;
}

function updateViewControls(details, loading = false) {
  const view = details.dataset.view;
  details.querySelector(".context-title").textContent = view === "stratified"
    ? "Stratified contexts"
    : "Strongest contexts";
  for (const button of details.querySelectorAll(".view-button")) {
    const selected = button.dataset.view === view;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-pressed", String(selected));
    button.disabled = loading;
  }
}

function renderContextList(contexts, feature) {
  const list = element("div", "contexts");
  for (const context of contexts) list.append(renderContext(context, feature));
  return list;
}

const sampleGroups = [
  ["Top activations", "10 strongest token activations", ["Top"]],
  ["Activation range", "Samples across the 25th–99th percentiles", ["25-50", "50-75", "75-90", "90-99"]],
  ["Random positives", "5 random activating examples", ["Random positive"]],
];

function renderView(details, feature, payload, view) {
  const content = details.querySelector(".context-content");
  if (view === "strongest") {
    content.replaceChildren(renderContextList(payload.contexts, feature));
    appendLoadMore(details, feature, payload);
    return;
  }

  if (!payload.contexts.length) {
    content.replaceChildren(element("p", "empty-state", "Not enough activation data for a stratified sample."));
    return;
  }

  const fragment = document.createDocumentFragment();
  for (const [title, description, buckets] of sampleGroups) {
    const contexts = payload.contexts.filter((context) => buckets.includes(context.sample.bucket));
    const section = element("section", "sample-group");
    const heading = element("header", "sample-heading");
    heading.append(element("h3", "", title), element("p", "", description));
    section.append(heading, renderContextList(contexts, feature));
    fragment.append(section);
  }
  content.replaceChildren(fragment);
}

function appendLoadMore(details, feature, payload) {
  const nextOffset = payload.offset + payload.contexts.length;
  if (nextOffset >= payload.context_count) return;

  const button = element(
    "button",
    "load-more",
    `Load ${Math.min(PAGE_SIZE, payload.context_count - nextOffset)} more`,
  );
  button.type = "button";
  button.addEventListener("click", async () => {
    button.disabled = true;
    button.textContent = "Loading…";
    try {
      const next = await fetchJson(contextUrl(feature, "strongest", nextOffset));
      if (details.dataset.view !== "strongest") return;
      button.remove();
      const list = details.querySelector(".context-content > .contexts");
      for (const context of next.contexts) list.append(renderContext(context, feature));
      appendLoadMore(details, feature, next);
    } catch (error) {
      button.disabled = false;
      button.textContent = "Try again";
      button.title = error.message;
    }
  });
  details.querySelector(".context-content").append(button);
}

async function loadView(details, feature, view) {
  if (details.dataset.view === view) return;
  details.dataset.view = view;
  updateViewControls(details, true);
  const content = details.querySelector(".context-content");
  content.replaceChildren(element("p", "loading", "Loading contexts…"));

  try {
    const payload = await fetchJson(contextUrl(feature, view));
    if (details.dataset.view === view) renderView(details, feature, payload, view);
  } catch (error) {
    if (details.dataset.view === view) {
      content.replaceChildren(element("p", "empty-state", `Could not load contexts: ${error.message}`));
    }
  } finally {
    updateViewControls(details);
  }
}

function renderContextBrowser(details, feature) {
  const browser = element("section", "context-browser");
  const toolbar = element("header", "context-toolbar");
  const switcher = element("div", "view-switch");
  switcher.setAttribute("role", "group");
  switcher.setAttribute("aria-label", "Context view");
  for (const [view, label] of [["strongest", "Strongest"], ["stratified", "Stratified"]]) {
    const button = element("button", "view-button", label);
    button.type = "button";
    button.dataset.view = view;
    button.addEventListener("click", () => loadView(details, feature, view));
    switcher.append(button);
  }

  const controls = element("div", "context-controls");
  const legend = element("span", "legend", `0 — ${formatActivation(feature.max_activation)}`);
  legend.prepend(element("span", "legend-ramp"));
  controls.append(switcher, legend);
  toolbar.append(element("h3", "context-title"), controls);
  browser.append(toolbar, element("div", "context-content"));
  return browser;
}

async function loadFeature(details, feature) {
  const body = details.querySelector(".feature-body");
  body.replaceChildren(element("p", "loading", "Loading feature…"));
  try {
    const payload = await fetchJson(contextUrl(feature, "strongest"));
    body.replaceChildren(renderOverview(feature, payload), renderContextBrowser(details, feature));
    details.dataset.view = "strongest";
    updateViewControls(details);
    renderView(details, feature, payload, "strongest");
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
    metadata = payload.metadata;
    renderMetadata();
    renderFeatureList();
  } catch (error) {
    ui.count.textContent = "Could not load analysis";
    showError(error.message);
  }
}

initialize();
