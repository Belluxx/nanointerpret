const PAGE_SIZE = 20;

const state = {
  features: [],
  metadata: {},
};

const featureList = document.querySelector("#feature-list");
const resultCount = document.querySelector("#result-count");
const searchInput = document.querySelector("#search-input");
const sortSelect = document.querySelector("#sort-select");
const datasetMeta = document.querySelector("#dataset-meta");
const errorMessage = document.querySelector("#error-message");

function el(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function formatNumber(value) {
  return new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

function formatActivation(value) {
  if (value >= 100) return value.toFixed(0);
  if (value >= 10) return value.toFixed(1);
  if (value >= 1) return value.toFixed(2);
  return value.toPrecision(3);
}

function formatPercent(numerator, denominator) {
  if (!denominator) return "—";
  const percent = (100 * numerator) / denominator;
  if (percent > 0 && percent < 0.01) return "<0.01%";
  if (percent >= 10) return `${percent.toFixed(1)}%`;
  return `${percent.toFixed(2)}%`;
}

function formatToken(token) {
  return token;
}

async function getJson(url) {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

function renderMetadata() {
  const metadata = state.metadata;
  const items = [
    ["Model", metadata.model_id || "—"],
    ["Tokens", metadata.processed_tokens ? formatNumber(metadata.processed_tokens) : "—"],
    ["Layer", metadata.layer_index ?? "—"],
    ["SAE width", metadata.d_sae ? formatNumber(metadata.d_sae) : "—"],
  ];

  datasetMeta.replaceChildren();
  for (const [label, value] of items) {
    const item = el("div");
    item.append(el("dt", "", label), el("dd", "", String(value)));
    datasetMeta.append(item);
  }
}

function filteredFeatures() {
  const query = searchInput.value.trim().toLocaleLowerCase();
  const features = query
    ? state.features.filter((feature) => {
        const title = feature.title || "";
        return String(feature.id).includes(query) || title.toLocaleLowerCase().includes(query);
      })
    : [...state.features];

  switch (sortSelect.value) {
    case "max":
      features.sort((a, b) => b.max_activation - a.max_activation || a.id - b.id);
      break;
    case "count":
      features.sort((a, b) => b.activation_count - a.activation_count || a.id - b.id);
      break;
    case "title":
      features.sort((a, b) => (a.title || `Feature ${a.id}`).localeCompare(b.title || `Feature ${b.id}`));
      break;
    default:
      features.sort((a, b) => a.id - b.id);
  }
  return features;
}

function renderFeatures() {
  const features = filteredFeatures();
  resultCount.textContent = features.length === state.features.length
    ? `${features.length.toLocaleString()} active features`
    : `${features.length.toLocaleString()} of ${state.features.length.toLocaleString()} features`;

  const fragment = document.createDocumentFragment();
  for (const feature of features) {
    const details = el("details", "feature");
    details.dataset.featureId = feature.id;

    const summary = el("summary");
    const title = el("span", "feature-title", feature.title || `Feature ${feature.id}`);
    title.title = title.textContent;
    summary.append(
      el("span", "feature-id", `#${feature.id}`),
      title,
      el("span", "feature-stat", feature.activation_count.toLocaleString()),
      el("span", "feature-stat", formatActivation(feature.max_activation)),
      el("span", "chevron"),
    );

    const body = el("div", "feature-body");
    details.append(summary, body);
    details.addEventListener("toggle", () => {
      if (details.open && !details.dataset.loaded) loadFeature(details, feature);
    });
    fragment.append(details);
  }

  featureList.replaceChildren(fragment);
}

function renderContext(context, feature) {
  const card = el("article", "context-card");
  const header = el("header", "context-header");
  const label = el("span", "context-label", `Context ${context.context_id}`);
  const stats = el("span", "context-stats");
  if (context.sample) {
    label.append(el("span", "context-bucket", context.sample.bucket));
    const target = el("span", "context-stat");
    const percentile = el("span", "context-stat");
    target.append("Selected ", el("strong", "", formatActivation(context.sample.activation)));
    percentile.append("Percentile ", el("strong", "", context.sample.percentile.toFixed(1)));
    stats.append(target, percentile);
  } else {
    const peak = el("span", "context-stat");
    const hits = el("span", "context-stat");
    peak.append("Peak ", el("strong", "", formatActivation(context.peak_activation)));
    hits.append("Active tokens ", el("strong", "", context.activation_count.toLocaleString()));
    stats.append(peak, hits);
  }
  header.append(label, stats);

  const tokens = el("pre", "tokens");
  for (let index = 0; index < context.tokens.length; index += 1) {
    const token = el("span", "token", context.tokens[index]);
    const activation = context.activations[index];
    if (activation > 0) {
      const ratio = Math.min(1, activation / feature.max_activation);
      const alpha = 0.12 + 0.76 * Math.sqrt(ratio);
      token.classList.add("active");
      token.style.backgroundColor = `rgba(233, 84, 56, ${alpha})`;
      token.title = `Activation ${formatActivation(activation)}`;
    }
    if (context.sample && index === context.sample.target_position) {
      token.classList.add("sample-target");
      token.title = `${context.sample.bucket} sample · activation ${formatActivation(context.sample.activation)} · percentile ${context.sample.percentile.toFixed(1)}`;
    }
    tokens.append(token);
  }

  card.append(header, tokens);
  return card;
}

function renderFeatureIntroduction(feature, payload) {
  const section = el("section", "feature-introduction");
  const heading = el("div", "feature-introduction-heading");
  const headingText = el("div");
  headingText.append(
    el("p", "feature-introduction-eyebrow", `Feature #${feature.id}`),
    el("h2", "feature-introduction-title", feature.title || `Feature ${feature.id}`),
  );
  const facts = el("ul", "feature-facts");
  const factItems = [
    [
      payload.activation_count.toLocaleString(),
      "activating tokens",
      `(${formatPercent(payload.activation_count, state.metadata.processed_tokens)})`,
    ],
    [
      payload.context_count.toLocaleString(),
      "contexts",
      `(${formatPercent(payload.context_count, state.metadata.context_count)})`,
    ],
    [formatActivation(feature.max_activation), "peak activation", ""],
  ];
  for (const [value, label, note] of factItems) {
    const item = el("li");
    item.append(el("strong", "", value), ` ${label}`);
    if (note) item.append(el("small", "", ` ${note}`));
    facts.append(item);
  }
  heading.append(headingText, facts);

  const tokensSection = el("section", "characteristic-tokens");
  const tokensHeading = el("div", "characteristic-tokens-heading");
  tokensHeading.append(
    el("h3", "", "Activating tokens"),
    el("p", "", "Hover for details"),
  );
  const tokenGroups = el("div", "activation-token-groups");
  const levelNames = { high: "High", med: "Med", low: "Low" };
  for (const group of payload.activation_token_groups) {
    const groupSection = el("section", "activation-token-group");
    const groupHeading = el("div", "activation-token-group-heading");
    groupHeading.append(
      el("h4", "", `${levelNames[group.level]} activation tokens`),
      el("span", "", `P${group.percentile}`),
    );
    const tokenList = el("ul", "characteristic-token-list");
    for (const token of group.tokens) {
      const item = el("li", "characteristic-token");
      const tokenName = el("code", "characteristic-token-name", formatToken(token.token));
      const hits = `${token.activation_count.toLocaleString()} ${token.activation_count === 1 ? "hit" : "hits"}`;
      item.title = `${hits} · mean ${formatActivation(token.mean_activation)} · peak ${formatActivation(token.max_activation)}`;
      item.append(tokenName);
      tokenList.append(item);
    }
    groupSection.append(groupHeading, tokenList);
    tokenGroups.append(groupSection);
  }
  tokensSection.append(tokensHeading, tokenGroups);
  section.append(heading, tokensSection);
  return section;
}

function contextUrl(feature, view, offset = 0) {
  return `/api/features/${feature.id}/contexts?view=${view}&offset=${offset}&limit=${PAGE_SIZE}`;
}

function setContextView(details, view, loading = false) {
  details.dataset.contextView = view;
  const titles = {
    strongest: "Strongest activation contexts",
    stratified: "Stratified activation contexts",
  };
  details.querySelector(".context-summary-title").textContent = titles[view];
  for (const button of details.querySelectorAll(".context-view-button")) {
    const selected = button.dataset.view === view;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-pressed", String(selected));
    button.disabled = loading;
  }
}

function appendLoadMore(details, feature, payload) {
  const content = details.querySelector(".context-view");
  const nextOffset = payload.offset + payload.contexts.length;
  if (nextOffset >= payload.context_count) return;

  const loadMore = el(
    "button",
    "load-more",
    `Load ${Math.min(PAGE_SIZE, payload.context_count - nextOffset)} more contexts`,
  );
  loadMore.type = "button";
  loadMore.addEventListener("click", async () => {
    loadMore.disabled = true;
    loadMore.textContent = "Loading…";
    try {
      const nextPayload = await getJson(contextUrl(feature, "strongest", nextOffset));
      if (details.dataset.contextView !== "strongest") return;
      loadMore.remove();
      const contexts = content.querySelector(".contexts");
      for (const context of nextPayload.contexts) {
        contexts.append(renderContext(context, feature));
      }
      appendLoadMore(details, feature, nextPayload);
    } catch (error) {
      loadMore.disabled = false;
      loadMore.textContent = "Try loading again";
      showError(error.message);
    }
  });
  content.append(loadMore);
}

function renderStrongestView(details, feature, payload) {
  const content = details.querySelector(".context-view");
  const contexts = el("div", "contexts");
  for (const context of payload.contexts) {
    contexts.append(renderContext(context, feature));
  }
  content.replaceChildren(contexts);
  appendLoadMore(details, feature, payload);
}

function renderStratifiedView(details, feature, payload) {
  const content = details.querySelector(".context-view");
  if (!payload.stratified_available) {
    content.replaceChildren(
      el(
        "div",
        "empty-state",
        "This feature does not have enough activation data for the complete interpretation evidence set.",
      ),
    );
    return;
  }

  const groups = [
    {
      category: "top",
      title: "Top activations",
      description: "The 10 strongest individual token activations",
    },
    {
      category: "stratified",
      title: "Stratified activations",
      description: "Two samples from each 25–50, 50–75, 75–90, and 90–99 percentile band",
    },
    {
      category: "random",
      title: "Random activating examples",
      description: "Five random positive activations",
    },
  ];
  const fragment = document.createDocumentFragment();
  for (const group of groups) {
    const section = el("section", "context-sample-group");
    const heading = el("header", "context-sample-heading");
    heading.append(el("h3", "", group.title), el("p", "", group.description));
    const contexts = el("div", "contexts");
    for (const context of payload.contexts) {
      if (context.sample.category === group.category) {
        contexts.append(renderContext(context, feature));
      }
    }
    section.append(heading, contexts);
    fragment.append(section);
  }
  content.replaceChildren(fragment);
}

async function loadContextView(details, feature, view) {
  if (details.dataset.contextView === view) return;
  const content = details.querySelector(".context-view");
  setContextView(details, view, true);
  content.replaceChildren(el("div", "loading", "Loading activation contexts…"));
  try {
    const payload = await getJson(contextUrl(feature, view));
    if (view === "stratified") {
      renderStratifiedView(details, feature, payload);
    } else {
      renderStrongestView(details, feature, payload);
    }
  } catch (error) {
    content.replaceChildren(el("div", "empty-state", `Could not load contexts: ${error.message}`));
  } finally {
    setContextView(details, view);
  }
}

function renderContextBrowser(details, feature) {
  const section = el("section", "context-browser");
  const summary = el("div", "context-summary");
  const actions = el("div", "context-summary-actions");
  const switcher = el("div", "context-view-switch");
  switcher.setAttribute("role", "group");
  switcher.setAttribute("aria-label", "Activation context view");
  for (const [view, label] of [["strongest", "Strongest"], ["stratified", "Stratified"]]) {
    const button = el("button", "context-view-button", label);
    button.type = "button";
    button.dataset.view = view;
    button.addEventListener("click", () => loadContextView(details, feature, view));
    switcher.append(button);
  }
  const legend = el("span", "legend");
  legend.append(
    "Activation",
    el("span", "legend-ramp"),
    `0 — ${formatActivation(feature.max_activation)}`,
  );
  actions.append(switcher, legend);
  summary.append(el("span", "context-summary-title"), actions);
  section.append(summary, el("div", "context-view"));
  return section;
}

async function loadFeature(details, feature) {
  const body = details.querySelector(".feature-body");
  body.replaceChildren(el("div", "loading", "Loading activation contexts…"));
  try {
    const payload = await getJson(contextUrl(feature, "strongest"));
    body.replaceChildren(
      renderFeatureIntroduction(feature, payload),
      renderContextBrowser(details, feature),
    );
    setContextView(details, "strongest");
    renderStrongestView(details, feature, payload);
    details.dataset.loaded = "true";
  } catch (error) {
    body.replaceChildren(el("div", "empty-state", `Could not load contexts: ${error.message}`));
  }
}

let renderFrame;
function scheduleRender() {
  cancelAnimationFrame(renderFrame);
  renderFrame = requestAnimationFrame(renderFeatures);
}

searchInput.addEventListener("input", scheduleRender);
sortSelect.addEventListener("change", renderFeatures);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && searchInput.value) {
    searchInput.value = "";
    renderFeatures();
    searchInput.focus();
  }
});

async function initialize() {
  try {
    const payload = await getJson("/api/summary");
    state.features = payload.features;
    state.metadata = payload.metadata;
    renderMetadata();
    renderFeatures();
  } catch (error) {
    resultCount.textContent = "Could not load analysis";
    showError(error.message);
  }
}

initialize();
