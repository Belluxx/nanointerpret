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
  const stats = el("span", "context-stats");
  const peak = el("span", "context-stat");
  const hits = el("span", "context-stat");
  peak.append("Peak ", el("strong", "", formatActivation(context.peak_activation)));
  hits.append("Active tokens ", el("strong", "", context.activation_count.toLocaleString()));
  stats.append(peak, hits);
  header.append(el("span", "", `Context ${context.context_id}`), stats);

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

async function loadContextPage(details, feature, offset, button) {
  if (button) {
    button.disabled = true;
    button.textContent = "Loading…";
  }

  const payload = await getJson(
    `/api/features/${feature.id}/contexts?offset=${offset}&limit=${PAGE_SIZE}`,
  );
  const body = details.querySelector(".feature-body");
  let contexts = body.querySelector(".contexts");

  if (offset === 0) {
    body.replaceChildren();
    const summary = el("div", "context-summary");
    summary.append(
      el("span", "context-summary-title", "Strongest activation contexts"),
    );
    const legend = el("span", "legend");
    legend.append(
      "Activation",
      el("span", "legend-ramp"),
      `0 — ${formatActivation(feature.max_activation)}`,
    );
    summary.append(legend);
    contexts = el("div", "contexts");
    body.append(renderFeatureIntroduction(feature, payload), summary, contexts);
  }

  for (const context of payload.contexts) {
    contexts.append(renderContext(context, feature));
  }

  if (button) button.remove();
  const nextOffset = offset + payload.contexts.length;
  if (nextOffset < payload.context_count) {
    const loadMore = el(
      "button",
      "load-more",
      `Load ${Math.min(PAGE_SIZE, payload.context_count - nextOffset)} more contexts`,
    );
    loadMore.type = "button";
    loadMore.addEventListener("click", async () => {
      try {
        await loadContextPage(details, feature, nextOffset, loadMore);
      } catch (error) {
        loadMore.disabled = false;
        loadMore.textContent = "Try loading again";
        showError(error.message);
      }
    });
    body.append(loadMore);
  }
}

async function loadFeature(details, feature) {
  const body = details.querySelector(".feature-body");
  body.replaceChildren(el("div", "loading", "Loading activation contexts…"));
  try {
    await loadContextPage(details, feature, 0);
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
