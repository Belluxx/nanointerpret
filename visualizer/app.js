const sandboxForm = document.querySelector(".intervention-form");
const formFields = sandboxForm.elements;
const samplingSettings = document.querySelector("#sampling-settings-dialog");
const config = window.NANOINTERPRET_CONFIG;
const staticData = Boolean(config.dataDirectory);
const featureFiles = new Map();

const ui = {
  list: document.querySelector("#feature-list"),
  count: document.querySelector("#result-count"),
  search: document.querySelector("#search-input"),
  filterButton: document.querySelector("#filter-button"),
  filterPopover: document.querySelector("#filter-popover"),
  clearFiltersButton: document.querySelector("#clear-filters-button"),
  activationCountRange: document.querySelector("#activation-count-range"),
  minimumActivationCount: document.querySelector("#minimum-activation-count"),
  maximumActivationCount: document.querySelector("#maximum-activation-count"),
  activationCountOutput: document.querySelector("#activation-count-output"),
  minimumActivationBound: document.querySelector("#minimum-activation-bound"),
  maximumActivationBound: document.querySelector("#maximum-activation-bound"),
  sort: document.querySelector("#sort-select"),
  sortDirection: document.querySelector("#sort-direction-button"),
  metadata: document.querySelector("#dataset-meta"),
  error: document.querySelector("#error-message"),
  pagination: document.querySelector("#feature-pagination"),
  previousPage: document.querySelector("#previous-page-button"),
  pageSelect: document.querySelector("#page-select"),
  pageCount: document.querySelector("#page-count"),
  nextPage: document.querySelector("#next-page-button"),
  sandboxForm,
  prompt: formFields.prompt,
  featureInput: formFields.feature_id,
  featureOptions: document.querySelector("#feature-options"),
  selectedFeatureTitle: sandboxForm.querySelector(".selected-feature-title"),
  interventionMode: formFields.mode,
  amountLabel: sandboxForm.querySelector(".amount-label"),
  amountMultiplier: sandboxForm.querySelector(".amount-preset-control input"),
  amountInput: formFields.amount,
  samplingSettings,
  samplingInputs: [...samplingSettings.querySelectorAll("input[type='number']")],
  generateButton: sandboxForm.querySelector("[type='submit']"),
  generationResults: document.querySelector("#generation-results"),
  baselineOutput: document.querySelector("#baseline-output"),
  intervenedOutput: document.querySelector("#intervened-output"),
};

let features = [];
let featuresById = new Map();
let featureCount = 0;
let amountIsCustom = false;
let renderedBaselineKey = null;
let minimumActivationCount = 0;
let maximumActivationCount = 0;
let reverseFeatureOrder = false;
let currentFeaturePage = 1;

const DEFAULT_CONTEXT_RANGE_START = 0.7;
const CONTEXT_LOAD_DELAY_MS = 120;
const FEATURES_PER_PAGE = 50;
const MAX_FEATURE_SUGGESTIONS = 8;
const generateButtonLabel = ui.generateButton.textContent.trim();

document.addEventListener("pointermove", (event) => {
  if (!event.buttons) return;
  const pointerTarget = document.elementFromPoint(event.clientX, event.clientY);
  for (const popover of document.querySelectorAll(".popover-dialog:popover-open")) {
    if (!popover.contains(pointerTarget)) popover.hidePopover();
  }
});

function positionFilterPopover() {
  const buttonRect = ui.filterButton.getBoundingClientRect();
  const preferredLeft = innerWidth <= 720
    ? buttonRect.left
    : buttonRect.right - 260;
  const left = Math.max(8, Math.min(preferredLeft, innerWidth - 268));

  ui.filterPopover.style.top = `${buttonRect.bottom + 8}px`;
  ui.filterPopover.style.left = `${left}px`;
}

ui.filterPopover.addEventListener("beforetoggle", (event) => {
  if (event.newState === "open") positionFilterPopover();
});

function featureUrl(featureId) {
  if (!staticData) return `/api/features/${featureId}`;
  const fileId = Math.floor(featureId / config.featuresPerFile);
  return `${config.dataDirectory}/features/${fileId}.json`;
}

async function fetchFeature(featureId) {
  const url = featureUrl(featureId);
  if (!staticData) return fetchJson(url);
  if (!featureFiles.has(url)) featureFiles.set(url, fetchJson(url));
  return (await featureFiles.get(url))[featureId];
}

function element(tag, className, text) {
  const result = document.createElement(tag);
  if (className) result.className = className;
  if (text !== undefined) result.textContent = text;
  return result;
}

const dropdowns = [];

function createDropdown(trigger, menu, onChoose, beforeOpen) {
  let activeIndex = 0;
  const root = menu.parentElement;
  const options = () => [...menu.querySelectorAll('[role="option"]')];

  function close() {
    menu.hidden = true;
    root.classList.remove("is-open");
    trigger.setAttribute("aria-expanded", "false");
    trigger.removeAttribute("aria-activedescendant");
  }

  function activate(index, scroll = true) {
    const items = options();
    if (!items.length) return;
    activeIndex = Math.max(0, Math.min(index, items.length - 1));
    items.forEach((item, itemIndex) => {
      item.classList.toggle("is-active", itemIndex === activeIndex);
    });
    trigger.setAttribute("aria-activedescendant", items[activeIndex].id);
    if (scroll) items[activeIndex].scrollIntoView({ block: "nearest" });
  }

  function open(index = 0) {
    for (const dropdown of dropdowns) {
      if (dropdown !== controller) dropdown.close();
    }
    menu.hidden = false;
    root.classList.add("is-open");
    trigger.setAttribute("aria-expanded", "true");
    activate(index);
  }

  function choose(index) {
    const item = options()[index];
    if (!item) return;
    onChoose(item);
    close();
  }

  trigger.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !menu.hidden) {
      event.preventDefault();
      close();
    } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      if (menu.hidden) {
        if (!beforeOpen || beforeOpen()) open();
      }
      else activate(activeIndex + (event.key === "ArrowDown" ? 1 : -1));
    } else if (!menu.hidden && (event.key === "Home" || event.key === "End")) {
      event.preventDefault();
      activate(event.key === "Home" ? 0 : options().length - 1);
    } else if (
      !menu.hidden
      && (event.key === "Enter" || (event.key === " " && trigger.matches("button")))
    ) {
      event.preventDefault();
      choose(activeIndex);
    } else if (event.key === "Tab") {
      close();
    }
  });
  menu.addEventListener("click", (event) => {
    const item = event.target.closest('[role="option"]');
    if (!item) return;
    event.preventDefault();
    choose(options().indexOf(item));
  });
  menu.addEventListener("pointermove", (event) => {
    const item = event.target.closest('[role="option"]');
    if (item) activate(options().indexOf(item), false);
  });

  const controller = { root, open, close };
  dropdowns.push(controller);
  return controller;
}

function createSelectControl(select, index) {
  const control = element("div", "custom-select");
  const trigger = element("button", "custom-select-trigger");
  const value = element("span", "custom-select-value");
  const menu = element("div", "dropdown-menu select-menu");
  const menuId = `select-menu-${index}`;
  const valueId = `select-value-${index}`;
  const labelledBy = select.getAttribute("aria-labelledby");
  const ariaLabel = select.getAttribute("aria-label");

  trigger.type = "button";
  trigger.setAttribute("role", "combobox");
  trigger.setAttribute("aria-expanded", "false");
  trigger.setAttribute("aria-controls", menuId);
  value.id = valueId;
  menu.id = menuId;
  menu.setAttribute("role", "listbox");
  menu.hidden = true;

  select.before(control);
  control.append(select, trigger, menu);
  trigger.append(value, element("span", "dropdown-chevron"));

  function render() {
    const selected = select.selectedOptions[0];
    value.textContent = selected?.textContent || "";
    if (labelledBy) trigger.setAttribute("aria-labelledby", `${labelledBy} ${valueId}`);
    else if (ariaLabel) trigger.setAttribute("aria-label", `${ariaLabel}: ${value.textContent}`);
    menu.replaceChildren(...[...select.options].map((option, optionIndex) => {
      const item = element("div", "custom-select-option", option.textContent);
      item.id = `${menuId}-option-${optionIndex}`;
      item.dataset.value = option.value;
      item.setAttribute("role", "option");
      item.setAttribute("aria-selected", String(option.selected));
      return item;
    }));
  }

  const dropdown = createDropdown(trigger, menu, (item) => {
    if (select.value === item.dataset.value) return;
    select.value = item.dataset.value;
    select.dispatchEvent(new Event("change", { bubbles: true }));
  });
  trigger.addEventListener("click", () => {
    if (menu.hidden) {
      dropdown.open(Math.max(0, select.selectedIndex));
    } else {
      dropdown.close();
    }
  });
  select.addEventListener("change", render);
  render();
  return { render };
}

const selectControls = new Map(
  [...document.querySelectorAll("select")].map((select, index) => [
    select,
    createSelectControl(select, index),
  ]),
);

function matchingFeatures(query) {
  const normalizedQuery = query.trim().toLocaleLowerCase();
  const matches = [];
  for (const feature of features) {
    const title = feature.title || "";
    if (normalizedQuery) {
      if (
        !String(feature.id).includes(normalizedQuery)
        && !title.toLocaleLowerCase().includes(normalizedQuery)
      ) continue;
    } else if (!title) {
      continue;
    }
    matches.push(feature);
    if (matches.length === MAX_FEATURE_SUGGESTIONS) break;
  }
  return matches;
}

function prepareFeatureSuggestions() {
  if (document.activeElement !== ui.featureInput || !features.length) {
    return false;
  }
  const matches = matchingFeatures(ui.featureInput.value);
  if (!matches.length) {
    ui.featureOptions.replaceChildren();
    return false;
  }

  ui.featureOptions.replaceChildren(...matches.map((feature) => {
    const option = element("div", "autocomplete-option");
    option.id = `feature-option-${feature.id}`;
    option.dataset.featureId = String(feature.id);
    option.setAttribute("role", "option");
    option.append(
      element("span", "autocomplete-option-title", feature.title || `Feature ${feature.id}`),
      element("span", "autocomplete-option-id", `#${feature.id}`),
    );
    return option;
  }));
  return true;
}

const featureDropdown = createDropdown(
  ui.featureInput,
  ui.featureOptions,
  (option) => {
    ui.featureInput.value = option.dataset.featureId;
    ui.featureInput.dispatchEvent(new Event("input", { bubbles: true }));
    ui.featureInput.focus();
  },
  prepareFeatureSuggestions,
);

function renderFeatureSuggestions() {
  if (prepareFeatureSuggestions()) featureDropdown.open();
  else featureDropdown.close();
}

ui.featureInput.addEventListener("focus", renderFeatureSuggestions);
document.addEventListener("pointerdown", (event) => {
  for (const dropdown of dropdowns) {
    if (!dropdown.root.contains(event.target)) dropdown.close();
  }
});

function renderGeneration(output, prompt, continuation) {
  output.classList.remove("is-loading");
  output.replaceChildren(
    element("span", "generation-prompt", prompt),
    continuation || "(No visible text)",
  );
}

function renderGenerationLoading(output) {
  const spinner = element("span", "generation-spinner");
  spinner.setAttribute("role", "status");
  spinner.setAttribute("aria-label", "Generating");
  output.classList.add("is-loading");
  output.replaceChildren(spinner);
}

const compactNumber = new Intl.NumberFormat("en", {
  notation: "compact",
  maximumFractionDigits: 1,
});

function compactCount(count) {
  return `~${compactNumber.format(count)}`;
}

function formatActivation(value) {
  if (value >= 100) return value.toFixed(0);
  if (value >= 10) return value.toFixed(1);
  if (value >= 1) return value.toFixed(2);
  return value.toPrecision(3);
}

function formatPercentage(value) {
  return value.toLocaleString("en", { style: "percent" });
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
  featureCount = metadata.d_sae;
  featuresById = new Map(features.map((feature) => [feature.id, feature]));
}

function selectedFeatureId() {
  const value = ui.featureInput.value.trim();
  const featureId = Number(value);
  return value
    && Number.isInteger(featureId)
    && featureId >= 0
    && featureId < featureCount
    ? featureId
    : null;
}

function updateSelectedFeatureTitle() {
  const featureId = selectedFeatureId();
  const title = featuresById.get(featureId)?.title || "";
  ui.selectedFeatureTitle.textContent = title;
  ui.selectedFeatureTitle.title = title;
  ui.selectedFeatureTitle.hidden = !title;
  if (title) ui.selectedFeatureTitle.href = `#feature-${featureId}`;
  else ui.selectedFeatureTitle.removeAttribute("href");
}

function openFeature(featureId) {
  let details = document.getElementById(`feature-${featureId}`);
  if (!details) {
    ui.search.value = "";
    resetActivationCountRange();
    const index = visibleFeatures().findIndex((feature) => feature.id === featureId);
    currentFeaturePage = Math.floor(Math.max(0, index) / FEATURES_PER_PAGE) + 1;
    renderFeatureList();
    details = document.getElementById(`feature-${featureId}`);
  }
  if (!details) return;

  details.open = true;
  details.scrollIntoView({ behavior: "smooth", block: "start" });
}

function selectedFeatureMaximum() {
  const featureId = selectedFeatureId();
  return featureId === null
    ? null
    : featuresById.get(featureId)?.max_activation ?? 0;
}

function updateAmountControl() {
  const clamping = ui.interventionMode.value === "clamp";
  const multiplier = ui.amountMultiplier.valueAsNumber;
  ui.amountLabel.textContent = clamping ? "Target activation" : "Steering strength";
  ui.amountInput.setAttribute(
    "aria-label",
    clamping ? "Target feature activation" : "Additive steering strength",
  );
  ui.amountMultiplier.setAttribute(
    "aria-valuetext",
    `${formatPercentage(multiplier)} of maximum activation`,
  );
  updateRangeProgress(ui.amountMultiplier);
  if (!amountIsCustom) {
    const maximum = selectedFeatureMaximum();
    ui.amountInput.value = maximum === null
      ? ""
      : String(Number((maximum * multiplier).toPrecision(4)));
  }
}

function updateAmountMultiplierFromCustomValue() {
  const maximum = selectedFeatureMaximum();
  const amount = ui.amountInput.valueAsNumber;
  if (!maximum || !Number.isFinite(amount)) return;

  ui.amountMultiplier.value = String(amount / maximum);
  updateAmountControl();
}

function addSynchronizedRange(input) {
  const range = input.cloneNode();
  range.type = "range";
  range.className = "progress-range";
  range.removeAttribute("id");
  range.removeAttribute("form");
  range.removeAttribute("name");
  range.removeAttribute("required");
  range.setAttribute("aria-label", input.labels[0].textContent.trim());
  input.before(range);

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

ui.interventionMode.addEventListener("change", updateAmountControl);
ui.amountMultiplier.addEventListener("input", () => {
  amountIsCustom = false;
  updateAmountControl();
});
ui.amountInput.addEventListener("input", () => {
  amountIsCustom = true;
  updateAmountMultiplierFromCustomValue();
});
ui.prompt.addEventListener("input", () => ui.prompt.setCustomValidity(""));
ui.featureInput.addEventListener("input", () => {
  ui.featureInput.setCustomValidity("");
  updateSelectedFeatureTitle();
  if (amountIsCustom) updateAmountMultiplierFromCustomValue();
  else updateAmountControl();
  renderFeatureSuggestions();
});
ui.selectedFeatureTitle.addEventListener("click", (event) => {
  event.preventDefault();
  const featureId = selectedFeatureId();
  if (featureId !== null) openFeature(featureId);
});
ui.samplingInputs.forEach(addSynchronizedRange);
ui.samplingSettings.addEventListener("invalid", () => {
  if (!ui.samplingSettings.matches(":popover-open")) {
    ui.samplingSettings.showPopover();
  }
}, true);
ui.sandboxForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!ui.prompt.value.trim()) {
    ui.prompt.setCustomValidity("Enter a prompt to continue.");
    ui.prompt.reportValidity();
    return;
  }

  const featureId = selectedFeatureId();
  if (featureId === null) {
    ui.featureInput.setCustomValidity("Choose a feature from the suggestions or enter its numeric ID.");
    ui.featureInput.reportValidity();
    return;
  }

  const request = Object.fromEntries(new FormData(ui.sandboxForm));
  request.feature_id = featureId;
  for (const input of ui.sandboxForm.elements) {
    if (input.type === "number") request[input.name] = input.valueAsNumber;
  }

  const baselineKey = JSON.stringify([
    request.prompt,
    ...ui.samplingInputs.map((input) => request[input.name]),
  ]);
  if (baselineKey !== renderedBaselineKey) {
    renderGenerationLoading(ui.baselineOutput);
  }
  renderGenerationLoading(ui.intervenedOutput);
  ui.generationResults.hidden = false;

  ui.generateButton.disabled = true;
  ui.generateButton.textContent = "Generating...";
  ui.sandboxForm.querySelector(".sandbox-error")?.remove();
  try {
    const payload = await fetchJson(config.interventionUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    renderGeneration(ui.baselineOutput, request.prompt, payload.baseline);
    renderGeneration(ui.intervenedOutput, request.prompt, payload.intervened);
    renderedBaselineKey = baselineKey;
  } catch (error) {
    renderedBaselineKey = null;
    ui.generationResults.hidden = true;
    const message = element("span", "sandbox-error", `Could not generate: ${error.message}`);
    message.setAttribute("role", "alert");
    ui.generateButton.after(message);
  } finally {
    ui.generateButton.disabled = false;
    ui.generateButton.textContent = generateButtonLabel;
  }
});
updateAmountControl();

function renderMetadata(metadata) {
  const items = [
    ["Model", metadata.model_id],
    ["Tokens", compactNumber.format(metadata.processed_tokens)],
    ["Layer", metadata.layer_index],
    ["SAE width", compactNumber.format(metadata.d_sae)],
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
  count: (a, b) => b.activation_count - a.activation_count || a.id - b.id,
  title: (a, b) => (a.title || `Feature ${a.id}`).localeCompare(b.title || `Feature ${b.id}`),
};

function visibleFeatures() {
  const query = ui.search.value.trim().toLocaleLowerCase();
  const selectedMinimumActivationCount = activationCountAt(
    ui.minimumActivationCount.valueAsNumber,
  );
  const selectedMaximumActivationCount = activationCountAt(
    ui.maximumActivationCount.valueAsNumber,
  );
  const visible = features.filter((feature) => {
    if (
      feature.activation_count < selectedMinimumActivationCount
      || feature.activation_count > selectedMaximumActivationCount
    ) return false;
    return !query
      || String(feature.id).includes(query)
      || (feature.title || "").toLocaleLowerCase().includes(query);
  });
  visible.sort(sorters[ui.sort.value]);
  return reverseFeatureOrder ? visible.reverse() : visible;
}

function activationCountAt(position) {
  const logarithmicMinimum = Math.log1p(minimumActivationCount);
  const logarithmicMaximum = Math.log1p(maximumActivationCount);
  return Math.round(Math.expm1(
    logarithmicMinimum + position * (logarithmicMaximum - logarithmicMinimum),
  ));
}

function keepRangeOrdered(minimumInput, maximumInput, changedInput) {
  if (minimumInput.valueAsNumber <= maximumInput.valueAsNumber) return;
  const source = changedInput || maximumInput;
  const target = source === minimumInput ? maximumInput : minimumInput;
  target.value = source.value;
}

function updateActivationCountRange(changedInput) {
  keepRangeOrdered(
    ui.minimumActivationCount,
    ui.maximumActivationCount,
    changedInput,
  );
  const minimumPosition = ui.minimumActivationCount.valueAsNumber;
  const maximumPosition = ui.maximumActivationCount.valueAsNumber;
  const selectedMinimum = activationCountAt(minimumPosition);
  const selectedMaximum = activationCountAt(maximumPosition);
  ui.activationCountRange.style.setProperty("--range-start", `${100 * minimumPosition}%`);
  ui.activationCountRange.style.setProperty("--range-end", `${100 * maximumPosition}%`);
  ui.activationCountOutput.value =
    `${selectedMinimum.toLocaleString()}–${selectedMaximum.toLocaleString()}`;
  ui.clearFiltersButton.disabled = minimumPosition === 0 && maximumPosition === 1;
}

function resetActivationCountRange() {
  ui.minimumActivationCount.value = ui.minimumActivationCount.min;
  ui.maximumActivationCount.value = ui.maximumActivationCount.max;
  updateActivationCountRange();
}

function renderFeatureList() {
  const visible = visibleFeatures();
  const pageCount = Math.max(1, Math.ceil(visible.length / FEATURES_PER_PAGE));
  currentFeaturePage = Math.min(currentFeaturePage, pageCount);
  const pageStart = (currentFeaturePage - 1) * FEATURES_PER_PAGE;
  const pageFeatures = visible.slice(pageStart, pageStart + FEATURES_PER_PAGE);
  ui.count.textContent = visible.length === features.length
    ? `${visible.length.toLocaleString()} active features`
    : `${visible.length.toLocaleString()} of ${features.length.toLocaleString()} features`;

  const fragment = document.createDocumentFragment();
  for (const feature of pageFeatures) {
    const details = element("details", "feature");
    details.id = `feature-${feature.id}`;
    const summary = element("summary");
    const title = feature.title || `Feature ${feature.id}`;
    const titleElement = element("span", "feature-title", title);
    titleElement.title = title;
    summary.append(
      element("span", "feature-id", `#${feature.id}`),
      titleElement,
      element(
        "span",
        "feature-stat",
        feature.activation_count.toLocaleString(),
      ),
      element("span", "chevron"),
    );
    details.append(summary, element("div", "feature-body"));
    details.addEventListener("toggle", () => {
      if (details.open && !details.dataset.loaded) {
        loadFeature(details, feature);
      }
    });
    fragment.append(details);
  }
  ui.list.replaceChildren(fragment);
  ui.pagination.hidden = pageCount === 1;
  if (pageCount === 1) return;

  if (ui.pageSelect.options.length !== pageCount) {
    ui.pageSelect.replaceChildren(...Array.from({ length: pageCount }, (_, index) => {
      const option = document.createElement("option");
      option.value = String(index + 1);
      option.textContent = (index + 1).toLocaleString();
      return option;
    }));
  }
  ui.pageSelect.value = String(currentFeaturePage);
  selectControls.get(ui.pageSelect).render();
  ui.pageCount.textContent = `of ${pageCount.toLocaleString()}`;
  ui.previousPage.disabled = currentFeaturePage === 1;
  ui.nextPage.disabled = currentFeaturePage === pageCount;
}

function resetFeaturePage() {
  currentFeaturePage = 1;
  renderFeatureList();
}

function showFeaturePage(page) {
  currentFeaturePage = page;
  renderFeatureList();
  ui.list.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderContext(context, feature) {
  const card = element("article", "context-card");
  const header = element("header", "context-header");
  header.append(
    element("span", "", `Context ${context.context_id}`),
    element(
      "span",
      "context-stats",
      `Peak ${formatActivation(context.peak_activation)}`,
    ),
  );

  const tokens = element("pre", "tokens");
  const sparseActivations = context.activation_positions
    ? new Map(context.activation_positions.map((position, index) => [
      position,
      context.activation_values[index],
    ]))
    : null;
  context.tokens.forEach((text, index) => {
    const token = element("span", "token", text);
    const activation = sparseActivations
      ? sparseActivations.get(index) || 0
      : context.activations[index];
    if (activation > 0) {
      const strength = Math.sqrt(Math.min(1, activation / feature.max_activation));
      token.classList.add("active");
      token.style.setProperty(
        "--activation-opacity",
        `${100 * (0.1 + 0.75 * strength)}%`,
      );
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
    ["Activating tokens", compactCount(payload.activation_count)],
    ["Contexts", compactCount(payload.context_count)],
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

function renderDistribution(feature, payload) {
  const counts = payload.activation_histogram;
  const exportedContexts = payload.contexts || [];
  const panel = element("section");
  const plot = element("div", "distribution-plot");
  plot.setAttribute("aria-hidden", "true");
  plot.style.setProperty("--distribution-bin-count", counts.length);
  const largestBin = Math.max(...counts, 1);
  const bars = counts.map((count, index) => {
    const bar = element("span", "distribution-bar");
    bar.style.height = `${100 * Math.sqrt(count / largestBin)}%`;
    const lower = feature.max_activation * index / counts.length;
    const upper = feature.max_activation * (index + 1) / counts.length;
    bar.title = `${formatActivation(lower)}-${formatActivation(upper)}: ${count.toLocaleString()} contexts`;
    plot.append(bar);
    return bar;
  });

  const selector = element("div", "dual-range range-selector");
  function rangeInput(label, value) {
    const input = document.createElement("input");
    input.type = "range";
    input.min = "0";
    input.max = "1";
    input.step = "0.001";
    input.value = String(value);
    input.setAttribute("aria-label", label);
    return input;
  }
  const minimumInput = rangeInput("Minimum peak activation", DEFAULT_CONTEXT_RANGE_START);
  const maximumInput = rangeInput("Maximum peak activation", 1);
  const minimumLabel = element("span", "range-value");
  const maximumLabel = element("span", "range-value");
  const valueLabels = element("div", "range-value-track");
  valueLabels.append(minimumLabel, maximumLabel);
  selector.append(
    minimumInput,
    maximumInput,
    valueLabels,
  );

  const resultCount = element("button", "range-result-count");
  resultCount.type = "button";
  resultCount.setAttribute("aria-label", "Reverse context order");
  const orderLabel = element("span");
  const results = element("div", "contexts");
  panel.append(
    element("h3", "range-heading", "Distribution of activation peaks"),
    plot,
    selector,
    resultCount,
    results,
  );

  let loadTimer;
  let requestId = 0;
  let descending = true;

  function updateSelection(changedInput) {
    keepRangeOrdered(minimumInput, maximumInput, changedInput);
    const minimumFraction = minimumInput.valueAsNumber;
    const maximumFraction = maximumInput.valueAsNumber;
    selector.style.setProperty("--range-start", `${100 * minimumFraction}%`);
    selector.style.setProperty("--range-end", `${100 * maximumFraction}%`);

    const minimum = feature.max_activation * minimumFraction;
    const maximum = feature.max_activation * maximumFraction;
    for (const [label, fraction, value] of [
      [minimumLabel, minimumFraction, minimum],
      [maximumLabel, maximumFraction, maximum],
    ]) {
      label.style.left = `${100 * fraction}%`;
      label.textContent = formatActivation(value);
    }
    bars.forEach((bar, index) => {
      const binStart = index / bars.length;
      const binEnd = (index + 1) / bars.length;
      bar.classList.toggle(
        "selected",
        binEnd >= minimumFraction && binStart <= maximumFraction,
      );
    });

    clearTimeout(loadTimer);
    const currentRequest = ++requestId;
    loadTimer = setTimeout(
      () => loadContexts(minimum, maximum, currentRequest),
      CONTEXT_LOAD_DELAY_MS,
    );
  }

  async function loadContexts(minimum, maximum, currentRequest) {
    resultCount.textContent = "Loading...";
    if (!results.childElementCount) {
      results.append(element("p", "loading", "Loading contexts..."));
    }
    try {
      let payload;
      if (staticData) {
        const contexts = exportedContexts.filter((context) => (
          context.peak_activation >= minimum && context.peak_activation <= maximum
        ));
        payload = { contexts };
      } else {
        const query = new URLSearchParams({ min: minimum, max: maximum });
        payload = await fetchJson(`${featureUrl(feature.id)}?${query}`);
      }
      if (currentRequest !== requestId) return;
      const contexts = descending ? payload.contexts : [...payload.contexts].reverse();
      const shown = contexts.length;
      const label = staticData
        ? `${shown.toLocaleString()} representative contexts`
        : shown === payload.matching_context_count
        ? `${shown.toLocaleString()} contexts`
        : `Sampled ${shown.toLocaleString()} of ${payload.matching_context_count.toLocaleString()}`;
      orderLabel.textContent = descending ? "(high to low)" : "(low to high)";
      resultCount.replaceChildren(label, orderLabel);
      if (shown) {
        results.replaceChildren(
          ...contexts.map((context) => renderContext(context, feature)),
        );
      } else {
        results.replaceChildren(
          element("p", "empty-state", "No contexts fall within this activation range."),
        );
      }
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
  resultCount.addEventListener("click", () => {
    descending = !descending;
    orderLabel.textContent = descending ? "(high to low)" : "(low to high)";
    results.replaceChildren(...[...results.children].reverse());
  });
  updateSelection();
  return panel;
}

async function loadFeature(details, feature) {
  const body = details.querySelector(".feature-body");
  details.dataset.loaded = "loading";
  body.replaceChildren(element("p", "loading", "Loading feature..."));
  try {
    const payload = await fetchFeature(feature.id);
    body.replaceChildren(
      renderOverview(feature, payload),
      renderDistribution(feature, payload),
    );
    details.dataset.loaded = "true";
  } catch (error) {
    delete details.dataset.loaded;
    body.replaceChildren(element("p", "empty-state", `Could not load feature: ${error.message}`));
  }
}

let renderFrame;
ui.search.addEventListener("input", () => {
  cancelAnimationFrame(renderFrame);
  renderFrame = requestAnimationFrame(resetFeaturePage);
});
for (const input of [ui.minimumActivationCount, ui.maximumActivationCount]) {
  input.addEventListener("input", () => updateActivationCountRange(input));
  input.addEventListener("change", resetFeaturePage);
}
ui.clearFiltersButton.addEventListener("click", () => {
  resetActivationCountRange();
  resetFeaturePage();
  ui.filterPopover.hidePopover();
});
ui.sort.addEventListener("change", resetFeaturePage);
ui.sortDirection.addEventListener("click", () => {
  reverseFeatureOrder = !reverseFeatureOrder;
  resetFeaturePage();
});
ui.pageSelect.addEventListener("change", () => {
  showFeaturePage(Number(ui.pageSelect.value));
});
ui.previousPage.addEventListener("click", () => showFeaturePage(currentFeaturePage - 1));
ui.nextPage.addEventListener("click", () => showFeaturePage(currentFeaturePage + 1));
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && ui.search.value) {
    ui.search.value = "";
    resetFeaturePage();
    ui.search.focus();
  }
});

async function initialize() {
  try {
    if (!config.interventionUrl) ui.sandboxForm.closest(".sandbox").hidden = true;
    const summaryUrl = staticData
      ? `${config.dataDirectory}/summary.json`
      : "/api/summary";
    const payload = await fetchJson(summaryUrl);
    features = payload.features;
    const activationCounts = features.map((feature) => feature.activation_count);
    minimumActivationCount = activationCounts.length
      ? Math.min(...activationCounts)
      : 0;
    maximumActivationCount = activationCounts.length
      ? Math.max(...activationCounts)
      : 0;
    ui.minimumActivationBound.textContent = minimumActivationCount.toLocaleString();
    ui.maximumActivationBound.textContent = maximumActivationCount.toLocaleString();
    renderMetadata(payload.metadata);
    initializeSandbox(payload.metadata);
    updateActivationCountRange();
    renderFeatureList();
  } catch (error) {
    ui.count.textContent = "Could not load activations";
    ui.error.textContent = error.message;
    ui.error.hidden = false;
  }
}

initialize();
