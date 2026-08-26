const playgroundForm = document.querySelector("#intervention-form");
const formFields = playgroundForm.elements;
const config = window.NANOINTERPRET_CONFIG;
const isStatic = Boolean(config.dataDirectory);
const coldStartNote = isStatic
  ? "The GPU may be cold, please wait a minute the first time"
  : null;

const ui = {
  list: document.querySelector("#feature-list"),
  count: document.querySelector("#result-count"),
  search: document.querySelector("#search-input"),
  filterPopover: document.querySelector("#filter-popover"),
  clearFiltersButton: document.querySelector("#clear-filters-button"),
  category: document.querySelector("#category-select"),
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
  prompt: formFields.prompt,
  featureInput: formFields.feature_id,
  amountMultiplier: playgroundForm.querySelector(".amount-preset-control input"),
  amountInput: formFields.amount,
  samplingSettings: document.querySelector("#sampling-settings-dialog"),
  samplingInputs: [
    ...document.querySelectorAll("#sampling-settings-dialog input[type='number']"),
  ],
  generateButton: playgroundForm.querySelector("[type='submit']"),
  generationResults: document.querySelector("#generation-results"),
  interventionExamples: document.querySelector("#intervention-examples"),
  interventionExampleList: document.querySelector("#intervention-example-list"),
  baselineOutput: document.querySelector("#baseline-output"),
  intervenedOutput: document.querySelector("#intervened-output"),
};

const DEFAULT_CONTEXT_RANGE_START = 0.7;
const CONTEXT_LOAD_DELAY_MS = 120;
const FEATURES_PER_PAGE = 50;
const SEARCH_RESULT_BATCH_SIZE = 100;
const generateButtonLabel = ui.generateButton.textContent.trim();
const featureFiles = new Map();

let features = [];
let featuresById = new Map();
let amountIsCustom = false;
let renderedBaselineKey = null;
let minimumActivationCount = 0;
let maximumActivationCount = 0;
let reverseFeatureOrder = false;
let currentFeaturePage = 1;

function featureUrl(featureId) {
  if (!isStatic) return `/api/features/${featureId}`;
  const fileId = Math.floor(featureId / config.featuresPerFile);
  return `${config.dataDirectory}/features/${fileId}.json`;
}

async function fetchFeature(featureId) {
  const url = featureUrl(featureId);
  if (!isStatic) return fetchJson(url);
  if (!featureFiles.has(url)) featureFiles.set(url, fetchJson(url));
  return (await featureFiles.get(url))[featureId];
}

function element(tag, className, text) {
  const result = document.createElement(tag);
  if (className) result.className = className;
  if (text !== undefined) result.textContent = text;
  return result;
}

function featureStar() {
  const star = element("span", "feature-star");
  star.setAttribute("aria-hidden", "true");
  return star;
}

function normalizeFeatureQuery(value) {
  return value.trim().toLocaleLowerCase();
}

function matchesFeatureQuery(id, title, query) {
  return !query
    || String(id).includes(query)
    || (title || "").toLocaleLowerCase().includes(query);
}

const dropdowns = [];

function createSelectControl(select, index) {
  const control = element("div", "custom-select");
  const trigger = element("button", "custom-select-trigger");
  const value = element("span", "custom-select-value");
  const menu = element("div", "dropdown-menu select-menu");
  const search = select.hasAttribute("data-feature-search")
    ? element("input", "select-search")
    : null;
  const optionsRoot = search ? element("div", "searchable-select-options") : menu;
  const empty = search ? element("p", "select-empty", "No features found") : null;
  const menuId = `select-menu-${index}`;
  const optionsId = search ? `${menuId}-options` : menuId;
  const valueId = `select-value-${index}`;
  const labelledBy = select.getAttribute("aria-labelledby");
  const ariaLabel = select.getAttribute("aria-label");
  const optionItems = () => [...optionsRoot.querySelectorAll('[role="option"]')];
  let activeIndex = 0;
  let choices = [];
  let matchingChoices = [];
  let renderedChoiceCount = 0;

  trigger.type = "button";
  trigger.setAttribute("role", "combobox");
  trigger.setAttribute("aria-expanded", "false");
  trigger.setAttribute("aria-controls", optionsId);
  value.id = valueId;
  menu.id = menuId;
  menu.hidden = true;
  optionsRoot.id = optionsId;
  optionsRoot.setAttribute("role", "listbox");

  if (search) {
    menu.classList.add("searchable-select-menu");
    search.type = "search";
    search.placeholder = "Search by ID or title";
    search.autocomplete = "off";
    search.setAttribute("aria-label", "Search features");
    search.setAttribute("aria-controls", optionsId);
    empty.hidden = true;
    const searchWrap = element("div", "select-search-wrap");
    searchWrap.append(search);
    menu.append(searchWrap, optionsRoot, empty);
  }

  select.before(control);
  control.append(select, trigger, menu);
  trigger.append(value, element("span", "dropdown-chevron"));

  function close() {
    menu.hidden = true;
    control.classList.remove("is-open");
    trigger.setAttribute("aria-expanded", "false");
    trigger.removeAttribute("aria-activedescendant");
  }

  function activate(index, scroll = true) {
    const items = optionItems();
    if (!items.length) {
      activeIndex = 0;
      trigger.removeAttribute("aria-activedescendant");
      return;
    }
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
    control.classList.add("is-open");
    trigger.setAttribute("aria-expanded", "true");
    activate(index);
  }

  function choose(index) {
    const item = optionItems()[index];
    if (!item) return false;
    if (select.value !== item.dataset.value) {
      select.value = item.dataset.value;
      select.dispatchEvent(new Event("change", { bubbles: true }));
    }
    close();
    return true;
  }

  function appendChoices() {
    const end = Math.min(
      renderedChoiceCount + (search ? SEARCH_RESULT_BATCH_SIZE : matchingChoices.length),
      matchingChoices.length,
    );
    const fragment = document.createDocumentFragment();
    for (let index = renderedChoiceCount; index < end; index += 1) {
      const choice = matchingChoices[index];
      const item = element("div", "custom-select-option");
      item.id = `${menuId}-option-${choice.index}`;
      item.dataset.value = choice.value;
      item.dataset.highQuality = String(choice.high_quality);
      item.setAttribute("role", "option");
      item.setAttribute("aria-selected", String(choice.value === select.value));
      if (choice.high_quality) item.append(featureStar());
      item.append(element("span", "feature-option-text", choice.text));
      fragment.append(item);
    }
    optionsRoot.append(fragment);
    renderedChoiceCount = end;
  }

  function renderChoices(nextChoices) {
    matchingChoices = nextChoices;
    renderedChoiceCount = 0;
    optionsRoot.replaceChildren();
    appendChoices();
  }

  function renderSearchResults() {
    const query = normalizeFeatureQuery(search.value);
    renderChoices(choices.filter((choice) => (
      matchesFeatureQuery(
        choice.value,
        featuresById.get(Number(choice.value))?.title,
        query,
      )
    )));
    optionsRoot.scrollTop = 0;
    empty.hidden = matchingChoices.length > 0;
  }

  function renderValue() {
    const selected = select.selectedOptions[0];
    value.replaceChildren();
    if (selected?.dataset.highQuality === "true") value.append(featureStar());
    value.append(element("span", "feature-option-text", selected?.textContent || ""));
    if (labelledBy) trigger.setAttribute("aria-labelledby", `${labelledBy} ${valueId}`);
    else if (ariaLabel) trigger.setAttribute("aria-label", `${ariaLabel}: ${value.textContent}`);
    for (const item of optionItems()) {
      item.setAttribute("aria-selected", String(item.dataset.value === select.value));
    }
  }

  function updateOptions() {
    choices = [...select.options]
      .filter((option) => !option.hidden)
      .map((option, optionIndex) => ({
        index: optionIndex,
        value: option.value,
        text: option.textContent,
        high_quality: option.dataset.highQuality === "true",
      }));
    if (search) renderSearchResults();
    else renderChoices(choices);
    renderValue();
  }

  const controller = { root: control, close };
  dropdowns.push(controller);

  trigger.addEventListener("click", () => {
    if (menu.hidden) {
      const selectedIndex = optionItems()
        .findIndex((item) => item.dataset.value === select.value);
      open(Math.max(0, selectedIndex));
      search?.focus();
    } else {
      close();
    }
  });
  trigger.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !menu.hidden) {
      event.preventDefault();
      close();
    } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      if (menu.hidden) open();
      else activate(activeIndex + (event.key === "ArrowDown" ? 1 : -1));
      search?.focus();
    } else if (!menu.hidden && (event.key === "Home" || event.key === "End")) {
      event.preventDefault();
      activate(event.key === "Home" ? 0 : optionItems().length - 1);
    } else if (!menu.hidden && (event.key === "Enter" || event.key === " ")) {
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
    choose(optionItems().indexOf(item));
  });
  menu.addEventListener("pointermove", (event) => {
    const item = event.target.closest('[role="option"]');
    if (item) activate(optionItems().indexOf(item), false);
  });

  if (search) {
    let searchFrame;
    search.addEventListener("input", () => {
      cancelAnimationFrame(searchFrame);
      searchFrame = requestAnimationFrame(() => {
        if (menu.hidden) return;
        renderSearchResults();
        open();
      });
    });
    optionsRoot.addEventListener("scroll", () => {
      if (
        renderedChoiceCount < matchingChoices.length
        && optionsRoot.scrollTop + optionsRoot.clientHeight >= optionsRoot.scrollHeight - 40
      ) appendChoices();
    });
    search.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        close();
        trigger.focus();
      } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        activate(activeIndex + (event.key === "ArrowDown" ? 1 : -1));
      } else if (event.key === "Enter") {
        event.preventDefault();
        if (choose(activeIndex)) trigger.focus();
      } else if (event.key === "Tab") {
        close();
      }
    });
  }
  select.addEventListener("change", renderValue);
  updateOptions();
  return { trigger, updateOptions };
}

const selectControls = new Map(
  [...document.querySelectorAll("select")].map((select, index) => [
    select,
    createSelectControl(select, index),
  ]),
);
const featureSelectControl = selectControls.get(ui.featureInput);

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

function renderLoading(output, label, noteText) {
  const spinner = element("span", "generation-spinner");
  spinner.setAttribute("role", "status");
  spinner.setAttribute("aria-label", label);
  output.classList.add("is-loading");
  output.replaceChildren(spinner);
  if (noteText) {
    output.append(element("span", "generation-wait-note", noteText));
  }
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

function starredFirst(sorter) {
  return (a, b) => (
    Number(b.high_quality) - Number(a.high_quality)
    || sorter(a, b)
  );
}

function initializePlayground(metadata) {
  featuresById = new Map(features.map((feature) => [feature.id, feature]));
  const featureChoices = Array.from(
    { length: metadata.d_sae },
    (_, id) => featuresById.get(id) || { id, activation_count: 0, high_quality: false },
  );
  featureChoices.sort(starredFirst((a, b) => (
    b.activation_count - a.activation_count || a.id - b.id
  )));

  const options = document.createDocumentFragment();
  for (const feature of featureChoices) {
    const label = feature.title ? `${feature.id}: ${feature.title}` : `Feature ${feature.id}`;
    const option = new Option(label, String(feature.id));
    option.dataset.highQuality = String(feature.high_quality);
    options.append(option);
  }
  ui.featureInput.append(options);
  featureSelectControl.updateOptions();
}

function tryInterventionExample(example) {
  ui.prompt.value = example.prompt;
  ui.prompt.setCustomValidity("");

  ui.featureInput.value = String(example.feature_id);
  ui.featureInput.dispatchEvent(new Event("change", { bubbles: true }));

  const multiplier = example.target_activation_pct;
  ui.amountMultiplier.value = String(multiplier);
  if (ui.amountMultiplier.valueAsNumber === multiplier) {
    amountIsCustom = false;
    updateAmountControl();
    return;
  }

  const featureMaximum = selectedFeatureMaximum();
  amountIsCustom = true;
  ui.amountInput.value = featureMaximum === null
    ? ""
    : String(Number((featureMaximum * multiplier).toPrecision(4)));
  updateAmountMultiplierFromCustomValue();
}

function renderInterventionExamples(examples) {
  if (!examples?.length) return;

  const fragment = document.createDocumentFragment();
  for (const example of examples) {
    const featureTitle = featuresById.get(example.feature_id)?.title;
    const featureLabel = featureTitle
      ? featureTitle.charAt(0).toUpperCase() + featureTitle.slice(1)
      : `Feature ${example.feature_id}`;
    const button = element("button", "intervention-example-button", "Try");
    button.type = "button";
    button.addEventListener("click", () => tryInterventionExample(example));

    const item = element("div", "intervention-example");
    item.append(
      element("span", "intervention-example-value example-feature", featureLabel),
      element("span", "intervention-example-value example-prompt", `“${example.prompt}”`),
      button,
    );
    fragment.append(item);
  }
  ui.interventionExampleList.replaceChildren(fragment);
  ui.interventionExamples.hidden = false;
}

function selectedFeatureId() {
  const value = ui.featureInput.value;
  return value === "" ? null : Number(value);
}

function selectedFeatureMaximum() {
  const featureId = selectedFeatureId();
  return featureId === null
    ? null
    : featuresById.get(featureId)?.max_activation ?? 0;
}

function updateAmountControl() {
  const multiplier = ui.amountMultiplier.valueAsNumber;
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

ui.amountMultiplier.addEventListener("input", () => {
  amountIsCustom = false;
  updateAmountControl();
});
ui.amountInput.addEventListener("input", () => {
  amountIsCustom = true;
  updateAmountMultiplierFromCustomValue();
});
ui.prompt.addEventListener("input", () => ui.prompt.setCustomValidity(""));
ui.featureInput.addEventListener("change", () => {
  featureSelectControl.trigger.removeAttribute("aria-invalid");
  if (amountIsCustom) updateAmountMultiplierFromCustomValue();
  else updateAmountControl();
});
ui.samplingInputs.forEach(addSynchronizedRange);
ui.samplingSettings.addEventListener("invalid", () => {
  if (!ui.samplingSettings.matches(":popover-open")) {
    ui.samplingSettings.showPopover();
  }
}, true);
playgroundForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!ui.prompt.value.trim()) {
    ui.prompt.setCustomValidity("Enter a prompt to continue.");
    ui.prompt.reportValidity();
    return;
  }

  const featureId = selectedFeatureId();
  if (featureId === null) {
    featureSelectControl.trigger.setAttribute("aria-invalid", "true");
    featureSelectControl.trigger.focus();
    return;
  }

  const request = Object.fromEntries(new FormData(playgroundForm));
  request.feature_id = featureId;
  for (const input of playgroundForm.elements) {
    if (input.type === "number") request[input.name] = input.valueAsNumber;
  }

  const baselineKey = JSON.stringify([
    request.prompt,
    ...ui.samplingInputs.map((input) => request[input.name]),
  ]);
  if (baselineKey !== renderedBaselineKey) {
    renderLoading(ui.baselineOutput, "Generating", coldStartNote);
  }
  renderLoading(ui.intervenedOutput, "Generating", coldStartNote);
  ui.generationResults.hidden = false;

  ui.generateButton.disabled = true;
  ui.generateButton.textContent = "Generating...";
  playgroundForm.querySelector(".playground-error")?.remove();
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
    const message = element("span", "playground-error", `Could not generate: ${error.message}`);
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
  const query = normalizeFeatureQuery(ui.search.value);
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
      || (ui.category.value && feature.category !== ui.category.value)
    ) return false;
    return matchesFeatureQuery(feature.id, feature.title, query);
  });
  const sorter = sorters[ui.sort.value];
  visible.sort(starredFirst((a, b) => (
    reverseFeatureOrder ? -sorter(a, b) : sorter(a, b)
  )));
  return visible;
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

function updateDualRange(range, minimumInput, maximumInput, changedInput) {
  keepRangeOrdered(
    minimumInput,
    maximumInput,
    changedInput,
  );
  const minimumPosition = minimumInput.valueAsNumber;
  const maximumPosition = maximumInput.valueAsNumber;
  range.style.setProperty("--range-start", `${100 * minimumPosition}%`);
  range.style.setProperty("--range-end", `${100 * maximumPosition}%`);
  return [minimumPosition, maximumPosition];
}

function updateActivationCountRange(changedInput) {
  const [minimumPosition, maximumPosition] = updateDualRange(
    ui.activationCountRange,
    ui.minimumActivationCount,
    ui.maximumActivationCount,
    changedInput,
  );
  const selectedMinimum = activationCountAt(minimumPosition);
  const selectedMaximum = activationCountAt(maximumPosition);
  ui.activationCountOutput.value =
    `${selectedMinimum.toLocaleString()}–${selectedMaximum.toLocaleString()}`;
  updateClearFiltersButton();
}

function updateClearFiltersButton() {
  ui.clearFiltersButton.disabled =
    ui.minimumActivationCount.valueAsNumber === 0
    && ui.maximumActivationCount.valueAsNumber === 1
    && !ui.category.value;
}

function resetActivationCountRange() {
  ui.minimumActivationCount.value = ui.minimumActivationCount.min;
  ui.maximumActivationCount.value = ui.maximumActivationCount.max;
  updateActivationCountRange();
}

function renderFeature(feature) {
  const details = element("details", "feature");
  details.id = `feature-${feature.id}`;

  const title = feature.title || `Feature ${feature.id}`;
  const titleElement = element("span", "feature-title");
  titleElement.title = title;
  if (feature.high_quality) titleElement.append(featureStar());
  titleElement.append(element("span", "feature-title-text", title));
  if (feature.category) {
    titleElement.append(
      element("span", "feature-category", feature.category),
    );
  }

  const summary = element("summary");
  summary.append(
    element("span", "feature-id", `#${feature.id}`),
    titleElement,
    element("span", "feature-stat", feature.activation_count.toLocaleString()),
    element("span", "chevron"),
  );
  details.append(summary, element("div", "feature-body"));
  details.addEventListener("toggle", () => {
    if (details.open && !details.dataset.loaded) loadFeature(details, feature);
  });
  return details;
}

function renderPagination(pageCount) {
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
  selectControls.get(ui.pageSelect).updateOptions();
  ui.pageCount.textContent = `of ${pageCount.toLocaleString()}`;
  ui.previousPage.disabled = currentFeaturePage === 1;
  ui.nextPage.disabled = currentFeaturePage === pageCount;
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
  ui.list.replaceChildren(...pageFeatures.map(renderFeature));
  renderPagination(pageCount);
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

async function contextsInRange(featureId, exportedContexts, minimum, maximum) {
  if (isStatic) {
    return {
      contexts: exportedContexts.filter((context) => (
        context.peak_activation >= minimum && context.peak_activation <= maximum
      )),
    };
  }
  const query = new URLSearchParams({ min: minimum, max: maximum });
  return fetchJson(`${featureUrl(featureId)}?${query}`);
}

function contextCountLabel(payload) {
  const shown = payload.contexts.length;
  if (isStatic) return `${shown.toLocaleString()} representative contexts`;
  if (shown === payload.matching_context_count) {
    return `${shown.toLocaleString()} contexts`;
  }
  return `Sampled ${shown.toLocaleString()} of ${payload.matching_context_count.toLocaleString()}`;
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
  const minimumInput = rangeInput(
    "Minimum peak activation",
    DEFAULT_CONTEXT_RANGE_START,
  );
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
    const [minimumFraction, maximumFraction] = updateDualRange(
      selector,
      minimumInput,
      maximumInput,
      changedInput,
    );

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
      const contextPayload = await contextsInRange(
        feature.id,
        exportedContexts,
        minimum,
        maximum,
      );
      if (currentRequest !== requestId) return;
      const contexts = descending
        ? contextPayload.contexts
        : [...contextPayload.contexts].reverse();
      const shown = contexts.length;
      orderLabel.textContent = descending ? "(high to low)" : "(low to high)";
      resultCount.replaceChildren(contextCountLabel(contextPayload), orderLabel);
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
  renderLoading(body, "Loading feature");
  try {
    const payload = await fetchFeature(feature.id);
    body.classList.remove("is-loading");
    body.replaceChildren(
      renderOverview(feature, payload),
      renderDistribution(feature, payload),
    );
    details.dataset.loaded = "true";
  } catch (error) {
    delete details.dataset.loaded;
    body.classList.remove("is-loading");
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
ui.category.addEventListener("change", () => {
  updateClearFiltersButton();
  resetFeaturePage();
});
ui.clearFiltersButton.addEventListener("click", () => {
  ui.category.value = "";
  selectControls.get(ui.category).updateOptions();
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
  if (!event.defaultPrevented && event.key === "Escape" && ui.search.value) {
    ui.search.value = "";
    resetFeaturePage();
    ui.search.focus();
  }
});

async function initialize() {
  try {
    if (!config.interventionUrl) playgroundForm.closest(".playground").hidden = true;
    const summaryUrl = isStatic
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
    initializePlayground(payload.metadata);
    renderInterventionExamples(payload.intervention_examples);
    updateActivationCountRange();
    renderFeatureList();
  } catch (error) {
    ui.count.textContent = "Could not load activations";
    ui.error.textContent = error.message;
    ui.error.hidden = false;
  }
}

initialize();
