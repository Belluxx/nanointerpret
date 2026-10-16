const main = document.getElementById('main');
const tooltip = document.getElementById('tooltip');
const listElement = document.getElementById('list');
const rangeFilter = document.getElementById('range-filter');
const query = document.getElementById('query');

const LIST_PAGE = 200;

const integer = new Intl.NumberFormat('en-US');
const list = { sort: 'density', low: null, high: null, offset: 0, done: false, loading: false, generation: 0 };
const observers = [];
const pendingCharts = [];
let overview = null;
let current = null;
let viewGeneration = 0;

/* DOM helpers: every string is inserted as text, never parsed as HTML. */

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  applyProps(node, props);
  for (const child of children.flat(Infinity)) {
    if (child == null || child === false) continue;
    node.append(child instanceof Node ? child : String(child));
  }
  return node;
}

function svg(tag, props = {}, ...children) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  applyProps(node, props);
  for (const child of children.flat(Infinity)) {
    if (child == null || child === false) continue;
    node.append(child instanceof Node ? child : String(child));
  }
  return node;
}

function applyProps(node, props) {
  for (const [key, value] of Object.entries(props)) {
    if (value == null || value === false) continue;
    if (key === 'class') node.setAttribute('class', value);
    else if (key === 'dataset') Object.assign(node.dataset, value);
    else if (key === 'style') Object.assign(node.style, value);
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? '' : value);
  }
}

async function api(path) {
  const response = await fetch(path);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

/* Formatting */

function compact(value) {
  for (const [scale, suffix] of [[1e9, 'B'], [1e6, 'M'], [1e3, 'k']]) {
    if (value >= scale) return `${Number((value / scale).toPrecision(3))}${suffix}`;
  }
  return String(Number(value.toPrecision(3)));
}

function percent(fraction) {
  const value = fraction * 100;
  if (value === 0) return '0%';
  if (value >= 10) return `${value.toFixed(1)}%`;
  if (value >= 1) return `${value.toFixed(2)}%`;
  return `${Number(value.toPrecision(2))}%`;
}

function axisPercent(fraction) {
  return `${Number((fraction * 100).toPrecision(2))}%`;
}

function signed(value, digits = 2) {
  return `${value < 0 ? '−' : '+'}${Math.abs(value).toFixed(digits)}`;
}

function quoted(text) {
  return JSON.stringify(text);
}

/* Tokens */

function tokenText(text) {
  if (!text) return el('span', { class: 'ws' }, '∅');
  const fragment = document.createDocumentFragment();
  let buffer = '';
  for (const char of text) {
    const mark = char === ' ' ? '·' : char === '\n' ? '⏎' : char === '\t' ? '⇥' : null;
    if (!mark) {
      buffer += char;
      continue;
    }
    if (buffer) fragment.append(buffer);
    buffer = '';
    fragment.append(el('span', { class: 'ws' }, mark));
  }
  if (buffer) fragment.append(buffer);
  return fragment;
}

function tokenChip(text) {
  return el('span', { class: 'token' }, tokenText(text));
}

function exampleToken(example, index, peak, multiline) {
  const text = example.tokens[index];
  const act = example.acts[index];
  const span = el('span', {
    class: 't',
    dataset: { tip: `${act > 0 ? act.toFixed(2) : 'not firing'}\n${quoted(text)}` },
  });
  if (example.specials.has(index)) {
    span.append(el('span', { class: 'special' }, text.replace(/^<\|?|\|?>$/g, '')));
  } else {
    const lines = text.split('\n');
    lines.forEach((line, lineIndex) => {
      if (lineIndex > 0) span.append(el('span', { class: 'nl' }, '⏎'), multiline ? '\n' : '');
      if (line) span.append(line.replace(/\t/g, '⇥'));
    });
  }
  if (act > 0) {
    const alpha = 8 + 82 * Math.min(act / peak, 1);
    span.style.background = `color-mix(in srgb, var(--ink) ${alpha.toFixed(1)}%, transparent)`;
    if (alpha >= 60) span.classList.add('on');
  }
  if (index === example.center) span.classList.add('peak');
  return span;
}

function exampleRow(example, peak) {
  example.specials = new Set(example.special);
  const tokens = example.tokens.map((_, index) => index);
  const row = el('div', { class: 'example', tabindex: 0, role: 'button', 'aria-expanded': 'false' },
    el('span', { class: 'act' }, example.acts[example.center].toFixed(2)),
    el('span', { class: 'left' }, tokens.slice(0, example.center).map((index) => exampleToken(example, index, peak, false))),
    el('span', { class: 'center' }, exampleToken(example, example.center, peak, false)),
    el('span', { class: 'right' }, tokens.slice(example.center + 1).map((index) => exampleToken(example, index, peak, false))),
  );
  let context = null;
  const toggle = () => {
    if (context) {
      context.remove();
      context = null;
    } else {
      context = el('div', { class: 'context' }, tokens.map((index) => exampleToken(example, index, peak, true)));
      row.after(context);
    }
    row.classList.toggle('open', Boolean(context));
    row.setAttribute('aria-expanded', String(Boolean(context)));
  };
  row.addEventListener('click', toggle);
  row.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      toggle();
    }
  });
  return row;
}

/* Tooltip */

function showTip(text, x, y) {
  const [lead, ...rest] = text.split('\n');
  tooltip.replaceChildren(el('b', {}, lead), ...rest.map((line) => el('div', { class: 'dim' }, line)));
  tooltip.hidden = false;
  placeTip(x, y);
}

function placeTip(x, y) {
  const { width, height } = tooltip.getBoundingClientRect();
  const left = Math.min(x + 14, window.innerWidth - width - 8);
  const top = y + 16 + height > window.innerHeight ? y - height - 12 : y + 16;
  tooltip.style.left = `${Math.max(8, left)}px`;
  tooltip.style.top = `${Math.max(8, top)}px`;
}

function hideTip() {
  tooltip.hidden = true;
}

document.addEventListener('pointerover', (event) => {
  const target = event.target.closest?.('[data-tip]');
  if (target) showTip(target.dataset.tip, event.clientX, event.clientY);
});
document.addEventListener('pointermove', (event) => {
  if (!tooltip.hidden) placeTip(event.clientX, event.clientY);
});
document.addEventListener('pointerout', (event) => {
  const target = event.target.closest?.('[data-tip]');
  if (target && !target.contains(event.relatedTarget)) hideTip();
});
document.addEventListener('focusin', (event) => {
  const target = event.target.closest?.('[data-tip]');
  if (!target) return;
  const rect = target.getBoundingClientRect();
  showTip(target.dataset.tip, rect.left, rect.bottom - 8);
});
document.addEventListener('focusout', hideTip);

/* Charts: drawn at the container's pixel width so text stays at its CSS size. */

function chart(height, draw, { clickable = false } = {}) {
  const box = el('div', { class: `chart${clickable ? ' clickable' : ''}` });
  let drawnWidth = 0;
  const redraw = () => {
    const width = Math.floor(box.clientWidth);
    if (!width || width === drawnWidth) return;
    drawnWidth = width;
    const root = svg('svg', { width, height, viewBox: `0 0 ${width} ${height}` });
    draw(root, width, height);
    box.replaceChildren(root);
  };
  pendingCharts.push({ box, redraw });
  return box;
}

function niceTicks(maximum, count = 4) {
  const rough = maximum / count;
  const power = 10 ** Math.floor(Math.log10(rough));
  const step = [1, 2, 2.5, 5, 10].map((factor) => factor * power).find((candidate) => candidate >= rough);
  const ticks = [];
  for (let tick = 0; tick <= maximum + step * 1e-9; tick += step) ticks.push(tick);
  return ticks;
}

function densityChart(histogram, onPick) {
  return chart(260, (root, width, height) => {
    const margin = { left: 36, right: 12, top: 10, bottom: 26 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const edges = histogram.feature_density_log10_bin_edges;
    const counts = histogram.feature_density_bin_counts;
    const total = histogram.total_features;
    // Percent of features per decade, so the wider lowest bin stays comparable (as in the training plot).
    const values = counts.map((count, index) => (100 * count) / total / (edges[index + 1] - edges[index]));
    const low = edges[0];
    const ticks = niceTicks(Math.max(...values) * 1.05, 4);
    const top = ticks[ticks.length - 1];
    const x = (logDensity) => margin.left + ((logDensity - low) / -low) * plotWidth;
    const y = (value) => margin.top + plotHeight - (value / top) * plotHeight;
    for (const tick of ticks) {
      root.append(
        svg('line', { class: tick ? 'grid' : 'axis', x1: margin.left, x2: width - margin.right, y1: y(tick), y2: y(tick) }),
        svg('text', { x: margin.left - 6, y: y(tick) + 3, 'text-anchor': 'end' }, tick),
      );
    }
    for (let decade = Math.ceil(low); decade <= 0; decade++) {
      const anchor = decade === 0 ? 'end' : 'middle';
      root.append(
        svg('line', { class: 'axis', x1: x(decade), x2: x(decade), y1: y(0), y2: y(0) + 4 }),
        svg('text', { x: x(decade), y: height - 8, 'text-anchor': anchor }, axisPercent(10 ** decade)),
      );
    }
    const hover = svg('rect', { class: 'hover', y: margin.top, height: plotHeight, visibility: 'hidden' });
    root.prepend(hover);
    let path = `M${x(edges[0])},${y(0)}`;
    values.forEach((value, index) => {
      path += `V${y(value).toFixed(1)}H${x(edges[index + 1]).toFixed(1)}`;
    });
    path += `V${y(0)}`;
    root.append(svg('path', { class: 'line', d: path }));
    values.forEach((value, index) => {
      const left = x(edges[index]);
      const right = x(edges[index + 1]);
      const range = index === 0 ? '1–9 fires' : `${percent(10 ** edges[index])}–${percent(10 ** edges[index + 1])}`;
      const hit = svg('rect', {
        class: 'hit', x: left, width: right - left, y: margin.top, height: plotHeight,
        dataset: { tip: `${integer.format(counts[index])} features\ndensity ${range}\nclick to list them` },
      });
      hit.addEventListener('pointerenter', () => {
        hover.setAttribute('x', left);
        hover.setAttribute('width', right - left);
        hover.setAttribute('visibility', 'visible');
      });
      hit.addEventListener('pointerleave', () => hover.setAttribute('visibility', 'hidden'));
      hit.addEventListener('click', () => onPick(edges[index], edges[index + 1], range));
      root.append(hit);
    });
  }, { clickable: true });
}

/* Shared view pieces */

function card(title, meta, ...body) {
  return el('section', { class: 'card' },
    el('header', { class: 'card-head' }, el('h2', {}, title), meta ? el('span', { class: 'meta' }, meta) : null),
    body);
}

function barList(rows, { value, format, tip, href }) {
  const largest = Math.max(...rows.map((row) => Math.abs(value(row))), 1e-12);
  return el('div', { class: 'bars' }, rows.map((row) => el(href ? 'a' : 'div',
    { class: 'bar-row', href: href?.(row), dataset: { tip: tip(row) } },
    tokenChip(row.token),
    el('span', { class: 'track' }, el('span', {
      class: 'fill',
      style: { width: `${((Math.abs(value(row)) / largest) * 100).toFixed(1)}%` },
    })),
    el('span', { class: 'num' }, format(row)))));
}

function tokenHref(token) {
  return token.trim() ? `#/token/${encodeURIComponent(token.trim())}` : null;
}

function button(label, onclick, title) {
  return el('button', { class: 'button', type: 'button', onclick, title }, label);
}

/* Views */

async function show(path, build) {
  const generation = ++viewGeneration;
  main.classList.add('loading');
  try {
    const data = await api(path);
    if (generation !== viewGeneration) return;
    render(build(data));
  } catch (error) {
    if (generation !== viewGeneration) return;
    render(el('div', { class: 'notice' }, error.message));
  } finally {
    if (generation === viewGeneration) main.classList.remove('loading');
  }
}

function render(view) {
  hideTip();
  observers.splice(0).forEach((observer) => observer.disconnect());
  main.replaceChildren(view);
  // Draw once the view is in the DOM so charts know their width, then follow resizes.
  for (const { box, redraw } of pendingCharts.splice(0)) {
    redraw();
    const observer = new ResizeObserver(redraw);
    observer.observe(box);
    observers.push(observer);
  }
  window.scrollTo({ top: 0 });
}

function overviewView() {
  return el('div', {},
    el('div', { class: 'page-head' }, el('div', {},
      el('h1', { class: 'text' }, overview.run),
      el('div', { class: 'sub muted' }, [
        overview.model_id, `layer ${overview.layer_index}`, `${integer.format(overview.d_sae)} features`,
        `k ${overview.k}`, `${compact(overview.tokens)} tokens recorded`,
      ].join(' · ')))),
    card('Activation frequency', '% of features per decade · click a bin to list it',
      el('div', { class: 'card-body' }, densityChart(overview.density_histogram, pickDensity))));
}

function featureView(data) {
  const peak = data.max;
  const head = el('div', { class: 'page-head' },
    el('div', {},
      el('h1', {}, el('span', { class: 'hash' }, '#'), data.id),
      el('div', { class: 'sub muted' }, data.fires
        ? `${percent(data.density)} of tokens · ${integer.format(data.fires)} fires · max ${peak.toFixed(2)}`
        : 'Never fires in the recorded tokens')),
    el('nav', {},
      button('←', () => goFeature(data.id - 1), 'Previous feature (←)'),
      button('→', () => goFeature(data.id + 1), 'Next feature (→)')));

  const promotes = card('Promotes', null, el('div', { class: 'card-body' }, barList(data.promotes, {
    value: (row) => row.logit, format: (row) => signed(row.logit), href: (row) => tokenHref(row.token),
    tip: (row) => `${signed(row.logit)} logit\n${quoted(row.token)}, centered over the vocabulary`,
  })));
  if (!data.fires) return el('div', {}, head, el('div', { class: 'columns' }, el('div'), el('div', { class: 'side' }, promotes)));

  const section = (label, meta, examples) => [
    el('div', { class: 'range-head' }, el('span', {}, label), meta ? el('span', { class: 'meta' }, meta) : null),
    examples.length
      ? examples.map((example) => exampleRow(example, peak))
      : el('div', { class: 'empty' }, 'No fires in this range.'),
  ];
  const examples = el('section', { class: 'card examples' },
    section('Top activations', null, data.top),
    data.ranges.map((range) => section(
      `${range.low * 100}–${range.high * 100}% of max`, `${percent(range.fires / data.fires)} of fires`, range.examples)));

  const firesOn = card('Fires on', null, el('div', { class: 'card-body' }, barList(data.fires_on, {
    value: (row) => row.share, format: (row) => percent(row.share), href: (row) => tokenHref(row.token),
    tip: (row) => `${percent(row.share)} of this feature's fires\n${quoted(row.token)}: fires on ${percent(row.precision)} of its occurrences`,
  })));
  return el('div', {}, head, el('div', { class: 'columns' }, examples, el('div', { class: 'side' }, firesOn, promotes)));
}

function tokenView(data) {
  const occurrences = data.matches.reduce((total, match) => total + match.occurrences, 0);
  const head = el('div', { class: 'page-head' }, el('div', {},
    el('h1', { class: 'text' }, quoted(data.query)),
    data.matches.length
      ? el('div', { class: 'sub' }, data.matches.map((match) => tokenChip(match.token)),
        el('span', { class: 'muted' }, `${integer.format(occurrences)} occurrences`))
      : null));

  if (!data.matches.length) {
    return el('div', {}, head, el('div', { class: 'notice' },
      data.pieces.length ? `No single token reads ${quoted(data.query.trim())}. It tokenizes into these pieces:` : 'Type a token to search.',
      el('div', { class: 'chips' }, data.pieces.map((piece) => el('a', { class: 'chip', href: tokenHref(piece) }, tokenText(piece))))));
  }

  const largest = Math.max(...data.features.map((row) => row.mean), 1e-12);
  return el('div', {}, head, el('section', { class: 'card' }, el('div', { class: 'table-wrap' }, el('table', { class: 'table' },
    el('thead', {}, el('tr', {},
      el('th', {}, 'Feature'), el('th', {}, 'Peaks on'),
      el('th', { class: 'num', dataset: { tip: 'Mean activation\nover sampled occurrences of this token' } }, 'Mean act.'),
      el('th', { class: 'num', dataset: { tip: 'Share of the feature\'s fires\nthat land on this token' } }, 'Of its fires'))),
    el('tbody', {}, data.features.map((row) => el('tr', { onclick: () => goFeature(row.id) },
      el('td', {}, el('a', { class: 'id', href: `#/feature/${row.id}` }, `#${row.id}`)),
      el('td', {}, row.peak_token == null ? '—' : tokenChip(row.peak_token)),
      el('td', { class: 'num' }, el('div', { class: 'cell-bar' },
        el('span', { style: { width: `${((row.mean / largest) * 60).toFixed(1)}px` } }), row.mean.toFixed(2))),
      el('td', { class: 'num' }, percent(row.share)))))))));
}

/* Sidebar list */

function pickDensity(low, high, label) {
  list.low = low;
  list.high = high;
  rangeFilter.replaceChildren(el('span', {}, `Density ${label}`), el('button', { type: 'button', onclick: clearDensity }, 'Clear'));
  rangeFilter.hidden = false;
  resetList();
}

function clearDensity() {
  list.low = null;
  list.high = null;
  rangeFilter.hidden = true;
  resetList();
}

const sentinel = el('div', { class: 'list-end' });
new IntersectionObserver((entries) => {
  if (entries.some((entry) => entry.isIntersecting)) loadMore();
}, { root: listElement, rootMargin: '400px' }).observe(sentinel);

function resetList() {
  list.offset = 0;
  list.done = false;
  list.loading = false;
  list.generation += 1;
  listElement.scrollTop = 0;
  listElement.replaceChildren(sentinel);
  loadMore();
}

async function loadMore() {
  if (list.loading || list.done) return;
  list.loading = true;
  const generation = list.generation;
  const params = new URLSearchParams({ sort: list.sort, offset: list.offset, limit: LIST_PAGE });
  if (list.low != null) params.set('low', list.low);
  if (list.high != null) params.set('high', list.high);
  try {
    const data = await api(`/api/features?${params}`);
    if (generation !== list.generation) return;
    sentinel.before(...data.items.map(listRow));
    list.offset += data.items.length;
    list.done = list.offset >= data.total;
    sentinel.textContent = list.done ? (data.total ? '' : 'No features match') : 'Loading…';
  } catch (error) {
    sentinel.textContent = error.message;
  } finally {
    if (generation === list.generation) list.loading = false;
  }
  if (generation === list.generation && !list.done && sentinel.getBoundingClientRect().top < window.innerHeight + 400) loadMore();
}

function listRow(item) {
  return el('a', {
    class: `row${item.id === current ? ' active' : ''}`, href: `#/feature/${item.id}`, role: 'listitem', dataset: { id: item.id },
  },
  el('span', { class: 'id' }, `#${item.id}`),
  el('span', { class: 'peak' }, item.peak_token == null ? el('span', { class: 'muted' }, '—') : tokenText(item.peak_token)),
  el('span', { class: 'density' }, percent(item.density)));
}

function markActive() {
  for (const row of listElement.querySelectorAll('.row')) {
    const active = Number(row.dataset.id) === current;
    row.classList.toggle('active', active);
    if (active) row.scrollIntoView({ block: 'nearest' });
  }
}

/* Routing and controls */

function goFeature(id) {
  location.hash = `#/feature/${(id + overview.d_sae) % overview.d_sae}`;
}

async function goRandom() {
  try {
    const { id } = await api('/api/random');
    goFeature(id);
  } catch (error) {
    render(el('div', { class: 'notice' }, error.message));
  }
}

function route() {
  const [view, argument] = location.hash.replace(/^#\/?/, '').split('/');
  current = view === 'feature' ? Number(argument) : null;
  markActive();
  if (view === 'feature' && Number.isInteger(current)) {
    document.title = `#${current} · Feature Explorer`;
    show(`/api/feature/${current}`, featureView);
  } else if (view === 'token' && argument) {
    const text = decodeURIComponent(argument);
    document.title = `${quoted(text)} · Feature Explorer`;
    show(`/api/token?q=${encodeURIComponent(text)}`, tokenView);
  } else {
    document.title = 'Feature Explorer';
    viewGeneration += 1;
    main.classList.remove('loading');
    render(overviewView());
  }
}

document.getElementById('search').addEventListener('submit', (event) => {
  event.preventDefault();
  const text = query.value;
  if (/^\s*#?\d+\s*$/.test(text)) goFeature(Number(text.replace(/[#\s]/g, '')));
  else if (text.trim()) location.hash = `#/token/${encodeURIComponent(text.trim())}`;
  query.blur();
});

document.getElementById('random').addEventListener('click', goRandom);

document.getElementById('sort').addEventListener('change', (event) => {
  list.sort = event.target.value;
  resetList();
});

document.addEventListener('keydown', (event) => {
  if (event.metaKey || event.ctrlKey || event.altKey) return;
  if (event.target.closest('input, select, textarea')) {
    if (event.key === 'Escape') event.target.blur();
    return;
  }
  if (event.key === '/') {
    event.preventDefault();
    query.focus();
    query.select();
  } else if (event.key === 'r') {
    goRandom();
  } else if (current != null && event.key === 'ArrowLeft') {
    goFeature(current - 1);
  } else if (current != null && event.key === 'ArrowRight') {
    goFeature(current + 1);
  }
});

window.addEventListener('hashchange', route);

overview = await api('/api/overview');
resetList();
route();
