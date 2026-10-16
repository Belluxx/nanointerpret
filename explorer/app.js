const main = document.getElementById('main');
const listElement = document.getElementById('list');
const tooltip = document.getElementById('tooltip');
const query = document.getElementById('query');
const sort = document.getElementById('sort');
const rangeButton = document.getElementById('range');

const LIST_PAGE = 200;
const integer = new Intl.NumberFormat('en-US');
const list = { offset: 0, done: false, loading: false, generation: 0, range: null };
let overview;
let current = null;
let viewGeneration = 0;

/* DOM: strings are always inserted as text, never parsed as HTML. */

function create(namespace, tag, props, children) {
  const node = namespace ? document.createElementNS(namespace, tag) : document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value == null) continue;
    if (key === 'dataset' || key === 'style') Object.assign(node[key], value);
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  node.append(...children.flat(Infinity).filter((child) => child != null && child !== false));
  return node;
}

const el = (tag, props = {}, ...children) => create(null, tag, props, children);
const svg = (tag, props = {}, ...children) => create('http://www.w3.org/2000/svg', tag, props, children);

async function api(path) {
  const response = await fetch(path);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error);
  return data;
}

function percent(fraction) {
  const value = fraction * 100;
  return `${value >= 1 ? value.toFixed(value >= 10 ? 1 : 2) : Number(value.toPrecision(2))}%`;
}

const quoted = (text) => JSON.stringify(text);
const featureHref = (id) => `#/feature/${(id + overview.d_sae) % overview.d_sae}`;
const tokenHref = (token) => (token.trim() ? `#/token/${encodeURIComponent(token.trim())}` : null);

/* Tokens: whitespace is made visible in token labels and kept real in example text. */

function tokenLabel(text, className = 'token') {
  return el('span', { class: className }, text.split(/([ \n\t])/).map((part) => {
    const mark = { ' ': '·', '\n': '⏎', '\t': '⇥' }[part];
    return mark ? el('span', { class: 'ws' }, mark) : part;
  }));
}

function exampleToken(example, index, peak, multiline) {
  const text = example.tokens[index];
  const act = example.acts[index];
  const alpha = act > 0 ? 8 + 82 * Math.min(act / peak, 1) : 0;
  const content = example.special.includes(index)
    ? el('span', { class: 'special' }, text.replace(/^<\|?|\|?>$/g, ''))
    : text.split('\n').flatMap((line, lineIndex) => (lineIndex ? [el('span', { class: 'nl' }, multiline ? '⏎\n' : '⏎'), line] : [line]));
  return el('span', {
    class: `t${index === example.center ? ' peak' : ''}${alpha >= 60 ? ' on' : ''}`,
    style: alpha ? { background: `color-mix(in srgb, var(--ink) ${alpha.toFixed(1)}%, transparent)` } : null,
    dataset: { tip: `${act > 0 ? act.toFixed(2) : 'not firing'}\n${quoted(text)}` },
  }, content);
}

function exampleRow(example, peak) {
  const tokens = (from, to) => example.tokens.slice(from, to).map((_, offset) => exampleToken(example, from + offset, peak, false));
  return el('details', { class: 'example' },
    el('summary', {}, el('div', { class: 'kwic' },
      el('span', { class: 'act' }, example.acts[example.center].toFixed(2)),
      el('span', { class: 'left' }, tokens(0, example.center)),
      exampleToken(example, example.center, peak, false),
      el('span', { class: 'right' }, tokens(example.center + 1)))),
    el('div', { class: 'context' }, example.tokens.map((_, index) => exampleToken(example, index, peak, true))));
}

/* Tooltip: any element with data-tip, first line bold. */

function moveTip(x, y) {
  const { width, height } = tooltip.getBoundingClientRect();
  tooltip.style.left = `${Math.max(8, Math.min(x + 14, innerWidth - width - 8))}px`;
  tooltip.style.top = `${y + 16 + height > innerHeight ? y - height - 12 : y + 16}px`;
}

document.addEventListener('pointerover', (event) => {
  const target = event.target.closest('[data-tip]');
  tooltip.hidden = !target;
  if (!target) return;
  const [lead, ...rest] = target.dataset.tip.split('\n');
  tooltip.replaceChildren(el('b', {}, lead), ...rest.map((line) => el('div', {}, line)));
  moveTip(event.clientX, event.clientY);
});
document.addEventListener('pointermove', (event) => tooltip.hidden || moveTip(event.clientX, event.clientY));
document.documentElement.addEventListener('pointerleave', () => { tooltip.hidden = true; });

/* Activation frequency chart, redrawn at the container's pixel width so text keeps its CSS size. */

function densityChart(histogram) {
  const box = el('div', { class: 'chart' });
  new ResizeObserver(() => box.replaceChildren(drawDensity(histogram, box.clientWidth, 240))).observe(box);
  return box;
}

function drawDensity({ feature_density_log10_bin_edges: edges, feature_density_bin_counts: counts, total_features: total }, width, height) {
  const margin = { left: 30, right: 4, top: 6, bottom: 24 };
  // Percent of features per decade, so the wider lowest bin stays comparable (as in the training plot).
  const values = counts.map((count, index) => (100 * count) / total / (edges[index + 1] - edges[index]));
  const rough = Math.max(...values) / 5;
  const power = 10 ** Math.floor(Math.log10(rough));
  const step = [1, 2, 5, 10].map((factor) => factor * power).find((candidate) => candidate >= rough);
  const top = Math.ceil(Math.max(...values) / step) * step;
  const x = (logDensity) => margin.left + (1 - logDensity / edges[0]) * (width - margin.left - margin.right);
  const y = (value) => margin.top + (1 - value / top) * (height - margin.top - margin.bottom);

  const root = svg('svg', { width, height });
  counts.forEach((count, index) => {
    const range = index ? `${percent(10 ** edges[index])}–${percent(10 ** edges[index + 1])}` : '1–9 fires';
    root.append(svg('rect', {
      x: x(edges[index]), width: x(edges[index + 1]) - x(edges[index]), y: margin.top, height: y(0) - margin.top,
      dataset: { tip: `${integer.format(count)} features\ndensity ${range}\nclick to list them` },
      onclick: () => filterDensity(edges[index], edges[index + 1], range),
    }));
  });
  for (let tick = 0; tick <= top; tick += step) {
    root.append(
      svg('line', { class: tick ? 'grid' : 'axis', x1: margin.left, x2: width - margin.right, y1: y(tick), y2: y(tick) }),
      svg('text', { x: margin.left - 6, y: y(tick) + 3, 'text-anchor': 'end' }, tick),
    );
  }
  for (let decade = Math.ceil(edges[0]); decade <= 0; decade++) {
    const label = `${Number((100 * 10 ** decade).toPrecision(1))}%`;
    root.append(svg('text', { x: x(decade), y: height - 6, 'text-anchor': decade ? 'middle' : 'end' }, label));
  }
  const steps = values.map((value, index) => `V${y(value)}H${x(edges[index + 1])}`).join('');
  root.append(svg('path', { d: `M${x(edges[0])},${y(0)}${steps}V${y(0)}` }));
  return root;
}

/* Views */

function card(title, ...body) {
  return el('section', { class: 'card' }, el('h2', {}, title), body);
}

function barList(rows, value, format, tip) {
  const largest = Math.max(...rows.map(value));
  return el('div', { class: 'bars' }, rows.map((row) => el('a', { href: tokenHref(row.token), dataset: { tip: tip(row) } },
    tokenLabel(row.token),
    el('span', { class: 'track' }, el('span', { style: { width: `${(100 * value(row)) / largest}%` } })),
    el('span', { class: 'num' }, format(row)))));
}

function pageHead(title, subtitle, ...extra) {
  return el('div', { class: 'page-head' }, el('div', {}, title, el('p', {}, subtitle)), extra);
}

function overviewView() {
  const facts = [overview.model_id, `layer ${overview.layer_index}`, `${integer.format(overview.d_sae)} features`, `k ${overview.k}`,
    `${integer.format(overview.tokens)} tokens recorded`];
  return el('div', {},
    pageHead(el('h1', { class: 'text' }, overview.run), facts.join(' · ')),
    card('Activation frequency, % of features per decade', densityChart(overview.density_histogram)));
}

function featureView(data) {
  const sections = [['Top activations', null, data.top], ...data.ranges.map((range) => [
    `${range.low * 100}–${range.high * 100}% of max`, `${percent(range.share)} of fires`, range.examples])];
  return el('div', {},
    pageHead(
      el('h1', {}, el('span', { class: 'hash' }, '#'), data.id),
      data.fires ? `${percent(data.density)} of tokens · ${integer.format(data.fires)} fires · max ${data.max.toFixed(2)}` : 'Never fires in the recorded tokens',
      el('nav', {},
        el('a', { class: 'button', href: featureHref(data.id - 1), title: 'Previous feature (←)' }, '←'),
        el('a', { class: 'button', href: featureHref(data.id + 1), title: 'Next feature (→)' }, '→'))),
    el('div', { class: 'columns' },
      data.fires > 0 && el('section', { class: 'card' }, sections.map(([title, meta, examples]) => [
        el('h2', {}, title, meta && el('span', {}, meta)), examples.map((example) => exampleRow(example, data.max))])),
      el('div', { class: 'side' },
        data.fires > 0 && card('Fires on', barList(data.fires_on, (row) => row.share, (row) => percent(row.share),
          (row) => `${percent(row.share)} of this feature's fires\n${quoted(row.token)}: fires on ${percent(row.precision)} of its occurrences`)),
        card('Promotes', barList(data.promotes, (row) => row.logit, (row) => `+${row.logit.toFixed(2)}`,
          (row) => `+${row.logit.toFixed(2)} logit\n${quoted(row.token)}, centered over the vocabulary`)))));
}

function tokenView(text, { matches, pieces, features }) {
  if (!matches.length) {
    return pageHead(el('h1', { class: 'text' }, quoted(text)), ['No single token matches. Its pieces:',
      pieces.map((piece) => el('a', { href: tokenHref(piece) }, tokenLabel(piece)))]);
  }
  const occurrences = matches.reduce((sum, match) => sum + match.occurrences, 0);
  return el('div', {},
    pageHead(el('h1', { class: 'text' }, quoted(text)), [matches.map((match) => tokenLabel(match.token)), `${integer.format(occurrences)} occurrences`]),
    el('section', { class: 'card scroll' }, el('table', {},
      el('thead', {}, el('tr', {}, ['Feature', 'Peaks on', 'Mean act.', 'Of its fires'].map((label) => el('th', {}, label)))),
      el('tbody', {}, features.map((row) => el('tr', { onclick: () => { location.hash = featureHref(row.id); } },
        el('td', {}, el('a', { href: featureHref(row.id) }, `#${row.id}`)),
        el('td', {}, tokenLabel(row.peak_token)),
        el('td', {}, row.mean.toFixed(2)),
        el('td', {}, percent(row.share))))))));
}

async function show(path, build) {
  const generation = ++viewGeneration;
  main.classList.add('loading');
  let view;
  try {
    view = build(path && await api(path));
  } catch (error) {
    view = el('p', { class: 'notice' }, error.message);
  }
  if (generation !== viewGeneration) return;
  main.classList.remove('loading');
  tooltip.hidden = true;
  main.replaceChildren(view);
  scrollTo(0, 0);
}

function route() {
  const [view, argument] = location.hash.slice(2).split('/');
  current = view === 'feature' ? Number(argument) : null;
  for (const row of listElement.children) row.classList.toggle('active', Number(row.dataset.id) === current);
  listElement.querySelector('.active')?.scrollIntoView({ block: 'nearest' });
  if (view === 'feature') {
    document.title = `#${argument} · Feature Explorer`;
    show(`/api/feature/${argument}`, featureView);
  } else if (view === 'token') {
    const text = decodeURIComponent(argument);
    document.title = `${quoted(text)} · Feature Explorer`;
    show(`/api/token?q=${argument}`, (data) => tokenView(text, data));
  } else {
    document.title = 'Feature Explorer';
    show(null, overviewView);
  }
}

/* Sidebar list, loaded a page at a time as it scrolls. */

const sentinel = el('div');
new IntersectionObserver(([entry]) => entry.isIntersecting && loadMore(), { root: listElement, rootMargin: '400px' }).observe(sentinel);

function filterDensity(low, high, label) {
  list.range = low == null ? null : { low, high };
  rangeButton.textContent = `Density ${label} ×`;
  rangeButton.hidden = !list.range;
  resetList();
}

function resetList() {
  Object.assign(list, { offset: 0, done: false, loading: false, generation: list.generation + 1 });
  listElement.replaceChildren(sentinel);
  loadMore();
}

async function loadMore() {
  if (list.loading || list.done) return;
  const generation = list.generation;
  list.loading = true;
  const range = list.range ? `&low=${list.range.low}&high=${list.range.high}` : '';
  const rows = await api(`/api/features?sort=${sort.value}&offset=${list.offset}&limit=${LIST_PAGE}${range}`);
  if (generation !== list.generation) return;
  sentinel.before(...rows.map((row) => el('a', { class: row.id === current ? 'row active' : 'row', href: featureHref(row.id), dataset: { id: row.id } },
    el('span', {}, `#${row.id}`),
    row.peak_token == null ? el('span') : tokenLabel(row.peak_token, 'peak'),
    el('span', { class: 'density' }, percent(row.density)))));
  Object.assign(list, { offset: list.offset + rows.length, done: rows.length < LIST_PAGE, loading: false });
}

/* Controls */

async function goRandom() {
  location.hash = featureHref((await api('/api/random')).id);
}

document.getElementById('search').addEventListener('submit', (event) => {
  event.preventDefault();
  const text = query.value.trim();
  if (text) location.hash = /^#?\d+$/.test(text) ? featureHref(Number(text.replace('#', ''))) : tokenHref(text);
  query.blur();
});

document.addEventListener('keydown', (event) => {
  if (event.metaKey || event.ctrlKey || event.altKey || event.target.closest('input, select')) return;
  const actions = {
    '/': () => query.focus(),
    r: goRandom,
    ArrowLeft: () => current != null && (location.hash = featureHref(current - 1)),
    ArrowRight: () => current != null && (location.hash = featureHref(current + 1)),
  };
  if (!actions[event.key]) return;
  event.preventDefault();
  actions[event.key]();
});

document.getElementById('random').addEventListener('click', goRandom);
sort.addEventListener('change', resetList);
rangeButton.addEventListener('click', () => filterDensity(null));
window.addEventListener('hashchange', route);

overview = await api('/api/overview');
resetList();
route();
