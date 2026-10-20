const main = document.getElementById('main');
const listElement = document.getElementById('list');
const tooltip = document.getElementById('tooltip');
const query = document.getElementById('query');
const sort = document.getElementById('sort');
const rangeButton = document.getElementById('range');

const LIST_PAGE = 200;
const integer = new Intl.NumberFormat('en-US');
const list = { offset: 0, done: false, loading: false, generation: 0, range: null, group: null };
const GROUPS = ['Trusted', 'Low confidence', 'Unnamed'];
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
const icon = (path) => svg('svg', { class: 'icon', viewBox: '0 0 16 16', 'aria-hidden': 'true' }, svg('path', { d: path }));
const kbd = (key) => el('kbd', {}, key);
const track = (fraction) => el('span', { class: 'track' }, el('span', { style: { width: `${100 * fraction}%` } }));

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
  const alpha = act > 0 ? 6 + 72 * Math.min(act / peak, 1) : 0;
  const content = example.special.includes(index)
    ? el('span', { class: 'special' }, text.replace(/^<\|?|\|?>$/g, ''))
    : text.split('\n').flatMap((line, lineIndex) => (lineIndex ? [el('span', { class: 'nl' }, multiline ? '⏎\n' : '⏎'), line] : [line]));
  return el('span', {
    class: `t${index === example.center ? ' peak' : ''}${alpha >= 55 ? ' on' : ''}`,
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

/* Feature density: the overview histogram and a sparkline of it. */

// Percent of features per decade, so the wider lowest bin stays comparable (as in the training plot).
const perDecade = ({ feature_density_log10_bin_edges: edges, feature_density_bin_counts: counts, total_features: total }) =>
  counts.map((count, index) => (100 * count) / total / (edges[index + 1] - edges[index]));

function densityChart(histogram) {
  const box = el('div', { class: 'chart' });
  new ResizeObserver(() => box.replaceChildren(drawDensity(histogram, box.clientWidth, 280))).observe(box);
  return box;
}

function drawDensity(histogram, width, height) {
  const { feature_density_log10_bin_edges: edges, feature_density_bin_counts: counts } = histogram;
  const margin = { left: 30, right: 4, top: 6, bottom: 24 };
  const values = perDecade(histogram);
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
      class: list.range?.low === edges[index] ? 'selected' : null,
      x: x(edges[index]), width: x(edges[index + 1]) - x(edges[index]), y: margin.top, height: y(0) - margin.top,
      dataset: { low: edges[index], tip: `${integer.format(count)} features\ndensity ${range}\nclick to list them` },
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

function densitySpark(density) {
  const { feature_density_log10_bin_edges: edges, feature_density_bin_counts: counts, total_features: total } = overview.density_histogram;
  const [width, height] = [120, 30];
  const values = perDecade(overview.density_histogram);
  const x = (logDensity) => (1 - logDensity / edges[0]) * width;
  const y = (value) => height * (1 - value / Math.max(...values));
  const steps = values.map((value, index) => `V${y(value)}H${x(edges[index + 1])}`).join('');
  // The feature's whole bin counts as denser, so the rank is at the histogram's resolution.
  const denser = counts.reduce((sum, count, index) => sum + (edges[index + 1] > Math.log10(density) ? count : 0), 0);
  const marker = x(Math.log10(density));
  return svg('svg', { class: 'spark', width, height, dataset: { tip: `top ${percent(denser / total)} by density\nof all ${integer.format(total)} features` } },
    svg('path', { d: `M0,${height}${steps}V${height}Z` }),
    svg('line', { class: 'axis', x1: 0, x2: width, y1: height, y2: height }),
    svg('line', { x1: marker, x2: marker, y1: 6, y2: height }),
    svg('rect', { x: marker - 2.5, y: 3.5, width: 5, height: 5 }));
}

/* Views */

function card(title, meta, ...body) {
  return el('section', { class: 'card' }, el('h2', {}, title, meta && el('span', { class: 'meta' }, meta)), body);
}

function barList(rows, value, format, tip) {
  const largest = Math.max(...rows.map(value));
  return el('div', { class: 'bars' }, rows.map((row) => el('a', { href: tokenHref(row.token), dataset: { tip: tip(row) } },
    tokenLabel(row.token),
    track(value(row) / largest),
    el('span', { class: 'num' }, format(row)))));
}

function pageHead(eyebrow, title, ...extra) {
  return el('div', { class: 'page-head' }, el('div', {}, el('p', { class: 'eyebrow' }, eyebrow), title), extra);
}

function stats(items) {
  return el('dl', { class: 'stats' }, items.map(([label, ...value]) => el('div', {}, el('dt', {}, label), el('dd', {}, value))));
}

// Four rising bars, with the one for this quarter of the max activation filled in.
function quartile(low) {
  return el('span', { class: 'quartile' }, [0, 0.25, 0.5, 0.75].map((edge) => el('span', { class: edge === low ? 'on' : null })));
}

function overviewView() {
  return el('div', {},
    pageHead('Run', el('h1', { class: 'text' }, overview.run)),
    stats([['Model', overview.model_id], ['Layer', overview.layer_index], ['Features', integer.format(overview.d_sae)], ['k', overview.k],
      ['Tokens recorded', integer.format(overview.tokens)]]),
    card('Activation frequency', '% of features per decade of density · click a bin to list it', densityChart(overview.density_histogram)),
    el('p', { class: 'hint' },
      el('span', {}, kbd('/'), 'Search'),
      el('span', {}, kbd('←'), kbd('→'), 'Step through features')));
}

const share = (fraction) => `${Math.round(100 * fraction)}%`;
// Rare features can lack test examples of some kind.
const rate = (fraction) => (fraction == null ? '–' : share(fraction));

function checkStats({ detection, pairs }) {
  const says = detection.says_fires;
  const passed = pairs.filter((pair) => pair.passed).length;
  return [
    ['Detection', el('span', { dataset: { tip: [
      `${share(detection.balanced_accuracy)} balanced accuracy`,
      'A judge given only the name says whether held-out marked tokens fire.',
      `Says yes to activations in the top 5%: ${rate(says.q3)}, 80–95%: ${rate(says.q2)}, 50–80%: ${rate(says.q1)}, bottom half: ${rate(says.q0)}`,
      `Says yes to silent tokens: random ${rate(says.random)}, same token ${rate(says.same_token)}, closest feature's ${rate(says.neighbor)}`,
    ].join('\n') } }, share(detection.balanced_accuracy))],
    ['Generated tests', el('span', { dataset: { tip: `${passed} of ${pairs.length} pairs pass\nTexts written from the name, run through the model:\nthe contrast must reach under half the activation of the text meant to fire` } },
      `${passed} / ${pairs.length}`)],
  ];
}

function confidenceTag({ detection, pairs }) {
  const passed = pairs.filter((pair) => pair.passed).length;
  const failed = [
    detection.balanced_accuracy < overview.trusted_detection && `Detection ${share(detection.balanced_accuracy)}, needs ${share(overview.trusted_detection)}`,
    passed < overview.trusted_pairs && `${passed} of ${pairs.length} generated tests pass, needs ${overview.trusted_pairs}`,
  ];
  return el('span', { class: 'tag', dataset: { tip: ['Low confidence', ...failed.filter(Boolean)].join('\n') } }, 'Low confidence');
}

function testsCard(pairs) {
  const line = (text, activation) => el('div', { class: 'test-line' }, el('span', { class: 'num' }, share(activation)), el('span', {}, text));
  return card('Generated tests', 'activation / max', el('div', { class: 'tests' }, pairs.map((pair) =>
    el('div', { class: pair.passed ? 'test' : 'test failed' },
      line(pair.fires, pair.fires_activation), line(pair.silent, pair.silent_activation)))));
}

function featureView(data) {
  const sections = [['Top activations', 'one per context', data.top], ...data.ranges.map((range) => [
    [quartile(range.low), `${range.low * 100}–${range.high * 100}% of max`], `${percent(range.share)} of fires`, range.examples])];
  const named = data.interpretation;
  return el('div', {},
    pageHead('Feature', el('h1', {}, el('span', { class: 'hash' }, '#'), data.id),
      el('nav', { class: 'stepper' },
        el('a', { class: 'button', href: featureHref(data.id - 1), title: 'Previous feature (←)' }, icon('M10 3.5 5.5 8l4.5 4.5')),
        el('a', { class: 'button', href: featureHref(data.id + 1), title: 'Next feature (→)' }, icon('M6 3.5 10.5 8 6 12.5')))),
    named && el('div', { class: 'naming' },
      el('p', { class: 'name' }, named.title, !data.trusted && confidenceTag(named)),
      el('p', { class: 'description' }, named.description)),
    data.fires
      ? stats([['Density', percent(data.density), densitySpark(data.density)], ['Fires', integer.format(data.fires)],
        ['Max activation', data.max.toFixed(2)], ['Peaks on', tokenLabel(data.peak_token)], ...(named ? checkStats(named) : [])])
      : el('p', { class: 'lede' }, 'Never fires in the recorded tokens'),
    el('div', { class: 'columns' },
      data.fires > 0 && el('div', { class: 'stack' },
        sections.map(([title, meta, examples]) => card(title, meta, examples.map((example) => exampleRow(example, data.max))))),
      el('div', { class: 'stack side' },
        named && testsCard(named.pairs),
        named && card('Steers toward', `${named.steering.strength}× max · KL ${named.steering.kl.toFixed(2)}`,
          el('div', { class: 'chips' }, named.steering.tokens.map((token) => el('a', { href: tokenHref(token) }, tokenLabel(token))))),
        data.fires > 0 && card('Fires on', 'share of fires', barList(data.fires_on, (row) => row.share, (row) => percent(row.share),
          (row) => `${percent(row.share)} of this feature's fires\n${quoted(row.token)}: fires on ${percent(row.precision)} of its occurrences`)),
        card('Promotes', 'logit effect', barList(data.promotes, (row) => row.logit, (row) => `+${row.logit.toFixed(2)}`,
          (row) => `+${row.logit.toFixed(2)} logit\n${quoted(row.token)}, centered over the vocabulary`)))));
}

function tokenView(text, { matches, pieces, features }) {
  const head = pageHead('Token', el('h1', { class: 'text' }, quoted(text)));
  if (!matches.length) {
    return el('div', {}, head, el('p', { class: 'lede' }, 'No single token matches. Its pieces:',
      pieces.map((piece) => el('a', { href: tokenHref(piece) }, tokenLabel(piece)))));
  }
  const occurrences = matches.reduce((sum, match) => sum + match.occurrences, 0);
  return el('div', {}, head,
    stats([['Matches', matches.map((match) => tokenLabel(match.token))], ['Occurrences', integer.format(occurrences)]]),
    el('section', { class: 'card scroll' }, el('table', {},
      el('thead', {}, el('tr', {}, ['Feature', 'Peaks on', 'Mean act.', 'Of its fires'].map((label) => el('th', {}, label)))),
      el('tbody', {}, features.map((row) => el('tr', { onclick: () => { location.hash = featureHref(row.id); } },
        el('td', {}, el('a', { href: featureHref(row.id) }, `#${row.id}`)),
        el('td', {}, tokenLabel(row.peak_token)),
        el('td', {}, row.mean.toFixed(2)),
        el('td', {}, track(row.share), percent(row.share))))))));
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
  main.scrollTop = 0;
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
  rangeButton.textContent = `Density ${label}`;
  rangeButton.hidden = !list.range;
  for (const bin of document.querySelectorAll('.chart rect')) bin.classList.toggle('selected', Number(bin.dataset.low) === low);
  resetList();
}

function resetList() {
  Object.assign(list, { offset: 0, done: false, loading: false, generation: list.generation + 1, group: null });
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
  sentinel.before(...rows.flatMap((row) => {
    const group = row.title ? (row.trusted ? 0 : 1) : 2;
    // A heading where the group changes, unless no feature in the list has a title.
    const heading = group !== list.group && !(list.group == null && group === 2) && el('div', { class: 'group' }, GROUPS[group]);
    list.group = group;
    return [heading, el('a', { class: row.id === current ? 'row active' : 'row', href: featureHref(row.id), dataset: { id: row.id } },
      el('span', {}, `#${row.id}`),
      row.title ? el('span', { class: row.trusted ? 'label' : 'label weak' }, row.title) : row.peak_token == null ? el('span') : tokenLabel(row.peak_token, 'peak'),
      el('span', { class: 'density' }, percent(row.density)))].filter(Boolean);
  }));
  Object.assign(list, { offset: list.offset + rows.length, done: rows.length < LIST_PAGE, loading: false });
}

/* Controls */

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
    ArrowLeft: () => current != null && (location.hash = featureHref(current - 1)),
    ArrowRight: () => current != null && (location.hash = featureHref(current + 1)),
  };
  if (!actions[event.key]) return;
  event.preventDefault();
  actions[event.key]();
});

sort.addEventListener('change', resetList);
rangeButton.addEventListener('click', () => filterDensity(null));
window.addEventListener('hashchange', route);

overview = await api('/api/overview');
document.getElementById('count').textContent = integer.format(overview.d_sae);
resetList();
route();
