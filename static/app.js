'use strict';
/* llama-manager UI. Plain JS, no build step; talks to the JSON API in web.go and chatapi.go. */

/* ===== helpers ===== */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const icon = (id, cls = 'i i-sm') => `<svg class="${cls}"><use href="#i-${id}"/></svg>`;
const GB = 1024;
const fmtGB = mb => (mb / GB).toFixed(mb >= 10 * GB ? 1 : 2).replace(/\.0+$/, '') + ' GB';
const fmtSize = mb => mb >= GB ? (mb / GB).toFixed(1) + ' GB' : Math.round(mb) + ' MB';
const fmtBytes = b => b >= 1048576 ? (b / 1048576).toFixed(1) + ' MB' : b >= 1024 ? Math.round(b / 1024) + ' KB' : b + ' B';
const kfmt = n => n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1) + 'k' : String(n);
const kctx = n => n >= 1024 ? Math.round(n / 1024) + 'k' : String(n);
function fmtDur(sec) {
  sec = Math.max(0, Math.floor(sec)); const d = Math.floor(sec / 86400), h = Math.floor(sec % 86400 / 3600), m = Math.floor(sec % 3600 / 60), s = sec % 60;
  if (d) return `${d}d ${h}h`; if (h) return `${h}h ${m}m`; if (m) return `${m}m ${s}s`; return `${s}s`;
}
const isMac = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
const MOD = isMac ? '⌘' : 'Ctrl+';
function pref(k, v) { try { if (v === undefined) return localStorage.getItem(k); if (v === null) localStorage.removeItem(k); else localStorage.setItem(k, v); } catch (_) { return null; } return null; }
const sleep = ms => new Promise(r => setTimeout(r, ms));
const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

class ApiError extends Error { constructor(msg, status) { super(msg); this.status = status; } }
async function api(path, opts = {}) {
  const init = { method: opts.method || 'GET', headers: {} };
  if (opts.body !== undefined) {
    if (opts.body instanceof FormData) init.body = opts.body;
    else { init.body = JSON.stringify(opts.body); init.headers['Content-Type'] = 'application/json'; }
  }
  let r;
  try { r = await fetch(path, init); } catch (e) { throw new ApiError("Can't reach llama-manager. Check that it's running.", 0); }
  const text = await r.text();
  if (!r.ok) throw new ApiError((text || r.statusText).trim(), r.status);
  if (!text) return null;
  try { return JSON.parse(text); } catch (_) { return text; }
}

const STATE = {
  running: ['Running', 'var(--success)'], starting: ['Starting', 'var(--info)'], restarting: ['Restarting', 'var(--violet)'],
  unhealthy: ['Unhealthy', 'var(--warning)'], crashed: ['Crashed', 'var(--danger)'], stopped: ['Stopped', 'var(--text-tertiary)'],
};
const stateOf = st => STATE[st] || [st, 'var(--text-tertiary)'];
const pill = st => `<span class="pill st-${esc(st)}"><span class="dot"></span>${esc(stateOf(st)[0])}</span>`;
const SERIES = ['#3987e5', '#199e70', '#d55181', '#c98500', '#9085e9', '#d95926', '#e66767', '#008300'];

const QUANT_RE = /[-_.]((?:UD-)?I?Q\d(?:_[A-Z0-9]+)*|MXFP4|BF16|F16|F32)$/i;
function parseModel(ref) {
  if (!ref) return { name: '', quant: '', owner: '' };
  if (ref.startsWith('/') || /^[A-Za-z]:\\/.test(ref)) {
    let base = ref.split(/[\\/]/).pop().replace(/-\d{5}-of-\d{5}\.gguf$/i, '').replace(/\.gguf$/i, '');
    const u = base.lastIndexOf('GGUF_'); if (u > 0) base = base.slice(u + 5);
    const q = base.match(QUANT_RE);
    return { name: q ? base.slice(0, q.index) : base, quant: q ? q[1] : '', owner: 'Local file' };
  }
  const [repo, quant = ''] = ref.split(':');
  const [owner, ...rest] = repo.split('/');
  let name = rest.join('/').replace(/-GGUF$/i, '');
  if (owner === 'bartowski' && name.includes('_')) name = name.slice(name.indexOf('_') + 1);
  return { name: name || repo, quant, owner };
}
function spark(vals, max) {
  const n = vals.length, top = Math.max(max, 1);
  const pts = vals.map((v, i) => [(i / (n - 1)) * 100, 24 - (Math.min(v, top) / top) * 22]);
  const line = pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' ');
  const flat = vals.every(v => !v);
  const endY = (pts[n - 1][1] / 26 * 100).toFixed(1);
  return `<div class="spark-wrap${flat ? ' flat' : ''}"><svg class="spark" viewBox="0 0 100 26" preserveAspectRatio="none" aria-hidden="true"><line class="base" x1="0" x2="100" y1="24.5" y2="24.5"/><path class="area" d="${line} L100 26 L0 26 Z"/><path class="line" d="${line}"/></svg><span class="spark-end" style="top:${endY}%"></span></div>`;
}
function toast(msg, err) {
  const t = document.createElement('div'); t.className = 'toast' + (err ? ' err' : ''); t.setAttribute('role', err ? 'alert' : 'status');
  t.innerHTML = icon(err ? 'alert' : 'check', 'i') + `<span>${esc(msg)}</span>`;
  $('#toasts').appendChild(t); setTimeout(() => t.remove(), err ? 6000 : 3200);
}
const toastErr = (prefix, e) => toast(prefix ? `${prefix}: ${e.message}` : e.message, true);
async function copy(text, msg = 'Copied') {
  try { await navigator.clipboard.writeText(text); toast(msg); return; } catch (_) {}
  try { const ta = document.createElement('textarea'); ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0'; document.body.appendChild(ta); ta.select(); const ok = document.execCommand('copy'); ta.remove(); if (ok) { toast(msg); return; } } catch (_) {}
  toast('Copy is blocked in this browser. Select the text instead.', true);
}
function fileToBase64(file) {
  return new Promise((res, rej) => { const r = new FileReader(); r.onload = () => res(String(r.result).split(',')[1] || ''); r.onerror = () => rej(r.error); r.readAsDataURL(file); });
}

/* ===== state ===== */
const S = {
  status: null, instances: [], metrics: {}, gpus: [], gpuBackend: '', gpuErr: '', configs: [], settings: null, models: [], cacheDir: '',
  hist: {},      // per instance: { h: [...36 live tok/s], lastTotal, lastAt }
  gpuHist: {},   // per GPU index: [...36 util]
  lastLog: {},   // instance -> last log line, for crash callouts
  dl: null, logFor: null, logLen: 200, page: 'chat',
};
const byName = n => S.instances.find(x => x.name === n);
const confOf = n => S.configs.find(x => x.name === n);

/* ===== overlays: drawers, menus, dialog ===== */
let openEl = null, lastFocus = null;
function openDrawer(sel) {
  closeMenu(); if (openEl && openEl !== $(sel)) openEl.classList.remove('open');
  lastFocus = document.activeElement; openEl = $(sel); openEl.classList.add('open'); $('#scrim').classList.add('open');
  setTimeout(() => { const f = openEl.querySelector('input:not([type=checkbox]):not([readonly]), [data-close]'); f && f.focus({ preventScroll: true }); }, 60);
}
function closeDrawer() {
  if (!openEl) return; openEl.classList.remove('open'); $('#scrim').classList.remove('open');
  if (openEl.id === 'log-drawer') { S.logFor = null; $$('.r-item').forEach(r => r.classList.remove('selected')); }
  openEl = null; lastFocus && lastFocus.focus && lastFocus.focus({ preventScroll: true });
}
$('#scrim').addEventListener('click', () => { if ($('#dialog').classList.contains('open')) dlgDone(false); else closeDrawer(); });
document.addEventListener('click', e => { if (e.target.closest('[data-close]')) closeDrawer(); });
document.addEventListener('keydown', e => {
  if (e.key !== 'Escape') return;
  if ($('#menu').classList.contains('open')) closeMenu();
  else if ($('#dialog').classList.contains('open')) dlgDone(false);
  else if (openEl) closeDrawer();
});
function openMenu(anchor, items) {
  const m = $('#menu');
  m.innerHTML = items.map((it, i) => it ? `<button role="menuitem" data-i="${i}" class="${it[3] || ''}">${icon(it[0], 'i')}${esc(it[1])}</button>` : '<hr>').join('');
  m.classList.add('open');
  const r = anchor.getBoundingClientRect(), mw = m.offsetWidth, mh = m.offsetHeight;
  let top = r.bottom + 6; if (top + mh > innerHeight - 8) top = Math.max(8, r.top - mh - 6);
  m.style.left = Math.min(innerWidth - mw - 8, Math.max(8, r.right - mw)) + 'px'; m.style.top = top + 'px';
  m.onclick = e => { const b = e.target.closest('[data-i]'); if (!b) return; closeMenu(); items[+b.dataset.i][2](); };
  setTimeout(() => { const b = m.querySelector('button'); b && b.focus(); }, 0);
}
function closeMenu() { $('#menu').classList.remove('open'); }
document.addEventListener('pointerdown', e => { if (!e.target.closest('#menu') && !e.target.closest('[aria-haspopup="menu"]')) closeMenu(); });
let dlgResolve = null;
function confirmDlg(title, text, ok) {
  $('#dlg-title').textContent = title; $('#dlg-text').textContent = text; $('#dlg-ok').textContent = ok;
  $('#dialog').classList.add('open'); $('#scrim').classList.add('open'); setTimeout(() => $('#dlg-cancel').focus(), 30);
  return new Promise(r => dlgResolve = r);
}
function dlgDone(v) { $('#dialog').classList.remove('open'); if (!openEl) $('#scrim').classList.remove('open'); dlgResolve && dlgResolve(v); dlgResolve = null; }
$('#dlg-ok').addEventListener('click', () => dlgDone(true));
$('#dlg-cancel').addEventListener('click', () => dlgDone(false));
document.addEventListener('click', e => { const b = e.target.closest('[data-copy]'); if (b && b.dataset.copy) copy(b.dataset.copy, 'Address copied'); });

/* ===== shell: pages, theme, sidebar, API address ===== */
const PAGES = ['chat', 'instances', 'models', 'config'];
function go(p, replace = true) {
  if (!PAGES.includes(p)) p = 'chat';
  closeDrawer(); S.page = p;
  $$('.page').forEach(s => s.classList.toggle('active', s.id === 'page-' + p));
  $$('[data-page]').forEach(b => b.dataset.page === p ? b.setAttribute('aria-current', 'page') : b.removeAttribute('aria-current'));
  $('#content').scrollTop = 0;
  if (p === 'chat') onChatPage();
  if (p === 'models') { loadModels(); pollDownload(); }
  if (p === 'config') loadSettings();
  if (p === 'instances') { refresh(); loadConfigs(); }
  if (replace && location.hash !== '#' + p) history.replaceState(null, '', '#' + p);
}
$$('[data-page]').forEach(b => b.addEventListener('click', () => go(b.dataset.page)));
window.addEventListener('hashchange', () => go(location.hash.slice(1), false));

function setTheme(t) {
  document.documentElement.setAttribute('data-theme', t);
  $('#theme-btn').innerHTML = icon(t === 'dark' ? 'sun' : 'moon', 'i');
  $('#theme-btn').setAttribute('aria-label', t === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
  $('meta[name="theme-color"]').content = t === 'dark' ? '#09090b' : '#f4f4f5';
  pref('lm-theme', t);
}
$('#theme-btn').addEventListener('click', () => setTheme(document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light'));

function setSidebar(collapsed, remember) {
  $('.app').classList.toggle('sb-collapsed', collapsed);
  const t = $('#sb-toggle'); t.setAttribute('aria-label', collapsed ? 'Expand sidebar' : 'Collapse sidebar'); t.dataset.tip = collapsed ? 'Expand sidebar' : 'Collapse sidebar';
  if (remember) pref('lm-sidebar', collapsed ? 'collapsed' : 'expanded');
}
$('#sb-toggle').addEventListener('click', () => setSidebar(!$('.app').classList.contains('sb-collapsed'), true));
document.addEventListener('keydown', e => {
  if ((e.metaKey || e.ctrlKey) && !e.shiftKey && !e.altKey && e.key.toLowerCase() === 'b' && !e.target.closest('input, textarea, [contenteditable]')) { e.preventDefault(); setSidebar(!$('.app').classList.contains('sb-collapsed'), true); }
});
const tip = $('#tip');
function showTip(el) {
  if (!$('.app').classList.contains('sb-collapsed') || !el.dataset.tip) return;
  tip.textContent = el.dataset.tip + (el.id === 'sb-toggle' ? `  ${MOD}B` : ''); const r = el.getBoundingClientRect();
  tip.style.left = (r.right + 10) + 'px'; tip.style.top = (r.top + r.height / 2 - 13) + 'px'; tip.classList.add('show');
}
const sidebar = $('.sidebar');
sidebar.addEventListener('pointerover', e => { const el = e.target.closest('[data-tip]'); if (el) showTip(el); });
sidebar.addEventListener('pointerout', () => tip.classList.remove('show'));
sidebar.addEventListener('focusin', e => { const el = e.target.closest('[data-tip]'); if (el && el.matches(':focus-visible')) showTip(el); });
sidebar.addEventListener('focusout', () => tip.classList.remove('show'));
sidebar.addEventListener('click', () => tip.classList.remove('show'));

// apiAddresses lists URLs other machines can use, best first. The browser's
// own address goes first when it isn't localhost, since it's proven to work.
function apiAddresses() {
  const st = S.status, port = (st && st.port) || location.port || 80, out = [], seen = new Set();
  const add = (label, host) => { const u = `http://${host}${port == 80 ? '' : ':' + port}/v1`; if (!seen.has(u)) { seen.add(u); out.push([label, u]); } };
  const here = location.hostname;
  const local = here === 'localhost' || here === '127.0.0.1' || here === '::1' || here === '[::1]';
  if (!local && /^\d+\.\d+\.\d+\.\d+$/.test(here)) add('This address', here);
  for (const a of (st && st.addresses) || []) add(a.kind === 'lan' ? `LAN (${a.iface})` : a.kind === 'vpn' ? `VPN (${a.iface})` : a.iface, a.ip);
  if (st && st.name) add('Hostname', /\./.test(st.name) ? st.name.toLowerCase() : st.name.toLowerCase() + '.local');
  if (!local && !/^\d+\.\d+\.\d+\.\d+$/.test(here)) add('This address', here);
  if (!out.length) add('This machine', here || 'localhost');
  return out;
}
function renderApiChip() {
  const first = apiAddresses()[0][1];
  $('#api-url').innerHTML = `<span class="scheme">http://</span>${esc(first.replace(/^http:\/\//, ''))}`;
  $$('.api-copy, #page-config [data-copy]').forEach(b => b.dataset.copy = first);
  const s = $('#set-api-url'); if (s) s.textContent = first;
}
$('#api-open').addEventListener('click', e => {
  const p = $('#api-pop'); if (p.classList.contains('open')) { p.classList.remove('open'); return; }
  const run = S.instances.filter(x => x.state === 'running'), model = run[0] ? run[0].name : (S.instances[0] ? S.instances[0].name : 'my-instance');
  const addrs = apiAddresses();
  p.innerHTML = `<h3>Connect other apps</h3><p class="hint">Any OpenAI-compatible client works, such as opencode, aider or Open WebUI. Use an instance name as the model.</p>
    ${addrs.map(([k, u]) => `<div class="addr"><span>${esc(k)}</span><code>${esc(u)}</code><button class="icon-btn" data-copy="${esc(u)}" aria-label="Copy ${esc(k)} address">${icon('copy')}</button></div>`).join('')}
    <div class="hint" style="margin:12px 0 0">Model ids you can use right now</div>
    <div class="idchips">${run.map(x => `<button data-copyid="${esc(x.name)}" title="Copy model id">${esc(x.name)}${icon('copy')}</button>`).join('') || '<span class="hint">Nothing is running. Start an instance first.</span>'}</div>
    <div class="snippet" id="api-snip"><span>curl ${esc(addrs[0][1])}/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"model": "${esc(model)}", "messages": [{"role": "user", "content": "Hello"}]}'</span><button class="icon-btn" data-copysnip aria-label="Copy command">${icon('copy')}</button></div>`;
  p.classList.add('open'); const r = e.currentTarget.closest('.api').getBoundingClientRect();
  p.style.top = (r.bottom + 8) + 'px'; p.style.left = Math.max(8, Math.min(r.right - p.offsetWidth, innerWidth - p.offsetWidth - 8)) + 'px';
});
$('#api-pop').addEventListener('click', e => {
  const id = e.target.closest('[data-copyid]'); if (id) copy(id.dataset.copyid, 'Model id copied');
  if (e.target.closest('[data-copysnip]')) copy($('#api-snip span').textContent.trim(), 'Command copied');
});
document.addEventListener('pointerdown', e => { if (!e.target.closest('#api-pop') && !e.target.closest('#api-open')) $('#api-pop').classList.remove('open'); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') $('#api-pop').classList.remove('open'); });

/* ===== polling ===== */
let refreshing = false, lastRefresh = 0;
async function refresh() {
  if (refreshing) return; refreshing = true;
  try {
    const [st, insts, mets, gp] = await Promise.all([
      api('/api/status').catch(() => S.status), api('/api/instances').catch(() => null),
      api('/api/metrics').catch(() => ({})), api('/api/gpus').catch(() => null),
    ]);
    const now = performance.now();
    S.status = st;
    if (insts) S.instances = insts;
    S.metrics = mets || {};
    if (gp) { S.gpus = gp.gpus || []; S.gpuBackend = gp.backend || ''; S.gpuErr = gp.error || ''; }
    for (const x of S.instances) {
      const h = S.hist[x.name] || (S.hist[x.name] = { h: Array(36).fill(0), lastTotal: null, lastAt: 0 });
      const m = S.metrics[x.name];
      let rate = 0;
      if (m && h.lastTotal != null && m.predicted_total >= h.lastTotal && now > h.lastAt) rate = (m.predicted_total - h.lastTotal) / ((now - h.lastAt) / 1000);
      if (m) { h.lastTotal = m.predicted_total; h.lastAt = now; } else h.lastTotal = null;
      if (lastRefresh) { h.h.push(rate); h.h.shift(); }
      if ((x.state === 'crashed' || x.state === 'unhealthy') && x.last_error && !(x.name in S.lastLog)) {
        S.lastLog[x.name] = '';
        api(`/api/instances/${encodeURIComponent(x.name)}/logs?n=20`).then(lines => {
          const pick = (lines || []).filter(l => l.trim()).reverse().find(l => /error|fail|abort|unable|cannot|out of memory/i.test(l)) || '';
          S.lastLog[x.name] = pick.slice(0, 300); if (S.page === 'instances') renderRoster();
        }).catch(() => {});
      }
      if (x.state === 'running') delete S.lastLog[x.name];
    }
    for (const g of S.gpus) { const a = S.gpuHist[g.index] || (S.gpuHist[g.index] = Array(36).fill(0)); a.push(g.util_pct || 0); a.shift(); }
    lastRefresh = now;
    renderShell();
    if (S.page === 'instances') { renderBays(); renderRoster(); }
    if (S.page === 'chat') chatOnInstances();
    if (S.logFor && openEl && openEl.id === 'log-drawer') syncLogHead();
    const d = $('#live-dot'); d.classList.remove('ping'); void d.offsetWidth; d.classList.add('ping');
    $('#live-text').textContent = 'Live';
  } catch (e) {
    $('#live-text').textContent = 'Offline';
  } finally { refreshing = false; }
}
function renderShell() {
  const st = S.status;
  if (st) { $('#host-name').textContent = st.name || 'llama-manager'; $('#host-uptime').textContent = fmtDur(st.manager_uptime_sec || 0); }
  renderApiChip();
  const run = S.instances.filter(x => x.state === 'running').length;
  $('#nav-count-inst').textContent = S.instances.length ? `${run}/${S.instances.length}` : '';
}
setInterval(() => { if (!document.hidden) refresh(); }, 5000);
document.addEventListener('visibilitychange', () => { if (!document.hidden) { refresh(); if (S.page === 'chat') chatResync(); } });

/* ===== instances: GPU rack ===== */
function renderBays() {
  const has = S.gpus.length > 0;
  $('#rack').hidden = !has;
  const used = S.gpus.reduce((a, g) => a + (g.mem_used_mb || 0), 0), total = S.gpus.reduce((a, g) => a + (g.mem_total_mb || 0), 0);
  const backend = { vulkan: 'Vulkan', cuda: 'CUDA', rocm: 'ROCm', rocm_rocr: 'ROCm', metal: 'Metal' }[S.gpuBackend] || S.gpuBackend;
  $('#gpu-sub').textContent = has ? `${S.gpus.length} on ${backend}, ${fmtGB(used)} of ${fmtGB(total)} VRAM in use`
    : S.gpuErr ? `Couldn't read GPU stats: ${S.gpuErr}` : `Live GPU stats come from rocm-smi or nvidia-smi, so they show with the ROCm and CUDA backends. This server uses ${backend || 'another backend'}.`;
  if (!has) { $('#bays').innerHTML = ''; return; }
  $('#bays').innerHTML = S.gpus.map(g => {
    const pct = g.mem_total_mb ? g.mem_used_mb / g.mem_total_mb : 0, cells = 16, per = 1 / cells;
    const fill = pct > .95 ? 'var(--danger)' : pct > .85 ? 'var(--warning)' : 'var(--accent)';
    const cellHtml = Array.from({ length: cells }, (_, i) => `<span class="cell"><i style="height:${(Math.max(0, Math.min(1, (pct - i * per) / per)) * 100).toFixed(0)}%;--fill:${fill}"></i></span>`).join('');
    const tenants = S.instances.filter(x => (x.gpu_ids || []).includes(g.index));
    const busy = tenants.some(x => (S.metrics[x.name] || {}).requests_processing > 0);
    return `<article class="bay${busy ? ' busy' : ''}" aria-label="GPU ${g.index}">
      <div class="bay-head"><span class="bay-idx" aria-hidden="true">${g.index}</span>
        <div style="min-width:0"><div class="bay-name">${esc(g.name || 'GPU')}</div>
        <div class="bay-sensors">${g.temp_c ? `<span class="${g.temp_c >= 85 ? 'hot' : ''}" title="Temperature">${icon('temp', 'i')}${Math.round(g.temp_c)}°C</span>` : ''}${g.power_w ? `<span title="Power draw">${icon('zap', 'i')}${Math.round(g.power_w)} W</span>` : ''}</div></div></div>
      <div><div class="vram-top"><span>VRAM</span><span><b>${(g.mem_used_mb / GB).toFixed(1)}</b> of ${(g.mem_total_mb / GB).toFixed(0)} GB</span></div>
        <div class="cells" role="img" aria-label="${Math.round(pct * 100)}% of VRAM used">${cellHtml}</div></div>
      <div class="load-row"><span>Load</span>${spark(S.gpuHist[g.index] || [0, 0], 100)}<b>${Math.round(g.util_pct || 0)}%</b></div>
      <div class="tenants">${tenants.length ? tenants.map(x => `<span class="tenant" title="${esc(x.name)}: ${esc(stateOf(x.state)[0])}"><span class="dot" style="background:${stateOf(x.state)[1]}"></span><span>${esc(x.name)}</span></span>`).join('') : '<span class="idle">No instance assigned</span>'}</div>
    </article>`;
  }).join('');
  $('#rack-dots').innerHTML = S.gpus.map((_, i) => `<i class="${i === (window.bayIdx || 0) ? 'on' : ''}"></i>`).join('');
}
$('#bays').addEventListener('scroll', e => {
  const el = e.currentTarget, w = el.firstElementChild ? el.firstElementChild.offsetWidth + 8 : 1;
  const idx = window.bayIdx = Math.min(S.gpus.length - 1, Math.round(el.scrollLeft / w));
  $$('#rack-dots i').forEach((d, i) => d.classList.toggle('on', i === idx));
}, { passive: true });

/* ===== instances: roster ===== */
function rowSig(x) { const c = confOf(x.name) || {}; return [x.state, x.model, x.port, (x.gpu_ids || []).join(','), c.auto_start, x.last_error, S.lastLog[x.name] || ''].join('|'); }
function rowHtml(x) {
  const pm = parseModel(x.model), st = x.state, live = st === 'running', c = confOf(x.name) || {};
  const m = S.metrics[x.name], canStart = st === 'stopped' || st === 'crashed';
  const toggle = canStart
    ? `<button class="icon-btn go" data-act="start" title="Start" aria-label="Start ${esc(x.name)}">${icon('play', 'i')}</button>`
    : `<button class="icon-btn halt" data-act="stop" title="Stop" aria-label="Stop ${esc(x.name)}">${icon('stop', 'i')}</button>`;
  const log = S.lastLog[x.name];
  return `<div class="r-item" data-name="${esc(x.name)}" data-sig="${esc(rowSig(x))}" style="--stripe:${stateOf(st)[1]}">
    <div class="r-row">
      <div class="r-ident">
        <button class="r-name" data-act="logs" title="Open logs"><span>${esc(x.name)}</span><span class="r-port">:${x.port}</span></button>
        <div class="r-model" title="${esc(x.model)}"><span class="mname">${esc(pm.name)}</span>${pm.quant ? `<span class="chip quant">${esc(pm.quant)}</span>` : ''}</div>
        <span class="tag">${esc(pm.owner)}${c.auto_start ? `<span style="display:inline-flex;align-items:center;gap:4px;margin-left:8px" title="Starts when the manager starts">${icon('power', 'i')}Boot</span>` : ''}</span>
      </div>
      <div class="r-gpus"><span class="m-label">GPUs</span><span class="chips">${(x.gpu_ids || []).map(g => `<span class="chip">GPU ${g}</span>`).join('')}</span></div>
      <div class="r-state">${pill(st)}<span class="meta" data-f="meta"></span></div>
      <div class="gen metric${live ? '' : ' dim'}"><span class="m-label">Generation</span><div><b data-f="g">–</b><small>t/s</small> <span data-f="busy"></span></div><div data-f="spark"></div></div>
      <div class="metric r-prompt${live ? '' : ' dim'}"><span class="m-label">Prompt</span><b data-f="p">–</b><small>t/s</small></div>
      <div class="kv"><span class="m-label">KV cache</span><span class="num" data-f="kv">–</span><span class="kv-bar"><i data-f="kvbar" style="width:0"></i></span></div>
      <div class="r-actions">
        ${toggle}
        <button class="icon-btn" data-act="restart" title="Restart" aria-label="Restart ${esc(x.name)}" ${canStart ? 'disabled' : ''}>${icon('restart', 'i')}</button>
        <button class="icon-btn" data-act="logs" title="Logs" aria-label="Logs for ${esc(x.name)}">${icon('term', 'i')}</button>
        <button class="icon-btn more" data-act="menu" title="More" aria-label="More actions for ${esc(x.name)}" aria-haspopup="menu">${icon('more', 'i')}</button>
      </div>
    </div>
    ${x.last_error && st !== 'running' ? `<div class="r-error">${icon('alert')}<span>Last exit: <code>${esc(x.last_error)}</code>${log ? `. Log: <code>${esc(log)}</code>` : ''}</span><button class="btn btn-sm btn-ghost" data-act="logs" style="color:inherit">Open logs</button></div>` : ''}
  </div>`;
}
function fillRow(el, x) {
  const f = k => $(`[data-f="${k}"]`, el), m = S.metrics[x.name], h = S.hist[x.name];
  f('spark').innerHTML = spark(h ? h.h : [0, 0], Math.max(20, ...(h ? h.h : [0])));
  const live = x.state === 'running' && m;
  f('g').textContent = live ? m.predicted_tokens_sec.toFixed(1) : '–';
  f('p').textContent = live ? Math.round(m.prompt_tokens_sec) : '–';
  f('kv').textContent = live ? Math.round(m.kv_cache_usage * 100) + '%' : '–';
  f('kvbar').style.width = live ? (m.kv_cache_usage * 100) + '%' : '0';
  f('busy').innerHTML = live && m.requests_processing > 0 ? '<span class="busy-tag">Generating</span>' : '';
  const st = x.state;
  f('meta').textContent = st === 'running' ? `Up ${fmtDur(x.uptime_sec)}, ${x.restart_count} restart${x.restart_count === 1 ? '' : 's'}`
    : st === 'crashed' ? (x.restart_count ? `Gave up after ${x.restart_count} restart${x.restart_count === 1 ? '' : 's'}` : 'Exited')
    : st === 'stopped' ? ((confOf(x.name) || {}).auto_start ? 'Starts at boot' : 'Manual start') : st === 'starting' ? 'Loading model' : '';
}
function renderRoster() {
  const root = $('#roster');
  if (!S.instances.length) {
    root.innerHTML = `<div class="empty">${icon('server', 'i')}<div>No instances yet. Add one to serve a model on your GPUs.</div><button class="btn btn-primary" data-empty-add>${icon('plus')}Add instance</button></div>`;
  } else {
    if (root.querySelector('.empty')) root.innerHTML = '';
    const want = S.instances.map(x => x.name);
    $$('.r-item', root).forEach(el => { if (!want.includes(el.dataset.name)) el.remove(); });
    S.instances.forEach((x, i) => {
      let el = $(`.r-item[data-name="${CSS.escape(x.name)}"]`, root);
      if (!el || el.dataset.sig !== rowSig(x)) {
        const tmp = document.createElement('div'); tmp.innerHTML = rowHtml(x); const fresh = tmp.firstElementChild;
        if (el) el.replaceWith(fresh); else root.appendChild(fresh);
        el = fresh;
      }
      if (root.children[i] !== el) root.insertBefore(el, root.children[i]);
      el.classList.toggle('selected', S.logFor === x.name);
      fillRow(el, x);
    });
  }
  $('.r-head').hidden = !S.instances.length; $$('[data-bulk], #bulk-menu-btn').forEach(b => b.disabled = !S.instances.length);
  const c = {}; S.instances.forEach(x => c[x.state] = (c[x.state] || 0) + 1);
  $('#inst-summary').innerHTML = Object.keys(STATE).filter(k => c[k]).map(k => `<span class="pill st-${k}"><span class="dot"></span>${c[k]} ${STATE[k][0].toLowerCase()}</span>`).join('');
  $('#roster-sub').textContent = S.instances.length ? `${c.running || 0} of ${S.instances.length} running` : '';
}
async function act(name, a) {
  const x = byName(name);
  try {
    await api(`/api/instances/${encodeURIComponent(name)}/${a}`, { method: 'POST' });
    if (x) { x.state = a === 'stop' ? 'stopped' : a === 'start' ? 'starting' : 'restarting'; renderRoster(); }
    toast({ start: `Starting ${name}`, stop: `Stopped ${name}`, restart: `Restarting ${name}` }[a]);
  } catch (e) { toastErr(`Couldn't ${a} ${name}`, e); }
  setTimeout(refresh, 600); setTimeout(refresh, 2500);
}
$('#roster').addEventListener('click', e => {
  if (e.target.closest('[data-empty-add]')) return openEditor();
  const b = e.target.closest('[data-act]'); if (!b) return;
  const name = b.closest('.r-item').dataset.name, a = b.dataset.act;
  if (a === 'logs') openLogs(name);
  else if (a === 'menu') openMenu(b, [
    ['pencil', 'Edit', () => openEditor(name, 'edit')],
    ['copy', 'Clone', () => openEditor(name, 'clone')],
    ['link', 'Copy model id', () => copy(name, 'Model id copied')],
    null,
    ['trash', 'Delete instance', () => confirmDlg(`Delete ${name}?`, 'The instance is stopped and removed from config.yaml. The model file stays on disk.', 'Delete instance').then(async ok => {
      if (!ok) return;
      try { await api(`/api/config/instances/${encodeURIComponent(name)}`, { method: 'DELETE' }); toast(`Deleted ${name}`); } catch (er) { toastErr(`Couldn't delete ${name}`, er); }
      loadConfigs(); refresh();
    }), 'danger'],
  ]);
  else act(name, a);
});
async function bulk(a) {
  const n = S.instances.filter(x => x.state !== 'stopped' && x.state !== 'crashed').length;
  if (a === 'stop' && !(await confirmDlg(`Stop all ${n} running instance${n === 1 ? '' : 's'}?`, 'Anything connected to them, like editors or agents, loses its model until you start them again.', 'Stop all'))) return;
  try { await api(`/api/instances/all/${a}`, { method: 'POST' }); toast({ start: 'Starting all instances', stop: 'Stopped all instances', restart: 'Restarting all instances' }[a]); } catch (e) { toastErr('Bulk action failed', e); }
  setTimeout(refresh, 700); setTimeout(refresh, 2500);
}
$$('[data-bulk]').forEach(b => b.addEventListener('click', () => bulk(b.dataset.bulk)));
$('#bulk-menu-btn').addEventListener('click', e => openMenu(e.currentTarget, [['play', 'Start all', () => bulk('start')], ['restart', 'Restart all', () => bulk('restart')], null, ['stop', 'Stop all', () => bulk('stop'), 'danger']]));
async function loadConfigs() { try { S.configs = await api('/api/config/instances') || []; if (S.page === 'instances') renderRoster(); } catch (_) {} }

/* ===== instances: logs ===== */
let logTimer = null;
function logClass(l) {
  if (/\b(error|failed|fatal|abort|exiting)\b/i.test(l)) return 'l-err';
  if (/\bwarn(ing)?\b/i.test(l)) return 'l-warn';
  if (/listening on|model loaded|server is listening/i.test(l)) return 'l-ok';
  if (/tokens per second/.test(l)) return 'l-perf';
  return '';
}
async function renderLog() {
  const name = S.logFor; if (!name) return;
  let lines;
  try { lines = await api(`/api/instances/${encodeURIComponent(name)}/logs?n=${S.logLen}`); } catch (e) { $('#log').innerHTML = `<p class="empty" style="font-family:var(--font-sans)">${esc(e.message)}</p>`; return; }
  if (S.logFor !== name) return;
  const el = $('#log'), atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  el.innerHTML = lines && lines.length ? lines.map(l => { const c = logClass(l), t = esc(l); return `<div class="${c}">${c === 'l-perf' ? t.replace(/([\d.]+ tokens per second)/, '<b>$1</b>') : t}</div>`; }).join('')
    : '<p class="empty" style="font-family:var(--font-sans)">No output yet. Start the instance to see its log here.</p>';
  if ($('#log-follow').checked || atBottom) el.scrollTop = el.scrollHeight;
}
function syncLogHead() {
  const x = byName(S.logFor); if (!x) return;
  $('#log-title').textContent = x.name;
  $('#log-sub').textContent = `Port ${x.port}, GPU ${(x.gpu_ids || []).join(' + ')}`;
  $('#log-pill').outerHTML = `<span class="pill st-${esc(x.state)}" id="log-pill"><span class="dot"></span>${esc(stateOf(x.state)[0])}</span>`;
  $('#log-restart').innerHTML = x.state === 'stopped' || x.state === 'crashed' ? `${icon('play')}Start` : `${icon('restart')}Restart`;
}
function openLogs(name) {
  S.logFor = name; $$('.r-item').forEach(r => r.classList.toggle('selected', r.dataset.name === name));
  syncLogHead(); $('#log').innerHTML = ''; renderLog(); openDrawer('#log-drawer');
  clearInterval(logTimer); logTimer = setInterval(() => { if (!S.logFor) return clearInterval(logTimer); if ($('#log-follow').checked && !document.hidden) renderLog(); }, 3000);
}
$$('[data-loglen]').forEach(b => b.addEventListener('click', () => { $$('[data-loglen]').forEach(o => o.setAttribute('aria-pressed', o === b)); S.logLen = +b.dataset.loglen; renderLog(); }));
$('#log-copy').addEventListener('click', () => copy($$('#log div').map(d => d.textContent).join('\n'), 'Log copied'));
$('#log-restart').addEventListener('click', () => { const x = byName(S.logFor); if (x) act(x.name, x.state === 'stopped' || x.state === 'crashed' ? 'start' : 'restart'); });

/* ===== instances: add / edit / clone ===== */
let editing = null;
function nextPort() { const used = new Set([...S.configs.map(x => x.port), (S.status || {}).port]); let p = Math.max(9089, ...S.configs.map(x => x.port)) + 1; while (used.has(p)) p++; return p; }
async function openEditor(name, mode = 'add') {
  if (!S.models.length) await loadModels(true);
  if (!S.settings) await loadSettings(true);
  const c = name ? confOf(name) || (await loadConfigs(), confOf(name)) : null;
  editing = mode === 'edit' ? name : null;
  $('#ie-title').textContent = mode === 'edit' ? `Edit ${name}` : mode === 'clone' ? `Clone ${name}` : 'Add instance';
  $('#ie-sub').textContent = mode === 'edit' ? 'Saving restarts the instance if it is set to start at boot' : 'Runs one llama-server on the GPUs you pick';
  $('#ie-save').textContent = mode === 'edit' ? 'Save changes' : 'Add instance';
  const sel = $('#ie-model');
  sel.innerHTML = '<option value="">Choose a downloaded model</option>' + S.models.filter(m => !m.projector).map(m => { const pm = parseModel(m.path); return `<option value="${esc(m.path)}">${esc(pm.name || m.name)} ${esc(pm.quant)} (${fmtSize(m.size_mb)})</option>`; }).join('');
  if (c && c.model && ![...sel.options].some(o => o.value === c.model)) sel.insertAdjacentHTML('beforeend', `<option value="${esc(c.model)}">${esc(parseModel(c.model).name)} ${esc(parseModel(c.model).quant)}</option>`);
  const ids = S.gpus.length ? S.gpus.map(g => g.index) : (() => { const mx = Math.max(0, ...S.configs.flatMap(x => x.gpu_ids || [])); const n = S.gpuBackend === 'metal' ? 1 : Math.max(4, mx + 2); return Array.from({ length: n }, (_, i) => i); })();
  $('#ie-gpus').innerHTML = ids.map(i => { const g = S.gpus.find(y => y.index === i); return `<label class="gpu-opt"><input type="checkbox" value="${i}"><span><b>GPU ${i}</b><small>${g ? ((g.mem_total_mb - g.mem_used_mb) / GB).toFixed(1) + ' GB free' : S.gpuBackend === 'metal' ? 'Apple GPU' : 'Device ' + i}</small></span></label>`; }).join('');
  $('#ie-gpu-hint').hidden = ids.length < 2;
  $('#ie-name').value = c ? (mode === 'clone' ? c.name + '-copy' : c.name) : '';
  $('#ie-port').value = c && mode === 'edit' ? c.port : nextPort();
  $('#ie-port-hint').textContent = mode === 'edit' ? '' : `Next free port is ${nextPort()}`;
  sel.value = c ? c.model : '';
  const pick = c ? c.gpu_ids : [S.gpus.length ? S.gpus.slice().sort((a, b) => a.mem_used_mb - b.mem_used_mb)[0].index : 0];
  $$('#ie-gpus input').forEach(i => i.checked = pick.includes(+i.value));
  $('#ie-auto').checked = c ? !!c.auto_start : false;
  const s = S.settings || {};
  $('#ie-ngl').placeholder = `Server default (${s.ngl ?? 99})`; $('#ie-ctx').placeholder = `Server default (${s.context_length ?? ''})`;
  $('#ie-ctk').options[0].textContent = `Server default (${s.cache_type_k || 'f16'})`; $('#ie-ctv').options[0].textContent = `Server default (${s.cache_type_v || 'f16'})`;
  $('#ie-ngl').value = c && c.ngl != null ? c.ngl : ''; $('#ie-ctx').value = c && c.context_length != null ? c.context_length : '';
  $('#ie-ctk').value = (c && c.cache_type_k) || ''; $('#ie-ctv').value = (c && c.cache_type_v) || '';
  $('#ie-over').open = !!(c && (c.ngl != null || c.context_length != null || c.cache_type_k || c.cache_type_v)); overCount();
  editCarry = c ? { extra_args: c.extra_args, jinja: c.jinja } : null;
  $('#ie-err').hidden = true; openDrawer('#inst-drawer');
}
let editCarry = null;
function overCount() { const n = ['#ie-ngl', '#ie-ctx', '#ie-ctk', '#ie-ctv'].filter(s => $(s).value !== '').length; $('#ie-over-count').textContent = n ? `${n} override${n > 1 ? 's' : ''}` : 'Using server defaults'; }
['#ie-ngl', '#ie-ctx', '#ie-ctk', '#ie-ctv'].forEach(s => $(s).addEventListener('input', overCount));
$('#add-inst-btn').addEventListener('click', () => openEditor());
$('#ie-form').addEventListener('submit', async e => {
  e.preventDefault();
  const showErr = m => { $('#ie-err-text').textContent = m; $('#ie-err').hidden = false; };
  const p = { name: $('#ie-name').value.trim(), model: $('#ie-model').value, port: parseInt($('#ie-port').value, 10) || 0, gpu_ids: $$('#ie-gpus input:checked').map(c => +c.value), auto_start: $('#ie-auto').checked };
  if (!p.name) return showErr('Give the instance a name.');
  if (!p.model) return showErr('Choose a model. Download one on the Models page if the list is empty.');
  if (!p.port) return showErr('Enter a port.');
  if (!p.gpu_ids.length) return showErr('Pick at least one GPU.');
  if ($('#ie-ngl').value !== '') p.ngl = parseInt($('#ie-ngl').value, 10);
  if ($('#ie-ctx').value !== '') p.context_length = parseInt($('#ie-ctx').value, 10);
  if ($('#ie-ctk').value) p.cache_type_k = $('#ie-ctk').value;
  if ($('#ie-ctv').value) p.cache_type_v = $('#ie-ctv').value;
  if (editCarry) { if (editCarry.extra_args !== undefined) p.extra_args = editCarry.extra_args; if (editCarry.jinja !== undefined) p.jinja = editCarry.jinja; }
  const btn = $('#ie-save'); btn.disabled = true;
  try {
    if (editing) await api(`/api/config/instances/${encodeURIComponent(editing)}`, { method: 'PUT', body: p });
    else await api('/api/config/instances', { method: 'POST', body: p });
    toast(editing ? `Saved ${p.name}` : `Added ${p.name}${p.auto_start ? ', starting it now' : ''}`);
    closeDrawer(); await loadConfigs(); refresh();
  } catch (er) { showErr(er.message); } finally { btn.disabled = false; }
});

/* ===== models ===== */
let pickedQuant = null;
async function loadModels(quiet) {
  try { const d = await api('/api/models'); S.models = d.models || []; S.cacheDir = d.cache_dir || ''; } catch (e) { if (!quiet) toastErr("Couldn't list models", e); }
  if (S.page === 'models') renderLib();
}
function usedBy(m) {
  return S.instances.filter(x => {
    if (x.model === m.path) return true;
    if (x.model.startsWith('/')) return false;
    const [repo, quant = ''] = x.model.split(':'), p = m.path.toLowerCase();
    return p.includes(repo.toLowerCase().replace('/', '_')) && (!quant || p.includes(quant.toLowerCase()));
  });
}
function renderLib() {
  $('#cache-dir').textContent = S.cacheDir || 'the llama.cpp cache';
  const q = $('#lib-filter').value.trim().toLowerCase(), total = S.models.reduce((a, m) => a + m.size_mb, 0);
  $('#disk-sum').textContent = S.models.length ? `${fmtSize(total)} across ${S.models.length} file${S.models.length === 1 ? '' : 's'}` : 'No models downloaded yet';
  $('#lib-sub').textContent = `${S.models.length} file${S.models.length === 1 ? '' : 's'}`;
  $('#disk-bar').innerHTML = S.models.map((m, i) => `<i style="width:${total ? (m.size_mb / total * 100).toFixed(2) : 0}%;background:${SERIES[i % SERIES.length]}" title="${esc(m.name)}: ${fmtSize(m.size_mb)}"></i>`).join('');
  $('#disk-bar').parentElement.hidden = !S.models.length;
  const list = S.models.map((m, i) => ({ m, i })).filter(({ m }) => !q || m.name.toLowerCase().includes(q) || m.path.toLowerCase().includes(q));
  $('#lib').innerHTML = !S.models.length ? `<div class="empty">${icon('box', 'i')}<div>No models on disk yet. Download a GGUF from Hugging Face above.</div></div>`
    : list.length ? list.map(({ m, i }) => {
      const pm = parseModel(m.path), users = usedBy(m);
      return `<div class="lib-row">
        <div class="lib-main">
          <div class="lib-title"><span class="swatch" style="background:${SERIES[i % SERIES.length]}" aria-hidden="true"></span><span>${esc(pm.name || m.name)}</span>${pm.quant ? `<span class="chip quant">${esc(pm.quant)}</span>` : ''}${m.projector ? '<span class="chip accent">Vision projector</span>' : ''}</div>
          <div class="lib-path" title="${esc(m.path)}"><bdi>${esc(m.path)}</bdi></div>
        </div>
        <div class="lib-size">${fmtSize(m.size_mb)}</div>
        <div class="lib-use">${users.map(x => `<span class="tenant"><span class="dot" style="background:${stateOf(x.state)[1]}"></span><span>${esc(x.name)}</span></span>`).join('')}</div>
        <button class="icon-btn halt lib-del" data-del="${i}" aria-label="Delete ${esc(m.name)}" title="Delete from disk">${icon('trash', 'i')}</button>
      </div>`;
    }).join('') : `<div class="empty">No model matches “${esc(q)}”.</div>`;
}
$('#lib-filter').addEventListener('input', renderLib);
$('#lib').addEventListener('click', async e => {
  const b = e.target.closest('[data-del]'); if (!b) return;
  const m = S.models[+b.dataset.del], users = usedBy(m);
  const ok = await confirmDlg(`Delete ${m.name}?`, users.length ? `${users.map(u => u.name).join(' and ')} use${users.length === 1 ? 's' : ''} this file and won't start until you pick another model. ${fmtSize(m.size_mb)} is freed.` : `The file is removed from disk and ${fmtSize(m.size_mb)} is freed.`, 'Delete file');
  if (!ok) return;
  try { await api('/api/models?path=' + encodeURIComponent(m.path), { method: 'DELETE' }); toast(`Deleted ${m.name}`); } catch (er) { toastErr("Couldn't delete the file", er); }
  loadModels();
});
$('#dl-find').addEventListener('click', findQuants);
$('#dl-repo').addEventListener('keydown', e => { if (e.key === 'Enter') findQuants(); });
async function findQuants() {
  const repo = $('#dl-repo').value.trim().replace(/^https?:\/\/huggingface\.co\//, '').replace(/\/+$/, '');
  $('#dl-repo').value = repo;
  if (!/^[\w.-]+\/[\w.-]+$/.test(repo)) { $('#quants').innerHTML = `<span class="hint" style="color:var(--danger)">Use the owner/name form, for example unsloth/Qwen3-0.6B-GGUF.</span>`; return; }
  $('#dl-find').disabled = true; $('#quants').innerHTML = '<span class="hint">Looking up files on Hugging Face…</span>';
  try {
    const qs = await api('/api/models/quants?repo=' + encodeURIComponent(repo)) || [];
    if (!qs.length) { $('#quants').innerHTML = '<span class="hint">No GGUF files found in that repository.</span>'; pickedQuant = null; renderJob(); return; }
    $('#quants').innerHTML = qs.map(q => `<button class="quant-opt" aria-pressed="false" data-q="${esc(q)}">${esc(q)}</button>`).join('');
    pickQuant(qs.find(q => /^Q4_K_M$/i.test(q)) || qs[0]);
  } catch (e) { $('#quants').innerHTML = `<span class="hint" style="color:var(--danger)">${esc(e.message)}</span>`; }
  finally { $('#dl-find').disabled = false; }
}
function pickQuant(q) { pickedQuant = q; $$('.quant-opt').forEach(b => b.setAttribute('aria-pressed', b.dataset.q === q)); renderJob(); }
$('#quants').addEventListener('click', e => { const b = e.target.closest('[data-q]'); if (b) pickQuant(b.dataset.q); });
$('#dl-start').addEventListener('click', async () => {
  const repo = $('#dl-repo').value.trim(); if (!repo || !pickedQuant) return;
  try { await api('/api/models/download', { method: 'POST', body: { repo, quant: pickedQuant } }); toast(`Downloading ${repo}:${pickedQuant}`); pollDownload(); } catch (e) { toastErr("Couldn't start the download", e); }
});
$('#job-stop').addEventListener('click', async () => { try { await api('/api/models/download/stop', { method: 'POST' }); } catch (e) { toastErr("Couldn't stop the download", e); } pollDownload(); });
$('#job-log-btn').addEventListener('click', e => { const open = $('#job-log').hidden; $('#job-log').hidden = !open; e.currentTarget.textContent = open ? 'Hide log' : 'Show log'; e.currentTarget.setAttribute('aria-expanded', open); });
let dlTimer = null, dlWasActive = false;
async function pollDownload() {
  clearTimeout(dlTimer);
  try { S.dl = await api('/api/models/download/status'); } catch (_) {}
  renderJob();
  const active = S.dl && S.dl.active;
  if (dlWasActive && !active) { loadModels(true); if (S.dl && S.dl.status === 'done') toast(`Downloaded ${S.dl.repo}:${S.dl.quant}`); if (S.dl && S.dl.status === 'failed') toast(`Download failed: ${S.dl.repo}`, true); }
  dlWasActive = !!active;
  if (active) dlTimer = setTimeout(pollDownload, 2000);
}
function renderJob() {
  const d = S.dl, job = $('#job'), active = !!(d && d.active);
  $('#nav-dl-dot').hidden = $('#tab-dl-dot').hidden = !active;
  $('#dl-start').disabled = active || !pickedQuant;
  $('#dl-start-lbl').textContent = active ? 'One download at a time' : pickedQuant ? `Download ${pickedQuant}` : 'Download';
  if (!d || !d.status) { job.hidden = true; return; }
  job.hidden = false;
  $('#job-name').textContent = d.repo + (d.quant ? ':' + d.quant : '');
  const label = { downloading: 'Downloading', done: 'Done', stopped: 'Stopped', failed: 'Failed' }[d.status] || d.status;
  $('#job-pill').className = `pill st-${d.status}`; $('#job-pill').lastElementChild.textContent = label;
  const logs = d.logs || [];
  let pct = d.status === 'done' ? 100 : 0, bytes = '';
  for (let i = logs.length - 1; i >= 0; i--) { const m = logs[i].match(/(\d+)\/(\d+) MB \((\d+)%\)/); if (m) { pct = d.status === 'done' ? 100 : +m[3]; bytes = d.status === 'done' ? `${fmtSize(+m[2])} downloaded` : `${fmtSize(+m[1])} of ${fmtSize(+m[2])} (${pct}%)`; break; } }
  $('#job-progress').className = 'progress' + (d.status === 'done' ? ' done' : ''); $('#job-progress i').style.width = pct + '%';
  $('#job-bytes').textContent = bytes || (d.status === 'done' ? 'Finished' : d.status === 'downloading' ? 'Starting…' : '');
  const el = !d.elapsed ? '' : /^0m$/.test(d.elapsed) ? 'under a minute' : d.elapsed;
  $('#job-elapsed').textContent = el ? (active ? `${el.charAt(0).toUpperCase() + el.slice(1)} elapsed` : `Took ${el}`) : '';
  $('#job-stop').hidden = !active;
  const log = $('#job-log'); log.textContent = logs.slice(-50).join('\n'); log.scrollTop = log.scrollHeight;
}

/* ===== settings ===== */
const SET_FIELDS = [['s-delay', 'restart_delay'], ['s-max', 'max_restarts', 'int'], ['s-hc', 'health_check_interval'], ['s-host', 'host'], ['s-ctx', 'context_length', 'int'],
  ['s-ngl', 'ngl', 'int'], ['s-mg', 'main_gpu', 'int'], ['s-ctk', 'cache_type_k'], ['s-ctv', 'cache_type_v'], ['s-extra', 'extra_args_line']];
let setInit = null;
const setValues = () => JSON.stringify([...SET_FIELDS.map(([id]) => $('#' + id).value), $('#s-jinja').checked, ($('input[name="backend"]:checked') || {}).value || '']);
async function loadSettings(quiet) {
  let s;
  try { s = await api('/api/settings'); } catch (e) { if (!quiet) toastErr("Couldn't load settings", e); return; }
  S.settings = s;
  if (setInit && setInit !== setValues() && !quiet) return; // keep unsaved edits
  $('#s-bin').textContent = s.server_bin || '';
  $('#s-port').value = s.manager_port || '';
  for (const [id, key] of SET_FIELDS) { const el = $('#' + id); el.value = s[key] ?? ''; }
  if (s.extra_args_line == null) $('#s-extra').value = (s.extra_args || []).join(' ');
  $('#s-jinja').checked = s.jinja !== false;
  $$('input[name="backend"]').forEach(r => r.checked = r.value === s.gpu_backend);
  setInit = setValues(); renderArgs(); $('#savebar').classList.remove('show');
}
function splitArgs(s) { const out = []; let cur = '', q = null, any = false; for (const ch of s) { if (q) { if (ch === q) q = null; else cur += ch; } else if (ch === '"' || ch === "'") { q = ch; any = true; } else if (/\s/.test(ch)) { if (cur || any) out.push(cur); cur = ''; any = false; } else cur += ch; } if (cur || any) out.push(cur); return { out, open: !!q }; }
function renderArgs() { const { out, open } = splitArgs($('#s-extra').value); $('#args-preview').innerHTML = out.map(a => `<span class="chip${a.startsWith('-') ? ' flag' : ''}">${esc(a)}</span>`).join('') + (open ? '<span class="hint" style="color:var(--danger)">Close the open quote</span>' : ''); }
$('#set-form').addEventListener('input', () => { $('#savebar').classList.toggle('show', setInit !== null && setValues() !== setInit); renderArgs(); });
$('#set-form').addEventListener('change', () => $('#savebar').classList.toggle('show', setInit !== null && setValues() !== setInit));
$('#discard-btn').addEventListener('click', () => { setInit = null; loadSettings(); });
$('#save-btn').addEventListener('click', async () => {
  const body = { jinja: $('#s-jinja').checked, gpu_backend: ($('input[name="backend"]:checked') || {}).value || '' };
  for (const [id, key, type] of SET_FIELDS) { const v = $('#' + id).value.trim(); if (type === 'int') { if (v !== '') body[key] = parseInt(v, 10); } else body[key] = v; }
  if (splitArgs(body.extra_args_line).open) return toast('Extra arguments have an unclosed quote', true);
  const btn = $('#save-btn'); btn.disabled = true;
  try { await api('/api/settings', { method: 'PUT', body }); setInit = null; await loadSettings(true); toast('Settings saved. Restart instances to apply server defaults.'); }
  catch (e) { toastErr("Settings weren't saved", e); } finally { btn.disabled = false; }
});
$('#export-btn').addEventListener('click', () => { location.href = '/api/config/export'; });
async function importConfig(file) {
  const fd = new FormData(); fd.append('file', file);
  try { const d = await api('/api/config/import', { method: 'POST', body: fd }); toast((d && d.message) || 'Config imported'); setInit = null; loadSettings(true); loadConfigs(); refresh(); }
  catch (e) { toastErr("Config wasn't imported", e); }
}
$('#import-file').addEventListener('change', e => { if (e.target.files.length) importConfig(e.target.files[0]); e.target.value = ''; });
const dz = $('#dropzone');
['dragenter', 'dragover'].forEach(t => dz.addEventListener(t, e => { e.preventDefault(); dz.classList.add('over'); }));
['dragleave', 'drop'].forEach(t => dz.addEventListener(t, e => { e.preventDefault(); dz.classList.remove('over'); if (t === 'drop' && e.dataTransfer.files.length) importConfig(e.dataTransfer.files[0]); }));
$$('#set-nav a').forEach(a => a.addEventListener('click', e => { e.preventDefault(); $(a.getAttribute('href')).scrollIntoView({ behavior: 'smooth', block: 'start' }); }));
$('#content').addEventListener('scroll', () => {
  if (S.page !== 'config') return;
  let on = null; $$('.set-section').forEach(s => { if (s.getBoundingClientRect().top < 200) on = s.id; });
  $$('#set-nav a').forEach(a => a.classList.toggle('on', a.getAttribute('href') === '#' + (on || 's-manager')));
}, { passive: true });

/* ===== chat: state ===== */
const DEFAULT_PARAMS = { temp: 0.7, top_p: 0.95, min_p: 0.05, top_k: 40, rep: 1.0, max: '', seed: '', think: true, sys: '' };
const chatDefaults = () => { try { return { ...DEFAULT_PARAMS, ...JSON.parse(pref('lm-chat-defaults') || '{}') }; } catch (_) { return { ...DEFAULT_PARAMS }; } };
const C = { list: [], q: '', cur: null, detail: null, draft: null, es: null, streamMsg: null, att: [], props: {}, sending: false, inited: false, warn: null, openThink: new Set() };
const curParams = () => C.detail ? { ...DEFAULT_PARAMS, ...(C.detail.chat.params || {}) } : C.draft.params;
const curInstance = () => C.detail ? C.detail.chat.instance : C.draft.instance;
const curMsgs = () => C.detail ? C.detail.messages : [];
const variant = m => (m.variants || [])[m.vi] || (m.variants || [])[m.variants.length - 1] || { content: '', status: 'done' };
const NOW = () => Date.now(), MIN = 6e4, HOUR = 36e5, DAY = 864e5;

/* ===== chat: markdown (escape first, then format, so model output can never inject HTML) ===== */
const KW = 'func return if else for range go defer var const type struct interface package import map chan select case switch break continue nil true false def class from as with try except raise lambda yield async await let function new this null undefined export default while do in of fi then echo local location server listen proxy_pass fn pub use mut impl enum match where SELECT FROM WHERE INSERT UPDATE DELETE JOIN'.split(' ');
const HASH_LANGS = ['py', 'python', 'bash', 'sh', 'shell', 'zsh', 'yaml', 'yml', 'toml', 'nginx', 'ruby', 'rb', 'dockerfile', 'perl', 'r', 'conf', 'ini'];
function hl(code, lang) {
  lang = (lang || '').toLowerCase();
  if (!lang || lang === 'text' || lang === 'txt' || lang === 'plaintext' || code.length > 60000) return esc(code);
  const com = HASH_LANGS.includes(lang) ? '#[^\\n]*' : lang === 'sql' ? '--[^\\n]*' : '\\/\\/[^\\n]*|\\/\\*[\\s\\S]*?\\*\\/';
  const re = new RegExp(`(${com})|("(?:\\\\.|[^"\\\\\\n])*"|'(?:\\\\.|[^'\\\\\\n])*'|\`[^\`]*\`)|\\b(\\d+(?:\\.\\d+)?)\\b|\\b([A-Za-z_]\\w*)(?=\\s*\\()|\\b([A-Za-z_]\\w*)\\b`, 'g');
  let out = '', last = 0, m;
  while ((m = re.exec(code))) {
    out += esc(code.slice(last, m.index)); last = re.lastIndex;
    const [t, c, s, n, f, w] = m;
    out += c ? `<span class="tk-c">${esc(t)}</span>` : s ? `<span class="tk-s">${esc(t)}</span>` : n ? `<span class="tk-n">${t}</span>`
      : f ? `<span class="${KW.includes(f) ? 'tk-k' : 'tk-f'}">${esc(t)}</span>` : KW.includes(w) ? `<span class="tk-k">${esc(t)}</span>` : esc(t);
  }
  return out + esc(code.slice(last));
}
function inline(s) {
  const codes = [];
  s = s.replace(/`([^`\n]+)`/g, (_, c) => { codes.push(c); return `\u0001${codes.length - 1}\u0001`; });
  s = esc(s)
    .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?!\w)/g, '$1<em>$2</em>')
    .replace(/(^|\s)_([^_\n]+)_(?=\s|$|[.,;:!?])/g, '$1<em>$2</em>')
    .replace(/~~([^~\n]+)~~/g, '<del>$1</del>')
    .replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
    .replace(/(^|[\s(])(https?:\/\/[^\s<)]+[^\s<).,;:!?'"])/g, '$1<a href="$2" target="_blank" rel="noopener noreferrer">$2</a>');
  return s.replace(/\u0001(\d+)\u0001/g, (_, i) => `<code>${esc(codes[+i])}</code>`);
}
function md(src) {
  const blocks = [];
  src = String(src || '').replace(/\r\n/g, '\n').replace(/(^|\n)```([\w+#.-]*)[^\n]*\n([\s\S]*?)(?:\n```[ \t]*(?=\n|$)|$(?![\s\S]))/g, (_, pre, lang, code) => { blocks.push({ lang, code }); return `${pre}\u0000${blocks.length - 1}\u0000`; });
  const L = src.split('\n'), out = []; let para = [];
  const flush = () => { if (para.length) { out.push(`<p>${para.map(inline).join('<br>')}</p>`); para = []; } };
  for (let i = 0; i < L.length; i++) {
    const l = L[i]; let m;
    if ((m = l.match(/^\u0000(\d+)\u0000$/))) { flush(); const b = blocks[+m[1]]; out.push(`<div class="codeblock"><div class="codeblock-head"><span>${esc(b.lang || 'code')}</span><button class="btn btn-sm btn-ghost" data-copycode>${icon('copy')}Copy</button></div><pre><code>${hl(b.code, b.lang)}</code></pre></div>`); continue; }
    if (!l.trim()) { flush(); continue; }
    if ((m = l.match(/^(#{1,4})\s+(.*)$/))) { flush(); const h = Math.min(5, m[1].length + 2); out.push(`<h${h}>${inline(m[2])}</h${h}>`); continue; }
    if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(l)) { flush(); out.push('<hr>'); continue; }
    if (l.trim().startsWith('|') && L[i + 1] && /^\s*\|?\s*:?-{2,}/.test(L[i + 1])) {
      flush(); const row = r => r.trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim());
      const head = row(l); i += 2; const body = [];
      while (i < L.length && L[i].trim().startsWith('|')) body.push(row(L[i++]));
      i--; out.push(`<div class="tbl"><table><thead><tr>${head.map(c => `<th>${inline(c)}</th>`).join('')}</tr></thead><tbody>${body.map(r => `<tr>${r.map(c => `<td>${inline(c)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`); continue;
    }
    if ((m = l.match(/^(\s*)([-*+]|\d+[.)])\s+/))) {
      flush(); const ordered = /\d/.test(m[2]), items = [];
      while (i < L.length && (m = L[i].match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/))) {
        let text = m[3]; const ind = m[1].length; i++;
        while (i < L.length && L[i].trim() && !/^\s*([-*+]|\d+[.)])\s+/.test(L[i]) && /^\s{2,}/.test(L[i])) text += ' ' + L[i++].trim();
        const task = text.match(/^\[([ xX])\]\s+(.*)$/);
        items.push(`<li${ind >= 2 ? ' style="margin-left:1.2em"' : ''}>${task ? `<input type="checkbox" disabled ${task[1] !== ' ' ? 'checked' : ''}> ${inline(task[2])}` : inline(text)}</li>`);
      }
      i--; out.push(`<${ordered ? 'ol' : 'ul'}>${items.join('')}</${ordered ? 'ol' : 'ul'}>`); continue;
    }
    if (l.startsWith('>')) { flush(); const q = []; while (i < L.length && L[i].startsWith('>')) q.push(L[i++].replace(/^>\s?/, '')); i--; out.push(`<blockquote>${md(q.join('\n'))}</blockquote>`); continue; }
    para.push(l);
  }
  flush();
  return out.join('');
}

/* ===== chat: rendering ===== */
function relTime(t) { const d = NOW() - t; if (d < HOUR) return Math.max(1, Math.round(d / MIN)) + 'm'; if (d < DAY) return Math.round(d / HOUR) + 'h'; if (d < 7 * DAY) return Math.round(d / DAY) + 'd'; return new Date(t).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }); }
function groupOf(c) {
  if (c.pinned) return 'Pinned';
  const start = new Date(); start.setHours(0, 0, 0, 0); const t0 = start.getTime();
  if (c.updated_at >= t0) return 'Today'; if (c.updated_at >= t0 - DAY) return 'Yesterday';
  if (c.updated_at >= t0 - 7 * DAY) return 'Previous 7 days'; if (c.updated_at >= t0 - 30 * DAY) return 'Previous 30 days'; return 'Older';
}
function markQ(text, q) { const t = esc(text); if (!q) return t; return t.replace(new RegExp(esc(q).replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'ig'), x => `<mark>${x}</mark>`); }
function renderHist() {
  const q = C.q.trim();
  const groups = {}; C.list.forEach(c => (groups[q ? 'Results' : groupOf(c)] ||= []).push(c));
  const keys = q ? ['Results'] : ['Pinned', 'Today', 'Yesterday', 'Previous 7 days', 'Previous 30 days', 'Older'];
  $('#hist-list').innerHTML = C.list.length ? keys.filter(k => groups[k]).map(k => `<div class="hist-group">${k}</div>` + groups[k].map(c => `
    <div class="hist-item${c.id === C.cur ? ' on' : ''}" data-id="${c.id}">
      <button class="open" data-open="${c.id}" title="${esc(c.title)}"><span class="t">${c.streaming ? '<span class="gen-dot" title="Replying"></span>' : ''}<span style="overflow:hidden;text-overflow:ellipsis">${markQ(c.title, q)}</span></span>${q && c.snippet ? `<span class="snip">${markQ(c.snippet, q)}</span>` : ''}</button>
      <button class="icon-btn more" data-hmenu="${c.id}" aria-label="Actions for ${esc(c.title)}" aria-haspopup="menu">${icon('more', 'i')}</button>
    </div>`).join('')).join('')
    : q ? `<div class="empty" style="padding:30px 10px">No chat mentions “${esc(q)}”.</div>` : `<div class="empty" style="padding:30px 10px">Your chats will show up here.</div>`;
}
async function propsOf(name) {
  const x = byName(name); if (!x || x.state !== 'running') return null;
  const start = Math.round(Date.now() / 1000 - (x.uptime_sec || 0)), c = C.props[name];
  if (c && Math.abs(c.start - start) < 10) return c.p;
  try { const p = await api(`/api/instances/${encodeURIComponent(name)}/props`); C.props[name] = { p, start }; return p; } catch (_) { return null; }
}
const cachedProps = name => (C.props[name] || {}).p || null;
function ctxUse() {
  const msgs = curMsgs();
  for (let i = msgs.length - 1; i >= 0; i--) {
    const m = msgs[i]; if (m.role !== 'assistant') continue;
    const s = variant(m).stats; if (s && s.prompt_tok) return s.prompt_tok + (s.tok || 0);
    break;
  }
  const chars = (curParams().sys || '').length + msgs.reduce((a, m) => a + (m.role === 'user' ? m.content.length : variant(m).content.length), 0);
  return Math.round(chars / 3.6);
}
function renderHead() {
  const name = curInstance(), x = byName(name), pm = x ? parseModel(x.model) : null, p = cachedProps(name);
  const btn = $('#model-btn');
  btn.querySelector('.dot').style.background = x ? stateOf(x.state)[1] : 'var(--danger)';
  btn.querySelector('.nm').textContent = x ? x.name : name ? `${name} (removed)` : 'Pick an instance';
  const q = btn.querySelector('.chip'); q.textContent = pm ? pm.quant : ''; q.hidden = !pm || !pm.quant;
  $('#conv-title').textContent = C.detail && curMsgs().length ? C.detail.chat.title : 'New chat';
  const used = ctxUse(), total = p && p.n_ctx ? p.n_ctx : 0, pc = total ? Math.min(100, used / total * 100) : 0;
  $('#ctx').hidden = !total;
  $('#ctx-ring').style.setProperty('--p', pc.toFixed(1)); $('#ctx-ring').style.setProperty('--ring', pc > 85 ? 'var(--danger)' : pc > 65 ? 'var(--warning)' : 'var(--accent)');
  $('#ctx-text').textContent = total ? `${kfmt(used)} of ${kctx(total)}` : '';
  $('#ctx').title = total ? `Context used: about ${used.toLocaleString()} of ${total.toLocaleString()} tokens` : '';
  const tt = $('#think-toggle'); tt.hidden = !(p && p.thinking); tt.setAttribute('aria-pressed', curParams().think !== false);
  $('#chat-input').placeholder = x && x.state === 'running' ? `Message ${x.name}` : 'Start an instance to chat';
  $('#conv-more').disabled = !C.detail;
  compWarn();
}
function compWarn() {
  const w = $('#comp-warn'), name = curInstance(), x = byName(name);
  let html = '';
  if (S.status && S.status.chat_history === false) html = `${icon('alert')}Chat history is off because the database couldn't be opened. See the llama-manager log.`;
  else if (C.warn) html = `${icon('alert')}<span>${esc(C.warn)}</span>`;
  else if (!S.instances.length) html = `${icon('alert')}<span>No instances yet. Add one on the Instances page to start chatting.</span>`;
  else if (!x) html = `${icon('alert')}<span>${name ? `${esc(name)} no longer exists. ` : ''}Pick an instance above.</span>`;
  else if (x.state !== 'running') html = `${icon('alert')}<span>${esc(x.name)} is ${esc(stateOf(x.state)[0].toLowerCase())}.</span>${x.state === 'stopped' || x.state === 'crashed' ? `<button type="button" class="btn btn-sm" data-start-inst="${esc(x.name)}" style="margin-left:auto">Start ${esc(x.name)}</button>` : ''}`;
  w.innerHTML = html; w.hidden = !html;
}
function attChip(a, removable, i) {
  const thumb = a.kind === 'image' && a.url ? `<img src="${esc(a.url)}" alt="">` : `<span class="ficon">${icon(a.kind === 'image' ? 'eye' : 'file', 'i')}</span>`;
  return `<span class="att">${thumb}<span class="nm">${esc(a.name)}<small>${a.kind === 'image' ? 'Image' : fmtBytes(a.size)}</small></span>${removable ? `<button type="button" class="icon-btn" data-rmatt="${i}" aria-label="Remove ${esc(a.name)}">${icon('x')}</button>` : ''}</span>`;
}
function msgBody(m) {
  const v = variant(m), live = v.status === 'streaming', thinking = live && !v.content && v.reasoning;
  let h = '';
  if (v.reasoning || (live && !v.content)) {
    const label = thinking || (live && !v.content) ? '<span class="think-label">Thinking<span class="think-dots"><i></i><i></i><i></i></span></span>' : `<span class="think-label">Thought for ${((v.reasoning_ms || 0) / 1000).toFixed(1)} s</span>`;
    h += `<details class="think${thinking || (live && !v.content) ? ' thinking' : ''}" data-vid="${esc(v.id || '')}" ${C.openThink.has(v.id) ? 'open' : ''}><summary>${icon('chev')}${label}</summary>${v.reasoning ? `<div class="think-body">${esc(v.reasoning)}</div>` : ''}</details>`;
  }
  if (v.content) h += `<div class="md${live ? ' caret' : ''}">${md(v.content)}</div>`;
  if (v.status === 'error') h += `<div class="msg-error">${icon('alert')}<span>${esc(v.error || 'The reply failed.')}</span><button class="btn btn-sm" data-act="regen">Try again</button></div>`;
  return h;
}
function msgHtml(m, last) {
  if (m.role === 'user') {
    if (m.editing) return `<div class="msg user editing" data-id="${m.id}"><div class="edit-box"><textarea class="textarea" id="edit-${m.id}" aria-label="Edit message">${esc(m.content)}</textarea><div class="row"><button class="btn btn-sm btn-ghost" data-act="edit-cancel">Cancel</button><button class="btn btn-sm btn-primary" data-act="edit-send">Send</button></div></div></div>`;
    const atts = (m.attachments || []).map(a => a.kind === 'image' ? `<a href="/api/attachments/${a.id}" target="_blank" rel="noopener"><img src="/api/attachments/${a.id}" alt="${esc(a.name)}" loading="lazy"></a>` : `<a href="/api/attachments/${a.id}" style="text-decoration:none;color:inherit">${attChip(a)}</a>`).join('');
    return `<div class="msg user" data-id="${m.id}">${atts ? `<div class="msg-att">${atts}</div>` : ''}<div class="bubble">${esc(m.content)}</div>
      <div class="msg-actions"><button class="icon-btn" data-act="edit" title="Edit and resend" aria-label="Edit message">${icon('pencil', 'i')}</button><button class="icon-btn" data-act="copy" title="Copy" aria-label="Copy message">${icon('copy', 'i')}</button></div></div>`;
  }
  const v = variant(m), live = v.status === 'streaming', n = (m.variants || []).length, s = v.stats || {};
  const nav = n > 1 ? `<span class="variants"><button class="icon-btn" data-act="prev" aria-label="Previous version" ${m.vi === 0 || live ? 'disabled' : ''}>${icon('chevl', 'i')}</button>${m.vi + 1}/${n}<button class="icon-btn" data-act="next" aria-label="Next version" ${m.vi === n - 1 || live ? 'disabled' : ''}>${icon('chevr', 'i')}</button></span>` : '';
  const stats = s.tps ? `<span class="msg-stats"><span><b>${s.tps.toFixed(1)}</b> t/s</span><span><b>${s.tok}</b> tokens</span>${s.ttft ? `<span>first token <b>${s.ttft}</b> ms</span>` : ''}</span>` : '';
  const stopped = v.status === 'stopped' ? `<span class="stopped">${icon('stop')}${v.error ? esc(v.error) : 'Stopped'}</span>` : s.finish === 'length' ? `<span class="stopped">${icon('alert')}Hit the max token limit</span>` : '';
  return `<div class="msg assistant${last ? ' last' : ''}" data-id="${m.id}">
    <div class="who"><span class="av"><svg><use href="#i-llama"/></svg></span><b>${esc(m.instance)}</b><span>${new Date(v.created_at || m.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span></div>
    <div class="body">${msgBody(m)}</div>
    <div class="msg-actions">${live ? '' : `<button class="icon-btn" data-act="copy" title="Copy" aria-label="Copy reply">${icon('copy', 'i')}</button><button class="icon-btn" data-act="regen" title="Regenerate" aria-label="Regenerate reply">${icon('restart', 'i')}</button>`}${nav}${stopped}${stats}</div>
  </div>`;
}
const STARTERS = [['Explain an error log', 'Paste a log and get the likely cause and a fix'], ['Review a diff', 'Attach a .diff or paste changes for a code review'], ['Write a commit message', 'Describe the change, get a clean message'], ['Plan a task', 'Turn rough notes into ordered steps']];
function renderThread(keepScroll) {
  const t = $('#thread'), msgs = curMsgs(), name = curInstance(), x = byName(name);
  if (!msgs.length) {
    const pm = x ? parseModel(x.model) : null, p = cachedProps(name), m = x && S.metrics[x.name];
    const sub = x ? [`${esc(pm.name)} ${esc(pm.quant)}`, (x.gpu_ids || []).length ? `GPU ${x.gpu_ids.join(' + ')}` : '', p && p.n_ctx ? `${kctx(p.n_ctx)} context` : '', m && m.predicted_tokens_sec ? `about ${Math.round(m.predicted_tokens_sec)} tokens/s` : ''].filter(Boolean).join(', ') : '';
    t.innerHTML = x && x.state === 'running'
      ? `<div class="welcome"><div><h2>${esc(x.name)}</h2><p>${sub}</p></div><div class="starters">${STARTERS.map(([a, b]) => `<button class="starter" data-starter="${esc(a)}"><b>${a}</b><span>${b}</span></button>`).join('')}</div></div>`
      : `<div class="welcome"><div><h2>${S.instances.some(y => y.state === 'running') ? 'Pick a running instance' : 'No model is running'}</h2><p>${S.instances.length ? 'Choose one with the picker above, or start one here.' : 'Add an instance on the Instances page first.'}</p></div>
         <div class="starters">${S.instances.slice(0, 4).map(y => `<button class="starter" data-pickinst="${esc(y.name)}"><b>${esc(y.name)}</b><span>${esc(stateOf(y.state)[0])}${y.state !== 'running' ? ', tap to start' : ''}</span></button>`).join('') || `<button class="starter" data-goto="instances"><b>Add an instance</b><span>Pick a model and GPUs</span></button>`}</div></div>`;
  } else t.innerHTML = msgs.map((m, i) => msgHtml(m, i === msgs.length - 1)).join('');
  if (!keepScroll || stick) $('#scroller').scrollTop = $('#scroller').scrollHeight;
  renderHead(); renderStats();
}
let patchQueued = null;
function patchMsg(m) {
  if (patchQueued) return; patchQueued = requestAnimationFrame(() => {
    patchQueued = null;
    const el = $(`#thread .msg[data-id="${m.id}"]`); if (!el) return renderThread(true);
    const msgs = curMsgs(); el.outerHTML = msgHtml(m, msgs[msgs.length - 1] === m);
    if (stick) $('#scroller').scrollTop = $('#scroller').scrollHeight;
    renderHead();
  });
}
function renderStats() {
  const vs = curMsgs().filter(m => m.role === 'assistant').map(variant).filter(v => v.stats && v.stats.tps);
  const tok = vs.reduce((a, v) => a + (v.stats.tok || 0), 0);
  const d = (k, v, hl) => `<dt>${k}</dt><dd class="${hl ? 'hl' : ''}">${v}</dd>`;
  $('#stats-total').innerHTML = d('Replies', vs.length) + d('Context used', `${ctxUse().toLocaleString()} tokens`) + d('Reply tokens', tok.toLocaleString()) + d('Average speed', vs.length ? (vs.reduce((a, v) => a + v.stats.tps, 0) / vs.length).toFixed(1) + ' t/s' : '–', 1);
}
function renderParams() {
  const p = curParams(), f = (n, d) => (+n).toFixed(d);
  $('#p-sys').value = p.sys || ''; $('#p-temp').value = f(p.temp, 2); $('#p-temp-r').value = p.temp; $('#p-topp').value = f(p.top_p, 2); $('#p-topp-r').value = p.top_p;
  $('#p-minp').value = f(p.min_p, 2); $('#p-minp-r').value = p.min_p; $('#p-topk').value = p.top_k; $('#p-rep').value = f(p.rep, 2); $('#p-max').value = p.max; $('#p-seed').value = p.seed;
}
function renderAtt() {
  $('#att-row').innerHTML = C.att.map((a, i) => attChip(a, true, i)).join('');
  const p = cachedProps(curInstance()), img = C.att.some(a => a.kind === 'image');
  $('#att-warn').hidden = !(img && !(p && p.vision));
  $('#att-warn').innerHTML = `${icon('alert')}${esc(curInstance() || 'This instance')} can't read images. Pick an instance with a vision projector (mmproj).`;
}

/* ===== chat: data and streaming ===== */
let stick = true;
async function loadChatList() {
  try { C.list = await api('/api/chats' + (C.q.trim() ? '?q=' + encodeURIComponent(C.q.trim()) : '')) || []; } catch (_) { C.list = C.list || []; }
  renderHist();
}
const searchChats = debounce(loadChatList, 200);
function defaultInstance() {
  const last = pref('lm-last-inst'), run = S.instances.filter(x => x.state === 'running');
  return (run.find(x => x.name === last) || run[0] || byName(last) || S.instances[0] || {}).name || '';
}
function newDraft() { C.cur = null; C.detail = null; C.draft = { instance: defaultInstance(), params: chatDefaults() }; }
async function openChat(id, opts = {}) {
  closeStream();
  if (!id) { newDraft(); } else {
    try { C.detail = await api(`/api/chats/${id}`); C.cur = id; } catch (e) { if (e.status === 404) { pref('lm-chat-cur', null); newDraft(); } else return toastErr("Couldn't open the chat", e); }
  }
  pref('lm-chat-cur', C.cur); C.att = []; C.warn = null; stick = true;
  $('#chat-input').value = pref('lm-draft-' + (C.cur || 'new')) || ''; autoGrow();
  if (C.detail) C.detail.messages.forEach(m => delete m.editing);
  renderHist(); renderParams(); renderAtt(); renderThread(); setSendMode(false);
  propsOf(curInstance()).then(() => { renderHead(); renderAtt(); if (!curMsgs().length) renderThread(true); });
  if (C.detail && (C.detail.chat.streaming || curMsgs().some(m => m.role === 'assistant' && variant(m).status === 'streaming'))) subscribe(C.cur);
  if (!opts.keepPanels) openHist(false);
}
function closeStream() { if (C.es) { C.es.close(); C.es = null; } C.streamMsg = null; }
function findVariant(mid, vid) { const m = curMsgs().find(x => x.id === mid); return m ? [m, (m.variants || []).find(v => v.id === vid)] : [null, null]; }
function subscribe(id) {
  closeStream();
  const es = new EventSource(`/api/chats/${id}/events`); C.es = es;
  const on = (ev, fn) => es.addEventListener(ev, e => { if (C.es !== es || C.cur !== id) return; let d = {}; try { d = JSON.parse(e.data || '{}'); } catch (_) {} fn(d); });
  on('snapshot', d => {
    let [m, v] = findVariant(d.message_id, d.variant_id);
    if (!v) { reloadDetail(); return; }
    v.content = d.content; v.reasoning = d.reasoning; if (!d.done) v.status = 'streaming';
    C.streamMsg = { mid: m.id, vid: v.id }; m.vi = m.variants.indexOf(v); setSendMode(!d.done); patchMsg(m);
  });
  on('delta', d => {
    const s = C.streamMsg; if (!s) return; const [m, v] = findVariant(s.mid, s.vid); if (!v) return;
    if (d.r) v.reasoning = (v.reasoning || '') + d.r; if (d.c) v.content = (v.content || '') + d.c; patchMsg(m);
  });
  on('done', d => {
    const [m, v] = findVariant(d.message_id, d.variant_id);
    if (v) Object.assign(v, { status: d.status, error: d.error || '', content: d.content, reasoning: d.reasoning || '', reasoning_ms: d.reasoning_ms || 0, stats: d.stats || {} });
    C.streamMsg = null; setSendMode(false);
    if (m) patchMsg(m); else reloadDetail();
    if (d.status === 'error') toast(d.error || 'The reply failed', true);
    loadChatList();
  });
  on('title', d => { if (C.detail && d.title) { C.detail.chat.title = d.title; renderHead(); } loadChatList(); });
  const end = () => { if (C.es === es) { es.close(); C.es = null; if (C.streamMsg) reloadDetail(); setSendMode(false); } };
  on('end', end); on('idle', end);
}
async function reloadDetail() { if (!C.cur) return; const id = C.cur; try { const d = await api(`/api/chats/${id}`); if (C.cur !== id) return; C.detail = d; renderThread(true); if (d.chat.streaming && !C.es) subscribe(id); } catch (_) {} }
function chatResync() { loadChatList(); if (C.cur && (C.streamMsg || (C.detail && C.detail.chat.streaming) || !C.es)) reloadDetail(); }
function applyDetail(d) { C.detail = d; C.cur = d.chat.id; pref('lm-chat-cur', C.cur); stick = true; renderThread(); subscribe(C.cur); loadChatList(); }
function chatFail(e) {
  if (e.status === 409 && /start it|doesn't exist/.test(e.message)) { C.warn = e.message; compWarn(); return; }
  toastErr(e.status === 409 ? '' : "The message wasn't sent", e);
}
async function ensureChat() {
  if (C.cur) return C.cur;
  const c = await api('/api/chats', { method: 'POST', body: { instance: C.draft.instance, params: C.draft.params } });
  C.detail = { chat: c, messages: [] }; C.cur = c.id; pref('lm-chat-cur', c.id);
  return c.id;
}
async function sendMsg(text) {
  if (C.sending || C.streamMsg) return;
  text = text.trim(); if (!text && !C.att.length) return;
  const x = byName(curInstance());
  if (!x || x.state !== 'running') { C.warn = null; compWarn(); $('#comp-warn').classList.add('shake'); setTimeout(() => $('#comp-warn').classList.remove('shake'), 400); return; }
  C.sending = true; $('#send').disabled = true;
  try {
    const attachments = await Promise.all(C.att.map(async a => ({ kind: a.kind, name: a.name, mime: a.mime, data: await fileToBase64(a.file) })));
    const id = await ensureChat(); await saveParamsNow();
    const d = await api(`/api/chats/${id}/messages`, { method: 'POST', body: { content: text, instance: curInstance(), attachments } });
    $('#chat-input').value = ''; autoGrow(); pref('lm-draft-new', null); pref('lm-draft-' + id, null);
    C.att.forEach(a => a.url && URL.revokeObjectURL(a.url)); C.att = []; renderAtt(); C.warn = null;
    applyDetail(d);
  } catch (e) { chatFail(e); }
  finally { C.sending = false; $('#send').disabled = false; }
}
async function stopGen() { if (!C.cur) return; try { await api(`/api/chats/${C.cur}/stop`, { method: 'POST' }); } catch (e) { toastErr("Couldn't stop the reply", e); } }
function setSendMode(on) { const b = $('#send'); b.classList.toggle('stop', on); b.innerHTML = icon(on ? 'stop' : 'send', 'i'); b.setAttribute('aria-label', on ? 'Stop reply' : 'Send'); b.type = on ? 'button' : 'submit'; }

$('#composer').addEventListener('submit', e => { e.preventDefault(); if (C.streamMsg) return stopGen(); sendMsg($('#chat-input').value); });
$('#send').addEventListener('click', e => { if (C.streamMsg) { e.preventDefault(); stopGen(); } });
function autoGrow() { const el = $('#chat-input'); el.style.height = 'auto'; el.style.height = Math.min(240, el.scrollHeight) + 'px'; el.style.overflowY = el.scrollHeight > 240 ? 'auto' : 'hidden'; }
$('#chat-input').addEventListener('input', () => { autoGrow(); pref('lm-draft-' + (C.cur || 'new'), $('#chat-input').value || null); });
$('#chat-input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing && !matchMedia('(pointer: coarse)').matches) { e.preventDefault(); $('#composer').requestSubmit(); }
  if (e.key === 'ArrowUp' && !e.target.value) { const u = [...curMsgs()].reverse().find(m => m.role === 'user'); if (u && !C.streamMsg) { e.preventDefault(); startEdit(u); } }
});
document.addEventListener('keydown', e => {
  const mod = e.metaKey || e.ctrlKey;
  if (e.key === 'Escape' && C.streamMsg && S.page === 'chat' && !openEl && !$('#dialog').classList.contains('open')) { stopGen(); return; }
  if (mod && !e.shiftKey && e.key.toLowerCase() === 'k') { e.preventDefault(); go('chat'); openHist(true); $('#hist-q').focus(); $('#hist-q').select(); }
  if (mod && e.shiftKey && e.key.toLowerCase() === 'o') { e.preventDefault(); go('chat'); newChat(); }
});
function startEdit(m) { curMsgs().forEach(y => delete y.editing); m.editing = true; renderThread(true); const ta = $('#edit-' + m.id); if (ta) { ta.focus(); ta.setSelectionRange(ta.value.length, ta.value.length); } }
async function runReply(path, method, body) {
  if (C.streamMsg) return;
  try { await saveParamsNow(); applyDetail(await api(path, { method, body })); } catch (e) { chatFail(e); }
}
$('#thread').addEventListener('click', async e => {
  const sum = e.target.closest('details.think > summary'); if (sum) { const d = sum.parentElement, id = d.dataset.vid; if (id) { if (d.open) C.openThink.delete(id); else C.openThink.add(id); } return; }
  const s = e.target.closest('[data-starter]'); if (s) { $('#chat-input').value = s.dataset.starter + ': '; autoGrow(); $('#chat-input').focus(); return; }
  const pi = e.target.closest('[data-pickinst]'); if (pi) { const x = byName(pi.dataset.pickinst); setInstance(x.name); if (x.state !== 'running') act(x.name, 'start'); return; }
  if (e.target.closest('[data-goto]')) return go('instances');
  const cc = e.target.closest('[data-copycode]'); if (cc) { copy(cc.closest('.codeblock').querySelector('pre').innerText, 'Code copied'); return; }
  const b = e.target.closest('[data-act]'); if (!b) return;
  const msgEl = b.closest('.msg'); if (!msgEl) return;
  const m = curMsgs().find(y => y.id === msgEl.dataset.id), a = b.dataset.act; if (!m) return;
  if (a === 'copy') copy(m.role === 'user' ? m.content : variant(m).content, 'Copied');
  if (a === 'edit') startEdit(m);
  if (a === 'edit-cancel') { delete m.editing; renderThread(true); }
  if (a === 'edit-send') { const val = $('#edit-' + m.id).value.trim(); if (val) runReply(`/api/chats/${C.cur}/messages/${m.id}`, 'PUT', { content: val, instance: curInstance() }); }
  if (a === 'regen') runReply(`/api/chats/${C.cur}/messages/${m.id}/regenerate`, 'POST', { instance: curInstance() });
  if ((a === 'prev' || a === 'next') && !C.streamMsg) {
    const vi = m.vi + (a === 'prev' ? -1 : 1);
    try { await api(`/api/chats/${C.cur}/messages/${m.id}`, { method: 'PATCH', body: { vi } }); m.vi = vi; patchMsg(m); } catch (er) { toastErr("Couldn't switch versions", er); }
  }
});
$('#thread').addEventListener('toggle', e => { const d = e.target; if (!d.matches || !d.matches('details.think') || !d.dataset.vid) return; if (d.open) C.openThink.add(d.dataset.vid); else C.openThink.delete(d.dataset.vid); }, true);
$('#thread').addEventListener('keydown', e => { if (e.target.matches('.edit-box textarea') && e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); e.target.closest('.msg').querySelector('[data-act="edit-send"]').click(); } });
$('#comp-warn').addEventListener('click', e => { const b = e.target.closest('[data-start-inst]'); if (b) { C.warn = null; act(b.dataset.startInst, 'start'); } });
$('#scroller').addEventListener('scroll', () => { const s = $('#scroller'); stick = s.scrollHeight - s.scrollTop - s.clientHeight < 80; $('#jump').classList.toggle('show', !stick); }, { passive: true });
$('#jump').addEventListener('click', () => { const s = $('#scroller'); s.scrollTo({ top: s.scrollHeight, behavior: 'smooth' }); });

/* attachments: picker, paste, drag and drop */
const IMG_TYPES = ['image/png', 'image/jpeg', 'image/gif', 'image/webp'];
function addFiles(list) {
  for (const f of list) {
    if (C.att.length >= 8) { toast('Attach at most 8 files per message', true); break; }
    if (f.type.startsWith('image/')) {
      if (!IMG_TYPES.includes(f.type)) { toast(`${f.name}: images must be PNG, JPEG, GIF or WebP`, true); continue; }
      if (f.size > 10 * 1048576) { toast(`${f.name} is over 10 MB. Attach a smaller image.`, true); continue; }
      C.att.push({ kind: 'image', name: f.name || 'pasted-image.png', mime: f.type, size: f.size, file: f, url: URL.createObjectURL(f) });
    } else if (f.size > 2 * 1048576) toast(`${f.name} is over 2 MB. Attach a smaller text file.`, true);
    else C.att.push({ kind: 'file', name: f.name, mime: 'text/plain', size: f.size, file: f });
  }
  renderAtt(); $('#chat-input').focus();
}
$('#att-file').addEventListener('change', e => { addFiles([...e.target.files]); e.target.value = ''; });
$('#att-row').addEventListener('click', e => { const b = e.target.closest('[data-rmatt]'); if (b) { const [a] = C.att.splice(+b.dataset.rmatt, 1); if (a && a.url) URL.revokeObjectURL(a.url); renderAtt(); } });
$('#chat-input').addEventListener('paste', e => { const fs = [...((e.clipboardData && e.clipboardData.files) || [])]; if (fs.length) { e.preventDefault(); addFiles(fs); } });
const conv = $('#conv'); let dragN = 0;
conv.addEventListener('dragenter', e => { if ([...e.dataTransfer.types].includes('Files')) { e.preventDefault(); dragN++; conv.classList.add('dragging'); } });
conv.addEventListener('dragover', e => { if (conv.classList.contains('dragging')) e.preventDefault(); });
conv.addEventListener('dragleave', () => { if (--dragN <= 0) { dragN = 0; conv.classList.remove('dragging'); } });
conv.addEventListener('drop', e => { e.preventDefault(); dragN = 0; conv.classList.remove('dragging'); addFiles([...e.dataTransfer.files]); });

/* ===== chat: history, picker, panels, params ===== */
function newChat() { if (!C.cur && !curMsgs().length) { $('#chat-input').focus(); openHist(false); return; } openChat(null); $('#chat-input').focus(); }
$('#new-chat').addEventListener('click', newChat); $('#new-chat-m').addEventListener('click', newChat);
$('#hist-q').addEventListener('input', e => { C.q = e.target.value; searchChats(); });
$('#hist-q').addEventListener('keydown', e => { if (e.key === 'Escape') { e.target.value = ''; C.q = ''; loadChatList(); } });
async function patchChat(id, body) { try { const c = await api(`/api/chats/${id}`, { method: 'PATCH', body }); if (C.detail && C.cur === id) Object.assign(C.detail.chat, c); return c; } catch (e) { toastErr("Couldn't save the chat", e); return null; } }
function renameChat(c) {
  const row = $(`.hist-item[data-id="${c.id}"]`), visible = row && row.offsetParent;
  const host = visible ? row : $('#conv-title');
  host.innerHTML = `<input class="hist-rename" value="${esc(c.title)}" aria-label="Chat name"${visible ? '' : ' style="text-align:center"'}>`;
  const inp = host.querySelector('input'); inp.focus(); inp.select();
  let done = false;
  const finish = async save => { if (done) return; done = true; const v = inp.value.trim(); if (save && v && v !== c.title) await patchChat(c.id, { title: v }); await loadChatList(); renderHead(); };
  inp.addEventListener('keydown', e => { e.stopPropagation(); if (e.key === 'Enter') finish(true); if (e.key === 'Escape') finish(false); });
  inp.addEventListener('click', e => e.stopPropagation());
  inp.addEventListener('blur', () => finish(true));
}
function deleteChat(c) {
  confirmDlg(`Delete “${c.title}”?`, 'The chat and its messages are removed from the history on this server, for everyone who uses it.', 'Delete chat').then(async ok => {
    if (!ok) return;
    try { await api(`/api/chats/${c.id}`, { method: 'DELETE' }); toast('Chat deleted'); } catch (e) { return toastErr("Couldn't delete the chat", e); }
    pref('lm-draft-' + c.id, null);
    if (C.cur === c.id) openChat(null); await loadChatList();
  });
}
const chatMenu = c => [['pencil', 'Rename', () => renameChat(c)], ['pin', c.pinned ? 'Unpin' : 'Pin to top', async () => { await patchChat(c.id, { pinned: !c.pinned }); loadChatList(); }],
  ['download', 'Export as Markdown', () => { location.href = `/api/chats/${c.id}/export`; }], null, ['trash', 'Delete chat', () => deleteChat(c), 'danger']];
$('#hist-list').addEventListener('click', e => {
  const o = e.target.closest('[data-open]'); if (o) return openChat(o.dataset.open);
  const mm = e.target.closest('[data-hmenu]'); if (mm) { const c = C.list.find(y => y.id === mm.dataset.hmenu); if (c) openMenu(mm, chatMenu(c)); }
});
$('#conv-more').addEventListener('click', e => { if (C.detail) openMenu(e.currentTarget, chatMenu(C.detail.chat)); });
$('#conv-title').addEventListener('click', () => { if (C.detail && curMsgs().length) renameChat(C.detail.chat); });

function setInstance(name) {
  pref('lm-last-inst', name);
  if (C.detail) { C.detail.chat.instance = name; patchChat(C.cur, { instance: name }); } else C.draft.instance = name;
  C.warn = null; renderHead(); renderAtt(); if (!curMsgs().length) renderThread(true);
  propsOf(name).then(() => { renderHead(); renderAtt(); if (!curMsgs().length) renderThread(true); });
}
$('#model-btn').addEventListener('click', async e => {
  const p = $('#picker'); if (p.classList.contains('open')) { p.classList.remove('open'); return; }
  const r = e.currentTarget.getBoundingClientRect(), cur = curInstance();
  const row = x => {
    const pm = parseModel(x.model), on = x.state === 'running', pr = cachedProps(x.name), m = S.metrics[x.name];
    const caps = [pr && pr.vision ? 'reads images' : '', pr && pr.thinking ? 'thinks' : ''].filter(Boolean).join(', ');
    return `<button class="pick${on ? '' : ' off'}" role="option" aria-checked="${x.name === cur}" data-pick="${esc(x.name)}"><span class="dot" style="background:${stateOf(x.state)[1]}"></span><b>${esc(x.name)}</b>${on ? `<span class="side">${m && m.predicted_tokens_sec ? Math.round(m.predicted_tokens_sec) + ' t/s<br>' : ''}${pr && pr.n_ctx ? kctx(pr.n_ctx) + ' ctx' : ''}</span>` : `<span class="side"><span class="btn btn-sm" data-pstart="${esc(x.name)}" role="button" tabindex="0">Start</span></span>`}<small>${esc(pm.name)} ${esc(pm.quant)}${caps ? ', ' + caps : ''}</small></button>`;
  };
  const draw = () => {
    const run = S.instances.filter(x => x.state === 'running'), rest = S.instances.filter(x => x.state !== 'running');
    p.innerHTML = S.instances.length ? `<div class="grp">Running</div>${run.map(row).join('') || '<div class="hint" style="padding:6px 10px">Nothing is running</div>'}${rest.length ? `<div class="grp">Not running</div>${rest.map(row).join('')}` : ''}`
      : '<div class="hint" style="padding:10px">No instances yet. Add one on the Instances page.</div>';
  };
  draw(); p.classList.add('open'); p.style.left = Math.max(8, Math.min(r.left, innerWidth - p.offsetWidth - 8)) + 'px'; p.style.top = (r.bottom + 6) + 'px';
  await Promise.all(S.instances.filter(x => x.state === 'running').map(x => propsOf(x.name)));
  if (p.classList.contains('open')) draw();
});
$('#picker').addEventListener('click', e => {
  const st = e.target.closest('[data-pstart]'); if (st) { e.stopPropagation(); act(st.dataset.pstart, 'start'); setInstance(st.dataset.pstart); $('#picker').classList.remove('open'); return; }
  const b = e.target.closest('[data-pick]'); if (!b || b.classList.contains('off')) return;
  setInstance(b.dataset.pick); $('#picker').classList.remove('open');
});
document.addEventListener('pointerdown', e => { if (!e.target.closest('#picker') && !e.target.closest('#model-btn')) $('#picker').classList.remove('open'); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') $('#picker').classList.remove('open'); });

const wide = () => matchMedia('(min-width: 1100px)').matches;
function applyPanels() {
  const sh = $('#chat-shell'), histOpen = pref('lm-hist') !== 'closed', parOpen = pref('lm-params') === 'open';
  sh.classList.toggle('hist-closed', !histOpen); sh.classList.toggle('params-open', parOpen);
  $('#hist-toggle').setAttribute('aria-pressed', wide() ? histOpen : sh.classList.contains('hist-m'));
  $('#params-toggle').setAttribute('aria-pressed', wide() ? parOpen : sh.classList.contains('params-m'));
}
function openHist(force) { const sh = $('#chat-shell'); if (wide()) { if (force) { pref('lm-hist', 'open'); applyPanels(); } return; } sh.classList.toggle('hist-m', !!force); sh.classList.remove('params-m'); applyPanels(); }
$('#hist-toggle').addEventListener('click', () => { if (wide()) { pref('lm-hist', pref('lm-hist') === 'closed' ? 'open' : 'closed'); applyPanels(); } else openHist(!$('#chat-shell').classList.contains('hist-m')); });
$('#params-toggle').addEventListener('click', () => { const sh = $('#chat-shell'); if (wide()) pref('lm-params', pref('lm-params') === 'open' ? 'closed' : 'open'); else { sh.classList.toggle('params-m'); sh.classList.remove('hist-m'); } applyPanels(); });
$('#params-close').addEventListener('click', () => { if (wide()) pref('lm-params', 'closed'); $('#chat-shell').classList.remove('params-m'); applyPanels(); });
$('#chat-scrim').addEventListener('click', () => { $('#chat-shell').classList.remove('hist-m', 'params-m'); applyPanels(); });
window.addEventListener('resize', debounce(applyPanels, 100));

let paramsDirty = false;
const saveParamsNow = async () => { if (paramsDirty && C.detail) { paramsDirty = false; await patchChat(C.cur, { params: C.detail.chat.params }); } };
const saveParams = debounce(saveParamsNow, 500);
function setParam(k, v) {
  if (C.detail) { C.detail.chat.params = { ...DEFAULT_PARAMS, ...(C.detail.chat.params || {}), [k]: v }; paramsDirty = true; saveParams(); } else C.draft.params[k] = v;
}
[['p-temp', 'temp', 2], ['p-topp', 'top_p', 2], ['p-minp', 'min_p', 2]].forEach(([id, k, d]) => {
  $('#' + id + '-r').addEventListener('input', e => { setParam(k, +e.target.value); $('#' + id).value = (+e.target.value).toFixed(d); });
  $('#' + id).addEventListener('change', e => { const v = parseFloat(e.target.value); if (!isNaN(v)) { setParam(k, v); $('#' + id + '-r').value = v; } });
});
[['p-topk', 'top_k', 1], ['p-rep', 'rep', 1], ['p-max', 'max', 0], ['p-seed', 'seed', 0], ['p-sys', 'sys', 0]].forEach(([id, k, num]) => $('#' + id).addEventListener('input', e => { const v = e.target.value; setParam(k, num ? (parseFloat(v) || 0) : v); if (k === 'sys') renderHead(); }));
$('#p-reset').addEventListener('click', () => { const sys = curParams().sys; Object.entries({ ...DEFAULT_PARAMS, sys }).forEach(([k, v]) => setParam(k, v)); renderParams(); renderHead(); toast('Sampling reset to defaults'); });
$('#p-default').addEventListener('click', () => { pref('lm-chat-defaults', JSON.stringify(curParams())); toast('New chats will start with these settings'); });
$('#think-toggle').addEventListener('click', () => { setParam('think', curParams().think === false); renderHead(); });

function fixDraftInstance() {
  if (!C.draft || C.detail || curMsgs().length) return;
  const x = byName(C.draft.instance);
  if (!x || (x.state !== 'running' && S.instances.some(y => y.state === 'running'))) C.draft.instance = defaultInstance();
}
function onChatPage() {
  applyPanels(); fixDraftInstance();
  if (!C.inited) { C.inited = true; loadChatList(); openChat(pref('lm-chat-cur'), { keepPanels: true }); return; }
  renderHead(); renderThread(true); loadChatList();
}
let lastInstSig = '';
function chatOnInstances() {
  const sig = S.instances.map(x => x.name + x.state).join();
  if (!C.inited) return;
  const before = C.draft && C.draft.instance; fixDraftInstance(); if (C.draft && before !== C.draft.instance) { lastInstSig = ''; }
  renderHead();
  if (sig !== lastInstSig) { lastInstSig = sig; propsOf(curInstance()).then(() => { renderHead(); renderAtt(); }); if (!curMsgs().length) renderThread(true); }
}
setInterval(() => { if (!document.hidden && S.page === 'chat' && C.list.some(c => c.streaming)) loadChatList(); }, 4000);

/* ===== start ===== */
(async function init() {
  setTheme(document.documentElement.getAttribute('data-theme') || 'dark');
  setSidebar($('.app').classList.contains('sb-collapsed'), false);
  $('#kbd-b').textContent = MOD + 'B'; $('#kbd-new').textContent = isMac ? '⌘⇧O' : 'Ctrl+Shift+O'; $('#kbd-k').textContent = MOD + 'K';
  newDraft();
  await refresh();
  loadConfigs(); pollDownload(); loadSettings(true);
  go(location.hash.slice(1) || 'chat', false);
})();
