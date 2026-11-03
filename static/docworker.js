'use strict';
/* Reads Word, PowerPoint, Excel, OpenDocument and RTF files for docread.js,
   off the page, so a big or hostile file can't freeze the chat: docread.js
   starts one worker per file and terminates it when done or too slow. The
   libraries load on first use. Output is Markdown-like text: headings, tables
   as | rows |, [Slide n] and [Sheet: name] markers. */

const LIB = { zip: '/static/vendor/fflate-0.8.3/fflate.min.js', xlsx: '/static/vendor/sheetjs-0.20.3/xlsx.full.min.js' };
const loaded = new Set();
function need(k) { if (!loaded.has(k)) { importScripts(LIB[k]); loaded.add(k); } }

const MAX_CHARS = 2300000;          // a bit over what the server keeps; the page cuts exactly
const MAX_ENTRY = 256 << 20;        // one unpacked part of a zip
const MAX_UNPACKED = 512 << 20;     // all parts we unpack

class DocErr extends Error { constructor(msg, reason) { super(msg); this.reason = reason || 'unreadable'; } }

/* ---------- zip ---------- */
const PARTS = /^(_rels\/\.rels|mimetype|content\.xml|word\/[^/]+\.xml|word\/_rels\/[^/]+\.rels|ppt\/presentation\.xml|ppt\/_rels\/presentation\.xml\.rels|ppt\/slides\/[^/]+\.xml|ppt\/slides\/_rels\/[^/]+\.rels|ppt\/notesSlides\/[^/]+\.xml|xl\/workbook\.xml)$/;
const isOle = u => u[0] === 0xD0 && u[1] === 0xCF && u[2] === 0x11 && u[3] === 0xE0;
const isZip = u => u[0] === 0x50 && u[1] === 0x4B;

function unzip(u8) {
  need('zip');
  let total = 0;
  try {
    return fflate.unzipSync(u8, {
      filter: f => {
        if (!PARTS.test(f.name)) return false;
        total += f.originalSize;
        if (f.originalSize > MAX_ENTRY || total > MAX_UNPACKED) throw new DocErr('unpacks to far more data than a document should (it looks like a zip bomb)', 'unreadable');
        return true;
      },
    });
  } catch (e) {
    if (e instanceof DocErr) throw e;
    throw new DocErr('is damaged and could not be unpacked (' + (e && e.message || e) + ')');
  }
}
const utf8 = new TextDecoder('utf-8');
const str = b => (b ? utf8.decode(b) : '');

/* ---------- XML ---------- */
const ENT = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'" };
const cp = n => (n > 0 && n <= 0x10FFFF ? String.fromCodePoint(n) : '');
const unent = s => (s.indexOf('&') < 0 ? s : s.replace(/&(#[xX][0-9a-fA-F]+|#\d+|\w+);/g, (m, e) => e[0] === '#' ? cp(e[1] === 'x' || e[1] === 'X' ? parseInt(e.slice(2), 16) : +e.slice(1)) : (ENT[e] ?? m)));
const ATTR = new Map();
function attr(a, n) {
  let re = ATTR.get(n);
  if (!re) ATTR.set(n, re = new RegExp('(?:^|\\s)' + n.replace(/[.:]/g, '\\$&') + '\\s*=\\s*(?:"([^"]*)"|\'([^\']*)\')'));
  const m = re.exec(a);
  return m ? unent(m[1] ?? m[2]) : null;
}
/* walk calls tag(name, attrs, kind) for each element, kind 1 = open,
   2 = close, 3 = empty, and text(t) for the text between tags. */
const TOK = /<(\/?)([A-Za-z_][\w.:-]*)((?:[^>"']|"[^"]*"|'[^']*')*)>|<!--[\s\S]*?-->|<\?[\s\S]*?\?>|<!\[CDATA\[([\s\S]*?)\]\]>|<![^>]*>|([^<]+)/g;
function walk(xml, tag, text) {
  const re = new RegExp(TOK.source, 'g');
  let m;
  while ((m = re.exec(xml))) {
    if (m[2]) {
      const empty = m[3].endsWith('/');
      tag(m[2], empty ? m[3].slice(0, -1) : m[3], m[1] ? 2 : empty ? 3 : 1);
    } else if (m[4] !== undefined) text(m[4]);
    else if (m[5] !== undefined) text(unent(m[5]));
  }
}

/* ---------- output ---------- */
const cellText = c => c.replace(/\s*\n\s*/g, ' ').replace(/\|/g, '\\|').trim();
function mdTable(rows) {
  rows = rows.filter(r => r.some(c => c.trim()));
  if (!rows.length) return '';
  const w = Math.max(...rows.map(r => r.length));
  const line = r => '| ' + Array.from({ length: w }, (_, i) => cellText(r[i] || '')).join(' | ') + ' |';
  return [line(rows[0]), '|' + ' --- |'.repeat(w), ...rows.slice(1).map(line)].join('\n');
}
// Blocks joins paragraphs with blank lines, list items with single breaks.
// One worker reads one file, so `written` counts everything it produced.
let written = 0;
const full = () => written > MAX_CHARS;
class Blocks {
  constructor() { this.b = []; }
  add(text, list) {
    text = text.replace(/[ \t]+$/gm, '');
    if (!text.trim()) return;
    this.b.push({ text, list });
    written += text.length + 2;
  }
  toString() { return this.b.map((x, i) => (i ? (x.list && this.b[i - 1].list ? '\n' : '\n\n') : '') + x.text).join(''); }
}

/* Paragraphs can nest (a text box inside a paragraph) and tables can nest; a
   nested table is flattened into the cell that holds it. */
function paraTables(target) {
  const paras = [], tables = [];
  const top = a => a[a.length - 1];
  return {
    get para() { return top(paras); },
    openPara(p) { paras.push(Object.assign({ t: '', h: 0, li: false }, p)); },
    closePara() {
      const p = paras.pop(); if (!p) return;
      const t = p.t.trim();
      const tb = top(tables);
      if (tb && tb.cell) { if (t) tb.cell.push(t); return; }
      target().add((p.h ? '#'.repeat(Math.min(p.h, 6)) + ' ' : p.li ? '- ' : '') + t, p.li && !p.h);
    },
    openTable() { tables.push({ rows: [], row: null, cell: null }); },
    openRow() { const tb = top(tables); if (tb) tb.row = []; },
    openCell() { const tb = top(tables); if (tb) tb.cell = []; },
    closeCell() { const tb = top(tables); if (tb && tb.row && tb.cell) tb.row.push(tb.cell.join(' ')); if (tb) tb.cell = null; },
    closeRow() { const tb = top(tables); if (tb && tb.row) tb.rows.push(tb.row); if (tb) tb.row = null; },
    closeTable() {
      const tb = tables.pop(); if (!tb) return;
      const outer = top(tables);
      if (outer && outer.cell) { outer.cell.push(tb.rows.map(r => r.filter(Boolean).join(' / ')).filter(Boolean).join('; ')); return; }
      target().add(mdTable(tb.rows));
    },
  };
}

/* ---------- Word ---------- */
function relMap(xml, base) {
  const m = {};
  walk(xml, (n, a) => {
    if (n !== 'Relationship') return;
    let t = attr(a, 'Target') || '';
    if (attr(a, 'TargetMode') === 'External') return;
    t = t.startsWith('/') ? t.slice(1) : base + t;
    const parts = [];
    for (const s of t.split('/')) { if (s === '..') parts.pop(); else if (s && s !== '.') parts.push(s); }
    m[attr(a, 'Id')] = { target: parts.join('/'), type: attr(a, 'Type') || '' };
  }, () => {});
  return m;
}
function paraStyles(xml) {
  // Style ids are localized ("Heading1", "1", "Titre1"); built-in style names
  // are always English. A style can also make its paragraphs list items.
  const map = {}; let id = null;
  walk(xml, (n, a, k) => {
    if (n === 'w:style') { id = k === 2 ? null : attr(a, 'w:styleId'); if (id) map[id] = { h: 0, li: false }; }
    else if (n === 'w:name' && id) {
      const v = (attr(a, 'w:val') || '').toLowerCase(), m = /^heading\s*(\d)$/.exec(v);
      if (m) map[id].h = +m[1]; else if (v === 'title') map[id].h = 1;
      if (/^list (bullet|number)/.test(v)) map[id].li = true;
    } else if (n === 'w:numPr' && id) map[id].li = true;
  }, () => {});
  return map;
}
// Skipped: the old-format copy Word keeps next to drawings, moved-away text, deleted text.
const WORD_SKIP = new Set(['mc:Fallback', 'w:moveFrom', 'w:del']);
function wordText(xml, styles, out) {
  const pt = paraTables(() => out);
  let inT = false, skip = 0, inTabs = false;
  walk(xml, (n, a, k) => {
    if (full()) return;
    if (WORD_SKIP.has(n) && k !== 3) { skip += k === 1 ? 1 : -1; return; }
    if (skip) return;
    const p = pt.para;
    switch (n) {
      case 'w:p': if (k === 1) pt.openPara(); else if (k === 2) pt.closePara(); break;
      case 'w:pStyle': { const st = p && styles[attr(a, 'w:val')]; if (st) { p.h = st.h; p.li = p.li || st.li; } break; }
      case 'w:numPr': if (p) p.li = true; break;
      case 'w:t': inT = k === 1; break;
      case 'w:tabs': inTabs = k === 1; break;
      case 'w:tab': if (p && !inTabs && k !== 2) p.t += '\t'; break;
      case 'w:br': case 'w:cr': if (p && k !== 2) p.t += '\n'; break;
      case 'w:noBreakHyphen': if (p) p.t += '-'; break;
      case 'w:tbl': if (k === 1) pt.openTable(); else if (k === 2) pt.closeTable(); break;
      case 'w:tr': if (k === 1) pt.openRow(); else if (k === 2) pt.closeRow(); break;
      case 'w:tc': if (k === 1) pt.openCell(); else if (k === 2) pt.closeCell(); break;
    }
  }, t => { if (inT && !skip && pt.para) pt.para.t += t; });
}
function readDocx(z) {
  const root = Object.values(relMap(str(z['_rels/.rels']), '')).find(r => /\/officeDocument$/.test(r.type));
  const main = root && z[root.target] ? root.target : 'word/document.xml';
  if (!z[main]) throw new DocErr("isn't a Word document (its text part is missing)");
  const styles = paraStyles(str(z['word/styles.xml']));
  const out = new Blocks(); wordText(str(z[main]), styles, out);
  const notes = new Blocks();
  for (const f of ['word/footnotes.xml', 'word/endnotes.xml']) if (z[f]) wordText(str(z[f]), styles, notes);
  const text = String(out) + (notes.b.length ? '\n\n[Footnotes]\n' + notes : '');
  return { text, truncated: full() };
}

/* ---------- PowerPoint ---------- */
const PH_SKIP = new Set(['sldNum', 'ftr', 'dt', 'hdr', 'sldImg']);
function slideText(xml) {
  const out = new Blocks(), pt = paraTables(() => out);
  let ph = null, inT = false;
  walk(xml, (n, a, k) => {
    switch (n) {
      case 'p:sp': case 'p:graphicFrame': case 'p:pic': case 'p:cxnSp': if (k === 1) ph = null; break;
      case 'p:ph': ph = attr(a, 'type') || 'body'; break;
      case 'a:p':
        if (k === 1) pt.openPara({ h: ph === 'title' || ph === 'ctrTitle' ? 1 : 0, skip: PH_SKIP.has(ph) });
        else if (k === 2) { if (pt.para && pt.para.skip) pt.para.t = ''; pt.closePara(); }
        break;
      case 'a:t': inT = k === 1; break;
      case 'a:br': if (pt.para) pt.para.t += '\n'; break;
      case 'a:tbl': if (k === 1) pt.openTable(); else if (k === 2) pt.closeTable(); break;
      case 'a:tr': if (k === 1) pt.openRow(); else if (k === 2) pt.closeRow(); break;
      case 'a:tc': if (k === 1) pt.openCell(); else if (k === 2) pt.closeCell(); break;
    }
  }, t => { if (inT && pt.para) pt.para.t += t; });
  return String(out);
}
function readPptx(z) {
  const rels = relMap(str(z['ppt/_rels/presentation.xml.rels']), 'ppt/');
  let order = [];
  walk(str(z['ppt/presentation.xml']), (n, a) => { if (n === 'p:sldId' && rels[attr(a, 'r:id')]) order.push(rels[attr(a, 'r:id')].target); }, () => {});
  if (!order.length) order = Object.keys(z).filter(f => /^ppt\/slides\/slide\d+\.xml$/.test(f)).sort((a, b) => parseInt(a.slice(16)) - parseInt(b.slice(16)));
  const parts = [];
  order.forEach((path, i) => {
    if (full() || !z[path]) return;
    const body = slideText(str(z[path]));
    const srels = relMap(str(z[path.replace(/([^/]+)$/, '_rels/$1.rels')]), path.replace(/[^/]+$/, ''));
    const nr = Object.values(srels).find(r => /\/notesSlide$/.test(r.type));
    const notes = nr && z[nr.target] ? slideText(str(z[nr.target])).replace(/^#+ /gm, '') : '';
    const part = `[Slide ${i + 1}]` + (body ? '\n' + body : '') + (notes ? '\n\nNotes: ' + notes : '');
    parts.push(part);
  });
  return { text: parts.join('\n\n'), slides: order.length, truncated: full() };
}

/* ---------- OpenDocument (Writer and Impress) ---------- */
// Skipped: footnote bodies (they'd land mid-sentence), comments, the text of tracked deletions.
const ODF_SKIP = new Set(['text:note', 'office:annotation', 'text:tracked-changes']);
function readOdf(xml, slides) {
  const doc = new Blocks(), pages = [];
  let notes = null, skip = 0, li = 0;
  const pt = paraTables(() => notes || pages[pages.length - 1] || doc);
  walk(xml, (n, a, k) => {
    if (full()) return;
    if (n === 'presentation:notes') {
      if (k === 1) notes = new Blocks();
      else if (k === 2) { const pg = pages[pages.length - 1]; if (pg) pg.notes = String(notes); notes = null; }
      return;
    }
    if (ODF_SKIP.has(n) && k !== 3) { skip += k === 1 ? 1 : -1; return; }
    if (skip) return;
    const p = pt.para;
    switch (n) {
      case 'draw:page': if (k === 1) pages.push(new Blocks()); break;
      case 'text:h': if (k === 1) pt.openPara({ h: +(attr(a, 'text:outline-level') || 1) }); else if (k === 2) pt.closePara(); break;
      case 'text:p': if (k === 1) pt.openPara({ li: li > 0 }); else if (k === 2) pt.closePara(); break;
      case 'text:list-item': li += k === 1 ? 1 : k === 2 ? -1 : 0; break;
      case 'text:s': if (p) p.t += ' '.repeat(Math.min(+(attr(a, 'text:c') || 1), 100)); break;
      case 'text:tab': if (p) p.t += '\t'; break;
      case 'text:line-break': if (p) p.t += '\n'; break;
      case 'table:table': if (k === 1) pt.openTable(); else if (k === 2) pt.closeTable(); break;
      case 'table:table-row': if (k === 1) pt.openRow(); else if (k === 2) pt.closeRow(); break;
      case 'table:table-cell': case 'table:covered-table-cell':
        if (k === 1) pt.openCell(); else if (k === 2) pt.closeCell(); else { pt.openCell(); pt.closeCell(); }
        break;
    }
  }, t => { if (!skip && pt.para) pt.para.t += t.replace(/[ \t\r\n]+/g, ' '); });
  if (!slides) return { text: String(doc), truncated: full() };
  const text = pages.map((s, i) => `[Slide ${i + 1}]` + (s.b.length ? '\n' + s : '') + (s.notes ? '\n\nNotes: ' + s.notes : '')).join('\n\n');
  return { text: text + (doc.b.length ? '\n\n' + doc : ''), slides: pages.length, truncated: full() };
}

/* ---------- spreadsheets ---------- */
function readSheets(u8) {
  need('xlsx');
  let wb;
  try {
    wb = XLSX.read(u8, { type: 'array', dense: true, sheetRows: 100001, cellFormula: false, cellHTML: false, cellStyles: false, bookVBA: false });
  } catch (e) {
    const m = String(e && e.message || e);
    if (/password|encrypt/i.test(m)) throw new DocErr('is password-protected. Remove the password and attach it again.', 'encrypted');
    throw new DocErr("couldn't be read as a spreadsheet (" + m + ')');
  }
  const parts = []; let size = 0, cut = false;
  wb.SheetNames.forEach((n, i) => {
    if (size > MAX_CHARS) { cut = true; return; }
    const ws = wb.Sheets[n], meta = wb.Workbook && wb.Workbook.Sheets && wb.Workbook.Sheets[i];
    if (ws['!fullref'] && ws['!fullref'] !== ws['!ref']) cut = true;
    const csv = XLSX.utils.sheet_to_csv(ws, { blankrows: false, strip: true }).trim();
    const part = `[Sheet: ${n}${meta && meta.Hidden ? ', hidden' : ''}]\n${csv || '(empty)'}`;
    parts.push(part); size += part.length;
  });
  return { text: parts.join('\n\n'), sheets: wb.SheetNames.length, truncated: cut };
}

/* ---------- RTF ---------- */
const RTF_SKIP = new Set(['fonttbl', 'colortbl', 'stylesheet', 'info', 'pict', 'object', 'header', 'headerl', 'headerr', 'headerf', 'footer', 'footerl', 'footerr', 'footerf', 'themedata', 'colorschememapping', 'datastore', 'latentstyles', 'listtable', 'listoverridetable', 'rsidtbl', 'generator', 'xmlnstbl', 'mmathPr', 'fldinst', 'listtext', 'pntext', 'pntxta', 'pntxtb', 'filetbl', 'revtbl', 'userprops', 'expandedcolortbl']);
const RTF_CHAR = { par: '\n', line: '\n', sect: '\n\n', page: '\n\n', row: '\n', cell: ' | ', tab: '\t', emdash: '\u2014', endash: '\u2013', bullet: '\u2022', lquote: '\u2018', rquote: '\u2019', ldblquote: '\u201C', rdblquote: '\u201D', emspace: ' ', enspace: ' ', qmspace: ' ' };
function codepage(n) {
  const name = n === 65001 ? 'utf-8' : n === 10000 ? 'macintosh' : n === 932 ? 'shift_jis' : n === 936 ? 'gbk' : n === 949 ? 'euc-kr' : n === 950 ? 'big5' : 'windows-' + n;
  try { return new TextDecoder(name); } catch (_) { return new TextDecoder('windows-1252'); }
}
function readRtf(u8) {
  const s = new TextDecoder('latin1').decode(u8);
  if (!s.startsWith('{\\rtf')) throw new DocErr("isn't an RTF document");
  let dec = codepage(1252), out = '', bytes = [], st = { skip: false, uc: 1 }, skipN = 0;
  const stack = [];
  const flush = () => { if (bytes.length) { out += dec.decode(new Uint8Array(bytes)); bytes = []; } };
  const emit = t => { if (skipN > 0) { skipN--; return; } if (!st.skip) { flush(); out += t; } };
  for (let i = 0; i < s.length && out.length < MAX_CHARS; i++) {
    const c = s[i];
    if (c === '{') { flush(); stack.push(st); st = { skip: st.skip, uc: st.uc, first: true }; continue; }
    if (c === '}') { flush(); st = stack.pop() || st; continue; }
    if (c === '\r' || c === '\n') continue;
    if (c !== '\\') { emit(c); st.first = false; continue; }
    const d = s[i + 1];
    if (d === '\\' || d === '{' || d === '}') { emit(d); i++; st.first = false; continue; }
    if (d === '\n' || d === '\r') { emit('\n'); i++; continue; } // a backslash before a line break means \\par
    if (d === '~') { emit(' '); i++; continue; }
    if (d === '_') { emit('-'); i++; continue; }
    if (d === '-') { i++; continue; }
    if (d === '*') { st.skip = true; i++; continue; }
    if (d === "'") {
      const b = parseInt(s.substr(i + 2, 2), 16); i += 3;
      if (skipN > 0) { skipN--; continue; }
      if (!st.skip && !isNaN(b)) bytes.push(b);
      continue;
    }
    const m = /^([a-zA-Z]{1,32})(-?\d{1,10})? ?/.exec(s.slice(i + 1, i + 46));
    if (!m) { i++; continue; }
    i += m[0].length;
    const w = m[1], n = m[2] === undefined ? null : +m[2];
    if (st.first && RTF_SKIP.has(w)) st.skip = true;
    st.first = false;
    if (w === 'ansicpg' && n) dec = codepage(n);
    else if (w === 'uc' && n !== null) st.uc = n;
    else if (w === 'u' && n !== null) { if (!st.skip) { flush(); out += String.fromCharCode(n < 0 ? n + 65536 : n); } skipN = st.uc; }
    else if (w === 'bin' && n) i += n;
    else if (RTF_CHAR[w] !== undefined) emit(RTF_CHAR[w]);
  }
  flush();
  return { text: out.replace(/ \| \n/g, ' |\n') };
}

/* ---------- entry ---------- */
function read(u8, fmt) {
  if (fmt === 'rtf') return readRtf(u8);
  if (fmt === 'sheet') return readSheets(u8);
  if (isOle(u8)) throw new DocErr('is password-protected, or an old .doc/.ppt file under a new name. Remove the password or save it again as .docx/.pptx.', 'encrypted');
  if (!isZip(u8)) throw new DocErr("isn't a valid document: the file is damaged or has the wrong extension");
  const z = unzip(u8);
  if (z['ppt/presentation.xml']) return readPptx(z);
  if (z['xl/workbook.xml']) return readSheets(u8);
  if (z.mimetype) {
    const mt = str(z.mimetype).trim();
    if (!z['content.xml']) throw new DocErr('is missing its content');
    if (/opendocument\.(text|text-template)$/.test(mt)) return readOdf(str(z['content.xml']), false);
    if (/opendocument\.(presentation|presentation-template)$/.test(mt)) return readOdf(str(z['content.xml']), true);
    throw new DocErr(`is an OpenDocument type the browser can't read (${mt})`);
  }
  return readDocx(z);
}

self.onmessage = e => {
  const { buf, fmt } = e.data;
  try {
    self.postMessage({ ok: true, res: read(new Uint8Array(buf), fmt) });
  } catch (err) {
    self.postMessage({ ok: false, error: err instanceof DocErr ? err.message : "couldn't be read (" + (err && err.message || err) + ')', reason: err && err.reason || 'unreadable' });
  }
};
