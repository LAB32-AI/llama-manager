'use strict';
/* Reads attached documents in the browser, for when Tika isn't set up or
   doesn't answer: PDF with pdf.js here, the rest in docworker.js (one worker
   per file). Nothing loads until a document is attached. */
const DocRead = (() => {
  const V = document.documentElement.dataset.v || '';
  const PDFJS = '/static/vendor/pdfjs-6.5.438/';
  const TIMEOUT = 90000;
  const NOTE = "\n\n[llama-manager cut the text here: it sends at most 2 MB of a document's text]";

  // extension -> how the browser reads it
  const BROWSER = {
    pdf: 'pdf', rtf: 'rtf',
    docx: 'zip', docm: 'zip', dotx: 'zip', dotm: 'zip', pptx: 'zip', pptm: 'zip', ppsx: 'zip', ppsm: 'zip', potx: 'zip',
    odt: 'zip', ott: 'zip', odp: 'zip', otp: 'zip',
    xlsx: 'sheet', xlsm: 'sheet', xlsb: 'sheet', xltx: 'sheet', xls: 'sheet', ods: 'sheet', ots: 'sheet', fods: 'sheet', numbers: 'sheet',
  };
  // documents only Tika reads
  const TIKA = new Set(['doc', 'dot', 'ppt', 'pps', 'pot', 'msg', 'eml', 'mbox', 'epub', 'fb2', 'pages', 'key', 'odg', 'wpd', 'wps', 'xps', 'oxps',
    'vsd', 'vsdx', 'pub', 'one', 'chm', 'mobi', 'zip', '7z', 'rar', 'tar', 'gz', 'tgz', 'bz2', 'xz', 'tif', 'tiff', 'heic', 'djvu']);
  const MIMES = {
    'application/pdf': 'pdf', 'application/rtf': 'rtf', 'text/rtf': 'rtf', 'application/msword': 'doc', 'application/vnd.ms-excel': 'xls', 'application/vnd.ms-powerpoint': 'ppt',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'pptx', 'application/vnd.oasis.opendocument.text': 'odt',
    'application/vnd.oasis.opendocument.spreadsheet': 'ods', 'application/vnd.oasis.opendocument.presentation': 'odp', 'application/epub+zip': 'epub', 'message/rfc822': 'eml',
  };
  const LABEL = {
    pdf: 'PDF', rtf: 'RTF', doc: 'Word', docx: 'Word', docm: 'Word', dot: 'Word', dotx: 'Word', dotm: 'Word', xls: 'Excel', xlsx: 'Excel', xlsm: 'Excel', xlsb: 'Excel', xltx: 'Excel',
    ppt: 'PowerPoint', pps: 'PowerPoint', pptx: 'PowerPoint', pptm: 'PowerPoint', ppsx: 'PowerPoint', potx: 'PowerPoint', odt: 'Writer', ott: 'Writer', ods: 'Calc', fods: 'Calc',
    odp: 'Impress', odg: 'Draw', epub: 'EPUB', eml: 'Email', msg: 'Outlook message', numbers: 'Numbers', pages: 'Pages', key: 'Keynote', zip: 'ZIP', csv: 'CSV',
  };
  const extOf = name => ((/\.([a-z0-9]{1,8})$/i.exec(name || '') || [])[1] || '').toLowerCase();
  const typeOf = f => { const e = extOf(f.name); return BROWSER[e] || TIKA.has(e) ? e : MIMES[(f.type || '').split(';')[0]] || e; };

  /* where(f): 'browser' if the browser can read it, 'tika' if only Tika can,
     '' if it isn't a known document (it may still be text). */
  const where = f => { const t = typeOf(f); return BROWSER[t] ? 'browser' : TIKA.has(t) ? 'tika' : ''; };
  const label = f => LABEL[typeOf(f)] || (extOf(f.name) || 'File').toUpperCase();

  function err(msg, reason) { const e = new Error(msg); e.reason = reason || 'unreadable'; return e; }
  const aborted = () => { const e = new Error('cancelled'); e.name = 'AbortError'; return e; };

  /* Same tidying as the server's cleanDocText, so token estimates match. */
  function clean(s, pdf) {
    s = String(s).replace(/\r\n?/g, '\n').replace(/[\f\v\u2028\u2029]/g, '\n').replace(/[\u00a0\u2007\u202f]/g, ' ')
      .replace(/\u00ad/g, pdf ? '-' : '').replace(/[\ufeff\u200b\x00-\x08\x0e-\x1f\x7f-\x9f]/g, '');
    return s.replace(/[ \t]+$/gm, '').replace(/\n{3,}/g, '\n\n').trim();
  }
  /* cut keeps at most maxBytes of UTF-8. */
  function cut(s, maxBytes) {
    if (s.length * 3 <= maxBytes) return [s, false];
    const budget = maxBytes - new TextEncoder().encode(NOTE).length;
    let bytes = 0, i = 0;
    for (const ch of s) {
      const c = ch.codePointAt(0), n = c < 0x80 ? 1 : c < 0x800 ? 2 : c < 0x10000 ? 3 : 4;
      if (bytes + n > budget) break;
      bytes += n; i += ch.length;
    }
    if (i >= s.length) return [s, false];
    let t = s.slice(0, i);
    const nl = t.lastIndexOf('\n'); if (nl > t.length * 0.9) t = t.slice(0, nl);
    return [t.trimEnd() + NOTE, true];
  }
  /* A rough token count: about 4 Latin characters, 2.6 Cyrillic/Greek/Arabic
     ones or one CJK character per token, and one per digit (Qwen and others
     split numbers into digits, so tables of numbers cost far more). It errs high. */
  function tokens(s) {
    let a = 0, d = 0, o = 0, c = 0;
    for (let i = 0; i < s.length; i++) { const k = s.charCodeAt(i); if (k >= 48 && k <= 57) d++; else if (k < 0x80) a++; else if (k < 0x3000) o++; else c++; }
    return Math.ceil(a / 4 + d + o / 2.6 + c * 1.1);
  }

  /* ---------- PDF ---------- */
  let lib = null;
  const pdfjs = () => lib || (lib = import(PDFJS + 'pdf.min.mjs').then(m => { m.GlobalWorkerOptions.workerSrc = PDFJS + 'pdf.worker.min.mjs'; return m; }).catch(e => { lib = null; throw e; }));
  async function openPdf(file, signal) {
    const m = await pdfjs();
    const task = m.getDocument({ data: new Uint8Array(await file.arrayBuffer()), wasmUrl: PDFJS + 'wasm/', disableFontFace: true, enableXfa: false, stopAtErrors: false, verbosity: 0 });
    const stop = () => task.destroy();
    signal && signal.addEventListener('abort', stop, { once: true });
    const timer = setTimeout(stop, TIMEOUT);
    try { return { doc: await task.promise, done: () => { clearTimeout(timer); signal && signal.removeEventListener('abort', stop); task.destroy(); } }; }
    catch (e) {
      clearTimeout(timer);
      if (signal && signal.aborted) throw aborted();
      if (e && e.name === 'PasswordException') throw err('is password-protected. Remove the password and attach it again.', 'encrypted');
      if (e && e.name === 'InvalidPDFException') throw err("isn't a valid PDF: the file is damaged or has the wrong extension");
      throw err("couldn't be read: " + (e && e.message || e));
    }
  }
  async function readPdf(file, signal, maxBytes) {
    const { doc, done } = await openPdf(file, signal);
    try {
      const pages = []; let size = 0, textPages = 0;
      for (let i = 1; i <= doc.numPages; i++) {
        if (signal && signal.aborted) throw aborted();
        const page = await doc.getPage(i);
        const tc = await page.getTextContent();
        let t = '';
        for (const it of tc.items) if (typeof it.str === 'string') t += it.str + (it.hasEOL ? '\n' : '');
        page.cleanup();
        t = clean(t, true);
        if (t) textPages++;
        pages.push(doc.numPages > 1 ? `[Page ${i}]\n${t}` : t);
        if ((size += t.length) > maxBytes) break;
      }
      return { text: textPages ? pages.join('\n\n') : '', pages: doc.numPages, type: 'application/pdf' };
    } catch (e) {
      if (signal && signal.aborted) throw aborted();
      throw e.reason ? e : err("couldn't be read: " + (e && e.message || e));
    } finally { done(); }
  }
  /* pdfImages renders the first pages of a PDF without text (a scan) as
     JPEGs, for models that can read images. */
  async function pdfImages(file, max, signal) {
    const { doc, done } = await openPdf(file, signal);
    try {
      const n = Math.min(doc.numPages, max), out = [], base = file.name.replace(/\.pdf$/i, '');
      for (let i = 1; i <= n; i++) {
        if (signal && signal.aborted) throw aborted();
        const page = await doc.getPage(i);
        const one = page.getViewport({ scale: 1 });
        const vp = page.getViewport({ scale: Math.min(2.5, 1568 / Math.max(one.width, one.height)) });
        const canvas = document.createElement('canvas');
        canvas.width = Math.ceil(vp.width); canvas.height = Math.ceil(vp.height);
        await page.render({ canvas, viewport: vp, background: '#fff' }).promise;
        page.cleanup();
        const blob = await new Promise(r => canvas.toBlob(r, 'image/jpeg', 0.85));
        if (blob) out.push(new File([blob], `${base} page ${i}.jpg`, { type: 'image/jpeg' }));
      }
      return { files: out, total: doc.numPages };
    } finally { done(); }
  }

  /* ---------- everything else ---------- */
  function inWorker(file, fmt, signal) {
    return new Promise((resolve, reject) => {
      const w = new Worker('/static/docworker.js?v=' + V);
      let over = false;
      const finish = (fn, v) => { if (over) return; over = true; clearTimeout(timer); w.terminate(); signal && signal.removeEventListener('abort', onAbort); fn(v); };
      const onAbort = () => finish(reject, aborted());
      const timer = setTimeout(() => finish(reject, err(`took more than ${TIMEOUT / 1000} seconds to read, so reading stopped`, 'timeout')), TIMEOUT);
      w.onmessage = e => e.data.ok ? finish(resolve, e.data.res) : finish(reject, err(e.data.error, e.data.reason));
      w.onerror = e => { e.preventDefault(); finish(reject, err("couldn't be read: the reader stopped (" + (e.message || 'out of memory?') + ')')); };
      signal && signal.addEventListener('abort', onAbort, { once: true });
      file.arrayBuffer().then(buf => { if (!over) w.postMessage({ buf, fmt }, [buf]); }, e => finish(reject, err("couldn't be opened: " + e.message)));
    });
  }

  /* read returns { text, type, pages|slides|sheets, truncated, via } with text
     cleaned and cut to maxBytes. The text is '' when the file has none (a
     scanned PDF). Errors carry .reason: encrypted, timeout or unreadable. */
  // Two files at a time: eight big PDFs at once would swamp a phone.
  let running = 0;
  const queue = [];
  async function slot(signal) {
    while (running >= 2) {
      await new Promise(r => queue.push(r));
      if (signal && signal.aborted) { const next = queue.shift(); if (next) next(); throw aborted(); }
    }
    running++;
  }
  function release() { running--; const next = queue.shift(); if (next) next(); }
  async function read(file, signal, maxBytes) {
    const fmt = BROWSER[typeOf(file)];
    if (!fmt) throw err("can't be read in the browser", 'unsupported');
    await slot(signal);
    try {
      const r = fmt === 'pdf' ? await readPdf(file, signal, maxBytes) : await inWorker(file, fmt, signal);
      const [text, wasCut] = cut(clean(r.text || '', fmt === 'pdf'), maxBytes);
      return { ...r, text, type: r.type || file.type || '', truncated: !!(r.truncated || wasCut), via: 'browser' };
    } finally { release(); }
  }

  return { where, label, read, pdfImages, clean, cut, tokens, typeOf };
})();
