import { html, render, useState, useEffect, useRef, useCallback, useLayoutEffect } from "/static/vendor/preact-htm.module.js";

// ---- api ------------------------------------------------------------------------------
async function api(path, opts = {}) {
  const res = await fetch(path, { credentials: "same-origin", headers: { "Content-Type": "application/json" }, ...opts });
  const body = res.headers.get("content-type")?.includes("json") ? await res.json() : null;
  if (!res.ok) throw Object.assign(new Error((body && body.detail) || `HTTP ${res.status}`), { status: res.status });
  return body;
}
const PRESETS = [
  { id: "nvidia", name: "NVIDIA", base_url: "https://integrate.api.nvidia.com/v1", model: "nvidia/nemotron-3.5-super-vl-120b-a12b", hint: "nvapi-… from build.nvidia.com" },
  { id: "openrouter", name: "OpenRouter", base_url: "https://openrouter.ai/api/v1", model: "openai/gpt-4o-mini", hint: "sk-or-… from openrouter.ai" },
  { id: "openai", name: "OpenAI", base_url: "https://api.openai.com/v1", model: "gpt-4o-mini", hint: "sk-… from platform.openai.com" },
  { id: "anthropic", name: "Anthropic (OpenAI-compatible)", base_url: "https://api.anthropic.com/v1", model: "claude-sonnet-4-5", hint: "sk-ant-… API key" },
  { id: "custom", name: "Other OpenAI-compatible", base_url: "https://", model: "", hint: "the endpoint's API key (leave empty if none)" },
];
const MODES = [
  ["off", "Off", "Requests are sent as they are."],
  ["mask", "Mask", "Emails, phones, account and ID numbers become placeholders; real values return in the reply."],
  ["mask_check", "Mask + check", "As Mask, and requests that still describe a person are refused (Laya)."],
  ["keep_private", "Keep private", "Personal requests go only to upstreams marked private, unmasked."],
];
const BUILTINS = ["email", "phone", "iban", "payment_card", "us_ssn", "uk_nino", "ie_ppsn", "passport", "date_of_birth"];
const EXAMPLES = [
  ["Email + phone", "Write a short note to ann.kelly@acme.ie confirming we will call her back on +353 87 123 4567 tomorrow."],
  ["IBAN + PPSN", "Refund the customer to IBAN IE29 AIBK 9311 5212 3456 78; their PPSN is 1234567T."],
  ["Claim number", "Approve claim CLM-2026-0048213 and tell the adjuster."],
  ["Name + health", "Our employee Priya Nair has been on sick leave for depression since June; plan her return-to-work."],
  ["Clean", "Explain the difference between TCP and UDP in two sentences."],
];
const ago = (ts) => { const s = Math.max(0, Math.round(Date.now() / 1000 - ts)); return s < 60 ? `${s}s ago` : s < 3600 ? `${Math.round(s / 60)}m ago` : `${Math.round(s / 3600)}h ago`; };
const host = (u) => { try { return new URL(u).host; } catch { return u; } };

// ---- login ----------------------------------------------------------------------------
function Login({ onIn }) {
  const [key, setKey] = useState(""); const [err, setErr] = useState("");
  const go = async (e) => { e.preventDefault(); setErr(""); try { await api("/api/login", { method: "POST", body: JSON.stringify({ key }) }); onIn(); } catch (x) { setErr(x.message); } };
  return html`<form class="panel login" onSubmit=${go}>
    <div class="brand" style="margin-bottom:12px">Inference Router</div>
    <label class="field"><span>Admin key</span><input type="password" value=${key} onInput=${(e) => setKey(e.target.value)} autofocus /></label>
    ${err && html`<div class="err">${err}</div>`}
    <button class="btn primary">Sign in</button>
    <p class="note">The admin key is LITELLM_MASTER_KEY in /etc/inference-router/secrets.env.</p>
  </form>`;
}

// ---- overview -------------------------------------------------------------------------
function resolveUpstream(status, ev) {
  if (!ev.target || !status) return null;
  const m = ev.target.match(/^ir\/(.+?)\/(private\/)?(\d+)$/); if (!m) return null;
  const route = status.routes.find((r) => r.name === m[1]); if (!route) return null;
  const ups = Object.fromEntries(status.upstreams.map((u) => [u.id, u]));
  const live = route.targets.filter((t) => ups[t.upstream]?.enabled && (!m[2] || ups[t.upstream]?.private));
  return live[Number(m[3])]?.upstream || null;
}

function Job({ ev, upName, refFn }) {
  const cls = ev.decision === "blocked" ? "blocked" : ev.outcome === "error" ? "error" : ev.decision === "private" ? "private" : "";
  const masked = Object.entries(ev.masked || {}).map(([k, v]) => `${v} ${k.toUpperCase()}`).join(", ");
  return html`<div class=${`job ${cls}`} ref=${refFn}>
    <div class="title">${ev.route || "unknown model"}
      ${ev.decision === "blocked" ? html`<span class="pill bad">refused</span>` : ev.reason === "masked" ? html`<span class="pill ok">masked</span>` : ev.decision === "private" ? html`<span class="pill private">kept private</span>` : html`<span class="pill">clean</span>`}
      ${ev.outcome === "error" && html`<span class="pill warn">error</span>`}</div>
    <div class="meta">
      ${ev.key ? html`From ${ev.key}<br />` : ""}
      ${ev.decision === "blocked" ? html`Not sent · ${ev.reason}` : html`Sent to ${upName || "—"}${ev.served_by ? html` · ${ev.served_by}` : ""}`}<br />
      ${masked && html`${masked} hidden<br />`}
      ${ev.total_ms != null ? `${(ev.total_ms / 1000).toFixed(1)} s · ` : ""}${ago(ev.ts)}${ev.stream ? " · stream" : ""}
    </div></div>`;
}

function UpstreamCard({ up, stats, refFn, onTest, testing }) {
  const t = up.test;
  const healthy = t ? t.ok : null;
  return html`<div class="card" ref=${refFn}>
    <div class="head">
      <div class=${`ring ${healthy ? "ok" : ""} ${up.private ? "private" : ""}`}><span class=${`dot ${healthy ? "ok" : healthy === false ? "bad" : ""}`}></span></div>
      <div style="flex:1">
        <div class="row"><span class="name">${up.name}</span>
          ${up.private && html`<span class="pill private">private</span>`}
          ${!up.enabled && html`<span class="pill warn">disabled</span>`}
          ${!up.key_set && html`<span class="pill warn">no key</span>`}</div>
        <div class="sub">${host(up.base_url)}${t ? html` · last test ${ago(t.at)}: ${t.ok ? `OK in ${t.latency_ms} ms` : (t.error || `HTTP ${t.chat_status || t.models_status}`)}` : ""}</div>
      </div>
      <button class="btn small" onClick=${onTest} disabled=${testing}>${testing ? "Testing…" : "Test"}</button>
    </div>
    <div class="stats">
      <div class="stat"><div class="v">${stats.requests}</div><div class="l">requests (live)</div></div>
      <div class="stat"><div class="v">${stats.masked}</div><div class="l">masked</div></div>
      <div class="stat"><div class="v">${stats.errors}</div><div class="l">errors</div></div>
      <div class="stat"><div class="v">${stats.avg ? (stats.avg / 1000).toFixed(1) + " s" : "—"}</div><div class="l">avg time</div></div>
    </div>
    <div class="models">${up.models.map((m) => html`<span class="pill mono">${m}</span>`)}</div>
  </div>`;
}

function Overview({ status, events, reload }) {
  const [testing, setTesting] = useState({});
  const box = useRef(null); const jobRefs = useRef({}); const cardRefs = useRef({});
  const [paths, setPaths] = useState([]);
  const shown = events.slice(-8).reverse();
  const stats = {};
  for (const u of status.upstreams) stats[u.id] = { requests: 0, masked: 0, errors: 0, total: 0, n: 0, avg: 0 };
  for (const ev of events) {
    const id = resolveUpstream(status, ev); if (!id || !stats[id]) continue;
    const s = stats[id]; s.requests++; if (ev.reason === "masked") s.masked++; if (ev.outcome === "error") s.errors++;
    if (ev.total_ms != null) { s.total += ev.total_ms; s.n++; s.avg = s.total / s.n; }
  }
  useLayoutEffect(() => {
    if (!box.current) return;
    const base = box.current.getBoundingClientRect(); const out = [];
    shown.forEach((ev, i) => {
      const id = resolveUpstream(status, ev); const j = jobRefs.current[i]; const c = id && cardRefs.current[id];
      if (!j || !c) return;
      const a = j.getBoundingClientRect(), b = c.getBoundingClientRect();
      const x1 = a.right - base.left, y1 = a.top + a.height / 2 - base.top, x2 = b.left - base.left, y2 = b.top + 36 - base.top;
      const mx = (x1 + x2) / 2;
      out.push({ d: `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`, color: ev.decision === "private" ? "#b48cff" : "#5aa9ff" });
    });
    setPaths(out);
  }, [events, status]);
  const test = async (id) => { setTesting((t) => ({ ...t, [id]: true })); try { await api(`/api/upstreams/${id}/test`, { method: "POST", body: "{}" }); } catch {} setTesting((t) => ({ ...t, [id]: false })); reload(); };
  const ups = Object.fromEntries(status.upstreams.map((u) => [u.id, u]));
  return html`<div class="overview" ref=${box}>
    <svg class="links">${paths.map((p) => html`<path d=${p.d} fill="none" stroke=${p.color} stroke-width="1.6" opacity="0.75" />`)}</svg>
    <div class="feed">
      <h3>Requests <span class="pill">live · nothing stored</span></h3>
      ${shown.length === 0 ? html`<div class="empty">No requests yet</div>` : shown.map((ev, i) =>
        html`<${Job} ev=${ev} upName=${ups[resolveUpstream(status, ev)]?.name} refFn=${(el) => (jobRefs.current[i] = el)} />`)}
    </div>
    <div class="cards">
      <h3 class="section-title">Upstreams</h3>
      ${status.upstreams.length === 0 && html`<div class="empty">No upstreams yet. Add one under Settings.</div>`}
      ${status.upstreams.map((u) => html`<${UpstreamCard} up=${u} stats=${stats[u.id]} testing=${testing[u.id]}
          refFn=${(el) => (cardRefs.current[u.id] = el)} onTest=${() => test(u.id)} />`)}
    </div>
  </div>`;
}

// ---- experimenter ---------------------------------------------------------------------
function Highlighted({ text, spans }) {
  const parts = []; let at = 0;
  for (const s of [...spans].sort((a, b) => a.start - b.start)) {
    if (s.start < at) continue;
    parts.push(text.slice(at, s.start));
    parts.push(html`<mark>${text.slice(s.start, s.end)}<span class="t">${s.type.toUpperCase()}</span></mark>`);
    at = s.end;
  }
  parts.push(text.slice(at));
  return html`<div class="text">${parts}</div>`;
}
const withPlaceholders = (text) => String(text || "").split(/(<[A-Z][A-Z_]*_[0-9a-f]{8,16}>)/).map((p, i) => i % 2 ? html`<span class="ph">${p}</span>` : p);

function Experimenter({ status }) {
  const [route, setRoute] = useState(status.routes[0]?.name || "");
  const [mode, setMode] = useState("");
  const [text, setText] = useState(EXAMPLES[0][1]);
  const [res, setRes] = useState(null); const [err, setErr] = useState(""); const [busy, setBusy] = useState(false);
  const run = async (send) => {
    setBusy(true); setErr("");
    try { setRes(await api("/api/experiment", { method: "POST", body: JSON.stringify({ route, mode: mode || undefined, send, messages: [{ role: "user", content: text }] }) })); }
    catch (x) { setErr(x.message); setRes(null); }
    setBusy(false);
  };
  const lastUser = res?.would_send ? [...res.would_send].reverse().find((m) => m.role === "user") : null;
  return html`<div class="grid2">
    <div class="panel">
      <div class="section-title">Try a message <span class="pill">dry run by default · nothing is stored</span></div>
      <div class="row" style="margin-bottom:10px">${EXAMPLES.map(([label, t]) => html`<button class="btn small" onClick=${() => setText(t)}>${label}</button>`)}</div>
      <label class="field"><span>Message</span><textarea value=${text} onInput=${(e) => setText(e.target.value)} /></label>
      <div class="grid2">
        <label class="field"><span>Route (model name)</span><select value=${route} onChange=${(e) => setRoute(e.target.value)}>
          ${status.routes.map((r) => html`<option value=${r.name}>${r.name}</option>`)}</select></label>
        <label class="field"><span>Privacy</span><select value=${mode} onChange=${(e) => setMode(e.target.value)}>
          <option value="">Current setting (${status.privacy.mode})</option>
          ${MODES.map(([v, l]) => html`<option value=${v}>${l}</option>`)}</select></label>
      </div>
      <div class="row"><button class="btn primary" onClick=${() => run(false)} disabled=${busy || !route}>Dry run</button>
        <button class="btn" onClick=${() => run(true)} disabled=${busy || !route}>Send for real</button>
        ${busy && html`<span class="note">working…</span>`}</div>
      <p class="note">Send for real calls the route's first upstream with the masked message and shows both replies here only.</p>
      ${err && html`<div class="err">${err}</div>`}
    </div>
    <div class="steps">
      ${!res ? html`<div class="panel empty">Run an example to see what is detected, what would leave the server, and where it would go.</div>` : html`
        <div class="step"><h4>1 · Detected</h4>
          ${(res.detected[0]?.spans?.length) ? html`<${Highlighted} text=${text} spans=${res.detected[0].spans} />` : html`<div class="note">Nothing detected.</div>`}</div>
        <div class="step"><h4>2 · What leaves the server</h4>
          ${res.decision === "blocked" ? html`<div class="note">Nothing: the request is refused.</div>`
            : html`<div class="text">${withPlaceholders(lastUser?.content)}</div>
              ${res.masked_values ? html`<p class="note">${res.masked_values} value(s) replaced; the model also gets a short note on how to treat placeholders.</p>` : ""}`}</div>
        <div class="step"><h4>3 · Personal-data check</h4>
          <div>${res.laya_max == null ? html`<span class="note">Laya not used in this mode.</span>` : html`Laya score ${res.laya_max} (refused at 0.4 or above)`}</div>
          ${res.reasons?.length ? html`<div class="note">Reasons: ${res.reasons.join(", ")}</div>` : ""}</div>
        <div class="step"><h4>4 · Decision</h4>
          <div class=${`verdict ${res.decision}`}>${res.decision === "blocked" ? "Refused" : res.decision === "private" ? `Kept private → ${res.target.upstream}` : `Sent → ${res.target.upstream}`}</div>
          ${res.decision !== "blocked" ? html`<div class="note">${res.target.model}${res.target.fallbacks.length ? html` · then ${res.target.fallbacks.join(", then ")}` : ""} · decided in ${res.decide_ms} ms</div>`
            : html`<div class="text" style="margin-top:6px">${res.refusal}</div>`}</div>
        ${(res.raw_reply != null || res.send_error) && html`<div class="step"><h4>5 · Reply</h4>
          ${res.send_error ? html`<div class="err">${res.send_error}</div>` : html`
            <div class="note">From the model (${res.upstream_ms} ms):</div><div class="text">${withPlaceholders(res.raw_reply)}</div>
            <div class="note" style="margin-top:8px">After the real values were put back:</div><div class="text">${res.restored_reply}</div>`}</div>`}
      `}
    </div>
  </div>`;
}

// ---- settings -------------------------------------------------------------------------
function Settings({ reload }) {
  const [cfg, setCfg] = useState(null); const [keysSet, setKeysSet] = useState({});
  const [err, setErr] = useState(""); const [ok, setOk] = useState(""); const [keys, setKeys] = useState({});
  const load = useCallback(async () => { const r = await api("/api/config"); setCfg(r.config); setKeysSet(r.keys_set || {}); setErr(r.error || ""); }, []);
  useEffect(() => { load(); }, []);
  if (!cfg) return html`<div class="empty">Loading…</div>`;
  const c = JSON.parse(JSON.stringify(cfg)); c.privacy ??= { mode: "off" }; c.privacy.rules ??= {}; c.upstreams ??= []; c.routes ??= [];
  const set = (fn) => { const n = JSON.parse(JSON.stringify(c)); fn(n); setCfg(n); setOk(""); };
  const save = async () => { setErr(""); setOk(""); try { const r = await api("/api/config", { method: "PUT", body: JSON.stringify(c) }); setOk(r.note); await load(); reload(); } catch (x) { setErr(x.message); } };
  const saveKey = async (id) => { setErr(""); try { await api(`/api/upstreams/${id}/key`, { method: "PUT", body: JSON.stringify({ key: keys[id] || "" }) }); setKeys({ ...keys, [id]: "" }); setOk(`Key saved for ${id}. The router restarts with it.`); await load(); reload(); } catch (x) { setErr(x.message); } };
  const addUpstream = (p) => set((n) => {
    let id = p.id; let i = 2; while (n.upstreams.some((u) => u.id === id)) id = `${p.id}-${i++}`;
    n.upstreams.push({ id, name: p.name, base_url: p.base_url, private: false });
    if (p.model) { if (!n.routes.length) n.routes.push({ name: "default", targets: [] }); n.routes[0].targets.push({ upstream: id, model: p.model }); }
  });
  const rules = c.privacy.rules; rules.custom ??= []; rules.exceptions ??= []; rules.builtin ??= {};
  return html`<div>
    ${err && html`<div class="panel err">${err}</div>`}${ok && html`<div class="panel okmsg">${ok}</div>`}
    <div class="panel">
      <div class="section-title">Privacy</div>
      <div class="modes">${MODES.map(([v, l, d]) => html`<div class=${`mode ${c.privacy.mode === v ? "on" : ""}`}
          onClick=${() => set((n) => { n.privacy.mode = v; if (v === "mask_check") n.privacy.laya = true; })}><b>${l}</b><span>${d}</span></div>`)}</div>
      <label class="row" style="margin-top:12px"><input type="checkbox" checked=${!!c.privacy.laya} disabled=${c.privacy.mode === "mask_check"}
        onChange=${(e) => set((n) => { n.privacy.laya = e.target.checked; })} /> Use the Laya classifier (names with health, money or work details; ~0.6 s per new message)</label>
      <div class="section-title" style="margin-top:16px">Built-in detectors</div>
      <div class="row">${BUILTINS.map((b) => html`<label class="pill"><input type="checkbox" checked=${rules.builtin[b] !== false}
        onChange=${(e) => set((n) => { n.privacy.rules.builtin = { ...(n.privacy.rules.builtin || {}), [b]: e.target.checked }; })} /> ${b}</label>`)}</div>
      <div class="section-title" style="margin-top:16px">Custom rules <span class="note">the name becomes the placeholder type; test them in the Experimenter</span></div>
      <table><tr><th>Name</th><th>Pattern (regex)</th><th>Keywords before it</th><th>Check</th><th></th></tr>
        ${rules.custom.map((r, i) => html`<tr>
          <td><input type="text" value=${r.name} onInput=${(e) => set((n) => { n.privacy.rules.custom[i].name = e.target.value.toUpperCase(); })} /></td>
          <td><input type="text" class="mono" value=${r.pattern} onInput=${(e) => set((n) => { n.privacy.rules.custom[i].pattern = e.target.value; })} /></td>
          <td><input type="text" value=${(r.keywords || []).join(", ")} onInput=${(e) => set((n) => { n.privacy.rules.custom[i].keywords = e.target.value.split(",").map((s) => s.trim()).filter(Boolean); })} /></td>
          <td><select value=${r.validator || ""} onChange=${(e) => set((n) => { const v = e.target.value; if (v) n.privacy.rules.custom[i].validator = v; else delete n.privacy.rules.custom[i].validator; })}>
            <option value="">none</option><option value="luhn">Luhn</option><option value="mod97">mod-97</option></select></td>
          <td><button class="btn small danger" onClick=${() => set((n) => n.privacy.rules.custom.splice(i, 1))}>Remove</button></td></tr>`)}
      </table>
      <button class="btn small" style="margin-top:8px" onClick=${() => set((n) => n.privacy.rules.custom.push({ name: "CUSTOM_ID", pattern: "", keywords: [] }))}>Add rule</button>
      <label class="field" style="margin-top:12px"><span>Never mask (one per line; @domain.com for a whole email domain)</span>
        <textarea style="min-height:70px" value=${rules.exceptions.join("\n")} onInput=${(e) => set((n) => { n.privacy.rules.exceptions = e.target.value.split("\n").map((s) => s.trim()).filter(Boolean); })} /></label>
    </div>

    <div class="panel">
      <div class="section-title">Upstreams</div>
      <table><tr><th>Name</th><th>Base URL</th><th>Private</th><th>Enabled</th><th>API key</th><th></th></tr>
        ${c.upstreams.map((u, i) => html`<tr>
          <td><input type="text" value=${u.name} onInput=${(e) => set((n) => { n.upstreams[i].name = e.target.value; })} /><div class="note mono">${u.id}</div></td>
          <td><input type="text" class="mono" value=${u.base_url} onInput=${(e) => set((n) => { n.upstreams[i].base_url = e.target.value; })} /></td>
          <td><input type="checkbox" checked=${!!u.private} onChange=${(e) => set((n) => { n.upstreams[i].private = e.target.checked; })} /></td>
          <td><input type="checkbox" checked=${u.enabled !== false} onChange=${(e) => set((n) => { n.upstreams[i].enabled = e.target.checked; })} /></td>
          <td><div class="row"><input type="password" placeholder=${keysSet[u.id] ? "set — enter to replace" : "not set"} value=${keys[u.id] || ""}
              onInput=${(e) => setKeys({ ...keys, [u.id]: e.target.value })} style="width:170px" />
            <button class="btn small" disabled=${!keys[u.id] || !cfg.upstreams?.some((x) => x.id === u.id)} onClick=${() => saveKey(u.id)}>Save key</button></div></td>
          <td><button class="btn small danger" onClick=${() => set((n) => { n.upstreams.splice(i, 1); n.routes.forEach((r) => (r.targets = r.targets.filter((t) => t.upstream !== u.id))); })}>Remove</button></td></tr>`)}
      </table>
      <div class="row" style="margin-top:10px">Add: ${PRESETS.map((p) => html`<button class="btn small" title=${p.hint} onClick=${() => addUpstream(p)}>${p.name}</button>`)}</div>
      <p class="note">Save the configuration first, then enter the new upstream's key.</p>
    </div>

    <div class="panel">
      <div class="section-title">Routes <span class="note">the model names callers use; targets are tried in order</span></div>
      ${c.routes.map((r, i) => html`<div class="step" style="margin-bottom:10px">
        <div class="row"><input type="text" value=${r.name} style="width:220px" onInput=${(e) => set((n) => { n.routes[i].name = e.target.value; })} />
          <span class="spacer"></span><button class="btn small danger" onClick=${() => set((n) => n.routes.splice(i, 1))}>Remove route</button></div>
        ${r.targets.map((t, j) => html`<div class="row" style="margin-top:8px"><span class="note" style="width:18px">${j + 1}.</span>
          <select style="width:200px" value=${t.upstream} onChange=${(e) => set((n) => { n.routes[i].targets[j].upstream = e.target.value; })}>
            ${c.upstreams.map((u) => html`<option value=${u.id}>${u.name}</option>`)}</select>
          <input type="text" class="mono" style="flex:1" value=${t.model} placeholder="model name at that upstream" onInput=${(e) => set((n) => { n.routes[i].targets[j].model = e.target.value; })} />
          <button class="btn small" disabled=${j === 0} onClick=${() => set((n) => { const a = n.routes[i].targets; [a[j - 1], a[j]] = [a[j], a[j - 1]]; })}>↑</button>
          <button class="btn small danger" onClick=${() => set((n) => n.routes[i].targets.splice(j, 1))}>✕</button></div>`)}
        <button class="btn small" style="margin-top:8px" disabled=${!c.upstreams.length} onClick=${() => set((n) => n.routes[i].targets.push({ upstream: c.upstreams[0].id, model: "" }))}>Add target</button>
      </div>`)}
      <button class="btn small" onClick=${() => set((n) => n.routes.push({ name: `route-${n.routes.length + 1}`, targets: [] }))}>Add route</button>
    </div>
    <div class="row"><button class="btn primary" onClick=${save}>Save configuration</button><button class="btn" onClick=${load}>Discard changes</button></div>
  </div>`;
}

// ---- keys -----------------------------------------------------------------------------
const PRIVACY_LABEL = { "": "Router setting", off: "Off", mask: "Mask", mask_check: "Mask + check", keep_private: "Keep private" };
const fmtDate = (ts) => ts ? new Date(ts * 1000).toLocaleDateString() : "—";

function NewKey({ info, onCreated }) {
  const [name, setName] = useState(""); const [more, setMore] = useState(false);
  const [routes, setRoutes] = useState(null); const [privacy, setPrivacy] = useState("");
  const [rpm, setRpm] = useState(String(info.default_rpm)); const [tpd, setTpd] = useState(""); const [expires, setExpires] = useState("");
  const [err, setErr] = useState("");
  const create = async (e) => {
    e.preventDefault(); setErr("");
    const body = { name };
    if (more) { body.routes = routes; body.privacy = privacy || null; body.rpm = rpm ? Number(rpm) : null; body.tpd = tpd ? Number(tpd) : null; body.expires_days = expires ? Number(expires) : null; }
    try { onCreated(await api("/api/keys", { method: "POST", body: JSON.stringify(body) })); setName(""); } catch (x) { setErr(x.message); }
  };
  const toggleRoute = (r) => { const cur = routes ?? info.routes; const next = cur.includes(r) ? cur.filter((x) => x !== r) : [...cur, r]; setRoutes(next.length === info.routes.length ? null : next); };
  return html`<form class="panel" onSubmit=${create}>
    <div class="section-title">New API key</div>
    <div class="row"><input type="text" placeholder="Who or what uses it, e.g. Ann (finance) or Support bot" value=${name} onInput=${(e) => setName(e.target.value)} style="flex:1;min-width:260px" />
      <button class="btn primary" disabled=${!name.trim()}>Create key</button>
      <button type="button" class="btn small" onClick=${() => setMore(!more)}>${more ? "Fewer options" : "More options"}</button></div>
    ${!more && html`<p class="note">Defaults: all routes · ${PRIVACY_LABEL[""].toLowerCase()} · ${info.default_rpm} requests/minute · no daily token limit · never expires.</p>`}
    ${more && html`<div class="grid2" style="margin-top:12px">
      <div><div class="note" style="margin-bottom:6px">Routes it may use</div><div class="row">${info.routes.map((r) => html`<label class="pill"><input type="checkbox" checked=${(routes ?? info.routes).includes(r)} onChange=${() => toggleRoute(r)} /> ${r}</label>`)}</div></div>
      <label class="field"><span>Privacy for this key</span><select value=${privacy} onChange=${(e) => setPrivacy(e.target.value)}>
        ${Object.entries(PRIVACY_LABEL).map(([v, l]) => html`<option value=${v} disabled=${(v === "mask_check" && !info.laya_available) || (v === "keep_private" && !info.has_private)}>${l}</option>`)}</select></label>
      <label class="field"><span>Requests per minute (empty = no limit)</span><input type="text" value=${rpm} onInput=${(e) => setRpm(e.target.value.replace(/\D/g, ""))} /></label>
      <label class="field"><span>Tokens per day (empty = no limit)</span><input type="text" value=${tpd} onInput=${(e) => setTpd(e.target.value.replace(/\D/g, ""))} /></label>
      <label class="field"><span>Expires</span><select value=${expires} onChange=${(e) => setExpires(e.target.value)}>
        <option value="">Never</option><option value="7">In 7 days</option><option value="30">In 30 days</option><option value="90">In 90 days</option><option value="365">In a year</option></select></label>
    </div>`}
    ${err && html`<div class="err">${err}</div>`}
  </form>`;
}

function Keys({ status }) {
  const [info, setInfo] = useState(null); const [created, setCreated] = useState(null); const [err, setErr] = useState("");
  const load = useCallback(async () => { try { setInfo(await api("/api/keys")); } catch (x) { setErr(x.message); } }, []);
  useEffect(() => { load(); const t = setInterval(load, 15000); return () => clearInterval(t); }, []);
  if (!info) return html`<div class="empty">${err || "Loading…"}</div>`;
  const revoke = async (k) => { if (!confirm(`Revoke the key for ${k.name}? Apps using it stop working immediately.`)) return; await api(`/api/keys/${k.id}/revoke`, { method: "POST" }); load(); };
  const url = status.router.url;
  return html`<div>
    <${NewKey} info=${info} onCreated=${(r) => { setCreated(r); load(); }} />
    <div class="panel">
      <div class="section-title">Keys <span class="note">usage is counts only, never content</span></div>
      ${info.keys.length === 0 ? html`<div class="empty">No keys yet. Create one per person, team or app; keep the admin key to yourself.</div>` : html`
      <table><tr><th>Name</th><th>Key</th><th>Routes</th><th>Privacy</th><th>Limits</th><th>Today</th><th>30 days</th><th>Last used</th><th></th></tr>
        ${info.keys.map((k) => html`<tr style=${k.active ? "" : "opacity:.5"}>
          <td><b>${k.name}</b><div class="note">created ${fmtDate(k.created)}${k.expires ? ` · expires ${fmtDate(k.expires)}` : ""}</div></td>
          <td class="mono">${k.prefix}…</td>
          <td>${k.routes ? k.routes.join(", ") : html`<span class="note">all</span>`}</td>
          <td>${PRIVACY_LABEL[k.privacy || ""]}</td>
          <td class="note">${k.rpm ? `${k.rpm}/min` : "no rate limit"}<br />${k.tpd ? `${k.tpd.toLocaleString()} tokens/day` : "no token limit"}</td>
          <td class="note">${k.usage.today.requests} req · ${k.usage.today.tokens.toLocaleString()} tok${k.usage.today.masked ? html`<br />${k.usage.today.masked} masked` : ""}${k.usage.today.blocked ? html`<br />${k.usage.today.blocked} refused` : ""}</td>
          <td class="note">${k.usage["30d"].requests} req · ${k.usage["30d"].tokens.toLocaleString()} tok</td>
          <td class="note">${k.last_used ? ago(k.last_used) : "never"}</td>
          <td>${k.active ? html`<button class="btn small danger" onClick=${() => revoke(k)}>Revoke</button>` : html`<span class="pill bad">${k.revoked ? "revoked" : "expired"}</span>`}</td></tr>`)}
      </table>`}
    </div>
    ${created && html`<div class="modal-back"><div class="modal">
      <div class="section-title">Key for ${created.key.name}</div>
      <div class="row"><code style="word-break:break-all">${created.secret}</code></div>
      <div class="row" style="margin:10px 0"><button class="btn primary" onClick=${() => navigator.clipboard.writeText(created.secret)}>Copy key</button></div>
      <p class="err" style="color:var(--amber)">${created.note}</p>
      <pre class="panel mono" style="white-space:pre-wrap">${`curl ${url}/chat/completions \\\n  -H "Authorization: Bearer ${created.secret}" -H "Content-Type: application/json" \\\n  -d '{"model":"${(created.key.routes || info.routes)[0] || "default"}","messages":[{"role":"user","content":"Hello"}]}'`}</pre>
      <button class="btn" onClick=${() => setCreated(null)}>Done</button></div></div>`}
  </div>`;
}

// ---- endpoints ------------------------------------------------------------------------
function Endpoints({ status, onClose }) {
  const url = status.router.url;
  const curl = `curl ${url}/chat/completions \\\n  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \\\n  -d '{"model":"${status.routes[0]?.name || "default"}","messages":[{"role":"user","content":"Hello"}]}'`;
  return html`<div class="modal-back" onClick=${onClose}><div class="modal" onClick=${(e) => e.stopPropagation()}>
    <div class="section-title">Endpoint</div>
    <div class="row"><code>${url}</code><button class="btn small" onClick=${() => navigator.clipboard.writeText(url)}>Copy</button></div>
    <p class="note">Models: ${status.routes.map((r) => r.name).join(", ") || "none yet"}. Any OpenAI-compatible client works with this base URL.</p>
    <pre class="panel mono" style="white-space:pre-wrap">${curl}</pre>
    <p class="note">Give each person, team or app its own key (Keys tab); keep the admin key for administration.</p>
    <button class="btn" onClick=${onClose}>Close</button></div></div>`;
}

// ---- app ------------------------------------------------------------------------------
function App() {
  const [authed, setAuthed] = useState(null); const [tab, setTab] = useState("overview");
  const [status, setStatus] = useState(null); const [events, setEvents] = useState([]); const [showEp, setShowEp] = useState(false);
  const reload = useCallback(async () => { try { setStatus(await api("/api/status")); setAuthed(true); } catch (x) { if (x.status === 401) setAuthed(false); } }, []);
  useEffect(() => { reload(); const t = setInterval(reload, 10000); return () => clearInterval(t); }, []);
  useEffect(() => {
    if (!authed) return;
    const es = new EventSource("/api/events");
    es.onmessage = (m) => setEvents((list) => [...list.slice(-199), JSON.parse(m.data)]);
    return () => es.close();
  }, [authed]);
  if (authed === false) return html`<${Login} onIn=${reload} />`;
  if (!status) return html`<div class="empty">Loading…</div>`;
  const logout = async () => { await api("/api/logout", { method: "POST" }); setAuthed(false); };
  return html`<div class="shell">
    <div class="topbar"><span class="brand">Inference Router</span>
      <div class="tabs">${[["overview", "Overview"], ["keys", "Keys"], ["experiment", "Experimenter"], ["settings", "Settings"]].map(([k, l]) =>
        html`<button class=${`tab ${tab === k ? "active" : ""}`} onClick=${() => setTab(k)}>${l}</button>`)}</div>
      <span class="spacer"></span>
      <span class=${`pill ${status.router.healthy ? "ok" : "bad"}`}><span class=${`dot ${status.router.healthy ? "ok" : "bad"}`}></span>${status.router.healthy ? "router up" : "router down"}</span>
      <span class="pill">privacy: ${status.privacy.mode}</span>
      <button class="btn" onClick=${() => setShowEp(true)}>Endpoints</button>
      <button class="btn small" onClick=${logout}>Sign out</button></div>
    ${status.error && html`<div class="panel err">${status.error}</div>`}
    ${tab === "overview" && html`<${Overview} status=${status} events=${events} reload=${reload} />`}
    ${tab === "keys" && html`<${Keys} status=${status} />`}
    ${tab === "experiment" && html`<${Experimenter} status=${status} />`}
    ${tab === "settings" && html`<${Settings} reload=${reload} />`}
    ${showEp && html`<${Endpoints} status=${status} onClose=${() => setShowEp(false)} />`}
  </div>`;
}
render(html`<${App} />`, document.getElementById("app"));
