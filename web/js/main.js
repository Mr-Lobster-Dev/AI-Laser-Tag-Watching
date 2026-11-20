import { loadIndex, loadReplay, N } from "./replay.js";
import { ArenaScene } from "./scene.js";
import { Inspector } from "./inspector.js";

const $ = (s) => document.querySelector(s);
const SPEEDS = [0.5, 1, 2, 4, 8];
const THEME_KEY = "lt-theme";
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");

const state = {
  index: [],
  entry: null,
  replay: null,
  t: 0,
  playing: false,
  speed: 1,
  agent: 0,
  cam: "orbit",
  lastStep: -1,
  theme: store("get", THEME_KEY) || "system",
};

const scene = new ArenaScene($("#scene"));
const inspector = new Inspector($("#inspector"));

function store(op, k, v) {
  try { return op === "get" ? localStorage.getItem(k) : localStorage.setItem(k, v); } catch { return null; }
}

/* ---------------- Theme ---------------- */
const darkQuery = matchMedia("(prefers-color-scheme: dark)");
function effectiveTheme() {
  return state.theme === "system" ? (darkQuery.matches ? "dark" : "light") : state.theme;
}
function applyTheme() {
  const root = document.documentElement;
  if (state.theme === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", state.theme);
  scene.setTheme(effectiveTheme());
  inspector.readColors();
  inspector.redraw();
  drawMarks();
  $("#btn-theme").title = `Appearance: ${state.theme[0].toUpperCase() + state.theme.slice(1)} (A)`;
  const icon = $("#theme-icon");
  icon.innerHTML = state.theme === "system"
    ? '<circle cx="10" cy="10" r="6.5"/><path d="M10 3.5a6.5 6.5 0 0 1 0 13z" class="fill"/>'
    : state.theme === "light"
      ? '<circle cx="10" cy="10" r="3.2"/><path d="M10 2.5v1.6M10 15.9v1.6M2.5 10h1.6M15.9 10h1.6M4.7 4.7l1.1 1.1M14.2 14.2l1.1 1.1M4.7 15.3l1.1-1.1M14.2 5.8l1.1-1.1"/>'
      : '<path d="M15.5 12.2A6.5 6.5 0 0 1 7.8 4.5a6.5 6.5 0 1 0 7.7 7.7z"/>';
}
function cycleTheme() {
  state.theme = { system: "light", light: "dark", dark: "system" }[state.theme];
  store("set", THEME_KEY, state.theme);
  applyTheme();
  toast(`Appearance: ${state.theme[0].toUpperCase() + state.theme.slice(1)}`);
}
darkQuery.addEventListener("change", () => state.theme === "system" && applyTheme());

/* ---------------- Toast ---------------- */
let toastTimer;
function toast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.hidden = false;
  el.style.animation = "none"; void el.offsetWidth; el.style.animation = "";
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.hidden = true), 1600);
}

/* ---------------- Formatting ---------------- */
const fmtTime = (sec) => `${Math.floor(sec / 60)}:${(sec % 60).toFixed(1).padStart(4, "0")}`;
const fmtScore = (v) => (Number.isInteger(v) ? String(v) : v.toFixed(1));
const agentName = (i) => `${i < 4 ? "Green" : "Pink"} ${(i % 4) + 1}`;

/* ---------------- Match picker ---------------- */
function renderMatchList() {
  const list = $("#match-list");
  list.innerHTML = "";
  $("#match-count").textContent = `${state.index.length} match${state.index.length === 1 ? "" : "es"}`;
  for (const e of [...state.index].reverse()) {
    const b = document.createElement("button");
    b.className = "match-row";
    b.setAttribute("role", "option");
    b.setAttribute("aria-selected", String(state.entry?.file === e.file));
    const win = e.winner === "draw" ? "Draw" : `${e.winner[0].toUpperCase() + e.winner.slice(1)} won`;
    b.innerHTML = `
      <span class="m-title">Match ${e.match.toLocaleString()}</span>
      <span class="m-score"><span class="dot" style="background:var(--green)"></span>${fmtScore(e.green)} – ${fmtScore(e.pink)}<span class="dot" style="background:var(--pink);margin:0 0 0 4px"></span></span>
      <span class="m-sub muted">${win} · ${e.tags ?? "?"} tags from ${e.shots ?? "?"} shots · trained on ${(e.trained_matches ?? 0).toLocaleString()} matches</span>`;
    b.addEventListener("click", () => { closeMatchPicker(); openMatch(e); });
    list.appendChild(b);
  }
}
function openMatchPicker() {
  renderMatchList();
  $("#match-popover").hidden = false;
  $("#btn-match").setAttribute("aria-expanded", "true");
  $("#match-list [aria-selected=true]")?.focus();
}
function closeMatchPicker() {
  $("#match-popover").hidden = true;
  $("#btn-match").setAttribute("aria-expanded", "false");
}
function toggleMatchPicker() { $("#match-popover").hidden ? openMatchPicker() : closeMatchPicker(); }

async function refreshIndex(initial) {
  try {
    const idx = await loadIndex();
    const known = new Set(state.index.map((e) => e.file));
    const fresh = idx.filter((e) => !known.has(e.file));
    state.index = idx;
    if (!initial && fresh.length) {
      toast(`New replay: Match ${fresh[fresh.length - 1].match.toLocaleString()}`);
      if (!$("#match-popover").hidden) renderMatchList();
    }
  } catch (err) {
    if (initial) throw err;
  }
}

function stepMatch(dir) {
  if (!state.entry) return;
  const i = state.index.findIndex((e) => e.file === state.entry.file);
  const next = state.index[i + dir];
  if (next) openMatch(next);
  else toast(dir > 0 ? "This is the newest match" : "This is the first match");
}

/* ---------------- Loading a match ---------------- */
async function openMatch(entry) {
  const loading = $("#loading");
  const wasPlaying = state.playing;
  setPlaying(false);
  loading.classList.remove("hide");
  if (state.replay) loading.classList.add("overlay");
  $("#loading-text").textContent = `Loading match ${entry.match.toLocaleString()}…`;
  $("#loading-bar").style.width = "0%";
  try {
    const r = await loadReplay(entry.file, (p) => ($("#loading-bar").style.width = `${Math.round(p * 100)}%`));
    state.entry = entry;
    state.replay = r;
    if (!state.arenaBuilt) { scene.buildArena(r.arena); state.arenaBuilt = true; }
    inspector.setReplay(r);
    $("#seek").max = r.steps - 1;
    $("#time-total").textContent = fmtTime((r.steps - 1) * r.dt).replace(/\.\d$/, "");
    $("#match-label").textContent = `Match ${r.match.toLocaleString()}`;
    const fs = r.finalScores;
    const result = fs.winner === "draw" ? "Draw" : `${fs.winner[0].toUpperCase() + fs.winner.slice(1)} won ${fmtScore(fs[fs.winner])}–${fmtScore(fs[fs.winner === "green" ? "pink" : "green"])}`;
    $("#match-meta").textContent = `${result} · trained on ${r.trainedMatches.toLocaleString()} matches`;
    history.replaceState(null, "", `#match=${r.match}`);
    renderRadioChart();
    seek(0);
    drawMarks();
    if (wasPlaying || !state.playedOnce) setPlaying(true);
    state.playedOnce = true;
  } catch (err) {
    console.error(err);
    toast(err.message || "Could not load the replay");
  } finally {
    loading.classList.add("hide");
  }
}

/* ---------------- Sidebar ---------------- */
function buildAgentUI() {
  const list = $("#agent-list");
  const picker = $("#agent-picker");
  for (let i = 0; i < N; i++) {
    const team = i < 4 ? "green" : "pink";
    const li = document.createElement("li");
    li.innerHTML = `<button class="agent-row" data-agent="${i}" role="option">
        <span class="agent-shape ${team}"></span>
        <span class="agent-name">${agentName(i)}<span class="key">${i + 1}</span></span>
        <span class="agent-radio">----</span>
        <span class="agent-score">0</span>
      </button>`;
    li.firstElementChild.addEventListener("click", () => selectAgent(i, true));
    list.appendChild(li);
    const b = document.createElement("button");
    b.className = team;
    b.setAttribute("role", "radio");
    b.textContent = `${team === "green" ? "G" : "P"}${(i % 4) + 1}`;
    b.title = `${agentName(i)} (${i + 1})`;
    b.addEventListener("click", () => selectAgent(i));
    picker.appendChild(b);
  }
}

function selectAgent(i, focus) {
  state.agent = i;
  scene.selected = i;
  scene.followAgent = i;
  document.querySelectorAll(".agent-row").forEach((el, k) => el.setAttribute("aria-selected", String(k === i)));
  document.querySelectorAll("#agent-picker button").forEach((el, k) => el.setAttribute("aria-checked", String(k === i)));
  if (focus && state.cam === "orbit") scene.focusAgent(i, reducedMotion.matches);
  if (state.cam === "follow") scene.resetLook();
  state.lastStep = -1;
}

function renderRadioChart() {
  const el = $("#radio-chart");
  const r = state.replay;
  const total = r.steps * 4;
  el.innerHTML = "";
  state.radioRows = {};
  for (const team of ["green", "pink"]) {
    const top = [...r.radioCounts[team].entries()].sort((a, b) => b[1] - a[1]).slice(0, 6);
    const col = document.createElement("div");
    col.className = `radio-col ${team}`;
    col.innerHTML = `<h4>${team === "green" ? "Green" : "Pink"}</h4>`;
    const max = top[0]?.[1] || 1;
    state.radioRows[team] = {};
    for (const [code, n] of top) {
      const row = document.createElement("div");
      row.className = "rbar";
      row.title = `${code}: ${((n / total) * 100).toFixed(1)}% of the time`;
      row.innerHTML = `<span>${code}</span><span class="track"><i style="width:${(n / max) * 100}%"></i></span>`;
      col.appendChild(row);
      state.radioRows[team][code] = row;
    }
    el.appendChild(col);
  }
}

let lastScores = [0, 0];
function updateSidebar(step) {
  const r = state.replay;
  const fr = r.frames;
  const g = fr.teamScores[step * 2], p = fr.teamScores[step * 2 + 1];
  const gEl = $("#score-green"), pEl = $("#score-pink");
  gEl.textContent = fmtScore(g);
  pEl.textContent = fmtScore(p);
  if (g !== lastScores[0] && step) bump(gEl);
  if (p !== lastScores[1] && step) bump(pEl);
  lastScores = [g, p];
  const rows = document.querySelectorAll(".agent-row");
  const live = { green: new Set(), pink: new Set() };
  for (let i = 0; i < N; i++) {
    const k = step * N + i;
    const row = rows[i];
    row.classList.toggle("down", fr.down[k] === 1);
    row.querySelector(".agent-shape").classList.toggle("ready", fr.gunReady[k] === 1 && !fr.down[k]);
    row.querySelector(".agent-radio").textContent = fr.radio[k];
    row.querySelector(".agent-score").textContent = fmtScore(r.agentScore[k]);
    live[i < 4 ? "green" : "pink"].add(fr.radio[k]);
  }
  for (const team of ["green", "pink"]) {
    for (const [code, row] of Object.entries(state.radioRows?.[team] || {})) row.classList.toggle("live", live[team].has(code));
  }
}
function bump(el) {
  if (reducedMotion.matches) return;
  el.classList.add("bump");
  setTimeout(() => el.classList.remove("bump"), 220);
}

/* ---------------- Playback ---------------- */
function setPlaying(on) {
  if (on && state.replay && state.t >= state.replay.steps - 1) state.t = 0;
  state.playing = on && !!state.replay;
  $("#btn-play").setAttribute("aria-label", state.playing ? "Pause" : "Play");
  $("#play-icon").innerHTML = state.playing
    ? '<rect class="fill" x="5.5" y="4.5" width="3" height="11" rx="1"/><rect class="fill" x="11.5" y="4.5" width="3" height="11" rx="1"/>'
    : '<path class="fill" d="M7 4.8v10.4l8.2-5.2z"/>';
}
function seek(step) {
  if (!state.replay) return;
  state.t = Math.max(0, Math.min(state.replay.steps - 1, step));
  state.lastStep = -1;
}
function setSpeed(v) {
  state.speed = v;
  $("#btn-speed").textContent = `${v}×`;
  document.querySelectorAll("#speed-menu button").forEach((b) => b.setAttribute("aria-checked", String(Number(b.dataset.v) === v)));
}
function nudgeSpeed(dir) {
  const i = SPEEDS.indexOf(state.speed);
  const v = SPEEDS[Math.max(0, Math.min(SPEEDS.length - 1, i + dir))];
  setSpeed(v);
  toast(`Speed ${v}×`);
}

function drawMarks() {
  const cv = $("#cv-marks");
  const r = state.replay;
  if (!r || !cv) return;
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w * dpr; cv.height = h * dpr;
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const cs = getComputedStyle(document.documentElement);
  const pad = 8; // thumb radius
  const x = (s) => pad + (s / (r.steps - 1)) * (w - pad * 2);
  g.fillStyle = cs.getPropertyValue("--label-3");
  for (const s of r.shots) if (!s.tag) g.fillRect(x(s.step), h / 2 + 5, 1, 3);
  for (const t of r.tagSteps) {
    g.fillStyle = cs.getPropertyValue(t.team === "green" ? "--green" : "--pink");
    g.fillRect(x(t.step) - 0.75, h / 2 - 10, 1.5, 6);
  }
}
new ResizeObserver(drawMarks).observe($("#scrubber"));

function setCam(mode) {
  state.cam = mode;
  scene.setMode(mode);
  const seg = $("#camera-modes");
  seg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.cam === mode)));
  moveThumb();
}
function moveThumb() {
  const seg = $("#camera-modes");
  const btn = seg.querySelector(`[data-cam="${state.cam}"]`);
  const thumb = seg.querySelector(".seg-thumb");
  thumb.style.width = `${btn.offsetWidth}px`;
  thumb.style.transform = `translateX(${btn.offsetLeft}px)`;
}
window.addEventListener("resize", moveThumb);

/* ---------------- Inspector & sidebar toggles ---------------- */
function toggleInspector(force) {
  const el = $("#inspector");
  const open = force ?? el.hidden;
  $("#btn-inspector").setAttribute("aria-pressed", String(open));
  if (open) {
    el.classList.remove("closing");
    el.hidden = false;
    state.lastStep = -1;
  } else if (!el.hidden) {
    if (reducedMotion.matches) { el.hidden = true; return; }
    el.classList.add("closing");
    el.addEventListener("animationend", () => { if (el.classList.contains("closing")) { el.hidden = true; el.classList.remove("closing"); } }, { once: true });
  }
}
function toggleSidebar() {
  const el = $("#sidebar");
  const collapsed = el.classList.toggle("collapsed");
  $("#btn-sidebar").setAttribute("aria-pressed", String(!collapsed));
}
function toggleHelp(force) {
  const el = $("#help");
  el.hidden = !(force ?? el.hidden);
  if (!el.hidden) $("#btn-help-close").focus();
}

/* ---------------- Events ---------------- */
function bindUI() {
  $("#btn-play").addEventListener("click", () => setPlaying(!state.playing));
  $("#btn-back").addEventListener("click", () => { setPlaying(false); seek(Math.floor(state.t) - 1); });
  $("#btn-fwd").addEventListener("click", () => { setPlaying(false); seek(Math.floor(state.t) + 1); });
  const seekEl = $("#seek");
  seekEl.addEventListener("input", () => { seek(Number(seekEl.value)); state.scrubbing = true; });
  seekEl.addEventListener("change", () => (state.scrubbing = false));
  $("#btn-match").addEventListener("click", (e) => { e.stopPropagation(); toggleMatchPicker(); });
  document.addEventListener("click", (e) => {
    if (!$("#match-popover").contains(e.target)) closeMatchPicker();
    if (!$("#speed").contains(e.target)) $("#speed-menu").hidden = true;
  });
  $("#camera-modes").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) setCam(b.dataset.cam); });
  $("#btn-inspector").addEventListener("click", () => toggleInspector());
  $("#btn-close-inspector").addEventListener("click", () => toggleInspector(false));
  $("#btn-sidebar").addEventListener("click", toggleSidebar);
  $("#btn-theme").addEventListener("click", cycleTheme);
  $("#btn-help").addEventListener("click", () => toggleHelp(true));
  $("#btn-help-close").addEventListener("click", () => toggleHelp(false));
  $("#help").addEventListener("click", (e) => e.target.id === "help" && toggleHelp(false));

  const menu = $("#speed-menu");
  for (const v of SPEEDS) {
    const b = document.createElement("button");
    b.dataset.v = v;
    b.setAttribute("role", "menuitemradio");
    b.textContent = `${v}×`;
    b.addEventListener("click", () => { setSpeed(v); menu.hidden = true; });
    menu.appendChild(b);
  }
  $("#btn-speed").addEventListener("click", () => (menu.hidden = !menu.hidden));

  document.addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.target.closest("input:not([type=range]), textarea")) return;
    const k = e.key;
    if (!$("#help").hidden && (k === "Escape" || k === "?")) { toggleHelp(false); e.preventDefault(); return; }
    if (k === "Escape") { closeMatchPicker(); $("#speed-menu").hidden = true; return; }
    const r = state.replay;
    const big = r ? Math.round(1 / r.dt) : 10;
    switch (k) {
      case " ": case "k": setPlaying(!state.playing); break;
      case "ArrowRight": setPlaying(false); seek(Math.floor(state.t) + (e.shiftKey ? big : 1)); break;
      case "ArrowLeft": setPlaying(false); seek(Math.ceil(state.t) - (e.shiftKey ? big : 1)); break;
      case "Home": seek(0); break;
      case "End": seek(Infinity); setPlaying(false); break;
      case "[": nudgeSpeed(-1); break;
      case "]": nudgeSpeed(1); break;
      case "o": case "O": setCam("orbit"); break;
      case "t": case "T": setCam("top"); break;
      case "f": case "F": setCam("follow"); break;
      case "c": case "C": setCam({ orbit: "chase", chase: "follow", follow: "top", top: "orbit" }[state.cam]); break;
      case "v": case "V": setCam("chase"); break;
      case "i": case "I": toggleInspector(); break;
      case "s": case "S": toggleSidebar(); break;
      case "m": case "M": toggleMatchPicker(); break;
      case "a": case "A": cycleTheme(); break;
      case ",": case "<": stepMatch(-1); break;
      case ".": case ">": stepMatch(1); break;
      case "?": toggleHelp(true); break;
      case "r": case "R": if (state.cam === "follow") scene.resetLook(); else if (state.cam === "orbit") scene.resetView(reducedMotion.matches); break;
      default:
        if (k >= "1" && k <= "8") { selectAgent(Number(k) - 1, true); toast(agentName(Number(k) - 1)); break; }
        return;
    }
    if (k === " " || k.startsWith("Arrow") || k === "Home" || k === "End") e.preventDefault();
    // Range input should not also move on arrows.
    if (document.activeElement === seekEl && k.startsWith("Arrow")) e.preventDefault();
  });
}

/* ---------------- Main loop ---------------- */
let last = performance.now();
function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  const r = state.replay;
  if (r) {
    if (state.playing && !state.scrubbing) {
      state.t += (dt / r.dt) * state.speed;
      if (state.t >= r.steps - 1) { state.t = r.steps - 1; setPlaying(false); }
    }
    const step = Math.floor(state.t);
    scene.update(r, state.t);
    if (step !== state.lastStep) {
      state.lastStep = step;
      updateSidebar(step);
      inspector.update(step, state.agent);
      const seekEl = $("#seek");
      if (!state.scrubbing) seekEl.value = step;
      seekEl.style.setProperty("--p", `${(step / (r.steps - 1)) * 100}%`);
      $("#time-now").textContent = fmtTime(step * r.dt);
    }
  }
  scene.render();
  requestAnimationFrame(frame);
}

/* ---------------- Boot ---------------- */
async function boot() {
  buildAgentUI();
  bindUI();
  applyTheme();
  setSpeed(1);
  setCam("orbit");
  selectAgent(0);
  requestAnimationFrame(frame);
  try {
    await refreshIndex(true);
  } catch (err) {
    $("#loading-text").textContent = err.message;
    $("#loading").querySelector(".spinner").hidden = true;
    return;
  }
  if (!state.index.length) {
    $("#loading-text").textContent = "No replays yet. Run scripts/sync_replays.py after training saves one.";
    $("#loading").querySelector(".spinner").hidden = true;
    return;
  }
  const want = Number(new URLSearchParams(location.hash.slice(1)).get("match"));
  const entry = state.index.find((e) => e.match === want) || state.index[state.index.length - 1];
  await openMatch(entry);
  setInterval(() => refreshIndex(false), 60_000);
  requestAnimationFrame(moveThumb);
}
boot();
