// Network inspector: draws one agent's network values for the current step.
import { N, netSlice } from "./replay.js";

const HEAD_LABELS = {
  forward: ["Back", "Stay", "Forward"],
  strafe: ["Left", "Stay", "Right"],
  turn: ["Hard left", "Left", "Straight", "Right", "Hard right"],
  pitch: ["Down", "Level", "Up"],
  crouch: ["Stand", "Crouch"],
  fire: ["Hold", "Fire"],
};
const HEAD_TITLES = { forward: "Move", strafe: "Strafe", turn: "Turn", pitch: "Aim", crouch: "Crouch", fire: "Trigger" };
const SHORT = { forward: "mov", strafe: "str", turn: "turn", pitch: "aim", crouch: "cr", fire: "fire", radio0: "r0", radio1: "r1", radio2: "r2", radio3: "r3" };

function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function setupCanvas(cv) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = cv.clientWidth || 300;
  const h = Number(cv.getAttribute("height"));
  cv.style.height = h + "px";
  if (cv.width !== Math.round(w * dpr) || cv._h !== h) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
    cv._h = h;
  }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { g, w, h };
}

function hexToRgb(hex) {
  const m = hex.replace("#", "");
  const n = parseInt(m.length === 3 ? m.split("").map((c) => c + c).join("") : m, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export class Inspector {
  constructor(root) {
    this.root = root;
    this.cv = {
      value: root.querySelector("#cv-value"),
      vision: root.querySelector("#cv-vision"),
      encV: root.querySelector("#cv-enc-vision"),
      encO: root.querySelector("#cv-enc-other"),
      memory: root.querySelector("#cv-memory"),
      links: root.querySelector("#cv-links"),
    };
    this.valueNum = root.querySelector("#value-num");
    this.headsEl = root.querySelector("#action-heads");
    this.radioEl = root.querySelector("#radio-heads");
    this.buf = { enc: new Float32Array(64), mem: new Float32Array(128), act: new Float32Array(58) };
    this.readColors();
    new ResizeObserver(() => this.redraw()).observe(root);
  }

  readColors() {
    this.c = {
      fill: css("--fill"), fill2: css("--fill-2"), label: css("--label"), label2: css("--label-2"), label3: css("--label-3"),
      accent: hexToRgb(css("--accent")), pos: hexToRgb(css("--pos")), neg: hexToRgb(css("--neg")),
      wall: hexToRgb(css("--vis-wall")), mate: hexToRgb(css("--vis-mate")), opp: hexToRgb(css("--vis-opp")), none: hexToRgb(css("--vis-none")),
      green: css("--green"), pink: css("--pink"), sep: css("--sep"),
    };
  }

  setReplay(replay) {
    this.replay = replay;
    this._buildHeads();
  }

  _buildHeads() {
    const heads = this.replay.network.heads;
    this.headsEl.innerHTML = "";
    this.radioEl.innerHTML = "";
    this.headRefs = [];
    let off = 0;
    for (const h of heads) {
      const ref = { name: h.name, off, size: h.size, items: [] };
      if (h.name.startsWith("radio")) {
        const row = document.createElement("div");
        row.className = "rh";
        row.innerHTML = `<span class="lbl">Digit ${Number(h.name.slice(5)) + 1}</span>`;
        for (let d = 0; d < h.size; d++) {
          const cell = document.createElement("div");
          cell.className = "cell";
          cell.innerHTML = `<i></i><em>${d}</em>`;
          row.appendChild(cell);
          ref.items.push(cell);
        }
        ref.row = row;
        this.radioEl.appendChild(row);
      } else {
        const wrap = document.createElement("div");
        const labels = HEAD_LABELS[h.name] || Array.from({ length: h.size }, (_, i) => String(i));
        wrap.innerHTML = `<div class="head-name"><span>${HEAD_TITLES[h.name] || h.name}</span><span class="pick"></span></div><div class="head-bars"></div>`;
        const bars = wrap.querySelector(".head-bars");
        bars.style.gridTemplateColumns = `repeat(${h.size}, 1fr)`;
        for (let i = 0; i < h.size; i++) {
          const b = document.createElement("div");
          b.className = "hb";
          b.innerHTML = `<i></i><span>${labels[i]}<b></b></span>`;
          bars.appendChild(b);
          ref.items.push(b);
        }
        ref.pick = wrap.querySelector(".pick");
        ref.labels = labels;
        this.headsEl.appendChild(wrap);
      }
      this.headRefs.push(ref);
      off += h.size;
    }
  }

  update(step, agent) {
    this.step = step;
    this.agent = agent;
    if (!this.replay || this.root.hidden) return;
    const net = this.replay.network;
    this._value(step, agent);
    this._vision(step, agent);
    this._strip(this.cv.encV, netSlice(net.encVision, step, agent, this.buf.enc), 4);
    this._strip(this.cv.encO, netSlice(net.encOther, step, agent, this.buf.enc), 4);
    const mem = netSlice(net.memory, step, agent, this.buf.mem);
    this._memory(mem);
    const act = netSlice(net.actionProbs, step, agent, this.buf.act);
    this._heads(act, step, agent);
    this._links(mem, agent);
  }

  redraw() { if (this.step != null) this.update(this.step, this.agent); }

  _value(step, agent) {
    const r = this.replay, net = r.network;
    const v = net.value[step * N + agent];
    this.valueNum.textContent = (v >= 0 ? "+" : "") + v.toFixed(2);
    const { g, w, h } = setupCanvas(this.cv.value);
    g.clearRect(0, 0, w, h);
    let [lo, hi] = r.valueRange[agent];
    lo = Math.min(lo, 0); hi = Math.max(hi, 0);
    if (hi - lo < 0.2) { hi += 0.1; lo -= 0.1; }
    const pad = 4;
    const y = (val) => pad + (1 - (val - lo) / (hi - lo)) * (h - pad * 2);
    const x = (s) => (s / (r.steps - 1)) * w;
    g.fillStyle = this.c.fill;
    g.fillRect(0, 0, w, h);
    g.strokeStyle = this.c.label3;
    g.setLineDash([3, 3]);
    g.beginPath(); g.moveTo(0, y(0)); g.lineTo(w, y(0)); g.stroke();
    g.setLineDash([]);
    const col = r.agents[agent].team === "green" ? this.c.green : this.c.pink;
    // Played part solid, rest faint.
    const stride = Math.max(1, Math.floor(r.steps / w));
    for (const [from, to, alpha] of [[0, step, 1], [step, r.steps - 1, 0.25]]) {
      g.globalAlpha = alpha;
      g.strokeStyle = col;
      g.lineWidth = 1.5;
      g.beginPath();
      for (let s = from; s <= to; s += stride) {
        const yy = y(net.value[s * N + agent]);
        s === from ? g.moveTo(x(s), yy) : g.lineTo(x(s), yy);
      }
      g.lineTo(x(to), y(net.value[to * N + agent]));
      g.stroke();
    }
    g.globalAlpha = 1;
    g.fillStyle = col;
    g.beginPath(); g.arc(x(step), y(v), 3.5, 0, Math.PI * 2); g.fill();
    // Tag markers for this agent.
    for (const t of this.replay.shots) {
      if (!t.tag) continue;
      if (t.shooter === agent || t.target === agent) {
        g.fillStyle = t.shooter === agent ? col : this.c.label2;
        g.fillRect(x(t.step) - 0.5, h - 6, 1.5, 6);
      }
    }
  }

  _vision(step, agent) {
    const vis = this.replay.network.vision;
    const { rows, cols } = vis;
    const { g, w, h } = setupCanvas(this.cv.vision);
    g.clearRect(0, 0, w, h);
    const off = (step * N + agent) * rows * cols;
    const gap = 1.5;
    const cw = (w - gap * (cols - 1)) / cols;
    const ch = (h - gap * (rows - 1)) / rows;
    const pal = [this.c.none, this.c.wall, this.c.mate, this.c.opp];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const b = vis.data[off + r * cols + c];
        const dist = ((b >> 2) / 63) * 40;
        const kind = b & 3;
        const near = kind === 0 ? 0 : Math.max(0.12, 1 - dist / 25);
        const [cr, cg, cb] = pal[kind];
        const [nr, ng, nb] = this.c.none;
        const k = kind === 0 ? 1 : near;
        g.fillStyle = `rgb(${nr + (cr - nr) * k | 0},${ng + (cg - ng) * k | 0},${nb + (cb - nb) * k | 0})`;
        roundRect(g, c * (cw + gap), r * (ch + gap), cw, ch, 2);
      }
    }
  }

  _strip(cv, vals, max) {
    const { g, w, h } = setupCanvas(cv);
    g.clearRect(0, 0, w, h);
    const n = vals.length, gap = 1;
    const cw = (w - gap * (n - 1)) / n;
    const [r, gg, b] = this.c.accent;
    for (let i = 0; i < n; i++) {
      const k = Math.min(1, vals[i] / max);
      g.fillStyle = this.c.fill;
      roundRect(g, i * (cw + gap), 0, cw, h, 2);
      if (k > 0.01) {
        g.fillStyle = `rgba(${r},${gg},${b},${0.15 + 0.85 * k})`;
        roundRect(g, i * (cw + gap), 0, cw, h, 2);
      }
    }
  }

  _memory(mem) {
    const { g, w, h } = setupCanvas(this.cv.memory);
    g.clearRect(0, 0, w, h);
    const cols = 16, rows = 8, gap = 2;
    const cw = (w - gap * (cols - 1)) / cols, ch = (h - gap * (rows - 1)) / rows;
    for (let i = 0; i < 128; i++) {
      const v = mem[i];
      const c = i % cols, r = (i / cols) | 0;
      g.fillStyle = this.c.fill;
      roundRect(g, c * (cw + gap), r * (ch + gap), cw, ch, 3);
      const [cr, cg, cb] = v >= 0 ? this.c.pos : this.c.neg;
      g.fillStyle = `rgba(${cr},${cg},${cb},${Math.min(1, Math.abs(v))})`;
      roundRect(g, c * (cw + gap), r * (ch + gap), cw, ch, 3);
    }
  }

  _heads(act, step, agent) {
    for (const ref of this.headRefs) {
      let best = 0;
      for (let i = 1; i < ref.size; i++) if (act[ref.off + i] > act[ref.off + best]) best = i;
      ref.items.forEach((el, i) => {
        const p = act[ref.off + i];
        el.classList.toggle("top", i === best);
        if (ref.row) {
          el.firstChild.style.height = `${Math.round(p * 100)}%`;
        } else {
          el.firstChild.style.width = `${Math.round(p * 100)}%`;
          el.querySelector("b").textContent = `${Math.round(p * 100)}%`;
        }
      });
      if (ref.pick) ref.pick.textContent = `most likely: ${ref.labels[best]}`;
    }
    // Mark the digit that the agent is actually broadcasting now.
    const code = this.replay.frames.radio[step * N + agent] || "";
    this.headRefs.filter((r) => r.row).forEach((ref, d) => {
      const digit = Number(code[d]);
      ref.items.forEach((el, i) => el.style.outline = i === digit ? `1.5px solid ${this.c.label2}` : "");
    });
  }

  _links(mem, agent) {
    const pol = this.replay.network.policy;
    const [, M, A] = pol.shape;
    const base = agent * M * A;
    const all = [];
    for (let m = 0; m < M; m++) {
      const mv = mem[m];
      if (Math.abs(mv) < 0.02) continue;
      for (let a = 0; a < A; a++) {
        const c = (pol.data[base + m * A + a] - 128) * pol.scale * mv;
        all.push([Math.abs(c), c, m, a]);
      }
    }
    all.sort((p, q) => q[0] - p[0]);
    const top = all.slice(0, 24);

    const { g, w, h } = setupCanvas(this.cv.links);
    g.clearRect(0, 0, w, h);
    const lx = 14, rx = w - 44, pad = 8;
    const my = (m) => pad + (m / (M - 1)) * (h - pad * 2);
    // Right-hand nodes grouped by head with small gaps.
    const heads = this.replay.network.heads;
    const gapPx = 6;
    const unit = (h - pad * 2 - gapPx * (heads.length - 1)) / (A - 1);
    const ay = [];
    let yy = pad;
    const groups = [];
    heads.forEach((hd, hi) => {
      const y0 = yy;
      for (let i = 0; i < hd.size; i++) { ay.push(yy); if (i < hd.size - 1) yy += unit; }
      groups.push([hd.name, y0, yy]);
      if (hi < heads.length - 1) yy += unit + gapPx;
    });

    // Nodes
    for (let m = 0; m < M; m++) {
      const v = mem[m];
      const [cr, cg, cb] = v >= 0 ? this.c.pos : this.c.neg;
      g.fillStyle = `rgba(${cr},${cg},${cb},${0.15 + 0.85 * Math.min(1, Math.abs(v))})`;
      g.fillRect(lx - 3, my(m) - 0.8, 6, 1.6);
    }
    g.fillStyle = this.c.label3;
    for (let a = 0; a < A; a++) g.fillRect(rx - 3, ay[a] - 0.8, 6, 1.6);
    g.font = `500 10px ${getComputedStyle(document.body).fontFamily}`;
    g.textBaseline = "middle";
    for (const [name, y0, y1] of groups) {
      g.fillStyle = this.c.label2;
      g.fillText(SHORT[name] || name, rx + 8, (y0 + y1) / 2);
      g.strokeStyle = this.c.sep;
      g.beginPath(); g.moveTo(rx + 4, y0); g.lineTo(rx + 4, y1); g.stroke();
    }
    if (!top.length) return;
    const maxC = top[0][0] || 1;
    for (let i = top.length - 1; i >= 0; i--) {
      const [ac, c, m, a] = top[i];
      const k = ac / maxC;
      const [cr, cg, cb] = c >= 0 ? this.c.pos : this.c.neg;
      g.strokeStyle = `rgba(${cr},${cg},${cb},${0.15 + 0.8 * k})`;
      g.lineWidth = 0.6 + 2 * k;
      g.beginPath();
      g.moveTo(lx + 3, my(m));
      g.bezierCurveTo((lx + rx) / 2, my(m), (lx + rx) / 2, ay[a], rx - 3, ay[a]);
      g.stroke();
    }
  }
}

function roundRect(g, x, y, w, h, r) {
  g.beginPath();
  if (g.roundRect) g.roundRect(x, y, w, h, Math.min(r, w / 2, h / 2));
  else g.rect(x, y, w, h);
  g.fill();
}
