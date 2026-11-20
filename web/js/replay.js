// Loading and decoding replay files (see docs/REPLAY_FORMAT.md).

export const N = 8;

export async function loadIndex() {
  const res = await fetch(`replays/index.json?t=${Date.now()}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Could not load the replay list (${res.status}).`);
  const list = await res.json();
  return list.sort((a, b) => a.match - b.match);
}

async function fetchBytes(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not load ${url} (${res.status}).`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress?.(Math.min(got / total, 1));
  }
  const out = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

async function gunzipToText(bytes) {
  // Some servers decompress .gz on the way; accept plain JSON too.
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return new TextDecoder().decode(bytes);
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return await new Response(stream).text();
}

function b64ToBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

const cache = new Map();

export async function loadReplay(file, onProgress) {
  if (cache.has(file)) return cache.get(file);
  const bytes = await fetchBytes(`replays/${file}`, (p) => onProgress?.(p * 0.7));
  onProgress?.(0.75);
  const raw = JSON.parse(await gunzipToText(bytes));
  onProgress?.(0.9);
  const replay = decode(raw);
  onProgress?.(1);
  cache.set(file, replay);
  if (cache.size > 3) cache.delete(cache.keys().next().value);
  return replay;
}

function decode(raw) {
  const S = raw.steps;
  const f = raw.frames;
  const frames = {
    pos: Float32Array.from(f.pos),
    yaw: Float32Array.from(f.yaw),
    pitch: Float32Array.from(f.pitch),
    crouch: Uint8Array.from(f.crouch),
    down: Uint8Array.from(f.down),
    gunReady: Uint8Array.from(f.gun_ready),
    radio: f.radio,
    teamScores: Float32Array.from(f.team_scores),
  };

  // Per-agent score over time, rebuilt from tag events (+5 tagger, -3 tagged).
  const agentScore = new Float32Array(S * N);
  const shotsByStep = new Map();
  const tagSteps = [];
  for (const s of raw.shots) {
    if (!shotsByStep.has(s.step)) shotsByStep.set(s.step, []);
    shotsByStep.get(s.step).push(s);
  }
  const running = new Float32Array(N);
  for (let st = 0; st < S; st++) {
    for (const s of shotsByStep.get(st) || []) {
      if (s.tag && s.target >= 0) {
        running[s.shooter] += 5;
        running[s.target] -= 3;
        tagSteps.push({ step: st, team: raw.agents[s.shooter].team });
      }
    }
    agentScore.set(running, st * N);
  }

  // Radio code usage per team (counted per agent-step).
  const radioCounts = { green: new Map(), pink: new Map() };
  for (let i = 0; i < S * N; i++) {
    const team = raw.agents[i % N].team;
    const code = frames.radio[i];
    const m = radioCounts[team];
    m.set(code, (m.get(code) || 0) + 1);
  }

  const net = raw.network;
  const q = (key) => {
    const e = net[key];
    return { size: e.size, lo: e.range[0], hi: e.range[1], data: b64ToBytes(e.data) };
  };
  const pw = net.policy_weights;
  const network = {
    vision: { rows: net.vision.rows, cols: net.vision.cols, data: b64ToBytes(net.vision.data) },
    encVision: q("enc_vision"),
    encOther: q("enc_other"),
    memory: q("memory"),
    actionProbs: q("action_probs"),
    heads: net.heads,
    value: Float32Array.from(net.value),
    policy: { shape: pw.shape, scale: pw.scale / 127, data: b64ToBytes(pw.data) },
  };

  // Value range per agent for the outlook chart.
  const valueRange = [];
  for (let a = 0; a < N; a++) {
    let lo = Infinity, hi = -Infinity;
    for (let st = 0; st < S; st++) {
      const v = network.value[st * N + a];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    valueRange.push([lo, hi]);
  }

  return {
    match: raw.match,
    trainedMatches: raw.trained_matches,
    dt: raw.dt,
    steps: S,
    arena: raw.arena,
    agents: raw.agents,
    finalScores: raw.final_scores,
    frames,
    shots: raw.shots,
    shotsByStep,
    tagSteps,
    agentScore,
    radioCounts,
    network,
    valueRange,
  };
}

/** Dequantized slice of one quantized network array for (step, agent). */
export function netSlice(q, step, agent, out) {
  const off = (step * N + agent) * q.size;
  const k = (q.hi - q.lo) / 255;
  for (let i = 0; i < q.size; i++) out[i] = q.lo + q.data[off + i] * k;
  return out;
}
