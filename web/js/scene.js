// Three.js arena, players, beams and cameras. World is z-up, metres.
import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";
import { N } from "./replay.js";

const BODY_R = 0.45;          // sphere radius / half cube side
const STAND_H = 1.75, CROUCH_H = 1.15;
const STAND_EYE = 1.6, CROUCH_EYE = 1.0;
const BARREL_LEN = 0.62;
const BEAM_STEPS = 3;         // how long a beam stays visible, in steps

const THEMES = {
  light: {
    bg: 0xe4e4ea, floor: 0xd6d6db, grid: 0x000000, gridOpacity: 0.06,
    kinds: { wall: 0x9c9ca3, cover: 0xb9b9bf, platform: 0xababb1, stair: 0xc6c6cb },
    edge: 0x3a3a3f, edgeOpacity: 0.28,
    green: 0x28b44f, pink: 0xff2d78, hemiSky: 0xffffff, hemiGround: 0x8a8a90, hemi: 1.6, sun: 2.0,
  },
  dark: {
    bg: 0x101012, floor: 0x2a2a2e, grid: 0xffffff, gridOpacity: 0.05,
    kinds: { wall: 0x5a5a60, cover: 0x75757b, platform: 0x66666c, stair: 0x808086 },
    edge: 0x000000, edgeOpacity: 0.55,
    green: 0x32d74b, pink: 0xff4f8f, hemiSky: 0xbfc4d6, hemiGround: 0x202024, hemi: 1.1, sun: 1.7,
  },
};

export class ArenaScene {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    this.scene = new THREE.Scene();
    this.theme = THEMES.light;

    // Cameras
    this.persp = new THREE.PerspectiveCamera(45, 1, 0.1, 400);
    this.persp.up.set(0, 0, 1);
    this.persp.position.set(15, -16, 22);
    this.orbit = new OrbitControls(this.persp, canvas);
    this.orbit.target.set(15, 10, 0);
    this.orbit.enableDamping = true;
    this.orbit.dampingFactor = 0.08;
    this.orbit.maxPolarAngle = Math.PI * 0.495;   // any angle above the floor
    this.orbit.minDistance = 1.5;
    this.orbit.maxDistance = 90;
    this.orbit.zoomToCursor = true;
    this.orbit.zoomSpeed = 0.8;
    this.orbit.rotateSpeed = 0.9;

    this.ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 200);
    this.ortho.position.set(15, 10, 60);
    this.ortho.up.set(0, 1, 0);
    this.ortho.lookAt(15, 10, 0);
    this.topControls = new OrbitControls(this.ortho, canvas);
    this.topControls.target.set(15, 10, 0);
    this.topControls.enableRotate = false;
    this.topControls.enableDamping = true;
    this.topControls.dampingFactor = 0.08;
    this.topControls.zoomToCursor = true;
    this.topControls.screenSpacePanning = true;
    this.topControls.minZoom = 0.6;
    this.topControls.maxZoom = 6;
    this.topControls.enabled = false;

    this.fp = new THREE.PerspectiveCamera(60, 1, 0.05, 200);
    this.fp.up.set(0, 0, 1);

    this.mode = "orbit";
    // Agent view: drag looks around and scroll zooms (field of view), all eased.
    this.look = { yaw: 0, pitch: 0, tYaw: 0, tPitch: 0, fov: 60, tFov: 60 };
    this.fpPose = null;
    this._bindLook(canvas);
    this.followAgent = 0;
    this.selected = 0;

    // Lights
    this.hemi = new THREE.HemisphereLight(0xffffff, 0x888888, 1.5);
    this.hemi.position.set(0, 0, 1);
    this.scene.add(this.hemi);
    this.sun = new THREE.DirectionalLight(0xffffff, 2);
    this.sun.position.set(6, -8, 24);
    this.sun.target.position.set(15, 10, 0);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    Object.assign(this.sun.shadow.camera, { left: -22, right: 22, top: 22, bottom: -22, near: 1, far: 60 });
    this.sun.shadow.bias = -0.0004;
    this.sun.shadow.normalBias = 0.02;
    this.scene.add(this.sun, this.sun.target);

    this.arenaGroup = new THREE.Group();
    this.scene.add(this.arenaGroup);
    this.agentGroup = new THREE.Group();
    this.scene.add(this.agentGroup);
    this.beamGroup = new THREE.Group();
    this.scene.add(this.beamGroup);

    this._buildAgents();
    this._buildBeamPool();
    this.resize();
    window.addEventListener("resize", () => this.resize());
    this.clock = new THREE.Clock();
  }

  setTheme(name) {
    this.themeName = name;
    this.theme = THEMES[name];
    const t = this.theme;
    this.scene.background = new THREE.Color(t.bg);
    this.scene.fog = new THREE.Fog(t.bg, 70, 160);
    this.hemi.color.set(t.hemiSky);
    this.hemi.groundColor.set(t.hemiGround);
    this.hemi.intensity = t.hemi;
    this.sun.intensity = t.sun;
    if (this.arena) this.buildArena(this.arena);
    for (const a of this.agents) {
      const c = new THREE.Color(a.team === "green" ? t.green : t.pink);
      a.bodyMat.color.copy(c);
      a.ringMat.color.copy(c);
      a.selMat.color.copy(c);
    }
    this.beamColors = { green: new THREE.Color(t.green), pink: new THREE.Color(t.pink) };
  }

  buildArena(arena) {
    this.arena = arena;
    const g = this.arenaGroup;
    for (const c of [...g.children]) { g.remove(c); c.geometry?.dispose(); }
    const t = this.theme;
    const W = arena.width, D = arena.depth;

    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(W + 0.6, D + 0.6),
      new THREE.MeshStandardMaterial({ color: t.floor, roughness: 0.95 })
    );
    floor.position.set(W / 2, D / 2, 0);
    floor.receiveShadow = true;
    g.add(floor);

    // 1 m grid on the floor
    const pts = [];
    for (let x = 0; x <= W; x++) pts.push(x, 0, 0.003, x, D, 0.003);
    for (let y = 0; y <= D; y++) pts.push(0, y, 0.003, W, y, 0.003);
    const gridGeo = new THREE.BufferGeometry();
    gridGeo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    g.add(new THREE.LineSegments(gridGeo, new THREE.LineBasicMaterial({ color: t.grid, transparent: true, opacity: t.gridOpacity })));

    const mats = {};
    for (const [k, col] of Object.entries(t.kinds)) mats[k] = new THREE.MeshStandardMaterial({ color: col, roughness: 0.85, metalness: 0 });
    const edgeMat = new THREE.LineBasicMaterial({ color: t.edge, transparent: true, opacity: t.edgeOpacity });
    for (const [x0, y0, z0, x1, y1, z1, kind] of arena.boxes) {
      const geo = new THREE.BoxGeometry(x1 - x0, y1 - y0, z1 - z0);
      const m = new THREE.Mesh(geo, mats[kind] || mats.wall);
      m.position.set((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2);
      m.castShadow = true;
      m.receiveShadow = true;
      g.add(m);
      const e = new THREE.LineSegments(new THREE.EdgesGeometry(geo), edgeMat);
      e.position.copy(m.position);
      g.add(e);
    }
    this.resize();
  }

  _buildAgents() {
    this.agents = [];
    const sphere = new THREE.SphereGeometry(BODY_R, 40, 24);
    const cube = new THREE.BoxGeometry(BODY_R * 1.8, BODY_R * 1.8, BODY_R * 1.8);
    const barrel = new THREE.CylinderGeometry(0.055, 0.065, BARREL_LEN, 16);
    barrel.rotateZ(-Math.PI / 2);
    barrel.translate(BARREL_LEN / 2 + BODY_R * 0.55, 0, 0);
    const tip = new THREE.SphereGeometry(0.08, 16, 12);
    const ring = new THREE.RingGeometry(0.5, 0.58, 48);
    const sel = new THREE.RingGeometry(0.66, 0.76, 64);

    for (let i = 0; i < N; i++) {
      const team = i < 4 ? "green" : "pink";
      const root = new THREE.Group();
      const bodyMat = new THREE.MeshStandardMaterial({ roughness: 0.35, metalness: 0.05, transparent: true });
      const body = new THREE.Mesh(team === "green" ? sphere : cube, bodyMat);
      body.castShadow = true;
      const aim = new THREE.Group();
      const barrelMat = new THREE.MeshStandardMaterial({ color: 0x2c2c30, roughness: 0.4, metalness: 0.4, transparent: true });
      const barrelMesh = new THREE.Mesh(barrel, barrelMat);
      barrelMesh.castShadow = true;
      const tipMat = new THREE.MeshBasicMaterial({ color: 0xffd60a, transparent: true });
      const tipMesh = new THREE.Mesh(tip, tipMat);
      tipMesh.position.x = BARREL_LEN + BODY_R * 0.55;
      aim.add(barrelMesh, tipMesh);
      const ringMat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.55, depthWrite: false });
      const ringMesh = new THREE.Mesh(ring, ringMat);
      ringMesh.position.z = 0.012;
      const selMat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.9, depthWrite: false });
      const selMesh = new THREE.Mesh(sel, selMat);
      selMesh.position.z = 0.014;
      const label = makeLabel(String(i + 1));
      root.add(body, aim, ringMesh, selMesh, label);
      this.agentGroup.add(root);
      this.agents.push({ team, root, body, bodyMat, aim, barrelMat, tipMesh, tipMat, ringMesh, ringMat, selMesh, selMat, label });
    }
  }

  _buildBeamPool() {
    this.beams = [];
    const geo = new THREE.CylinderGeometry(1, 1, 1, 10, 1, true);
    geo.translate(0, 0.5, 0);
    const spark = new THREE.SphereGeometry(1, 16, 12);
    for (let i = 0; i < 48; i++) {
      const glowMat = new THREE.MeshBasicMaterial({ transparent: true, blending: THREE.AdditiveBlending, depthWrite: false });
      const glow = new THREE.Mesh(geo, glowMat);
      const coreMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, depthWrite: false });
      const core = new THREE.Mesh(geo, coreMat);
      const sparkMat = new THREE.MeshBasicMaterial({ transparent: true, blending: THREE.AdditiveBlending, depthWrite: false });
      const sp = new THREE.Mesh(spark, sparkMat);
      const g = new THREE.Group();
      g.add(glow, core, sp);
      g.visible = false;
      this.beamGroup.add(g);
      this.beams.push({ g, glow, glowMat, core, coreMat, sp, sparkMat });
    }
  }

  resize() {
    const w = this.canvas.clientWidth || window.innerWidth;
    const h = this.canvas.clientHeight || window.innerHeight;
    this.renderer.setSize(w, h, false);
    const asp = w / h;
    this.persp.aspect = asp; this.persp.updateProjectionMatrix();
    this.fp.aspect = asp; this.fp.updateProjectionMatrix();
    // Fit the arena (30 x 20 plus margin) in the top-down view.
    const W = (this.arena?.width ?? 30) + 6, D = (this.arena?.depth ?? 20) + 8;
    const half = Math.max(D / 2, W / 2 / asp);
    Object.assign(this.ortho, { left: -half * asp, right: half * asp, top: half, bottom: -half });
    this.ortho.updateProjectionMatrix();
  }

  setMode(mode) {
    this.mode = mode;
    this.orbit.enabled = mode === "orbit" || mode === "chase";
    this.topControls.enabled = mode === "top";
    if (mode === "follow") this.fpPose = null;
    this.canvas.style.cursor = mode === "follow" ? "grab" : "";
  }

  _bindLook(canvas) {
    let drag = null;
    canvas.addEventListener("pointerdown", (e) => {
      if (this.mode !== "follow") return;
      drag = { x: e.clientX, y: e.clientY };
      canvas.setPointerCapture(e.pointerId);
      canvas.style.cursor = "grabbing";
    });
    canvas.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const k = (this.look.tFov * Math.PI / 180) / canvas.clientHeight;
      this.look.tYaw += (e.clientX - drag.x) * k;
      this.look.tPitch = Math.max(-1.2, Math.min(1.2, this.look.tPitch + (e.clientY - drag.y) * k));
      drag.x = e.clientX; drag.y = e.clientY;
    });
    const end = () => { drag = null; if (this.mode === "follow") canvas.style.cursor = "grab"; };
    canvas.addEventListener("pointerup", end);
    canvas.addEventListener("pointercancel", end);
    canvas.addEventListener("wheel", (e) => {
      if (this.mode !== "follow") return;
      e.preventDefault();
      this.look.tFov = Math.max(25, Math.min(95, this.look.tFov * Math.exp(e.deltaY * 0.001)));
    }, { passive: false });
    canvas.addEventListener("dblclick", () => this.resetLook());
  }

  resetLook() { Object.assign(this.look, { tYaw: 0, tPitch: 0, tFov: 60 }); }

  get camera() {
    return this.mode === "top" ? this.ortho : this.mode === "follow" ? this.fp : this.persp;
  }

  /** Pose the scene at fractional step t. */
  update(replay, t) {
    const S = replay.steps;
    const fr = replay.frames;
    const s0 = Math.max(0, Math.min(S - 1, Math.floor(t)));
    const s1 = Math.min(S - 1, s0 + 1);
    const a = Math.min(1, Math.max(0, t - s0));
    this._dt = this.clock.getDelta();
    const now = this.clock.elapsedTime;
    this.poses = this.poses || Array.from({ length: N }, () => ({ x: 0, y: 0, z: 0, yaw: 0, pitch: 0, crouch: 0, down: 0 }));

    for (let i = 0; i < N; i++) {
      const A = this.agents[i];
      const k0 = s0 * N + i, k1 = s1 * N + i;
      const x0 = fr.pos[k0 * 3], y0 = fr.pos[k0 * 3 + 1], z0 = fr.pos[k0 * 3 + 2];
      const x1 = fr.pos[k1 * 3], y1 = fr.pos[k1 * 3 + 1], z1 = fr.pos[k1 * 3 + 2];
      const jump = Math.hypot(x1 - x0, y1 - y0) > 1.5 || fr.down[k0] !== fr.down[k1];
      const b = jump ? 0 : a;
      const p = this.poses[i];
      p.x = x0 + (x1 - x0) * b;
      p.y = y0 + (y1 - y0) * b;
      p.z = z0 + (z1 - z0) * b;
      p.yaw = lerpAngle(fr.yaw[k0], fr.yaw[k1], b);
      p.pitch = fr.pitch[k0] + (fr.pitch[k1] - fr.pitch[k0]) * b;
      p.crouch = fr.crouch[k0] + (fr.crouch[k1] - fr.crouch[k0]) * b;
      p.down = fr.down[k0];
      const height = STAND_H + (CROUCH_H - STAND_H) * p.crouch;
      p.eye = STAND_EYE + (CROUCH_EYE - STAND_EYE) * p.crouch;
      const center = height - BODY_R - 0.1;

      A.root.position.set(p.x, p.y, p.z);
      A.body.position.z = center;
      A.body.rotation.z = p.yaw;
      A.aim.position.z = center;
      A.aim.rotation.set(0, 0, 0);
      A.aim.rotateZ(p.yaw);
      A.aim.rotateY(-p.pitch);
      A.label.position.z = height + 0.42;

      const ready = fr.gunReady[k0] === 1;
      A.tipMat.color.setHex(ready ? 0xffd60a : 0x48484c);
      A.tipMesh.scale.setScalar(ready ? 1 + 0.15 * Math.sin(now * 6 + i) : 0.8);

      const hidden = this.mode === "follow" && this.followAgent === i;
      A.root.visible = !hidden;
      if (p.down) {
        const blink = 0.18 + 0.22 * (0.5 + 0.5 * Math.sin(now * 14));
        A.bodyMat.opacity = blink;
        A.barrelMat.opacity = blink;
        A.tipMesh.visible = false;
        A.ringMat.opacity = 0.15;
        A.label.material.opacity = 0.4;
      } else {
        A.bodyMat.opacity = 1;
        A.barrelMat.opacity = 1;
        A.tipMesh.visible = true;
        A.ringMat.opacity = 0.5;
        A.label.material.opacity = 1;
      }
      A.bodyMat.depthWrite = !p.down;
      const isSel = this.selected === i;
      A.selMesh.visible = isSel;
      if (isSel) {
        const pulse = 1 + 0.06 * Math.sin(now * 3);
        A.selMesh.scale.setScalar(pulse);
        A.selMat.opacity = 0.75 + 0.2 * Math.sin(now * 3);
      }
      A.label.visible = this.mode !== "follow";
    }

    this._updateBeams(replay, t);
    this._updateCamera();
  }

  _updateBeams(replay, t) {
    let used = 0;
    const cur = Math.floor(t);
    const fr = replay.frames;
    const up = new THREE.Vector3(0, 1, 0);
    for (let st = Math.max(0, cur - BEAM_STEPS + 1); st <= cur; st++) {
      const list = replay.shotsByStep.get(st);
      if (!list) continue;
      for (const s of list) {
        if (used >= this.beams.length) break;
        const age = (t - st) / BEAM_STEPS; // 0..1
        if (age < 0 || age > 1) continue;
        const B = this.beams[used++];
        const k = st * N + s.shooter;
        const eye = (fr.crouch[k] ? CROUCH_EYE : STAND_EYE);
        const start = new THREE.Vector3(fr.pos[k * 3], fr.pos[k * 3 + 1], fr.pos[k * 3 + 2] + eye);
        const end = new THREE.Vector3(...s.end);
        const dir = end.clone().sub(start);
        const len = dir.length();
        if (len < 0.01) { used--; continue; }
        dir.normalize();
        B.g.position.copy(start);
        B.g.quaternion.setFromUnitVectors(up, dir);
        const fade = (1 - age) * (1 - age);
        const w = s.tag ? 1.5 : 1;
        B.glow.scale.set(0.07 * w, len, 0.07 * w);
        B.core.scale.set(0.018 * w, len, 0.018 * w);
        B.glowMat.color.copy(this.beamColors[replay.agents[s.shooter].team]);
        B.glowMat.opacity = 0.85 * fade;
        B.coreMat.opacity = 0.9 * fade;
        B.sp.position.set(0, len, 0);
        B.sp.scale.setScalar((s.tag ? 0.45 : 0.18) * (0.6 + age));
        B.sparkMat.color.copy(s.tag ? new THREE.Color(0xffffff) : this.beamColors[replay.agents[s.shooter].team]);
        B.sparkMat.opacity = fade;
        B.g.visible = true;
      }
    }
    for (let i = used; i < this.beams.length; i++) this.beams[i].g.visible = false;
  }

  _updateCamera() {
    const dt = Math.min(0.1, this._dt || 0.016);
    if (this.mode === "orbit") this.orbit.update();
    else if (this.mode === "top") this.topControls.update();
    else if (this.mode === "chase" && this.poses) {
      // Free orbit around the selected agent; the pivot glides after them.
      const p = this.poses[this.followAgent];
      const delta = new THREE.Vector3(p.x, p.y, p.z + 1.1).sub(this.orbit.target).multiplyScalar(1 - Math.exp(-dt * 7));
      this.orbit.target.add(delta);
      this.persp.position.add(delta);
      this.orbit.update();
    } else if (this.mode === "follow" && this.poses) {
      const p = this.poses[this.followAgent];
      const L = this.look;
      const kl = 1 - Math.exp(-dt * 10);
      L.yaw += (L.tYaw - L.yaw) * kl;
      L.pitch += (L.tPitch - L.pitch) * kl;
      L.fov += (L.tFov - L.fov) * kl;
      const want = { x: p.x, y: p.y, z: p.z + p.eye, yaw: p.yaw, pitch: p.pitch };
      if (!this.fpPose) this.fpPose = { ...want };
      const f = this.fpPose, kp = 1 - Math.exp(-dt * 20); // short glide when switching agents
      f.x += (want.x - f.x) * kp; f.y += (want.y - f.y) * kp; f.z += (want.z - f.z) * kp;
      f.yaw = lerpAngle(f.yaw, want.yaw, kp); f.pitch += (want.pitch - f.pitch) * kp;
      this.fp.fov = L.fov; this.fp.updateProjectionMatrix();
      this.fp.position.set(f.x, f.y, f.z);
      const yaw = f.yaw - L.yaw, pit = Math.max(-1.4, Math.min(1.4, f.pitch - L.pitch)), cp = Math.cos(pit);
      this.fp.lookAt(f.x + Math.cos(yaw) * cp, f.y + Math.sin(yaw) * cp, f.z + Math.sin(pit));
    }
  }

  /** Smoothly bring the orbit camera to look at an agent. */
  focusAgent(i, instant) {
    if (!this.poses) return;
    const p = this.poses[i];
    const target = new THREE.Vector3(p.x, p.y, p.z + 1);
    const offset = this.persp.position.clone().sub(this.orbit.target);
    if (offset.length() > 18) offset.setLength(18);
    this._tween(this.orbit.target.clone(), target, this.persp.position.clone(), target.clone().add(offset), instant);
  }

  resetView(instant) {
    this._tween(this.orbit.target.clone(), new THREE.Vector3(15, 10, 0), this.persp.position.clone(), new THREE.Vector3(15, -16, 22), instant);
  }

  _tween(t0, t1, p0, p1, instant) {
    if (instant) { this.orbit.target.copy(t1); this.persp.position.copy(p1); return; }
    const start = performance.now(), dur = 650;
    const step = () => {
      const k = Math.min(1, (performance.now() - start) / dur);
      const e = 1 - Math.pow(1 - k, 4);
      this.orbit.target.lerpVectors(t0, t1, e);
      this.persp.position.lerpVectors(p0, p1, e);
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }
}

function lerpAngle(a, b, t) {
  let d = ((b - a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
  return a + d * t;
}

function makeLabel(text) {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d");
  g.fillStyle = "rgba(20,20,22,0.72)";
  g.beginPath(); g.arc(32, 32, 26, 0, Math.PI * 2); g.fill();
  g.fillStyle = "#fff";
  g.font = "600 32px -apple-system, BlinkMacSystemFont, Inter, sans-serif";
  g.textAlign = "center"; g.textBaseline = "middle";
  g.fillText(text, 32, 34);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }));
  s.scale.set(0.42, 0.42, 1);
  s.renderOrder = 10;
  return s;
}
