// Browser UI for Quoridor. Reads immutable snapshots from engine.js and
// never changes coordinates: row 0 is always drawn at the top.
import * as E from "./engine.js";
import { search as botSearch } from "./bot.js";

const NS = "http://www.w3.org/2000/svg";

// ---------------------------------------------------------------- layout
const CELL = 52, GAP = 14, PITCH = CELL + GAP, FRAME = 14, LABEL = 26, RES = 96;
const N = E.SIZE;
const BOARD = N * CELL + (N - 1) * GAP;
const FRAME_PX = BOARD + 2 * FRAME;
const W = LABEL + FRAME_PX + 10;
const H = RES + LABEL + FRAME_PX + RES;
const FX = LABEL, FY = RES + LABEL;            // frame top-left
const OX = FX + FRAME, OY = FY + FRAME;         // first square top-left
const LONG = 2 * CELL + GAP;
const STUB_H = RES - 26;
// Reserve strips: Player 2 (top of board) above, Player 1 below.
const STRIP_Y = [FY + FRAME_PX + 6, 0];

const COLORS = {
  square: "#c4aa84", groove: "#241b16", frame: "#342620", frameEdge: "#4e3a2c",
  pawn: ["#ff525c", "#3aa0ff"], pawnDark: ["#a8202c", "#1456aa"],
  wall: "#ffc440", wallDark: "#b07814", ok: "#78e68c", bad: "#f05050",
  dot: "#6e7078", label: "#969cac",
};
const GOAL_TINT = 0.30;

const squareXY = (r, c) => [OX + c * PITCH, OY + r * PITCH];
const squareCenter = (r, c) => [OX + c * PITCH + CELL / 2, OY + r * PITCH + CELL / 2];

function wallRect([r, c, o]) {
  return o === "h"
    ? [OX + c * PITCH, OY + r * PITCH + CELL, LONG, GAP]
    : [OX + c * PITCH + CELL, OY + r * PITCH, GAP, LONG];
}

function stubRect(player, i) {
  const step = (BOARD - GAP - 24) / (E.WALLS_PER_PLAYER - 1);
  return [OX + 12 + i * step, STRIP_Y[player] + 13, GAP, STUB_H];
}

const inReserve = (player, [x, y]) =>
  x >= OX - 10 && x <= OX + BOARD + 10 && y >= STRIP_Y[player] && y <= STRIP_Y[player] + RES - 6;

function squareAt([x, y], forgiving = false) {
  const lx = x - OX, ly = y - OY;
  if (forgiving) {
    if (lx < -GAP || ly < -GAP || lx > BOARD + GAP || ly > BOARD + GAP) return null;
    const clamp = (v) => Math.min(Math.max(v, 0), N - 1);
    return [clamp(Math.round((ly - CELL / 2) / PITCH)), clamp(Math.round((lx - CELL / 2) / PITCH))];
  }
  if (lx < 0 || ly < 0) return null;
  const c = Math.floor(lx / PITCH), r = Math.floor(ly / PITCH);
  if (r >= N || c >= N || lx - c * PITCH > CELL || ly - r * PITCH > CELL) return null;
  return [r, c];
}

// Same snapping rule as quoridor/gui/layout.py: nearest groove decides the
// orientation; horizontal walls extend left of the hovered square, vertical
// walls extend down. Second entry is the one-square-shifted fallback.
function wallCandidates([x, y]) {
  let lx = x - OX, ly = y - OY;
  const tol = CELL / 2;
  if (lx < -tol || ly < -tol || lx > BOARD + tol || ly > BOARD + tol) return [];
  lx = Math.min(Math.max(lx, 0), BOARD - 1);
  ly = Math.min(Math.max(ly, 0), BOARD - 1);
  const col = Math.min(Math.floor(lx / PITCH), N - 1);
  const row = Math.min(Math.floor(ly / PITCH), N - 1);
  const groove = (v) => {
    const k = Math.min(Math.max(Math.round((v - CELL - GAP / 2) / PITCH), 0), N - 2);
    return [k, Math.abs(v - (k * PITCH + CELL + GAP / 2))];
  };
  const [kv, dv] = groove(lx), [kh, dh] = groove(ly);
  if (dv < dh) {
    const anchor = Math.min(row, N - 2);
    const out = [[anchor, kv, "v"]];
    if (row < N - 1 && anchor - 1 >= 0) out.push([anchor - 1, kv, "v"]);
    return out;
  }
  const anchor = Math.max(col - 1, 0);
  const out = [[kh, anchor, "h"]];
  if (col > 0 && anchor + 1 <= N - 2) out.push([kh, anchor + 1, "h"]);
  return out;
}

// ---------------------------------------------------------------- state
let game = E.newGame();
let snap = E.snapshot(game);
let history = [];
let validWallKeys = new Set();
const ui = {};
resetUi();

function resetUi() {
  Object.assign(ui, {
    selected: false, press: null, pawnDrag: false,
    wallDrag: false, preview: null, previewOk: false, mouse: [0, 0],
  });
}

// ---------------------------------------------------------------- bot
const settings = { mode: "hvh", depth: 4, timeMs: 5000, humanSide: 0 };
const isBot = (p) => settings.mode === "hvb" && p !== settings.humanSide;
let worker = null;
let workerBroken = false;
let requestId = 0;
let thinking = false;
let lastBotInfo = null;

function getWorker() {
  if (worker || workerBroken) return worker;
  try {
    worker = new Worker(new URL("./bot-worker.js", import.meta.url), { type: "module" });
    worker.onmessage = (evt) => onBotResult(evt.data);
    worker.onerror = () => {  // e.g. module workers unsupported: fall back to the main thread
      workerBroken = true;
      worker = null;
      if (thinking) { thinking = false; maybeBotTurn(); }
    };
  } catch {
    workerBroken = true;
    worker = null;
  }
  return worker;
}

function maybeBotTurn() {
  if (snap.winner !== null || !isBot(snap.current_player) || thinking || animating) return;
  thinking = true;
  const id = ++requestId;
  const msg = { id, snapshot: snap, depth: settings.depth, timeLimitMs: settings.timeMs || null };
  const w = getWorker();
  if (w) {
    w.postMessage(msg);
  } else {
    // Let the "thinking" status paint before blocking the main thread.
    setTimeout(() => onBotResult({ id, ...botSearch(msg.snapshot, msg.depth, msg.timeLimitMs) }), 50);
  }
  renderPanel();
}

function onBotResult({ id, move, info }) {
  if (id !== requestId || !thinking) return;  // stale answer (new game / undo)
  thinking = false;
  lastBotInfo = info;
  play(move);
}

function cancelBot() {
  requestId++;
  if (thinking && worker) { worker.terminate(); worker = null; }
  thinking = false;
}

function describeBotInfo(info) {
  if (!info) return "";
  const outcome = info.forced ? ` · sees a forced ${info.forced === "win" ? "win" : "loss"} for itself`
    : ` · score ${info.score > 0 ? "+" : ""}${info.score} (bot steps − your steps)`;
  return `Bot searched depth ${info.depth_reached}/${info.depth_requested} · ` +
    `${info.nodes.toLocaleString()} positions in ${info.seconds.toFixed(2)} s${outcome}`;
}

function setGame(g) {
  game = g;
  snap = E.snapshot(game);
  validWallKeys = new Set(snap.valid_walls.map((w) => w.join(",")));
}
setGame(game);

const names = () => [0, 1].map((p) => (isBot(p) ? "Bot" : $(`name${p}`).value.trim() || `Player ${p + 1}`));
const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- svg helpers
const svg = $("board");
svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

function mix(hexA, hexB, t) {
  const a = hexA.match(/\w\w/g).map((h) => parseInt(h, 16));
  const b = hexB.match(/\w\w/g).map((h) => parseInt(h, 16));
  return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(",")})`;
}

// Static defs + layers.
const defs = el("defs", {}, svg);
for (const p of [0, 1]) {
  const g = el("radialGradient", { id: `pg${p}`, cx: "35%", cy: "30%", r: "75%" }, defs);
  el("stop", { offset: "0%", "stop-color": mix(COLORS.pawn[p].slice(1), "ffffff", 0.5) }, g);
  el("stop", { offset: "45%", "stop-color": COLORS.pawn[p] }, g);
  el("stop", { offset: "100%", "stop-color": COLORS.pawnDark[p] }, g);
}
const shadow = el("filter", { id: "shadow", x: "-50%", y: "-50%", width: "200%", height: "200%" }, defs);
el("feDropShadow", { dx: 0, dy: 2, stdDeviation: 2, "flood-opacity": 0.45 }, shadow);

const layers = {};
for (const name of ["base", "hover", "walls", "preview", "dots", "pawns", "reserve", "float"]) {
  layers[name] = el("g", { class: name }, svg);
}

function drawBase() {
  const g = layers.base;
  el("rect", { x: FX, y: FY, width: FRAME_PX, height: FRAME_PX, rx: 14, fill: COLORS.frame,
               stroke: COLORS.frameEdge, "stroke-width": 3 }, g);
  el("rect", { x: OX - 4, y: OY - 4, width: BOARD + 8, height: BOARD + 8, rx: 6, fill: COLORS.groove }, g);
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      let fill = COLORS.square;
      if (r === E.GOAL_ROWS[0]) fill = mix(COLORS.square.slice(1), COLORS.pawn[0].slice(1), GOAL_TINT);
      if (r === E.GOAL_ROWS[1]) fill = mix(COLORS.square.slice(1), COLORS.pawn[1].slice(1), GOAL_TINT);
      const [x, y] = squareXY(r, c);
      el("rect", { x, y, width: CELL, height: CELL, rx: 6, fill }, g);
    }
  }
  for (let i = 0; i < N; i++) {
    const [cx] = squareCenter(0, i);
    const [, cy] = squareCenter(i, 0);
    el("text", { x: cx, y: FY - 9, "text-anchor": "middle", fill: COLORS.label, "font-size": 14 }, g).textContent = i;
    el("text", { x: LABEL / 2, y: cy + 5, "text-anchor": "middle", fill: COLORS.label, "font-size": 14 }, g).textContent = i;
  }
  for (const p of [0, 1]) {
    el("rect", { x: OX - 10, y: STRIP_Y[p], width: BOARD + 20, height: RES - 6, rx: 12,
                 fill: "#222636", class: `strip strip${p}` }, g);
  }
}
drawBase();

function wallEl([x, y, w, h], parent, { fill = COLORS.wall, stroke = COLORS.wallDark, opacity = 1 } = {}) {
  return el("rect", { x, y, width: w, height: h, rx: 4, fill, stroke, "stroke-width": 2,
                      opacity, filter: opacity === 1 ? "url(#shadow)" : "" }, parent);
}

function pawnEl(p, [x, y], lifted = false) {
  const g = el("g", { class: `pawn pawn${p}`, transform: `translate(${x} ${y})` }, layers.pawns);
  const r = CELL * (lifted ? 0.40 : 0.36);
  el("circle", { r: r + 6, fill: "none", stroke: COLORS.pawn[p], "stroke-width": 4, class: "glow", opacity: 0 }, g);
  el("circle", { r, fill: `url(#pg${p})`, stroke: COLORS.pawnDark[p], "stroke-width": 3, filter: "url(#shadow)" }, g);
  return g;
}

const clear = (g) => { while (g.firstChild) g.removeChild(g.firstChild); };

// ---------------------------------------------------------------- render
let pawnEls = [null, null];

function render() {
  const me = snap.current_player;
  const over = snap.winner !== null;

  clear(layers.walls);
  for (const w of snap.walls) {
    wallEl(wallRect(w), layers.walls).dataset.key = w.join(",");
  }

  clear(layers.reserve);
  for (const p of [0, 1]) {
    let shown = snap.walls_left[p];
    if (p === me && ui.wallDrag) shown -= 1;
    for (let i = 0; i < shown; i++) wallEl(stubRect(p, i), layers.reserve);
    layers.base.querySelector(`.strip${p}`).setAttribute(
      "stroke", !over && p === me ? "#ffd666" : "none");
    layers.base.querySelector(`.strip${p}`).setAttribute("stroke-width", 2);
  }

  clear(layers.dots);
  if (!over && (ui.selected || ui.pawnDrag)) {
    for (const [r, c] of snap.valid_pawn_moves) {
      const [x, y] = squareCenter(r, c);
      el("circle", { cx: x, cy: y, r: CELL * 0.14, fill: COLORS.dot, opacity: 0.85 }, layers.dots);
    }
  }

  clear(layers.pawns);
  pawnEls = [0, 1].map((p) => pawnEl(p, squareCenter(...snap.pawns[p])));
  if (!over) pawnEls[me].querySelector(".glow").classList.add("pulse");
  if (ui.pawnDrag) {
    layers.pawns.appendChild(pawnEls[me]);  // keep dragged pawn on top
    pawnEls[me].setAttribute("transform", `translate(${ui.mouse[0]} ${ui.mouse[1]}) scale(1.1)`);
  }

  renderPreview();
  renderHover();
  renderPanel();
}

function renderPreview() {
  clear(layers.preview);
  clear(layers.float);
  if (!ui.wallDrag) return;
  if (ui.preview) {
    wallEl(wallRect(ui.preview), layers.preview, {
      fill: ui.previewOk ? COLORS.ok : COLORS.bad, stroke: "none", opacity: ui.previewOk ? 0.6 : 0.75,
    });
  }
  if (!ui.previewOk) wallEl(floatingRect(), layers.float, { opacity: 0.85 });
}

function renderHover() {
  clear(layers.hover);
  if (snap.winner !== null || ui.wallDrag || animating) return;
  const sq = squareAt(ui.mouse, ui.pawnDrag);
  if (!sq || !snap.valid_pawn_moves.some(([r, c]) => r === sq[0] && c === sq[1])) return;
  const [x, y] = squareXY(...sq);
  el("rect", { x, y, width: CELL, height: CELL, rx: 6, fill: "#ffffff", opacity: 0.25 }, layers.hover);
}

function renderPanel() {
  const n = names();
  for (const p of [0, 1]) {
    const chip = $(`chip${p}`);
    chip.classList.toggle("active", snap.winner === null && snap.current_player === p);
    chip.querySelector(".walls").textContent = snap.walls_left[p];
  }
  const status = $("status");
  status.classList.toggle("thinking", thinking);
  $("bot-info").textContent = settings.mode === "hvb" ? describeBotInfo(lastBotInfo) : "";
  if (thinking) {
    status.textContent = "Bot is thinking";
    status.style.color = COLORS.pawn[snap.current_player];
  } else if (snap.winner !== null) {
    status.textContent = `${n[snap.winner]} wins!`;
    status.style.color = COLORS.pawn[snap.winner];
  } else {
    status.textContent = `${n[snap.current_player]}'s turn`;
    status.style.color = COLORS.pawn[snap.current_player];
  }
  $("undo").disabled = history.length === 0 || animating;
  const overlay = $("overlay");
  overlay.hidden = snap.winner === null || animating;
  if (snap.winner !== null) {
    $("winner-text").textContent = `${n[snap.winner]} wins!`;
    $("winner-text").style.color = COLORS.pawn[snap.winner];
  }
}

let messageTimer = null;
function flash(text) {
  const m = $("message");
  m.textContent = text;
  m.style.opacity = 1;
  clearTimeout(messageTimer);
  messageTimer = setTimeout(() => { m.style.opacity = 0; }, 2600);
}

// ---------------------------------------------------------------- animation
let animating = false;
const easeOutCubic = (t) => 1 - (1 - t) ** 3;
const easeOutBack = (t) => { const c1 = 1.4, c3 = c1 + 1; return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2; };
const lerp = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);

function tween(duration, easing, step) {
  animating = true;
  return new Promise((resolve) => {
    const t0 = performance.now();
    const frame = (now) => {
      const t = Math.min((now - t0) / duration, 1);
      step(easing(t));
      if (t < 1) requestAnimationFrame(frame);
      else { animating = false; resolve(); }
    };
    requestAnimationFrame(frame);
  });
}

function animatePawn(p, from, to, ms = 240) {
  const g = pawnEls[p];
  layers.pawns.appendChild(g);
  const dist = Math.hypot(to[0] - from[0], to[1] - from[1]) / PITCH;
  return tween(ms * (1 + 0.25 * Math.max(dist - 1, 0)), easeOutCubic, (t) => {
    const [x, y] = lerp(from, to, t);
    g.setAttribute("transform", `translate(${x} ${y})`);
  });
}

function animateRect(node, from, to, ms, easing) {
  return tween(ms, easing, (t) => {
    const [x, y, w, h] = lerp(from, to, t);
    node.setAttribute("x", x); node.setAttribute("y", y);
    node.setAttribute("width", Math.max(w, 1)); node.setAttribute("height", Math.max(h, 1));
  });
}

// ---------------------------------------------------------------- moves
async function play(move, origin) {
  const before = snap;
  const p = before.current_player;
  const err = E.moveError(game, move);
  if (err) { flash(err); render(); return; }
  history.push(game);
  setGame(E.applyMove(game, move));
  resetUi();
  render();
  if (move.type === "pawn") {
    const from = origin || squareCenter(...before.pawns[p]);
    pawnEls[p].setAttribute("transform", `translate(${from[0]} ${from[1]})`);
    $("overlay").hidden = true;
    await animatePawn(p, from, squareCenter(move.row, move.col));
  } else {
    const k = [move.row, move.col, move.orientation].join(",");
    const node = layers.walls.querySelector(`[data-key="${k}"]`);
    const from = origin || stubRect(p, before.walls_left[p] - 1);
    await animateRect(node, from, wallRect([move.row, move.col, move.orientation]), 170, easeOutBack);
  }
  render();
  maybeBotTurn();
}

function floatingRect() {
  const [w, h] = ui.preview ? wallRect(ui.preview).slice(2) : [GAP, LONG];
  return [ui.mouse[0] - w / 2, ui.mouse[1] - h / 2, w, h];
}

function updatePreview() {
  const cands = wallCandidates(ui.mouse);
  ui.preview = null;
  ui.previewOk = false;
  for (const w of cands) {
    if (validWallKeys.has(w.join(","))) { ui.preview = w; ui.previewOk = true; return; }
  }
  if (cands.length) ui.preview = cands[0];
}

async function returnWall(rect) {
  const me = snap.current_player;
  ui.wallDrag = false;
  ui.preview = null;
  renderPreview();
  const node = wallEl(rect, layers.float);
  await animateRect(node, rect, stubRect(me, snap.walls_left[me] - 1), 220, easeOutCubic);
  render();
}

async function returnPawn(from) {
  const me = snap.current_player;
  ui.pawnDrag = false;
  render();
  await animatePawn(me, from, squareCenter(...snap.pawns[me]), 200);
  render();
}

// ---------------------------------------------------------------- input
function toSvg(evt) {
  const pt = svg.createSVGPoint();
  pt.x = evt.clientX; pt.y = evt.clientY;
  const p = pt.matrixTransform(svg.getScreenCTM().inverse());
  return [p.x, p.y];
}

const isValidSquare = (sq) => sq && snap.valid_pawn_moves.some(([r, c]) => r === sq[0] && c === sq[1]);

svg.addEventListener("pointerdown", (evt) => {
  if (animating || snap.winner !== null || isBot(snap.current_player)) return;
  if (evt.button === 2) { cancelDrags(); return; }
  if (evt.button !== 0) return;
  const pt = toSvg(evt);
  ui.mouse = pt;
  const me = snap.current_player;
  const [px, py] = squareCenter(...snap.pawns[me]);
  svg.setPointerCapture(evt.pointerId);

  if (Math.hypot(pt[0] - px, pt[1] - py) <= CELL * 0.45) {
    ui.press = pt;
    return;
  }
  if (inReserve(me, pt)) {
    if (snap.walls_left[me] <= 0) { flash("You have no walls left."); return; }
    ui.selected = false;
    ui.wallDrag = true;
    updatePreview();
    render();
    return;
  }
  const sq = squareAt(pt);
  if (isValidSquare(sq)) { play({ type: "pawn", row: sq[0], col: sq[1] }); return; }
  if (sq && ui.selected && !(sq[0] === snap.pawns[me][0] && sq[1] === snap.pawns[me][1])) {
    flash("Your pawn can't move there.");
  }
  ui.selected = false;
  render();
});

svg.addEventListener("pointermove", (evt) => {
  if (animating) return;
  ui.mouse = toSvg(evt);
  if (ui.press && !ui.pawnDrag && Math.hypot(ui.mouse[0] - ui.press[0], ui.mouse[1] - ui.press[1]) > 5) {
    ui.pawnDrag = true;
    ui.selected = true;
    render();
    return;
  }
  if (ui.pawnDrag) {
    pawnEls[snap.current_player].setAttribute("transform", `translate(${ui.mouse[0]} ${ui.mouse[1]}) scale(1.1)`);
    renderHover();
  } else if (ui.wallDrag) {
    updatePreview();
    renderPreview();
  } else {
    renderHover();
  }
});

svg.addEventListener("pointerup", (evt) => {
  if (animating) return;
  const pt = toSvg(evt);
  ui.mouse = pt;
  const me = snap.current_player;
  if (ui.pawnDrag) {
    ui.press = null;
    const sq = squareAt(pt, true);
    if (isValidSquare(sq)) { play({ type: "pawn", row: sq[0], col: sq[1] }, pt); return; }
    if (sq && !(sq[0] === snap.pawns[me][0] && sq[1] === snap.pawns[me][1])) flash("Your pawn can't move there.");
    returnPawn(pt);
    return;
  }
  if (ui.press) {
    ui.press = null;
    ui.selected = !ui.selected;
    render();
    return;
  }
  if (ui.wallDrag) {
    const rect = floatingRect();
    if (ui.preview && ui.previewOk) {
      const [row, col, orientation] = ui.preview;
      play({ type: "wall", row, col, orientation }, rect);
      return;
    }
    if (ui.preview) flash(E.wallError(game, me, ui.preview) || "Illegal wall.");
    returnWall(rect);
  }
});

svg.addEventListener("pointercancel", () => cancelDrags());
svg.addEventListener("pointerleave", () => { if (!ui.wallDrag && !ui.pawnDrag) renderHover(); });
svg.addEventListener("contextmenu", (evt) => evt.preventDefault());

function cancelDrags() {
  if (ui.wallDrag) returnWall(floatingRect());
  else if (ui.pawnDrag) { ui.press = null; returnPawn(ui.mouse); }
  else { ui.press = null; ui.selected = false; render(); }
}

function undo() {
  if (animating) { setTimeout(undo, 60); return; }
  if (!history.length) return;
  cancelBot();
  let g = history.pop();
  // Against the bot, go back to the human's previous turn.
  while (isBot(g.current) && history.length) g = history.pop();
  setGame(g);
  resetUi();
  render();
  maybeBotTurn();
}

function newGame() {
  if (animating) { setTimeout(newGame, 60); return; }  // let a running animation finish
  cancelBot();
  history = [];
  lastBotInfo = null;
  setGame(E.newGame());
  resetUi();
  render();
  maybeBotTurn();
}

document.addEventListener("keydown", (evt) => {
  if (evt.target.tagName === "INPUT") return;
  if (evt.key === "Escape") cancelDrags();
  if ((evt.ctrlKey || evt.metaKey) && evt.key.toLowerCase() === "z") { evt.preventDefault(); undo(); }
});
$("undo").addEventListener("click", undo);
$("new").addEventListener("click", newGame);
$("again").addEventListener("click", newGame);

// Names: from ?p1=&p2= in the URL, else remembered locally.
const params = new URLSearchParams(location.search);
const humanNames = [0, 1].map((p) => {
  let saved = null;
  try { saved = localStorage.getItem(`quoridor.name${p}`); } catch { /* storage unavailable */ }
  return params.get(`p${p + 1}`) || saved || `Player ${p + 1}`;
});
function applyNameInputs() {
  for (const p of [0, 1]) {
    const input = $(`name${p}`);
    input.disabled = isBot(p);
    input.value = isBot(p) ? "Bot" : humanNames[p];
  }
}
for (const p of [0, 1]) {
  const input = $(`name${p}`);
  input.addEventListener("input", () => {
    humanNames[p] = input.value;
    try { localStorage.setItem(`quoridor.name${p}`, input.value); } catch { /* ignore */ }
    renderPanel();
  });
}

// ---------------------------------------------------------------- settings
function loadSettings() {
  try { Object.assign(settings, JSON.parse(localStorage.getItem("quoridor.settings") || "{}")); } catch { /* ignore */ }
  if (params.get("mode") === "bot") settings.mode = "hvb";
  if (params.get("mode") === "human") settings.mode = "hvh";
  if (params.get("depth")) settings.depth = Number(params.get("depth"));
  settings.depth = Math.max(1, Math.min(30, Math.round(settings.depth) || 4));
}
function saveSettings() {
  try { localStorage.setItem("quoridor.settings", JSON.stringify(settings)); } catch { /* ignore */ }
}
function renderSettings() {
  for (const b of document.querySelectorAll(".segmented [data-mode]")) {
    b.setAttribute("aria-checked", String(b.dataset.mode === settings.mode));
  }
  $("bot-options").hidden = settings.mode !== "hvb";
  $("depth").value = settings.depth;
  $("depth-value").textContent = settings.depth;
  $("time-limit").value = String(settings.timeMs);
  $("human-side").value = String(settings.humanSide);
  applyNameInputs();
}
for (const b of document.querySelectorAll(".segmented [data-mode]")) {
  b.addEventListener("click", () => {
    if (settings.mode === b.dataset.mode) return;
    settings.mode = b.dataset.mode;
    saveSettings();
    renderSettings();
    newGame();
  });
}
$("depth").addEventListener("input", () => {
  settings.depth = Number($("depth").value);
  $("depth-value").textContent = settings.depth;
  saveSettings();
});
$("time-limit").addEventListener("change", () => {
  settings.timeMs = Number($("time-limit").value);
  saveSettings();
});
$("human-side").addEventListener("change", () => {
  settings.humanSide = Number($("human-side").value);
  saveSettings();
  renderSettings();
  newGame();
});
loadSettings();
renderSettings();

// Small API for scripts/bots in the console: Quoridor.state() returns the
// same dict shape as Python's BoardState.to_dict(); Quoridor.play(move) plays.
window.Quoridor = {
  state: () => E.snapshot(game),
  play: (move) => play(move),
  botMove: (depth = settings.depth, timeLimitMs = settings.timeMs || null) => botSearch(E.snapshot(game), depth, timeLimitMs),
  settings,
};

render();
maybeBotTurn();
