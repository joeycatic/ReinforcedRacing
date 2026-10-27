import "./editor.css";
import { buildEditorTrack } from "./build";
import config from "./editor-config.json";
import { strokeToControls } from "./stroke";
import { canSubmit, MESSAGES } from "./issues";
import { decodeTrack, encodeTrack } from "./share";
import { crossingAt, drawPreview, type PreviewView } from "./preview";
import { TrialLap, describe } from "./trial";
import { AssetLoader } from "../assets/load";
import { RaceEngine } from "../race/engine";
import { RaceScene } from "../race/scene";
import { OnnxPolicy } from "../race/policy";
import { policyForTrack } from "../race/custom-policy";
import type { Vec2 } from "../sim/types";
const el = <T extends HTMLElement = HTMLElement>(id: string) =>
  document.getElementById(id)! as T;
const canvas = el<HTMLCanvasElement>("draw-canvas"),
  ctx = canvas.getContext("2d")!;
const submit = el<HTMLButtonElement>("submit"),
  share = el<HTMLButtonElement>("share"),
  race = el<HTMLAnchorElement>("race-it"),
  host = el("trial"),
  result = el("trial-result");
type Built = ReturnType<typeof buildEditorTrack>;
let built: Built | null = null,
  controls: Vec2[] | null = null,
  problem: string | null = null,
  damaged = false,
  running = false;
let view: PreviewView = { scale: 1, offset: [0, 0] },
  width = 0,
  height = 0;
let points: Vec2[] = [],
  pointer: number | null = null,
  down: Vec2 = [0, 0],
  crossing: number | null = null,
  moved = false,
  lastRebuild = 0;
let serial = 0;
type Active = {
  stopped: boolean;
  frame: number;
  scene: RaceScene | null;
  engine: RaceEngine | null;
  policy: OnnxPolicy | null;
  pending: Promise<void> | null;
};
let active: Active | null = null;
function stopTrial() {
  serial++;
  running = false;
  const old = active;
  active = null;
  if (old) {
    old.stopped = true;
    cancelAnimationFrame(old.frame);
    old.engine?.pause();
    void (async () => {
      await old.pending?.catch(() => {});
      old.scene?.destroy();
      await old.policy?.release();
    })();
  }
  host.hidden = true;
  result.textContent = "";
  result.className = "";
}
function code() {
  return controls && built
    ? encodeTrack(
        controls,
        (built.track.data.crossings ?? []).map((c) => c.over),
      )
    : null;
}
const trialNote = el("trial-policy-note"),
  defaultTrialNote = trialNote.textContent;
function render() {
  trialNote.textContent = built
    ? built.track.data.crossings?.length
      ? "The final model drives this overpass track with positive throttle capped at 20%. Sharp corners can still cause crashes."
      : "The final model drives this track with its original controls. Sharp corners can still cause crashes."
    : defaultTrialNote;
  const valid = !!built && canSubmit(problem, built.issues) && !damaged;
  submit.disabled = !valid || running;
  share.disabled = !valid;
  race.setAttribute("aria-disabled", String(!valid));
  const encoded = valid ? code() : null;
  if (encoded) race.href = "./race.html#track=" + encoded;
  else race.removeAttribute("href");
  el("track-stats").textContent = built
    ? `${Math.round(built.track.length).toLocaleString("en-US")} M · ${built.track.data.crossings?.length ?? 0} OVERPASS${built.track.data.crossings?.length === 1 ? "" : "ES"}`
    : "YOUR NEXT CIRCUIT";
  const issues = el("issues");
  issues.replaceChildren();
  for (const message of damaged
    ? ["This track link is damaged"]
    : problem
      ? [MESSAGES[problem]]
      : built
        ? [
            ...built.issues.map((i) => MESSAGES[i]),
            ...(built.tight.length ? [MESSAGES.tight] : []),
          ]
        : []) {
    const li = document.createElement("li");
    li.textContent = message;
    li.classList.toggle("warning", message === MESSAGES.tight);
    issues.append(li);
  }
  const buttons = el("crossing-controls");
  buttons.replaceChildren();
  built?.track.data.crossings?.forEach((c, i) => {
    const button = document.createElement("button");
    button.textContent = `Flip overpass ${i + 1}`;
    button.setAttribute(
      "aria-label",
      `Flip overpass ${i + 1}, ${c.over === 1 ? "later" : "earlier"} pass on top`,
    );
    button.addEventListener("click", () => flip(i));
    buttons.append(button);
  });
  el("drawing-prompt").hidden = !!built || points.length > 0;
  paint();
}
function paint() {
  if (width <= 0 || height <= 0) return;
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  if (built) {
    const d = built.track.data,
      pad = d.width / 2 + 12;
    const xs = d.centerline.map((p) => p[0]),
      ys = d.centerline.map((p) => p[1]);
    const minX = Math.min(...xs) - pad,
      maxX = Math.max(...xs) + pad,
      minY = Math.min(...ys) - pad,
      maxY = Math.max(...ys) + pad;
    view = {
      scale: Math.min(width / (maxX - minX), height / (maxY - minY)),
      offset: [0, 0],
    };
    view.offset = [
      width / 2 - ((minX + maxX) / 2) * view.scale,
      height / 2 + ((minY + maxY) / 2) * view.scale,
    ];
    drawPreview(ctx, built, view);
  } else {
    ctx.fillStyle = "#68a842";
    ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = "#5f9e3a";
    for (let y = 0; y < height; y += 56) ctx.fillRect(0, y, width, 28);
    if (points.length) {
      ctx.beginPath();
      points.forEach((p, i) => (i ? ctx.lineTo(...p) : ctx.moveTo(...p)));
      ctx.strokeStyle = "#f7f6ef";
      ctx.lineWidth = 5;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.stroke();
    }
  }
}
new ResizeObserver(() => {
  const rect = canvas.getBoundingClientRect();
  width = rect.width;
  height = rect.height;
  canvas.width = Math.round(width * devicePixelRatio);
  canvas.height = Math.round(height * devicePixelRatio);
  paint();
}).observe(canvas);
function invalidateShare() {
  el("share-status").textContent = "";
  el("share-fallback").hidden = true;
  if (location.hash)
    history.replaceState(null, "", location.pathname + location.search);
}
function rebuild(final: boolean) {
  const stroke = strokeToControls(
    final ? points : [...points, points[0]],
    config,
  );
  problem = final ? stroke.problem : null;
  controls = stroke.control;
  built = null;
  if (controls) {
    try {
      built = buildEditorTrack(controls, config);
    } catch {
      problem = "too-short";
      controls = null;
    }
  }
  render();
}
function flip(index: number) {
  if (!built || !controls) return;
  stopTrial();
  const over = (built.track.data.crossings ?? []).map((c) => c.over);
  over[index] = 1 - over[index];
  built = buildEditorTrack(controls, config, over);
  invalidateShare();
  render();
}
const position = (event: PointerEvent): Vec2 => {
  const r = canvas.getBoundingClientRect();
  return [event.clientX - r.left, event.clientY - r.top];
};
canvas.addEventListener("pointerdown", (event) => {
  if (pointer !== null || event.button !== 0) return;
  event.preventDefault();
  pointer = event.pointerId;
  canvas.setPointerCapture(pointer);
  down = position(event);
  crossing = built ? crossingAt(built, view, down) : null;
  moved = false;
  if (crossing === null) {
    stopTrial();
    invalidateShare();
    points = [down];
    built = null;
    controls = null;
    problem = null;
    damaged = false;
    render();
  }
  lastRebuild = performance.now();
});
canvas.addEventListener("pointermove", (event) => {
  if (event.pointerId !== pointer) return;
  event.preventDefault();
  const p = position(event);
  if (Math.hypot(p[0] - down[0], p[1] - down[1]) > 5 && !moved) {
    moved = true;
    if (crossing !== null) {
      stopTrial();
      invalidateShare();
      points = [down];
      built = null;
      controls = null;
      problem = null;
      damaged = false;
    }
  }
  if (crossing !== null && !moved) return;
  points.push(p);
  if (performance.now() - lastRebuild >= config.ui.rebuildMs) {
    lastRebuild = performance.now();
    rebuild(false);
  } else if (!built) paint();
});
function finish(event: PointerEvent) {
  if (event.pointerId !== pointer) return;
  pointer = null;
  if (canvas.hasPointerCapture(event.pointerId))
    canvas.releasePointerCapture(event.pointerId);
  if (crossing !== null && !moved) {
    flip(crossing);
    return;
  }
  points.push(position(event));
  rebuild(true);
}
canvas.addEventListener("pointerup", finish);
canvas.addEventListener("pointercancel", (event) => {
  if (event.pointerId !== pointer) return;
  pointer = null;
  problem = "not-closed";
  render();
});
el("clear").addEventListener("click", () => {
  stopTrial();
  invalidateShare();
  points = [];
  built = null;
  controls = null;
  problem = null;
  damaged = false;
  render();
});
function readHash() {
  stopTrial();
  points = [];
  built = null;
  controls = null;
  problem = null;
  damaged = false;
  el("share-fallback").hidden = true;
  el("share-status").textContent = "";
  if (location.hash.startsWith("#t=")) {
    const decoded = decodeTrack(location.hash.slice(3));
    if (decoded) {
      controls = decoded.control;
      built = buildEditorTrack(controls, config, decoded.over);
    } else damaged = true;
  }
  render();
}
window.addEventListener("hashchange", readHash);
share.addEventListener("click", async () => {
  const encoded = code();
  if (!encoded || share.disabled) return;
  const url = new URL(location.href);
  url.hash = "t=" + encoded;
  history.replaceState(null, "", url);
  try {
    await navigator.clipboard.writeText(url.href);
    el("share-status").textContent = "Track link copied.";
  } catch {
    el("share-status").textContent = "Select and copy your track link.";
    el("share-fallback").hidden = false;
    const field = el<HTMLInputElement>("share-url");
    field.value = url.href;
    field.focus();
    field.select();
  }
});
submit.addEventListener("click", () => {
  void startTrial();
});
async function startTrial() {
  if (!built || submit.disabled) return;
  stopTrial();
  const current = ++serial,
    selected = built;
  running = true;
  render();
  result.textContent = "LOADING FINAL POLICY…";
  const run: Active = {
    stopped: false,
    frame: 0,
    scene: null,
    engine: null,
    policy: null,
    pending: null,
  };
  active = run;
  const stale = () => run.stopped || serial !== current;
  try {
    const loader = await AssetLoader.open(
      new URL(`${import.meta.env.BASE_URL}assets/manifest.json`, location.href),
    );
    if (stale()) return;
    const [simulation, bytes] = await Promise.all([
      loader.simulation(),
      loader.bytes(
        loader.manifest.snapshots.find((s) => s.id === "final")!.model,
      ),
    ]);
    if (stale()) return;
    const policy = await OnnxPolicy.create(bytes);
    if (stale()) {
      await policy.release();
      return;
    }
    run.policy = policy;
    host.hidden = false;
    const scene = new RaceScene(host);
    scene.follow = "ai";
    scene.playerVisible = false;
    await scene.init();
    if (stale()) {
      scene.destroy();
      return;
    }
    run.scene = scene;
    scene.app.canvas.setAttribute(
      "aria-label",
      "Final AI driving your circuit",
    );
    scene.setTrack(selected.track);
    const engine = new RaceEngine(
        selected.track,
        simulation,
        policyForTrack(policy, true, !!selected.track.data.crossings?.length),
      ),
      trial = new TrialLap(engine, selected.tight);
    run.engine = engine;
    let previous = performance.now(),
      accumulator = 0;
    const frame = (now: number) => {
      if (stale()) return;
      run.pending = (async () => {
        const elapsed = document.hidden
          ? 0
          : Math.min(0.25, (now - previous) / 1000);
        previous = now;
        accumulator += elapsed;
        while (
          accumulator >= simulation.physics.dt &&
          !trial.result &&
          !stale()
        ) {
          await trial.tick();
          accumulator -= simulation.physics.dt;
        }
        if (stale()) return;
        scene.render(
          engine,
          accumulator / simulation.physics.dt,
          elapsed,
          null,
        );
        if (trial.result) {
          result.textContent = describe(trial.result);
          result.className = trial.result.kind;
          running = false;
          render();
        } else {
          result.textContent = `DRIVING · ${Math.floor(engine.ai.time / 60)}:${(engine.ai.time % 60).toFixed(3).padStart(6, "0")}`;
          run.frame = requestAnimationFrame(frame);
        }
      })().catch((cause) => {
        if (!stale()) {
          result.textContent = `Trial unavailable: ${cause instanceof Error ? cause.message : String(cause)}. Submit to try again.`;
          result.className = "error";
          running = false;
          render();
        }
      });
    };
    run.frame = requestAnimationFrame(frame);
  } catch (cause) {
    if (!stale()) {
      result.textContent = `Trial unavailable: ${cause instanceof Error ? cause.message : String(cause)}. Submit to try again.`;
      result.className = "error";
      running = false;
      render();
    }
  }
}
readHash();
