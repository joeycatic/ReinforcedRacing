import "../style.css";
import "./watch.css";
import { AssetLoader, type Heatmap } from "../assets/load";
import { RaceScene, minimapPath } from "../race/scene";
import { heatmapScale } from "../viewer/heatmap";
import {
  WatchTimeline,
  caption,
  snapshotLabel,
  type AttemptRun,
} from "./timeline";
const node = <T extends HTMLElement = HTMLElement>(id: string): T =>
  document.getElementById(id) as T;
const scene = new RaceScene(node("watch-canvas"));
const play = node<HTMLButtonElement>("watch-play");
const replay = node<HTMLButtonElement>("watch-replay");
const skip = node<HTMLButtonElement>("watch-skip");
const crash = node<HTMLInputElement>("crash-toggle");
const motion = matchMedia("(prefers-reduced-motion: reduce)");
let loader: AssetLoader;
let timeline: WatchTimeline | null = null;
let trackId = "sweep",
  revision = 0,
  heatRevision = 0,
  lastIndex = -1;
const maps = new Map<string, Promise<Heatmap[]>>();
function notice(message: string): void {
  node("notice").textContent = message;
  node("notice").hidden = false;
}
function controls(enabled: boolean): void {
  for (const button of [play, replay, skip]) button.disabled = !enabled;
}
function clearHeatmap(): void {
  scene.setHeatmap(null, false, false, { visits: 0, crashes: 0 });
}
async function heatmap(): Promise<void> {
  const serial = ++heatRevision,
    pageRevision = revision,
    id = trackId,
    current = timeline;
  clearHeatmap();
  if (!current || !crash.checked) {
    node("crash-stage").textContent = "Stochastic training-stage heatmap";
    return;
  }
  const step = Number(
    current.runs[current.current().index].data.metadata.snapshot_step,
  );
  const stages = loader.manifest.snapshots;
  const selected = stages.reduce(
    (best, stage, i) =>
      stage.step <= step && stage.step >= stages[best].step ? i : best,
    0,
  );
  node("crash-stage").textContent = "Loading training-stage crashes…";
  try {
    let pending = maps.get(id);
    if (!pending) {
      pending = Promise.all(stages.map((stage) => loader.heatmap(stage, id)));
      maps.set(id, pending);
    }
    const all = await pending;
    if (serial !== heatRevision || pageRevision !== revision || !crash.checked)
      return;
    const map = all[selected];
    scene.setHeatmap(map, false, true, heatmapScale(all));
    node("crash-stage").textContent =
      `CRASHES · ${snapshotLabel(stages[selected]).replace(" FINAL", "")}-STEP STAGE · ${map.metadata.runs} RUNS`;
  } catch {
    maps.delete(id);
    if (serial !== heatRevision || pageRevision !== revision) return;
    node("crash-stage").textContent = "Heatmap unavailable";
    notice(
      "Could not load training-stage crashes. Toggle crash locations to retry.",
    );
  }
}
function setText(id: string, value: string): void {
  const target = node(id);
  if (target.textContent !== value) target.textContent = value;
}
function update(): void {
  if (!timeline) return;
  const current = timeline.current(),
    run = timeline.runs[current.index];
  const text = caption(run, current.index, timeline.runs.length);
  setText("attempt-kicker", text.kicker);
  node("attempt-headline").hidden = text.headline === null;
  setText("attempt-headline", text.headline ?? "");
  setText("attempt-outcome", text.outcome);
  setText(
    "watch-state",
    current.holding ? "HOLDING" : timeline.playing ? "PLAYING" : "PAUSED",
  );
  setText("watch-play", timeline.playing ? "Pause" : "Play");
  play.setAttribute("aria-pressed", String(timeline.playing));
  const timer = node("lap-timer");
  timer.hidden = run.attempt.outcome !== "finished";
  setText("lap-timer", `${current.replaySeconds.toFixed(2)} s`);
  setText(
    "watch-rate",
    `${Number(timeline.slots[current.index].rate.toFixed(2))}× replay speed`,
  );
  for (const button of node("progress-bar").querySelectorAll<HTMLButtonElement>(
    "button",
  )) {
    const i = Number(button.dataset.attempt);
    button.classList.toggle("done", i < current.index);
    button.classList.toggle("current", i === current.index);
    if (i === current.index) button.setAttribute("aria-current", "step");
    else button.removeAttribute("aria-current");
  }
  if (lastIndex !== current.index) {
    lastIndex = current.index;
    void heatmap();
  }
}
async function selectTrack(id: string): Promise<void> {
  const serial = ++revision;
  ++heatRevision;
  timeline?.pause();
  timeline = null;
  lastIndex = -1;
  trackId = id;
  controls(false);
  clearHeatmap();
  scene.setAttempts([]);
  scene.app.render();
  node("progress-bar").replaceChildren();
  node("notice").hidden = true;
  node("attempt-kicker").textContent = "LOADING ATTEMPTS";
  node("attempt-headline").hidden = false;
  node("attempt-headline").textContent = "Preparing the next circuit.";
  node("attempt-outcome").textContent = "";
  node("lap-timer").hidden = true;
  node("watch-state").textContent = "LOADING";
  node("crash-stage").textContent = "Stochastic training-stage heatmap";
  node("watch-rate").textContent = "";
  for (const card of node("watch-tracks").querySelectorAll<HTMLButtonElement>(
    "button",
  ))
    card.setAttribute("aria-pressed", String(card.dataset.track === id));
  try {
    const track = await loader.track(id);
    if (serial !== revision) return;
    scene.setTrack(track);
    scene.app.render();
    node("watch-track-name").textContent = track.data.name.toUpperCase();
    scene.app.canvas.setAttribute(
      "aria-label",
      `Training attempts on ${track.data.name}, one car at a time.`,
    );
    const runs: AttemptRun[] = await Promise.all(
      loader.manifest.attempts[id].map(async (attempt) => ({
        attempt,
        ...(await loader.attemptReplay(attempt, id)),
      })),
    );
    if (serial !== revision) return;
    timeline = new WatchTimeline(runs);
    scene.setAttempts(runs);
    runs.forEach((run, index) => {
      const button = document.createElement("button"),
        text = caption(run, index, runs.length);
      button.className = "segment";
      button.dataset.attempt = String(index);
      button.style.flex = String(timeline!.slots[index].displaySeconds);
      button.setAttribute("aria-label", `${text.kicker} · ${text.outcome}`);
      button.title = `${text.kicker} · ${text.outcome}`;
      button.onclick = () => {
        timeline!.jump(index);
        update();
        scene.renderAttempt(timeline!);
      };
      node("progress-bar").append(button);
    });
    if (!motion.matches && !document.hidden) timeline.play();
    controls(true);
    update();
    scene.renderAttempt(timeline);
  } catch {
    if (serial !== revision) return;
    node("watch-state").textContent = "UNAVAILABLE";
    node("attempt-headline").textContent = "This replay could not load.";
    notice("Could not load training attempts. Select a circuit to retry.");
  }
}
play.onclick = () => {
  if (!timeline) return;
  if (timeline.playing) timeline.pause();
  else timeline.play();
  update();
};
replay.onclick = () => {
  timeline?.replay();
  update();
};
skip.onclick = () => {
  timeline?.skipToFinal();
  update();
};
crash.onchange = () => void heatmap();
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    timeline?.pause();
    update();
  }
});
motion.addEventListener("change", () => {
  if (motion.matches) {
    timeline?.pause();
    update();
  }
});
async function mount(): Promise<void> {
  [loader] = await Promise.all([
    AssetLoader.open(
      new URL(`${import.meta.env.BASE_URL}assets/manifest.json`, location.href),
    ),
    scene.init(),
  ]);
  for (const descriptor of loader.manifest.tracks) {
    const track = await loader.track(descriptor.id),
      map = minimapPath(track),
      button = document.createElement("button");
    button.className = "track-card";
    button.dataset.track = descriptor.id;
    button.innerHTML = `<svg viewBox="${map.viewBox}" aria-hidden="true"><path d="${map.path}"/></svg><strong></strong>`;
    button.querySelector("strong")!.textContent = track.data.name;
    button.onclick = () => void selectTrack(descriptor.id);
    node("watch-tracks").append(button);
  }
  await selectTrack("sweep");
  let previous = performance.now();
  const draw = (now: number): void => {
    if (timeline) {
      const delta = Math.max(0, Math.min(0.1, (now - previous) / 1000));
      if (motion.matches && timeline.playing) {
        const slot = timeline.slots[timeline.current().index],
          end = slot.start + slot.displaySeconds;
        // Keep the same attempt selected at its end, even after explicit Play.
        if (timeline.seconds + delta >= end) {
          timeline.seconds = end - 1e-7;
          timeline.pause();
        } else timeline.advance(delta);
      } else timeline.advance(delta);
      scene.renderAttempt(timeline);
      update();
    }
    previous = now;
    requestAnimationFrame(draw);
  };
  requestAnimationFrame(draw);
}
void mount().catch(() =>
  notice("Could not load training attempts. Reload to try again."),
);
