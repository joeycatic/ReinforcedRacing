import "./style.css";
import {
  AssetLoader,
  type Snapshot,
  type SimulationConfig,
} from "./assets/load";
import { Track } from "./sim/track";
import { speed } from "./sim/physics";
import { RaceEngine, FixedClock } from "./race/engine";
import { DriverInput } from "./race/input";
import { OnnxPolicy } from "./race/policy";
import { RaceScene, minimapPath } from "./race/scene";

function element<T extends HTMLElement = HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) throw new Error(`Missing interface element ${id}`);
  return found as T;
}
const trackSelect = element<HTMLSelectElement>("track-select"),
  snapshotSelect = element<HTMLSelectElement>("snapshot-select");
const start = element<HTMLButtonElement>("start-button"),
  pause = element<HTMLButtonElement>("pause-button");
const stage = element("circuit"),
  center = element("center-message"),
  notice = element("notice"),
  result = element("result");
const ghostToggle = element<HTMLInputElement>("ghost-toggle");
const labels = Object.fromEntries(
  [
    "system-status",
    "track-description",
    "track-length",
    "snapshot-note",
    "ai-label",
    "circuit-name",
    "race-state",
    "lap-timer",
    "checkpoint-label",
    "split-delta",
    "speed",
    "progress-percent",
    "ai-time",
    "result-title",
    "result-description",
    "result-player",
    "result-ai",
  ].map((id) => [id, element(id)]),
);
const descriptions: Record<string, string> = {
  sweep: "Wide, flowing corners. Find a rhythm.",
  ribbon: "Long arcs. Keep your speed through the bends.",
  technical: "A tighter circuit. Every steering input counts.",
};
const time = (seconds: number | null): string => {
  if (seconds === null) return "—";
  const milliseconds = Math.round(seconds * 1000);
  return `${String(Math.floor(milliseconds / 60000)).padStart(2, "0")}:${String(Math.floor(milliseconds / 1000) % 60).padStart(2, "0")}.${String(milliseconds % 1000).padStart(3, "0")}`;
};
const steps = (snapshot: Snapshot): string =>
  snapshot.id === "final"
    ? `FINAL / ${(snapshot.step / 1e6).toFixed(1).replace(/\.0$/, "")}M STEPS`
    : `${Number(snapshot.id).toLocaleString("en-US")} STEPS`;
let loader: AssetLoader, config: SimulationConfig;
const tracks = new Map<string, Track>();
let engine: RaceEngine | null = null,
  clock: FixedClock | null = null,
  policy: OnnxPolicy | null = null,
  policyId = "";
let ghostFrames: number[][] = [],
  loading = true,
  loadSerial = 0,
  lastPhase = "",
  lastCrashCount = 0;
let stepping: Promise<void> | null = null,
  lastFrame = performance.now(),
  lastHud = 0;
const scene = new RaceScene(element("canvas-host"));
const input = new DriverInput(
  begin,
  () => {
    engine?.respawn();
    clock?.clear();
  },
  togglePause,
);
function announce(message: string): void {
  notice.textContent = message;
  notice.classList.remove("error");
}
function message(
  title: string,
  subtitle: string,
  eyebrow: string,
  kind = "",
): void {
  center.hidden = false;
  center.className = "center-message " + kind;
  center.querySelector("strong")!.textContent = title;
  center.querySelector("p")!.textContent = subtitle;
  center.querySelector(".eyebrow")!.textContent = eyebrow;
}
function focusCircuit(): void {
  stage.focus({ preventScroll: true });
}
function begin(): void {
  if (loading || !engine || !clock) return;
  input.clear();
  clock.clear();
  scene.clearMarks();
  engine.start();
  lastPhase = "";
  lastCrashCount = 0;
  result.hidden = true;
  focusCircuit();
  announce(
    "Green in three. Arrows or WASD to drive. R returns you to the last checkpoint.",
  );
}
function togglePause(): void {
  if (!engine || loading) return;
  if (engine.phase === "paused") {
    engine.resume();
    focusCircuit();
    announce("Back on track.");
  } else {
    engine.pause();
    input.clear();
    announce("Race paused. Your lap clock is stopped.");
  }
  clock?.clear();
  lastPhase = "";
}
function error(error: unknown): void {
  loading = true;
  start.disabled = true;
  pause.disabled = true;
  labels["system-status"].textContent = "UNAVAILABLE";
  labels["race-state"].textContent = "OFFLINE";
  message(
    "CIRCUIT OFFLINE",
    "Reload once the race data is available.",
    "RACE DATA COULD NOT LOAD",
  );
  notice.textContent = `${error instanceof Error ? error.message : String(error)}. Check the asset export and reload.`;
  notice.classList.add("error");
}
async function selectRace(): Promise<void> {
  const serial = ++loadSerial;
  loading = true;
  engine?.pause();
  input.clear();
  result.hidden = true;
  start.disabled = true;
  pause.disabled = true;
  trackSelect.disabled = true;
  snapshotSelect.disabled = true;
  labels["system-status"].textContent = "LOADING POLICY";
  labels["race-state"].textContent = "LOADING";
  message("SETTING THE GRID", "Preparing your live opponent.", "NEXT RACE");
  try {
    if (stepping) await stepping;
    const snapshot = loader.manifest.snapshots.find(
      (s) => s.id === snapshotSelect.value,
    )!;
    const track = tracks.get(trackSelect.value)!;
    const replay = await loader.replay(snapshot, track.data.id);
    if (serial !== loadSerial) return;
    if (policyId !== snapshot.id) {
      const next = await OnnxPolicy.create(await loader.bytes(snapshot.model));
      if (serial !== loadSerial) {
        await next.release();
        return;
      }
      const previous = policy;
      policy = next;
      policyId = snapshot.id;
      await previous?.release();
    }
    engine = new RaceEngine(track, config, policy!);
    clock = new FixedClock(engine);
    ghostFrames = replay.frames;
    // Warm up the graph before the countdown, keeping the first driving tick smooth.
    await policy!.predict(
      engine.observations.build(
        engine.ai.state,
        engine.ai.progress.projection,
        [0, 0],
      ),
    );
    scene.setTrack(track);
    const map = minimapPath(track);
    for (const id of ["route-preview", "minimap"])
      document.getElementById(id)!.setAttribute("viewBox", map.viewBox);
    for (const id of ["route-path", "minimap-path"])
      document.getElementById(id)!.setAttribute("d", map.path);
    for (const id of ["minimap-player", "minimap-ai"])
      document
        .getElementById(id)!
        .setAttribute("r", String(Number(map.viewBox.split(" ")[2]) * 0.021));
    labels["track-description"].textContent =
      descriptions[track.data.id] ?? "One clean lap. Make it count.";
    labels["track-length"].textContent = Math.round(
      track.length,
    ).toLocaleString("en-US");
    labels["circuit-name"].textContent = track.data.name.toUpperCase();
    labels["ai-label"].textContent =
      snapshot.id === "final" ? "FINAL POLICY" : `${snapshot.id} STEPS`;
    labels["snapshot-note"].textContent =
      snapshot.id === "0"
        ? "No training. An AI at its very first start line."
        : snapshot.id === "final"
          ? `${snapshot.step.toLocaleString("en-US")} steps of practice. One lap to beat it.`
          : "An earlier policy. Experience makes the difference.";
    lastPhase = "";
    lastCrashCount = 0;
    loading = false;
    trackSelect.disabled = false;
    snapshotSelect.disabled = false;
    start.disabled = false;
    labels["system-status"].textContent = "LIVE POLICY READY";
    announce(
      matchMedia("(max-width:720px)").matches
        ? "Connect a keyboard or gamepad to drive. Your circuit is ready."
        : "One lap. No shortcuts. Take the wheel when you’re ready.",
    );
  } catch (cause) {
    if (serial === loadSerial) error(cause);
  }
}
function updateHud(): void {
  if (!engine || loading) return;
  const phase = engine.phase;
  labels["race-state"].textContent = {
    ready: "GRID READY",
    countdown: "START SEQUENCE",
    racing: "RACE IN PROGRESS",
    paused: "PAUSED",
    finished: "FINISHED",
  }[phase];
  labels["lap-timer"].textContent = time(
    engine.playerLapTime ?? engine.player.time,
  );
  labels["speed"].textContent = String(
    Math.round(speed(engine.player.state) * 3.6),
  ).padStart(3, "0");
  const progress = Math.max(
    0,
    Math.min(
      100,
      (engine.player.progress.unwrapped / engine.track.length) * 100,
    ),
  );
  labels["progress-percent"].textContent = `${Math.floor(progress)}%`;
  element("progress-fill").style.width = `${progress}%`;
  labels["ai-time"].textContent =
    engine.aiLapTime !== null
      ? time(engine.aiLapTime)
      : phase === "ready"
        ? "READY"
        : time(engine.ai.time);
  const checkpoint = engine.player.progress.checkpoint_index;
  labels["checkpoint-label"].textContent = checkpoint
    ? `CHECKPOINT ${String(checkpoint).padStart(2, "0")}`
    : "START LINE";
  const split = engine.split;
  labels["split-delta"].textContent = split
    ? `${split.delta <= 0 ? "−" : "+"}${Math.abs(split.delta).toFixed(3)}`
    : "—";
  labels["split-delta"].className = split
    ? split.delta <= 0
      ? "ahead"
      : "behind"
    : "";
  for (const [id, state] of [
    ["minimap-player", engine.player.state],
    ["minimap-ai", engine.ai.state],
  ] as const) {
    document.getElementById(id)!.setAttribute("cx", String(state.x));
    document.getElementById(id)!.setAttribute("cy", String(-state.y));
  }
  if (phase !== lastPhase) {
    lastPhase = phase;
    start.querySelector("span")!.textContent =
      phase === "ready" ? "TAKE THE WHEEL" : "RESTART RACE";
    pause.disabled = !["racing", "countdown", "paused"].includes(phase);
    pause.firstChild!.textContent =
      phase === "paused" ? "Resume race " : "Pause race ";
    if (phase === "ready")
      message(
        "THE GRID IS YOURS",
        "One lap. You, the circuit, and a live AI.",
        "READY WHEN YOU ARE",
        "ready",
      );
    if (phase === "paused")
      message(
        "TAKE A BREATHER",
        "Press Escape or Resume race to return.",
        "RACE PAUSED",
      );
    if (phase === "finished") {
      center.hidden = true;
      result.hidden = false;
      const win =
        engine.aiLapTime === null || engine.playerLapTime! <= engine.aiLapTime;
      labels["result-title"].textContent = win
        ? "YOU TOOK THE LAP"
        : "THE MACHINE TAKES IT";
      labels["result-description"].textContent =
        engine.aiLapTime === null
          ? "Across the line ahead of the AI."
          : `${Math.abs(engine.playerLapTime! - engine.aiLapTime).toFixed(3)} seconds ${win ? "ahead. That’s your racing line." : "back. There’s time to find on the next lap."}`;
      labels["result-player"].textContent = time(engine.playerLapTime);
      labels["result-ai"].textContent =
        engine.aiLapTime === null ? "NOT FINISHED" : time(engine.aiLapTime);
      announce(
        win
          ? "Lap complete. You beat the live AI."
          : "Lap complete. Race again to find a faster line.",
      );
      element("race-again").focus({ preventScroll: true });
    }
  }
  if (phase === "countdown")
    message(
      String(engine.countdown),
      "Get ready. Hold your throttle.",
      "LIGHTS OUT IN",
      "countdown",
    );
  else if (phase === "racing") {
    if (engine.player.respawn_ticks > 0)
      message(
        "BACK TO THE CHECKPOINT",
        "Your lap clock keeps running.",
        "WALL CONTACT",
      );
    else if (engine.raceTicks < 30)
      message("GO", "Find your racing line.", "LIGHTS OUT", "countdown");
    else center.hidden = true;
    if (engine.player.crashes !== lastCrashCount) {
      lastCrashCount = engine.player.crashes;
      announce(
        "Wall contact. Returning to your last checkpoint; the clock keeps running.",
      );
    }
  }
}
async function frame(now: number): Promise<void> {
  const elapsed = Math.max(0, (now - lastFrame) / 1000);
  lastFrame = now;
  try {
    if (engine && clock && !loading) {
      stepping = clock.advance(elapsed, input.action());
      await stepping;
      stepping = null;
      let ghost: number[] | null = null;
      if (ghostToggle.checked && ghostFrames.length) {
        const tick = Math.max(
          0,
          Math.min(
            ghostFrames.length - 1,
            engine.player.ticks - 1 + clock.alpha,
          ),
        );
        const index = Math.floor(tick),
          next = Math.min(index + 1, ghostFrames.length - 1),
          fraction = tick - index;
        ghost = ghostFrames[index].map((v, i) =>
          i === 2
            ? v +
              Math.atan2(
                Math.sin(ghostFrames[next][i] - v),
                Math.cos(ghostFrames[next][i] - v),
              ) *
                fraction
            : v + (ghostFrames[next][i] - v) * fraction,
        );
      }
      scene.render(engine, clock.alpha, elapsed, ghost);
      if (now - lastHud > 30) {
        updateHud();
        lastHud = now;
      }
    }
  } catch (cause) {
    error(cause);
  }
  requestAnimationFrame(frame);
}
start.addEventListener("click", begin);
element("race-again").addEventListener("click", begin);
pause.addEventListener("click", togglePause);
trackSelect.addEventListener("change", () => void selectRace());
snapshotSelect.addEventListener("change", () => void selectRace());
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    engine?.pause();
    clock?.clear();
    input.clear();
    lastPhase = "";
  }
});
async function boot(): Promise<void> {
  try {
    [loader] = await Promise.all([
      AssetLoader.open(
        new URL(
          `${import.meta.env.BASE_URL}assets/manifest.json`,
          location.href,
        ),
      ),
      scene.init(),
    ]);
    config = await loader.simulation();
    for (const descriptor of loader.manifest.tracks) {
      const track = await loader.track(descriptor.id);
      tracks.set(descriptor.id, track);
      const option = new Option(track.data.name, track.data.id);
      trackSelect.add(option);
    }
    trackSelect.remove(0);
    snapshotSelect.remove(0);
    for (const snapshot of loader.manifest.snapshots)
      snapshotSelect.add(new Option(steps(snapshot), snapshot.id));
    snapshotSelect.value = loader.manifest.snapshots.some(
      (s) => s.id === "final",
    )
      ? "final"
      : loader.manifest.snapshots.at(-1)!.id;
    await selectRace();
    requestAnimationFrame(frame);
  } catch (cause) {
    error(cause);
  }
}
void boot();
