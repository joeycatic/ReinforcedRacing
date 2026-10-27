if (location.hash === "#watch") location.replace("./watch.html");
import "../style.css";
import { visualConfig } from "../viewer/visual-config";
import { ViewerController } from "../viewer/controller";
import {
  AssetLoader,
  type MedalName,
  type SimulationConfig,
} from "../assets/load";
import { medalsForTrack, MEDAL_COLORS, earnedMedal } from "./medals";
import { Track } from "../sim/track";
import { speed } from "../sim/physics";
import { RaceEngine, FixedClock } from "./engine";
import { DriverInput } from "./input";
import { OnnxPolicy } from "./policy";
import { policyForTrack } from "./custom-policy";
import { RaceScene, minimapPath } from "./scene";
import { decodeTrack } from "../editor/share";
import { buildEditorTrack } from "../editor/build";
import editorConfig from "../editor/editor-config.json";
import { canSubmit } from "../editor/issues";

function element<T extends HTMLElement = HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) throw new Error(`Missing interface element ${id}`);
  return found as T;
}
let customCode: string | null = null;
let damagedLink = false;
let customTrack: Track | null = null;
let selectedTrack = "",
  selectedMedal: MedalName = "gold";
const prerace = element("prerace");
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
let loader: AssetLoader, config: SimulationConfig;
let viewer: ViewerController | undefined;
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
  prerace.hidden = true;
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
    prerace.hidden = true;
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

  labels["system-status"].textContent = "LOADING POLICY";
  labels["race-state"].textContent = "LOADING";
  message("SETTING THE GRID", "Preparing your live opponent.", "NEXT RACE");
  try {
    if (stepping) await stepping;
    const snapshot = loader.manifest.medals.find(
      (m) => m.medal === selectedMedal,
    )!;
    const track = tracks.get(selectedTrack)!;
    {
      const replay = customTrack
        ? null
        : await loader.medalReplay(snapshot, track.data.id);
      if (serial !== loadSerial) return;
      if (policyId !== snapshot.medal) {
        const next = await OnnxPolicy.create(
          await loader.bytes(snapshot.model),
        );
        if (serial !== loadSerial) {
          await next.release();
          return;
        }
        const previous = policy;
        policy = next;
        policyId = snapshot.medal;
        await previous?.release();
      }
      if (serial !== loadSerial) return;
      engine = new RaceEngine(
        track,
        config,
        policyForTrack(policy!, !!customTrack, !!track.data.crossings?.length),
      );
      clock = new FixedClock(engine);
      ghostFrames = replay?.frames ?? [];
      // Warm up the graph before the countdown, keeping the first driving tick smooth.
      await policy!.predict(
        engine.observations.build(
          engine.ai.state,
          engine.ai.progress.projection,
          [0, 0],
        ),
      );
    }
    if (serial !== loadSerial) return;
    scene.setTrack(track);
    if (!customTrack) await viewer?.setTrack(track);
    if (serial !== loadSerial) return;
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
    labels["circuit-name"].textContent = customTrack
      ? "YOUR TRACK"
      : track.data.name.toUpperCase();
    labels["ai-label"].textContent = customTrack
      ? "FINAL POLICY"
      : snapshot.medal.toUpperCase() + " MEDAL";
    labels["snapshot-note"].textContent = customTrack
      ? track.data.crossings?.length
        ? "Final model · positive throttle capped at 20% on this overpass track. Sharp corners can still cause crashes."
        : "Final model · original controls on this track. Sharp corners can still cause crashes."
      : snapshot.step.toLocaleString("en-US") +
        " steps of practice. A live policy to beat.";
    lastPhase = "";
    lastCrashCount = 0;
    loading = false;

    start.disabled = false;
    labels["system-status"].textContent = "LIVE POLICY READY";
    element<HTMLButtonElement>("mode-race").disabled = false;
    if (engine?.phase === "ready") prerace.hidden = false;
    if (engine?.phase === "paused") {
      announce("Your race is paused. Resume when you are ready.");
      return;
    }
    announce(
      (damagedLink
        ? "This track link is damaged. Choose a showcase circuit. "
        : "") +
        (matchMedia("(max-width:720px)").matches
          ? "Connect a keyboard or gamepad to drive. Your circuit is ready."
          : "One lap. No shortcuts. Take the wheel when you’re ready."),
    );
  } catch (cause) {
    if (serial === loadSerial) error(cause);
  }
}
function updateHud(): void {
  if (!engine || loading) return;
  const phase = engine.phase;
  const showSensors = element<HTMLInputElement>("sensors-toggle").checked;
  element("ai-inspector").hidden = !showSensors;
  scene.showSensors = showSensors;
  if (showSensors) {
    const [steer, throttle] = engine.controls;
    element<HTMLMeterElement>("ai-steer").value = steer;
    element<HTMLMeterElement>("ai-throttle").value = throttle;
    element("ai-steer-value").textContent = steer.toFixed(2);
    element("ai-throttle-value").textContent = throttle.toFixed(2);
    element("ai-value").textContent = engine.value.toFixed(2);
    element<HTMLMeterElement>("ai-value-meter").value = Math.tanh(
      engine.value / visualConfig.criticValueScale,
    );
  }
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
  const ordinal = Math.max(
    0,
    ...[...engine.playerSplits.keys()].filter(
      (i) =>
        i % visualConfig.stadium.gateEvery === 0 && engine!.aiSplits.has(i),
    ),
  );
  labels["checkpoint-label"].textContent = ordinal
    ? `CHECKPOINT ${String(ordinal).padStart(2, "0")}`
    : "START LINE";
  const split = ordinal
    ? {
        ordinal,
        delta:
          engine.playerSplits.get(ordinal)! - engine.aiSplits.get(ordinal)!,
      }
    : null;
  element("split-time").textContent = split
    ? time(engine.playerSplits.get(ordinal)!)
    : "—";
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
      const medal = customTrack
        ? null
        : earnedMedal(
            engine.playerLapTime!,
            loader.manifest.medals,
            selectedTrack,
          );
      labels["result-title"].textContent = customTrack
        ? "LAP COMPLETE"
        : medal
          ? medal.toUpperCase() + " MEDAL"
          : "NO MEDAL";
      labels["result-description"].textContent =
        engine.aiLapTime === null
          ? "Across the line ahead of the AI."
          : `${Math.abs(engine.playerLapTime! - engine.aiLapTime).toFixed(3)} seconds ${win ? "ahead. That’s your racing line." : "back. There’s time to find on the next lap."}`;
      labels["result-player"].textContent = time(engine.playerLapTime);
      labels["result-ai"].textContent = time(
        engine.aiLapTime ??
          (customTrack
            ? null
            : loader.manifest.medals.find((m) => m.medal === selectedMedal)!
                .lap_times[selectedTrack]),
      );
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
for (const button of document.querySelectorAll(".change-race"))
  button.addEventListener("click", () => {
    engine?.pause();
    input.clear();
    clock?.clear();
    lastPhase = "";
    prerace.hidden = false;
    result.hidden = true;
  });

document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    engine?.pause();
    clock?.clear();
    input.clear();
    lastPhase = "";
  }
});
function renderCards(): void {
  for (const card of document.querySelectorAll<HTMLElement>(".track-card"))
    card.setAttribute(
      "aria-pressed",
      String(card.dataset.track === selectedTrack),
    );
  if (customTrack) {
    const card = document.createElement("button");
    card.className = "medal-card";
    card.dataset.medal = "author";
    card.style.setProperty("--medal", MEDAL_COLORS.author);
    card.setAttribute("aria-pressed", "true");
    const label = document.createElement("strong");
    label.textContent = "AI · FINAL POLICY";
    label.style.gridColumn = "1 / -1";
    card.append(label);
    element("medal-cards").replaceChildren(card);
    return;
  }
  const medals = medalsForTrack(loader.manifest.medals, selectedTrack);
  if (!medals.some((m) => m.medal === selectedMedal))
    selectedMedal = (medals.find((m) => m.medal === "gold") ?? medals.at(-1)!)
      .medal;
  element("medal-cards").replaceChildren();
  for (const medal of medals) {
    const card = document.createElement("button");
    card.className = "medal-card";
    card.dataset.medal = medal.medal;
    card.style.setProperty("--medal", MEDAL_COLORS[medal.medal]);
    card.setAttribute("aria-pressed", String(medal.medal === selectedMedal));
    card.innerHTML =
      '<span class="medal-disc" aria-hidden="true"></span><strong>' +
      medal.medal.toUpperCase() +
      "</strong><span>" +
      time(medal.lap_times[selectedTrack]) +
      "</span>";
    card.addEventListener("click", () => {
      selectedMedal = medal.medal;
      renderCards();
      void selectRace();
    });
    element("medal-cards").append(card);
  }
}
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
    const params = new URLSearchParams(location.hash.slice(1));
    if (params.has("track")) {
      const code = params.get("track")!;
      const decoded = decodeTrack(code);
      if (decoded) {
        const built = buildEditorTrack(
          decoded.control,
          editorConfig,
          decoded.over,
        );
        if (canSubmit(null, built.issues)) {
          customTrack = built.track;
          customCode = code;
        }
      }
      damagedLink = !customTrack;
    }
    if (customTrack) {
      selectedMedal = "author";
      const back = element<HTMLAnchorElement>("back-to-editor");
      back.href = "./editor.html#t=" + customCode;
      back.hidden = false;
      element("circuit-count").textContent = "YOUR CIRCUIT";
      element("setup-title").textContent = "YOUR DRAWN CIRCUIT.";
      element("track-cards").setAttribute("aria-label", "Your drawn track");
      element("medal-cards").setAttribute("aria-label", "Live AI opponent");
      for (const id of [
        "ghost-toggle",
        "visits-toggle",
        "crashes-toggle",
        "heatmap-snapshot",
      ]) {
        const control = element<HTMLInputElement>(id);
        control.disabled = true;
        control.checked = false;
        control.title = "Not available on drawn tracks";
        if (control.parentElement?.tagName === "LABEL")
          control.parentElement.title = control.title;
      }
      element("heatmap-summary").textContent = "Not available on drawn tracks";
      element("heatmap-stage").textContent = "FINAL";
    } else viewer = new ViewerController(loader, scene, announce);
    const availableTracks = customTrack
      ? [customTrack]
      : await Promise.all(
          loader.manifest.tracks.map((descriptor) =>
            loader.track(descriptor.id),
          ),
        );
    for (const track of availableTracks) {
      tracks.set(track.data.id, track);
      const card = document.createElement("button");
      card.className = "track-card";
      card.dataset.track = track.data.id;
      const map = minimapPath(track);
      card.innerHTML =
        '<svg viewBox="' +
        map.viewBox +
        '" aria-hidden="true"><path d="' +
        map.path +
        '"/></svg><strong></strong><span></span><p></p>';
      card.querySelector("strong")!.textContent = customTrack
        ? "YOUR TRACK"
        : track.data.name;
      card.querySelector("span")!.textContent = Math.round(track.length) + " m";
      card.querySelector("p")!.textContent =
        descriptions[track.data.id] ?? "Your line. A live final policy.";
      card.addEventListener("click", () => {
        selectedTrack = track.data.id;
        renderCards();
        void selectRace();
      });
      element("track-cards").append(card);
    }
    selectedTrack = availableTracks[0].data.id;
    renderCards();
    await selectRace();
    requestAnimationFrame(frame);
  } catch (cause) {
    error(cause);
  }
}
void boot();
