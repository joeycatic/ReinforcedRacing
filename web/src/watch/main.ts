import "../style.css";
import { AssetLoader } from "../assets/load";
import { RaceScene } from "../race/scene";
import { WatchTimeline } from "./timeline";

const host = document.querySelector<HTMLElement>("#circuit")!;
const state = document.querySelector<HTMLElement>("#watch-state")!;
const play = document.querySelector<HTMLButtonElement>("#watch-play")!;
const segments = document.querySelector<HTMLElement>("#watch-segments")!;

async function mount(): Promise<void> {
  const loader = await AssetLoader.open("./assets/manifest.json");
  const [track, runs] = await Promise.all([
    loader.track("sweep"),
    Promise.all(
      loader.manifest.attempts.sweep.map(async (attempt) => ({
        attempt,
        ...(await loader.attemptReplay(attempt, "sweep")),
      })),
    ),
  ]);
  const scene = new RaceScene(host);
  await scene.init();
  scene.app.canvas.setAttribute(
    "aria-label",
    "Training attempts on Sweeping Circuit, one car at a time.",
  );
  scene.setTrack(track);
  scene.setAttempts(runs);
  const timeline = new WatchTimeline(runs);
  runs.forEach((_, index) => {
    const button = document.createElement("button");
    button.textContent = String(index + 1);
    button.setAttribute("aria-label", `Attempt ${index + 1}`);
    button.onclick = () => timeline.jump(index);
    segments.append(button);
  });
  play.disabled = false;
  play.onclick = () => (timeline.playing ? timeline.pause() : timeline.play());
  let previous = performance.now();
  const draw = (now: number): void => {
    timeline.advance(Math.max(0, Math.min(0.1, (now - previous) / 1000)));
    previous = now;
    scene.renderAttempt(timeline);
    state.textContent = timeline.current().holding
      ? "HOLDING"
      : timeline.playing
        ? "PLAYING"
        : "PAUSED";
    play.textContent = timeline.playing ? "Pause" : "Play";
    requestAnimationFrame(draw);
  };
  draw(previous);
}
void mount().catch(() => {
  state.textContent = "Could not load training attempts. Reload to try again.";
});
