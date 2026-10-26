import "./intro.css";
import { AssetLoader } from "../assets/load";
import { renderLearningChart } from "./chart";
import { replayParity, renderParityDemo } from "./parity";
import { formatStats } from "./format";
const host = (id: string): HTMLElement => document.getElementById(id)!;
async function boot(): Promise<void> {
  try {
    const loader = await AssetLoader.open(
      new URL(import.meta.env.BASE_URL + "assets/manifest.json", location.href),
    );
    const site = await loader.site();
    for (const stat of formatStats(site.stats)) {
      const tile = document.createElement("div");
      tile.className = "stat";
      const value = document.createElement("strong"),
        label = document.createElement("span");
      value.textContent = stat.value;
      label.textContent = stat.label;
      tile.append(value, label);
      host("stats").append(tile);
    }
    for (const target of document.querySelectorAll<HTMLElement>("[data-stat]"))
      target.textContent = String(
        site.stats[target.dataset.stat as keyof typeof site.stats],
      );
    document.querySelector<HTMLElement>('[data-seed="first"]')!.textContent =
      String(site.seeds[0].seed);
    for (const label of [
      site.stats.parallel_cars + " parallel cars",
      "Rollout",
      "GAE",
      "Clipped PPO update",
      "Repeat",
    ]) {
      const span = document.createElement("span");
      span.textContent = label;
      host("ppo-loop").append(span);
    }
    for (const seed of site.seeds) {
      const row = document.createElement("tr");
      for (const value of [
        String(seed.seed),
        seed.showcase_pct.toFixed(1) + "%",
        seed.held_out_pct.toFixed(1) + "%",
        String(seed.deterministic_showcase_finishes),
      ]) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      }
      host("seed-rows").append(row);
    }
    renderLearningChart(host("learning-chart"), site);
    try {
      const [demo, config] = await Promise.all([
        loader.parityDemo(),
        loader.simulation(),
      ]);
      const track = await loader.track(demo.track_id);
      renderParityDemo(
        host("parity-demo"),
        demo,
        replayParity(demo, track, config),
        track,
      );
      const { AttractLoop } = await import("./attract");
      const attract = new AttractLoop(
        host("hero-canvas"),
        loader,
        config,
        track,
      );
      void attract.start();
      window.addEventListener("pagehide", () => attract.stop(), { once: true });
    } catch (cause) {
      console.warn("Interactive preview unavailable", cause);
      host("hero-status").textContent = "STATIC · live preview unavailable";
    }
  } catch (cause) {
    console.warn("Intro data unavailable", cause);
    host("stats").textContent =
      "Measured results unavailable. Reload to try again.";
  }
}
void boot();
