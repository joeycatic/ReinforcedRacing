import { AssetLoader, type Heatmap } from "../assets/load";
import { Track } from "../sim/track";
import { RaceScene } from "../race/scene";
import { GhostPlayback, snapshotColor, snapshotLabel } from "./playback";
import { heatmapScale } from "./heatmap";

const node = <T extends HTMLElement = HTMLElement>(id: string): T =>
  document.getElementById(id) as T;
export class ViewerController {
  mode: "race" | "ghosts" = "race";
  playback: GhostPlayback | null = null;
  private track: Track | null = null;
  private revision = 0;
  private heatRevision = 0;
  private maps = new Map<string, Promise<Heatmap[]>>();
  constructor(
    private readonly loader: AssetLoader,
    private readonly scene: RaceScene,
    private readonly notify: (message: string) => void,
  ) {
    const slider = node<HTMLInputElement>("heatmap-snapshot");
    slider.max = String(loader.manifest.snapshots.length - 1);
    slider.value = slider.max;
    slider.addEventListener("input", () => void this.updateHeatmap());
    for (const id of ["visits-toggle", "crashes-toggle"])
      node<HTMLInputElement>(id).addEventListener(
        "change",
        () => void this.updateHeatmap(),
      );
    node("playback-play").addEventListener("click", () => {
      if (this.playback?.playing) this.playback.pause();
      else this.playback?.play();
      this.update();
    });
    node("playback-restart").addEventListener("click", () => {
      this.playback?.restart();
      this.update();
    });
    node<HTMLSelectElement>("playback-speed").addEventListener(
      "change",
      (event) =>
        this.playback?.setRate(
          Number((event.target as HTMLSelectElement).value),
        ),
    );
    node<HTMLInputElement>("playback-seek").addEventListener(
      "input",
      (event) => {
        this.playback?.seek(Number((event.target as HTMLInputElement).value));
        this.update();
      },
    );
    this.updateSnapshotLabel();
  }
  async setTrack(track: Track): Promise<void> {
    this.track = track;
    this.revision++;
    this.heatRevision++;
    this.scene.setHeatmap(null, false, false, { visits: 0, crashes: 0 });
    this.playback?.pause();
    this.playback = null;
    if (this.mode === "ghosts") {
      const revision = this.revision;
      const runs = await Promise.all(
        this.loader.manifest.snapshots.map(async (snapshot) => ({
          snapshot,
          ...(await this.loader.replay(snapshot, track.data.id)),
        })),
      );
      if (this.revision !== revision) return;
      this.playback = new GhostPlayback(runs);
      this.playback.setRate(
        Number(node<HTMLSelectElement>("playback-speed").value),
      );
      this.scene.setGhosts(runs);
      node<HTMLInputElement>("playback-seek").max = String(
        this.playback.duration,
      );
      const legend = node("ghost-lineup");
      legend.replaceChildren();
      runs.forEach((run, index) => {
        const row = document.createElement("div"),
          name = document.createElement("strong"),
          status = document.createElement("span");
        row.style.setProperty(
          "--ghost-color",
          `#${snapshotColor(index, runs.length).toString(16).padStart(6, "0")}`,
        );
        name.textContent = snapshotLabel(run.snapshot);
        name.title = `${run.snapshot.step.toLocaleString("en-US")} training steps`;
        status.dataset.ghostStatus = run.snapshot.id;
        row.append(name, status);
        legend.append(row);
      });
      this.update();
    }
    void this.updateHeatmap();
  }
  private updateSnapshotLabel(): void {
    const slider = node<HTMLInputElement>("heatmap-snapshot"),
      snapshot = this.loader.manifest.snapshots[Number(slider.value)];
    node("heatmap-stage").textContent = snapshotLabel(snapshot);
    slider.setAttribute(
      "aria-valuetext",
      `${snapshot.step.toLocaleString("en-US")} steps${snapshot.id === "final" ? ", final policy" : ""}`,
    );
  }
  async updateHeatmap(): Promise<void> {
    this.updateSnapshotLabel();
    if (!this.track) return;
    const visits = node<HTMLInputElement>("visits-toggle").checked,
      crashes = node<HTMLInputElement>("crashes-toggle").checked;
    const revision = ++this.heatRevision,
      track = this.track,
      selected = Number(node<HTMLInputElement>("heatmap-snapshot").value);
    // Clear the old stage immediately; never label stale data as a newly selected snapshot.
    this.scene.setHeatmap(null, false, false, { visits: 0, crashes: 0 });
    if (!visits && !crashes) {
      node("heatmap-summary").textContent = "50 stochastic runs per stage";
      return;
    }
    node("heatmap-summary").textContent = "Loading verified heatmaps…";
    try {
      let pending = this.maps.get(track.data.id);
      if (!pending) {
        pending = Promise.all(
          this.loader.manifest.snapshots.map((s) =>
            this.loader.heatmap(s, track.data.id),
          ),
        );
        this.maps.set(track.data.id, pending);
      }
      const maps = await pending;
      if (revision !== this.heatRevision || track !== this.track) return;
      const map = maps[selected];
      this.scene.setHeatmap(map, visits, crashes, heatmapScale(maps));
      const runs = map.metadata.runs as number;
      node("heatmap-summary").textContent =
        `${runs} runs · ${map.crash_count} crashes · ${map.visit_count.toLocaleString("en-US")} visits`;
    } catch (error) {
      this.maps.delete(track.data.id);
      if (revision === this.heatRevision) {
        node("heatmap-summary").textContent = "Heatmap unavailable";
        this.notify(error instanceof Error ? error.message : String(error));
      }
    }
  }
  update(): void {
    const p = this.playback;
    if (!p) return;
    const text = (id: string, value: string): void => {
      const target = node(id);
      if (target.textContent !== value) target.textContent = value;
    };
    text("playback-play", p.playing ? "Pause" : "Play");
    node("playback-play").setAttribute("aria-pressed", String(p.playing));
    const elapsed = `${p.seconds.toFixed(1)} / ${p.duration.toFixed(1)} s`;
    text("playback-time", elapsed);
    const seek = node<HTMLInputElement>("playback-seek");
    const position = String(Number(p.seconds.toFixed(1)));
    if (seek.value !== position) seek.value = position;
    seek.setAttribute(
      "aria-valuetext",
      `${p.seconds.toFixed(1)} seconds of ${p.duration.toFixed(1)}`,
    );
    for (const pose of p.poses()) {
      const status = node("ghost-lineup").querySelector<HTMLElement>(
        `[data-ghost-status="${CSS.escape(pose.run.snapshot.id)}"]`,
      );
      const label =
        pose.status === "timeout" ? "TIME LIMIT" : pose.status.toUpperCase();
      if (status && status.textContent !== label) status.textContent = label;
    }
  }
  advance(seconds: number): void {
    this.playback?.advance(seconds);
  }
  pause(): void {
    this.playback?.pause();
    this.update();
  }
}
