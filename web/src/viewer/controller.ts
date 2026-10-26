import { AssetLoader, type Heatmap } from "../assets/load";
import { Track } from "../sim/track";
import { RaceScene } from "../race/scene";
import { snapshotLabel } from "../watch/timeline";
import { heatmapScale } from "./heatmap";

const node = <T extends HTMLElement = HTMLElement>(id: string): T =>
  document.getElementById(id) as T;
export class ViewerController {
  private track: Track | null = null;
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
    this.updateSnapshotLabel();
  }
  async setTrack(track: Track): Promise<void> {
    this.track = track;
    this.heatRevision++;
    this.scene.setHeatmap(null, false, false, { visits: 0, crashes: 0 });
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
}
