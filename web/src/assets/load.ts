import { VERSIONS, requireVersions } from "./versions";
import type {
  ObservationConfig,
  PhysicsConfig,
  TrackConfig,
  TrackData,
  CarState,
  Action,
} from "../sim/types";
import { Track } from "../sim/track";

export interface Asset {
  path: string;
  sha256: string;
  bytes: number;
}
export interface Snapshot {
  id: string;
  step: number;
  checkpoint_sha256: string;
  model: Asset;
  replays: Record<string, Asset>;
  heatmaps: Record<string, Asset>;
}
export type MedalName = "bronze" | "silver" | "gold" | "author";
export interface Medal {
  medal: MedalName;
  step: number;
  checkpoint_sha256: string;
  model: Asset;
  lap_times: Record<string, number | null>;
  replays: Record<string, Asset>;
}
export interface SiteData {
  stats: {
    showcase_completion: number;
    held_out_completion: number;
    parity_max_error: number;
    training_steps: number;
    training_minutes: number;
    training_tracks: number;
    held_out_tracks: number;
    parallel_cars: number;
    rl_libraries: number;
  };
  curves: Record<string, [number, number][]>;
  snapshots: [number, number, number][];
  seeds: {
    seed: number;
    showcase_pct: number;
    held_out_pct: number;
    deterministic_showcase_finishes: number;
  }[];
}
export interface ParityDemo {
  track_id: string;
  initial_state: CarState;
  actions: Action[];
  python: [number, number, number][];
}
export interface Manifest {
  version: string;
  versions: Record<string, string>;
  simulation: Asset;
  medals: Medal[];
  site: Asset;
  parity_demo: Asset;
  tracks: (Asset & { id: string; name: string })[];
  snapshots: Snapshot[];
}
export interface SimulationConfig {
  versions: Record<string, string>;
  physics: PhysicsConfig;
  tracks: TrackConfig;
  observation: ObservationConfig;
}
export interface Replay {
  version: string;
  dt: number;
  columns: string[];
  scales: number[];
  deltas: number[][];
  metadata: {
    track_id: string;
    snapshot_id: string;
    physics_ticks: number;
    finished: boolean;
    crashed: boolean;
    [key: string]: unknown;
  };
}
export interface Heatmap {
  version: string;
  metadata: { track_id: string; snapshot_id: string; [key: string]: unknown };
  grid: {
    encoding: string;
    origin: [number, number];
    cell_size: number;
    width: number;
    height: number;
  };
  visits: [number, number][];
  crash_bins: [number, number][];
  crash_positions: [number, number][];
  visit_count: number;
  crash_count: number;
}
function validateAsset(asset: Asset): void {
  if (
    typeof asset?.path !== "string" ||
    !/^[a-zA-Z0-9_./-]+$/.test(asset.path) ||
    asset.path.startsWith("/") ||
    asset.path.split("/").some((p) => p === ".." || p === "." || p === "")
  )
    throw new Error("Invalid asset path");
  if (
    !/^[a-f0-9]{64}$/.test(asset.sha256) ||
    !Number.isSafeInteger(asset.bytes) ||
    asset.bytes <= 0
  )
    throw new Error("Invalid asset integrity descriptor");
}
export function validateManifest(data: Manifest): Manifest {
  if (data.version !== "manifest.v2")
    throw new Error("Unsupported manifest version");
  requireVersions(data.versions);
  validateAsset(data.simulation);
  if (
    !Array.isArray(data.tracks) ||
    !data.tracks.length ||
    !Array.isArray(data.snapshots) ||
    !data.snapshots.length
  )
    throw new Error("Empty asset manifest");
  const tracks = new Set(data.tracks.map((t) => t.id));
  if (
    tracks.size !== data.tracks.length ||
    new Set(data.snapshots.map((s) => s.id)).size !== data.snapshots.length
  )
    throw new Error("Duplicate asset identity");
  data.tracks.forEach(validateAsset);
  for (const snapshot of data.snapshots) {
    validateAsset(snapshot.model);
    if (!Number.isSafeInteger(snapshot.step) || snapshot.step < 0)
      throw new Error("Invalid snapshot step");
    for (const track of tracks) {
      validateAsset(snapshot.replays[track]);
      validateAsset(snapshot.heatmaps[track]);
    }
  }
  validateAsset(data.site);
  validateAsset(data.parity_demo);
  const order: MedalName[] = ["bronze", "silver", "gold", "author"];
  if (!Array.isArray(data.medals) || data.medals.at(-1)?.medal !== "author")
    throw new Error("Missing author medal");
  let previous = -1;
  for (const medal of data.medals) {
    const index = order.indexOf(medal.medal);
    if (index <= previous) throw new Error("Invalid medal order");
    previous = index;
    if (
      !Number.isSafeInteger(medal.step) ||
      medal.step < 0 ||
      !/^[a-f0-9]{64}$/.test(medal.checkpoint_sha256)
    )
      throw new Error("Invalid medal identity");
    validateAsset(medal.model);
    for (const track of tracks) {
      const lap = medal.lap_times?.[track];
      if (
        lap !== null &&
        (typeof lap !== "number" || !Number.isFinite(lap) || lap <= 0)
      )
        throw new Error("Invalid medal lap time");
      validateAsset(medal.replays?.[track]);
    }
  }
  return data;
}
export function decodeReplay(data: Replay): number[][] {
  if (
    data.version !== VERSIONS.replay ||
    data.columns.join(",") !== "x,y,psi,speed,steer,throttle"
  )
    throw new Error("Unsupported replay version or columns");
  if (
    data.scales.length !== 6 ||
    data.scales.some((s) => !Number.isFinite(s) || s <= 0) ||
    !Number.isFinite(data.dt) ||
    data.dt <= 0 ||
    !data.deltas.length
  )
    throw new Error("Invalid replay");
  const previous = [0, 0, 0, 0, 0, 0];
  return data.deltas.map((row) => {
    if (row.length !== 6 || !row.every(Number.isSafeInteger))
      throw new Error("Invalid replay deltas");
    return row.map((delta, i) => {
      previous[i] += delta;
      if (!Number.isSafeInteger(previous[i]))
        throw new Error("Replay overflow");
      return previous[i] / data.scales[i];
    });
  });
}
export function validateHeatmap(data: Heatmap): Heatmap {
  if (data.version !== VERSIONS.heatmap)
    throw new Error("Unsupported heatmap version");
  const g = data.grid;
  if (
    g.encoding !== "sparse.flat.v1" ||
    ![g.width, g.height].every((v) => Number.isSafeInteger(v) && v > 0) ||
    !Number.isFinite(g.cell_size) ||
    g.cell_size <= 0 ||
    g.origin.length !== 2 ||
    !g.origin.every(Number.isFinite)
  )
    throw new Error("Invalid heatmap grid");
  for (const counts of [data.visits, data.crash_bins]) {
    let previous = -1;
    for (const [index, count] of counts) {
      if (
        !Number.isSafeInteger(index) ||
        index <= previous ||
        index >= g.width * g.height ||
        !Number.isSafeInteger(count) ||
        count <= 0
      )
        throw new Error("Invalid sparse heatmap");
      previous = index;
    }
  }
  if (
    data.visits.reduce((sum, [, v]) => sum + v, 0) !== data.visit_count ||
    data.crash_bins.reduce((sum, [, v]) => sum + v, 0) !== data.crash_count ||
    data.crash_positions.length !== data.crash_count ||
    data.crash_positions.some(
      (p) => p.length !== 2 || !p.every(Number.isFinite),
    )
  )
    throw new Error("Heatmap counts disagree");
  return data;
}
/** Manifest versions are checked before any policy or simulation asset is fetched. */
export class AssetLoader {
  private constructor(
    readonly base: URL,
    readonly manifest: Manifest,
  ) {}
  static async open(manifestUrl: string | URL): Promise<AssetLoader> {
    const url = new URL(
      manifestUrl,
      globalThis.location?.href ?? "http://localhost/",
    );
    const response = await fetch(url);
    if (!response.ok)
      throw new Error(`Cannot load manifest: HTTP ${response.status}`);
    return new AssetLoader(
      new URL(".", url),
      validateManifest((await response.json()) as Manifest),
    );
  }
  async bytes(asset: Asset): Promise<ArrayBuffer> {
    validateAsset(asset);
    const response = await fetch(new URL(asset.path, this.base));
    if (!response.ok)
      throw new Error(`Cannot load ${asset.path}: HTTP ${response.status}`);
    const bytes = await response.arrayBuffer();
    if (bytes.byteLength !== asset.bytes)
      throw new Error(`Asset size mismatch: ${asset.path}`);
    const digest = await crypto.subtle.digest("SHA-256", bytes);
    const hex = Array.from(new Uint8Array(digest), (b) =>
      b.toString(16).padStart(2, "0"),
    ).join("");
    if (hex !== asset.sha256)
      throw new Error(`Asset checksum mismatch: ${asset.path}`);
    return bytes;
  }
  async json<T>(asset: Asset): Promise<T> {
    return JSON.parse(new TextDecoder().decode(await this.bytes(asset))) as T;
  }
  async simulation(): Promise<SimulationConfig> {
    const data = await this.json<SimulationConfig>(this.manifest.simulation);
    requireVersions(data.versions);
    return data;
  }
  async track(id: string): Promise<Track> {
    const asset = this.manifest.tracks.find((t) => t.id === id);
    if (!asset) throw new Error(`Unknown track: ${id}`);
    const track = new Track(await this.json<TrackData>(asset));
    if (track.data.id !== id) throw new Error("Track identity mismatch");
    return track;
  }
  async replay(
    snapshot: Snapshot,
    trackId: string,
  ): Promise<{ data: Replay; frames: number[][] }> {
    const data = await this.json<Replay>(snapshot.replays[trackId]);
    const frames = decodeReplay(data);
    if (
      data.metadata.track_id !== trackId ||
      data.metadata.snapshot_id !== snapshot.id ||
      frames.length !== data.metadata.physics_ticks + 1
    )
      throw new Error("Replay identity or tick count mismatch");
    return { data, frames };
  }
  async site(): Promise<SiteData> {
    return this.json<SiteData>(this.manifest.site);
  }
  async parityDemo(): Promise<ParityDemo> {
    return this.json<ParityDemo>(this.manifest.parity_demo);
  }
  async medalReplay(
    medal: Medal,
    trackId: string,
  ): Promise<{ data: Replay; frames: number[][] }> {
    return this.replay(
      {
        id: medal.medal === "author" ? "final" : `medal-${medal.medal}`,
        step: medal.step,
        checkpoint_sha256: medal.checkpoint_sha256,
        model: medal.model,
        replays: medal.replays,
        heatmaps: {},
      },
      trackId,
    );
  }
  async heatmap(snapshot: Snapshot, trackId: string): Promise<Heatmap> {
    const data = validateHeatmap(
      await this.json<Heatmap>(snapshot.heatmaps[trackId]),
    );
    if (
      data.metadata.track_id !== trackId ||
      data.metadata.snapshot_id !== snapshot.id
    )
      throw new Error("Heatmap identity mismatch");
    return data;
  }
}
