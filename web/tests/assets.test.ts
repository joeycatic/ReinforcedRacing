import { describe, it, expect, vi, afterEach } from "vitest";
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import {
  AssetLoader,
  decodeReplay,
  validateHeatmap,
  validateManifest,
  type Asset,
  type Manifest,
  type Replay,
  type Heatmap,
} from "../src/assets/load";
import { VERSIONS } from "../src/assets/versions";

const asset: Asset = {
  path: "models/final.onnx",
  bytes: 4,
  sha256: "a".repeat(64),
};
const manifest = (): Manifest => ({
  version: "manifest.v1",
  versions: { ...VERSIONS },
  simulation: asset,
  tracks: [{ ...asset, id: "sweep", name: "Sweep" }],
  snapshots: [
    {
      id: "final",
      step: 42,
      checkpoint_sha256: "b".repeat(64),
      model: asset,
      replays: { sweep: asset },
      heatmaps: { sweep: asset },
    },
  ],
});
const replay = (): Replay => ({
  version: "replay.v1",
  dt: 1 / 60,
  columns: ["x", "y", "psi", "speed", "steer", "throttle"],
  scales: [100, 100, 1000, 100, 10000, 10000],
  deltas: [
    [1234, -50, 3141, 0, 0, 0],
    [3, -5, -6282, 123, 500, -10000],
    [-2, 1, 1, -10, -1000, 20000],
  ],
  metadata: {
    track_id: "sweep",
    snapshot_id: "final",
    physics_ticks: 2,
    finished: false,
    crashed: true,
  },
});
afterEach(() => vi.unstubAllGlobals());
describe("versioned assets", () => {
  for (const name of Object.keys(VERSIONS))
    it(`rejects an incompatible ${name} before fetching assets`, async () => {
      const data = manifest();
      data.versions[name] = "future.v99";
      const fetch = vi
        .fn()
        .mockResolvedValue(new Response(JSON.stringify(data)));
      vi.stubGlobal("fetch", fetch);
      await expect(
        AssetLoader.open("http://localhost/assets/manifest.json"),
      ).rejects.toThrow(`Incompatible ${name}`);
      expect(fetch).toHaveBeenCalledTimes(1);
    });
  it("rejects missing assets, duplicate identities and unsafe paths", () => {
    const missing = manifest();
    delete missing.snapshots[0].replays.sweep;
    expect(() => validateManifest(missing)).toThrow();
    const duplicate = manifest();
    duplicate.tracks.push(duplicate.tracks[0]);
    expect(() => validateManifest(duplicate)).toThrow("Duplicate");
    const unsafe = manifest();
    unsafe.simulation = { ...asset, path: "../secret" };
    expect(() => validateManifest(unsafe)).toThrow("path");
  });
  it("decodes absolute-first integer deltas including heading wrap", () => {
    const frames = decodeReplay(replay());
    expect(frames).toEqual([
      [12.34, -0.5, 3.141, 0, 0, 0],
      [12.37, -0.55, -3.141, 1.23, 0.05, -1],
      [12.35, -0.54, -3.14, 1.13, -0.05, 1],
    ]);
    const bad = replay();
    bad.deltas[0][0] = 0.5;
    expect(() => decodeReplay(bad)).toThrow("deltas");
    expect(() => decodeReplay({ ...replay(), version: "replay.v2" })).toThrow(
      "version",
    );
  });
  it("validates sparse heatmap conservation", () => {
    const data: Heatmap = {
      version: "heatmap.v1",
      metadata: { track_id: "sweep", snapshot_id: "final" },
      grid: {
        encoding: "sparse.flat.v1",
        origin: [0, 0],
        width: 2,
        height: 2,
        cell_size: 0.5,
      },
      visits: [
        [0, 3],
        [2, 4],
      ],
      crash_bins: [[2, 1]],
      crash_positions: [[0.1, 0.7]],
      visit_count: 7,
      crash_count: 1,
    };
    expect(validateHeatmap(data)).toBe(data);
    expect(() => validateHeatmap({ ...data, visit_count: 8 })).toThrow(
      "counts",
    );
    expect(() => validateHeatmap({ ...data, visits: [[4, 7]] })).toThrow(
      "sparse",
    );
  });
  it("verifies checksums before using downloaded JSON/model bytes", async () => {
    const bytes = new TextEncoder().encode("test");
    const digest = Array.from(
      new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
      (b) => b.toString(16).padStart(2, "0"),
    ).join("");
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(manifest())))
      .mockResolvedValueOnce(new Response(bytes))
      .mockResolvedValueOnce(new Response("tEst"));
    vi.stubGlobal("fetch", fetch);
    const loader = await AssetLoader.open(
      "http://localhost/assets/manifest.json",
    );
    const checked = { ...asset, sha256: digest };
    expect(new Uint8Array(await loader.bytes(checked))).toEqual(bytes);
    await expect(loader.bytes(checked)).rejects.toThrow("checksum");
  });
});
// When a local export exists, exercise the exact production loader against every file.
const exported = fileURLToPath(new URL("../public/assets/", import.meta.url));
it.skipIf(!existsSync(exported + "manifest.json"))(
  "loads the complete exported bundle with integrity checks",
  async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: URL) => {
        const path = decodeURIComponent(url.pathname).replace(
          /^\/assets\//,
          "",
        );
        if (path.includes("..")) return new Response("", { status: 400 });
        return new Response(readFileSync(exported + path));
      }),
    );
    const loader = await AssetLoader.open(
      "http://localhost/assets/manifest.json",
    );
    await loader.simulation();
    for (const track of loader.manifest.tracks) await loader.track(track.id);
    for (const snapshot of loader.manifest.snapshots) {
      await loader.bytes(snapshot.model);
      for (const track of loader.manifest.tracks) {
        const replay = await loader.replay(snapshot, track.id);
        expect(replay.frames.length).toBeGreaterThan(1);
        const heatmap = await loader.heatmap(snapshot, track.id);
        expect(heatmap.metadata.runs).toBe(50);
      }
    }
  },
);
