export const VERSIONS = {
  physics: "physics.v1",
  obs: "obs.v1",
  track: "track.v1",
  replay: "replay.v1",
  heatmap: "heatmap.v1",
} as const;
export function requireVersions(versions: Record<string, string>): void {
  for (const [name, expected] of Object.entries(VERSIONS)) {
    if (versions?.[name] !== expected)
      throw new Error(
        `Incompatible ${name} version: expected ${expected}, received ${versions?.[name]}`,
      );
  }
}

export const TRACK_V2 = "track.v2";
