// A development-server-only acceptance harness. This is not a production entry point.
import { AssetLoader } from "../src/assets/load";
import { RaceEngine } from "../src/race/engine";
import { OnnxPolicy } from "../src/race/policy";
interface Validation {
  track: string;
  snapshot: string;
  browserLapTime: number;
  pythonLapTime: number;
  relativeError: number;
  ticks: number;
  inferences: number;
  crashes: number;
  maxPositionError: number;
  inferenceAndSimulationMs: number;
}
declare global {
  interface Window {
    validatePolicy: (
      trackId: string,
      snapshotId?: string,
    ) => Promise<Validation>;
  }
}
window.validatePolicy = async (trackId, snapshotId = "final") => {
  const loader = await AssetLoader.open(
    new URL("/assets/manifest.json", location.href),
  );
  const track = await loader.track(trackId),
    config = await loader.simulation();
  const medal = loader.manifest.medals.find((m) => m.medal === snapshotId);
  const snapshot =
    medal ?? loader.manifest.snapshots.find((s) => s.id === snapshotId)!;
  const policy = await OnnxPolicy.create(await loader.bytes(snapshot.model));
  try {
    const replay = medal
        ? await loader.medalReplay(medal, trackId)
        : await loader.replay(
            snapshot as import("../src/assets/load").Snapshot,
            trackId,
          ),
      engine = new RaceEngine(track, config, policy);
    engine.start();
    for (let i = 0; i < 180; i++) await engine.tick([0, 0]);
    let maxPositionError = 0;
    const started = performance.now();
    for (let i = 0; i < 12000 && engine.aiLapTime === null; i++) {
      await engine.tick([0, 0]);
      if (engine.ai.crashes)
        throw new Error(`AI crashed on ${trackId} at tick ${engine.ai.ticks}`);
      const frame =
        replay.frames[Math.min(engine.ai.ticks, replay.frames.length - 1)];
      maxPositionError = Math.max(
        maxPositionError,
        Math.hypot(frame[0] - engine.ai.state.x, frame[1] - engine.ai.state.y),
      );
    }
    if (engine.aiLapTime === null)
      throw new Error("AI did not finish before the evaluation limit");
    const pythonLapTime = Number(replay.data.metadata.lap_time);
    return {
      track: trackId,
      snapshot: snapshotId,
      browserLapTime: engine.aiLapTime,
      pythonLapTime,
      relativeError: Math.abs(engine.aiLapTime - pythonLapTime) / pythonLapTime,
      ticks: engine.ai.ticks,
      inferences: engine.inferences,
      crashes: engine.ai.crashes,
      maxPositionError,
      inferenceAndSimulationMs: performance.now() - started,
    };
  } finally {
    await policy.release();
  }
};
