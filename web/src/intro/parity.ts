import type { ParityDemo, SimulationConfig } from "../assets/load";
import { step } from "../sim/physics";
import type { Track } from "../sim/track";

export function replayParity(
  demo: ParityDemo,
  track: Track,
  config: SimulationConfig,
): { typescript: [number, number, number][]; maxError: number } {
  if (
    track.data.id !== demo.track_id ||
    demo.actions.length !== demo.python.length
  )
    throw new Error("Parity demo identity mismatch");
  let state = { ...demo.initial_state },
    maxError = 0;
  const typescript = demo.actions.map((action, i) => {
    // The exported fixture's `state` is its unrestricted physics trajectory.
    // Its separately recorded `session` trajectory includes crashes and resets.
    state = step(state, action, config.physics);
    const pose: [number, number, number] = [state.x, state.y, state.psi];
    pose.forEach((v, k) => {
      maxError = Math.max(maxError, Math.abs(v - demo.python[i][k]));
    });
    return pose;
  });
  return { typescript, maxError };
}

export function renderParityDemo(
  host: HTMLElement,
  demo: ParityDemo,
  result: ReturnType<typeof replayParity>,
  track: Track,
): void {
  const all = [...track.data.centerline, ...demo.python, ...result.typescript];
  const xs = all.map((p) => p[0]),
    ys = all.map((p) => -p[1]);
  const x = Math.min(...xs) - 20,
    y = Math.min(...ys) - 20,
    w = Math.max(...xs) - x + 20,
    h = Math.max(...ys) - y + 20;
  const path = (points: number[][]): string =>
    points.map((p, i) => `${i ? "L" : "M"}${p[0]},${-p[1]}`).join(" ");
  host.innerHTML = `<div class="parity-legend"><span>Python recording</span><span>TypeScript · computed here</span></div><svg role="img" aria-label="Overlapping Python and TypeScript trajectories" viewBox="${x} ${y} ${w} ${h}"><path d="${path(track.data.centerline)} Z" fill="none" stroke="#c9d2c1" stroke-width="${track.data.width}"/><path d="${path(demo.python)}" fill="none" stroke="#2a78d6" stroke-width="3"/><path d="${path(result.typescript)}" fill="none" stroke="#e45f23" stroke-width="1.5" stroke-dasharray="4 3"/></svg><p>Maximum position / heading error: <strong>${result.maxError === 0 ? "0 (exact match in this browser)" : result.maxError.toExponential(1)}</strong> · ${result.typescript.length} physics steps, run in your browser.</p>`;
}
