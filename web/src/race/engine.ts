import type { Action, CarState } from "../sim/types";
import { DriveSession } from "../sim/session";
import { Track } from "../sim/track";
import { ObservationBuilder } from "../sim/observation";
import type { SimulationConfig } from "../assets/load";
import type { Policy } from "./policy";

export type Phase = "ready" | "countdown" | "racing" | "paused" | "finished";
export class RaceEngine {
  readonly player: DriveSession;
  readonly ai: DriveSession;
  readonly observations: ObservationBuilder;
  phase: Phase = "ready";
  private beforePause: Phase = "ready";
  countdownTicks = 0;
  raceTicks = 0;
  inferences = 0;
  value = 0;
  aiLapTime: number | null = null;
  playerLapTime: number | null = null;
  previousPlayer: CarState;
  previousAi: CarState;
  readonly playerSplits = new Map<number, number>();
  readonly aiSplits = new Map<number, number>();
  private playerAction: Action = [0, 0];
  private aiAction: Action = [0, 0];
  private aiPreviousAction: Action = [0, 0];
  private generation = 0;
  constructor(
    readonly track: Track,
    readonly config: SimulationConfig,
    readonly policy: Policy,
  ) {
    this.player = new DriveSession(track, config.physics, config.tracks);
    this.ai = new DriveSession(track, config.physics, config.tracks);
    this.observations = new ObservationBuilder(track, config.observation);
    this.previousPlayer = this.player.state;
    this.previousAi = this.ai.state;
  }
  start(): void {
    this.reset();
    this.countdownTicks = Math.round(3 / this.config.physics.dt);
    this.phase = "countdown";
  }
  reset(): void {
    this.generation++;
    this.player.restart();
    this.ai.restart();
    this.raceTicks = 0;
    this.inferences = 0;
    this.aiLapTime = null;
    this.playerLapTime = null;
    this.value = 0;
    this.countdownTicks = 0;
    this.playerSplits.clear();
    this.aiSplits.clear();
    this.playerAction = [0, 0];
    this.aiAction = [0, 0];
    this.aiPreviousAction = [0, 0];
    this.previousPlayer = this.player.state;
    this.previousAi = this.ai.state;
    this.phase = "ready";
  }
  pause(): void {
    if (this.phase === "countdown" || this.phase === "racing") {
      this.beforePause = this.phase;
      this.phase = "paused";
      this.generation++;
    }
  }
  resume(): void {
    if (this.phase === "paused") this.phase = this.beforePause;
  }
  respawn(): void {
    if (this.phase !== "racing") return;
    this.player.respawn();
    this.previousPlayer = this.player.state;
    this.playerAction = [0, 0];
  }
  get countdown(): number {
    return Math.ceil(this.countdownTicks * this.config.physics.dt);
  }
  get split(): { ordinal: number; delta: number } | null {
    const common = [...this.playerSplits.keys()].filter((i) =>
      this.aiSplits.has(i),
    );
    if (!common.length) return null;
    const ordinal = Math.max(...common);
    return {
      ordinal,
      delta: this.playerSplits.get(ordinal)! - this.aiSplits.get(ordinal)!,
    };
  }
  private drive(
    session: DriveSession,
    action: Action,
    splits: Map<number, number>,
  ): void {
    const ordinal = session.progress.next_checkpoint,
      previous = session.progress.unwrapped;
    session.tick(action);
    for (
      let index = ordinal;
      index < session.progress.next_checkpoint;
      index++
    ) {
      const fraction =
        (session.progress.checkpointArc(index) - previous) /
        session.progress.delta;
      splits.set(
        index,
        session.time - session.physics.dt + fraction * session.physics.dt,
      );
    }
  }
  async tick(input: Action): Promise<void> {
    if (this.phase === "countdown") {
      this.countdownTicks--;
      if (this.countdownTicks <= 0) this.phase = "racing";
      return;
    }
    if (this.phase !== "racing") return;
    const generation = this.generation;
    if (this.raceTicks % this.config.physics.action_repeat === 0) {
      if (this.aiLapTime === null) {
        const raw = this.observations.build(
          this.ai.state,
          this.ai.progress.projection,
          this.aiPreviousAction,
        );
        const prediction = await this.policy.predict(raw);
        if (this.generation !== generation || this.phase !== "racing") return;
        this.aiAction = prediction.action;
        this.value = prediction.value;
        this.inferences++;
      }
      this.playerAction = [...input];
    }
    this.previousPlayer = this.player.state;
    this.previousAi = this.ai.state;
    this.drive(this.player, this.playerAction, this.playerSplits);
    if (this.aiLapTime === null) {
      this.drive(this.ai, this.aiAction, this.aiSplits);
      this.aiPreviousAction = [...this.aiAction];
      if (this.ai.progress.laps >= 1)
        this.aiLapTime = this.ai.progress.lap_times[0];
    }
    this.raceTicks++;
    if (this.player.progress.laps >= 1) {
      this.playerLapTime = this.player.progress.lap_times[0];
      this.phase = "finished";
    }
  }
}
/** Render clocks cannot change dt or skip physics ticks. Long stalls pause simulation time. */
export class FixedClock {
  accumulator = 0;
  private running = false;
  droppedSeconds = 0;
  constructor(
    readonly engine: RaceEngine,
    readonly maxCatchupTicks = 8,
  ) {}
  get alpha(): number {
    return Math.min(1, this.accumulator / this.engine.config.physics.dt);
  }
  clear(): void {
    this.accumulator = 0;
  }
  async advance(seconds: number, input: Action): Promise<void> {
    if (this.running) return;
    this.running = true;
    try {
      if (
        this.engine.phase === "paused" ||
        this.engine.phase === "ready" ||
        this.engine.phase === "finished"
      ) {
        this.accumulator = 0;
        return;
      }
      const dt = this.engine.config.physics.dt,
        maximum = dt * this.maxCatchupTicks;
      const elapsed = Math.max(0, seconds);
      this.droppedSeconds += Math.max(0, this.accumulator + elapsed - maximum);
      this.accumulator = Math.min(this.accumulator + elapsed, maximum);
      while (this.accumulator + 1e-12 >= dt) {
        await this.engine.tick(input);
        this.accumulator = Math.max(0, this.accumulator - dt);
      }
    } finally {
      this.running = false;
    }
  }
}
export function interpolate(
  previous: CarState,
  current: CarState,
  alpha: number,
): CarState {
  const angle = Math.atan2(
    Math.sin(current.psi - previous.psi),
    Math.cos(current.psi - previous.psi),
  );
  return {
    ...current,
    x: previous.x + (current.x - previous.x) * alpha,
    y: previous.y + (current.y - previous.y) * alpha,
    psi: previous.psi + angle * alpha,
  };
}
