import type { Action } from "../sim/types";
export class DriverInput {
  private keys = new Set<string>();
  constructor(
    private readonly onRestart: () => void,
    private readonly onRespawn: () => void,
    private readonly onPause: () => void,
  ) {
    window.addEventListener("keydown", this.keydown);
    window.addEventListener("keyup", this.keyup);
    window.addEventListener("blur", this.clear);
  }
  private keydown = (event: KeyboardEvent): void => {
    if (event.altKey || event.ctrlKey || event.metaKey) return;
    if (
      event.target instanceof HTMLElement &&
      /^(INPUT|SELECT|TEXTAREA|BUTTON|SUMMARY|A)$/.test(event.target.tagName)
    )
      return;
    if (
      [
        "ArrowUp",
        "ArrowDown",
        "ArrowLeft",
        "ArrowRight",
        "Space",
        "KeyW",
        "KeyA",
        "KeyS",
        "KeyD",
        "KeyR",
        "Enter",
        "Escape",
      ].includes(event.code)
    )
      event.preventDefault();
    this.keys.add(event.code);
    if (!event.repeat) {
      if (event.code === "Enter") this.onRestart();
      if (event.code === "KeyR") this.onRespawn();
      if (event.code === "Escape") this.onPause();
    }
  };
  private keyup = (event: KeyboardEvent): void => {
    this.keys.delete(event.code);
  };
  clear = (): void => {
    this.keys.clear();
  };
  action(): Action {
    let steer =
      Number(this.keys.has("ArrowLeft") || this.keys.has("KeyA")) -
      Number(this.keys.has("ArrowRight") || this.keys.has("KeyD"));
    let throttle =
      Number(this.keys.has("ArrowUp") || this.keys.has("KeyW")) -
      Number(this.keys.has("ArrowDown") || this.keys.has("KeyS"));
    if (this.keys.has("Space")) throttle = -1;
    const pad = navigator.getGamepads?.().find((p) => p?.connected);
    if (pad) {
      if (Math.abs(pad.axes[0]) > 0.15) steer = -pad.axes[0];
      const trigger =
        (pad.buttons[7]?.value ?? 0) - (pad.buttons[6]?.value ?? 0);
      if (Math.abs(trigger) > 0.05) throttle = trigger;
      if (pad.buttons[14]?.pressed) steer = 1;
      if (pad.buttons[15]?.pressed) steer = -1;
    }
    return [steer, throttle];
  }
  destroy(): void {
    window.removeEventListener("keydown", this.keydown);
    window.removeEventListener("keyup", this.keyup);
    window.removeEventListener("blur", this.clear);
  }
}
