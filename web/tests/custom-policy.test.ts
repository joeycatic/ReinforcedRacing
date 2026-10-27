import { expect, it, vi } from "vitest";
import { policyForTrack } from "../src/race/custom-policy";
import type { Policy, Prediction } from "../src/race/policy";

it.each([1, 0.2, 0.05, 0, -0.75, -1])(
  "drawn tracks cap positive throttle %s without changing steering, braking or value",
  async (throttle) => {
    const prediction: Prediction = { action: [-0.83, throttle], value: 4.5 };
    Object.freeze(prediction.action);
    Object.freeze(prediction);
    const predict = vi.fn(async () => prediction);
    const observation = [0.1, 0.2];
    expect(
      await policyForTrack({ predict }, true, true).predict(observation),
    ).toEqual({
      action: [-0.83, Math.min(throttle, 0.2)],
      value: 4.5,
    });
    expect(predict).toHaveBeenCalledExactlyOnceWith(observation);
    expect(prediction.action).toEqual([-0.83, throttle]);
  },
);
it.each([
  { drawn: false, crossings: false },
  { drawn: false, crossings: true },
  { drawn: true, crossings: false },
])(
  "raw actions remain unchanged for $drawn drawn / $crossings crossings",
  async ({ drawn, crossings }) => {
    const prediction: Prediction = { action: [0.6, 1], value: -3 };
    const raw: Policy = { predict: async () => prediction };
    const selected = policyForTrack(raw, drawn, crossings);
    expect(selected).toBe(raw);
    expect(await selected.predict([])).toBe(prediction);
  },
);
it("drawn-track inference failures propagate to existing loading and cancellation handling", async () => {
  const failure = new Error("inference failed");
  const raw: Policy = {
    predict: async () => {
      throw failure;
    },
  };
  await expect(policyForTrack(raw, true, true).predict([])).rejects.toBe(
    failure,
  );
});
