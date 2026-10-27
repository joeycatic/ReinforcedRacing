import type { Policy } from "./policy";

/** Owner-approved pace for drawn tracks with overpasses; the underlying model and simulation are unchanged. */
export function policyForTrack(
  policy: Policy,
  drawnTrack: boolean,
  hasCrossings: boolean,
): Policy {
  if (!drawnTrack || !hasCrossings) return policy;
  return {
    async predict(observations) {
      const prediction = await policy.predict(observations);
      return {
        action: [prediction.action[0], Math.min(prediction.action[1], 0.2)],
        value: prediction.value,
      };
    },
  };
}
