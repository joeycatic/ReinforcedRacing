import type { SiteData } from "../assets/load";
export function formatStats(
  stats: SiteData["stats"],
): { label: string; value: string }[] {
  return [
    {
      value: stats.showcase_completion.toFixed(1) + "%",
      label: "laps completed on showcase tracks",
    },
    {
      value: stats.held_out_completion.toFixed(1) + "%",
      label: "on " + stats.held_out_tracks + " tracks it never saw",
    },
    {
      value: stats.rl_libraries + " RL libs",
      label: "PPO, GAE, normalization by hand",
    },
    {
      value: stats.parity_max_error.toExponential(0).replace("e+", "e"),
      label: "Python ↔ browser physics error",
    },
  ];
}
