export const MESSAGES: Record<string, string> = {
  "not-closed": "Finish near the start to close the loop",
  "too-short": "Draw a bigger loop",
  fold: "The road folds over itself in a corner",
  overlap: "Roads overlap — cross them clearly or keep them apart",
  "shallow-crossing": "Roads cross at too shallow an angle",
  "crossing-spacing": "Crossings are too close together",
  "crossing-start": "A crossing is too close to the start line",
  tight: "Too tight — the AI never trained on corners this sharp",
};
export function canSubmit(problem: string | null, issues: string[]): boolean {
  return problem === null && issues.every((code) => code === "tight");
}
