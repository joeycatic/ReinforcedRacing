import type { Medal, MedalName } from "../assets/load";
export const MEDAL_COLORS: Record<MedalName, string> = {
  bronze: "#c9824a",
  silver: "#c9d1d9",
  gold: "#f5c542",
  author: "#3fe0b0",
};
export function medalsForTrack(medals: Medal[], trackId: string): Medal[] {
  return medals.filter(
    (m) =>
      typeof m.lap_times[trackId] === "number" &&
      Number.isFinite(m.lap_times[trackId]),
  );
}
export function earnedMedal(
  lap: number,
  medals: Medal[],
  trackId: string,
): MedalName | null {
  return (
    medalsForTrack(medals, trackId)
      .filter((m) => lap <= m.lap_times[trackId]!)
      .at(-1)?.medal ?? null
  );
}
