import type { Heatmap } from "../assets/load";

export interface HeatmapScale {
  visits: number;
  crashes: number;
}
export function heatmapScale(maps: Heatmap[]): HeatmapScale {
  let visits = 0,
    crashes = 0;
  for (const map of maps) {
    for (const [, count] of map.visits) visits = Math.max(visits, count);
    for (const [, count] of map.crash_bins) crashes = Math.max(crashes, count);
  }
  return { visits, crashes };
}
/** Convert lower-left world grid rows to top-left texture rows without changing cells. */
export function heatmapPixels(
  map: Heatmap,
  layer: "visits" | "crash_bins",
  peak: number,
): Uint8ClampedArray {
  const { width, height } = map.grid,
    pixels = new Uint8ClampedArray(width * height * 4);
  if (peak <= 0) return pixels;
  for (const [index, count] of map[layer]) {
    const x = index % width,
      y = Math.floor(index / width),
      offset = ((height - 1 - y) * width + x) * 4;
    const strength = Math.min(1, Math.log1p(count) / Math.log1p(peak));
    const cold = layer === "visits" ? [40, 130, 180] : [233, 70, 62],
      warm = layer === "visits" ? [134, 255, 217] : [255, 132, 104];
    for (let channel = 0; channel < 3; channel++)
      pixels[offset + channel] =
        cold[channel] + (warm[channel] - cold[channel]) * strength;
    pixels[offset + 3] = (layer === "visits" ? 165 : 220) * Math.sqrt(strength);
  }
  return pixels;
}
