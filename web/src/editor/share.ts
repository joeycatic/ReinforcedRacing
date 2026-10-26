import type { Vec2 } from "../sim/types";
import { buildEditorTrack, roundEven } from "./build";
import config from "./editor-config.json";
const MAX_COORD = 1_000_000,
  MAX_CONTROLS = 256,
  MAX_CROSSINGS = 128;
function put(n: number, bytes: number[]) {
  do {
    const b = n % 128;
    n = Math.floor(n / 128);
    bytes.push(b + (n ? 128 : 0));
  } while (n);
}
export function encodeTrack(control: Vec2[], over: number[]): string {
  if (
    control.length < 4 ||
    control.length > MAX_CONTROLS ||
    over.length > MAX_CROSSINGS ||
    over.some((v) => v !== 0 && v !== 1) ||
    control.some(
      (p) =>
        p.length !== 2 ||
        p.some((v) => !Number.isFinite(v) || Math.abs(v) > MAX_COORD),
    )
  )
    throw new Error("Invalid share controls");
  const bytes: number[] = [];
  put(control.length, bytes);
  put(over.length, bytes);
  over.forEach((v) => put(v, bytes));
  const prev = [0, 0];
  for (const p of control)
    for (let k = 0; k < 2; k++) {
      const v = roundEven(p[k] / config.quantum),
        d = v - prev[k];
      put(d < 0 ? -2 * d - 1 : 2 * d, bytes);
      prev[k] = v;
    }
  return (
    "1." +
    btoa(String.fromCharCode(...bytes))
      .replaceAll("+", "-")
      .replaceAll("/", "_")
      .replace(/=+$/, "")
  );
}
export function decodeTrack(
  code: string,
): { control: Vec2[]; over: number[] } | null {
  try {
    if (code.length > 10000 || !/^1\.[A-Za-z0-9_-]+$/.test(code)) return null;
    const raw = code.slice(2),
      binary = atob(raw.replaceAll("-", "+").replaceAll("_", "/"));
    if (
      btoa(binary)
        .replaceAll("+", "-")
        .replaceAll("/", "_")
        .replace(/=+$/, "") !== raw
    )
      return null;
    let at = 0;
    const get = () => {
      let n = 0,
        f = 1;
      for (let i = 0; i < 5; i++) {
        if (at >= binary.length) throw Error();
        const b = binary.charCodeAt(at++);
        n += (b & 127) * f;
        if (b < 128) {
          if (i > 0 && b === 0) throw Error();
          return n;
        }
        f *= 128;
      }
      throw Error();
    };
    const count = get(),
      crossings = get();
    if (count < 4 || count > MAX_CONTROLS || crossings > MAX_CROSSINGS)
      return null;
    const over = Array.from({ length: crossings }, get);
    if (over.some((v) => v !== 0 && v !== 1)) return null;
    const prev = [0, 0],
      control: Vec2[] = [];
    for (let i = 0; i < count; i++) {
      const p: Vec2 = [0, 0];
      for (let k = 0; k < 2; k++) {
        const v = get();
        prev[k] += v % 2 ? -(v + 1) / 2 : v / 2;
        p[k] = prev[k] * config.quantum;
        if (Math.abs(p[k]) > MAX_COORD) return null;
      }
      control.push(p);
    }
    if (at !== binary.length) return null;
    if (
      (buildEditorTrack(control, config, over).track.data.crossings?.length ??
        0) !== crossings
    )
      return null;
    return { control, over };
  } catch {
    return null;
  }
}
