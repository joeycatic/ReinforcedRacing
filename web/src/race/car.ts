import {
  Container,
  Graphics,
  Text,
  Sprite,
  type Renderer,
  type Texture,
} from "pixi.js";
export interface Livery {
  body: number;
  dark: number;
  light: number;
  accent: number;
  number: string;
}
export const LIVERIES: { player: Livery; ai: Livery } = {
  player: {
    body: 0x3fe0b0,
    dark: 0x167859,
    light: 0xb9ffe9,
    accent: 0xffffff,
    number: "07",
  },
  ai: {
    body: 0xff7a2f,
    dark: 0xa44016,
    light: 0xffc797,
    accent: 0xffffff,
    number: "AI",
  },
};
const textureCache = new WeakMap<Renderer, Map<string, Texture>>();
/** Bake once at 24 px/m; the longitudinal axis points along positive x. */
export function formulaCar(livery: Livery, renderer: Renderer): Sprite {
  let cache = textureCache.get(renderer);
  if (!cache) {
    cache = new Map();
    textureCache.set(renderer, cache);
  }
  const key = JSON.stringify(livery),
    cached = cache.get(key);
  if (cached) {
    const sprite = new Sprite(cached);
    sprite.anchor.set(0.5);
    sprite.scale.set(1 / 24);
    return sprite;
  }
  const source = new Container(),
    g = new Graphics();
  g.ellipse(0.1, 0.18, 2.38, 1.1).fill({ color: 0x172116, alpha: 0.18 });
  g.ellipse(0.05, 0.1, 2.15, 0.92).fill({ color: 0x172116, alpha: 0.16 });
  // Suspension and four exposed slick tyres.
  for (const x of [-1.35, 1.35]) {
    g.rect(x - 0.08, -0.95, 0.16, 1.9).fill(0x273139);
    for (const side of [-1, 1]) {
      g.roundRect(x - 0.48, side * 0.82 - 0.26, 0.96, 0.52, 0.1).fill(0x101519);
      g.rect(x - 0.35, side * 0.82 - 0.21, 0.7, 0.06).fill(0x586067);
    }
  }
  g.rect(-2.25, -1, 0.4, 2).fill(livery.dark);
  g.rect(-2.25, -1, 0.12, 2).fill(livery.accent);
  g.poly([
    -1.9, -0.35, -0.7, -0.57, 0.3, -0.38, 1.65, -0.17, 2.1, -0.12, 2.1, 0.12,
    1.65, 0.17, 0.3, 0.38, -0.7, 0.57, -1.9, 0.35,
  ]).fill(livery.body);
  g.poly([
    -1.7, -0.3, -0.8, -0.48, 0.4, -0.28, 1.9, -0.08, 0.1, -0.08, -0.7, -0.25,
  ]).fill(livery.light);
  g.poly([-1.7, 0.3, -0.7, 0.49, 0.4, 0.29, 1.9, 0.08, -0.4, 0.15]).fill(
    livery.dark,
  );
  g.rect(1.8, -0.9, 0.4, 1.8).fill(livery.body);
  g.rect(2.1, -0.9, 0.12, 1.8).fill(livery.accent);
  g.roundRect(-0.9, -0.28, 0.85, 0.56, 0.25).fill(0x17242b);
  g.circle(-0.43, 0, 0.19).fill(livery.accent);
  g.moveTo(-0.1, -0.3)
    .lineTo(0.05, 0)
    .lineTo(-0.1, 0.3)
    .stroke({ color: 0x212b32, width: 0.08 });
  source.addChild(g);
  const label = new Text({
    text: livery.number,
    style: {
      fontFamily: "Arial",
      fontSize: 8,
      fontWeight: "900",
      fill: 0x142322,
    },
  });
  label.anchor.set(0.5);
  label.scale.set(1 / 24);
  label.position.set(1.08, 0);
  label.rotation = Math.PI / 2;
  source.addChild(label);
  const bake = new Container();
  source.scale.set(24);
  bake.addChild(source);
  const texture = renderer.generateTexture({
    target: bake,
    resolution: 1,
    antialias: true,
  });
  cache.set(key, texture);
  const sprite = new Sprite(texture);
  sprite.anchor.set(0.5);
  sprite.scale.set(1 / 24);
  bake.destroy({ children: true });
  return sprite;
}
