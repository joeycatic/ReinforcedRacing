import type { SiteData } from "../assets/load";

export function renderLearningChart(host: HTMLElement, site: SiteData): void {
  const width = Math.max(360, Math.min(960, host.clientWidth)),
    height = width < 600 ? 550 : 520,
    left = 64,
    right = 22,
    panelHeight = 155;
  const plotWidth = width - left - right;
  const maxStep = Math.max(...site.snapshots.map((s) => s[0]));
  const returns = Object.values(site.curves)
    .flat()
    .map((p) => p[1]);
  const low = Math.min(0, ...returns),
    high = Math.ceil(Math.max(...returns) / 50) * 50;
  const x = (step: number): number => left + (plotWidth * step) / maxStep;
  const y = (v: number, top: number, min: number, max: number): number =>
    top + panelHeight * (1 - (v - min) / (max - min));
  const path = (
    points: [number, number][],
    top: number,
    min: number,
    max: number,
  ): string =>
    points
      .map(([s, v], i) => `${i ? "L" : "M"}${x(s)},${y(v, top, min, max)}`)
      .join(" ");
  const colors = ["#2a78d6", "#eb6834"];
  let markup = `<svg viewBox="0 0 ${width} ${height}" role="img" tabindex="0" aria-label="Interactive learning curves. Hover, touch, or use arrow keys to inspect training steps."><title>Training return and lap completion</title>`;
  for (const [top, min, max, title] of [
    [45, low, high, "Training return · smoothed per rollout"],
    [300, 0, 100, "Lap completion · stochastic evaluation"],
  ] as const) {
    markup += `<text x="${left}" y="${top - 20}" class="chart-title">${title}</text>`;
    for (let i = 0; i <= 4; i++) {
      const value = min + ((max - min) * i) / 4,
        py = y(value, top, min, max);
      markup += `<path d="M${left},${py}H${width - right}" class="chart-grid"/><text x="${left - 12}" y="${py + 4}" text-anchor="end">${Math.round(value)}${top === 300 ? "%" : ""}</text>`;
    }
  }
  for (const [i, [seed, points]] of Object.entries(site.curves).entries())
    markup += `<path class="chart-line" stroke="${colors[i % 2]}" d="${path(points, 45, low, high)}"/><text x="${left + i * 150}" y="226" fill="${colors[i % 2]}">Seed ${seed}</text>`;
  for (const [i, column] of [1, 2].entries()) {
    const points = site.snapshots.map(
      (s) => [s[0], s[column]] as [number, number],
    );
    markup += `<path class="chart-line" stroke="${colors[i]}" d="${path(points, 300, 0, 100)}"/>`;
    for (const [step, value] of points)
      markup += `<circle cx="${x(step)}" cy="${y(value, 300, 0, 100)}" r="4" fill="${colors[i]}"/>`;
  }
  for (let i = 0; i <= 6; i++) {
    const step = (maxStep * i) / 6;
    markup += `<text x="${x(step)}" y="480" text-anchor="middle">${(step / 1e6).toFixed(1)}M</text>`;
  }
  markup += `<text x="${left}" y="510" fill="${colors[0]}">Showcase</text><text x="${left + 150}" y="510" fill="${colors[1]}">Held-out</text><text x="${width - right}" y="${height - 10}" text-anchor="end">Training steps</text><path id="chart-crosshair" d="M${left},35V455"/></svg><output id="chart-tooltip" hidden></output>`;
  host.innerHTML = markup;
  const svg = host.querySelector("svg")!,
    tooltip = host.querySelector<HTMLOutputElement>("output")!,
    crosshair = host.querySelector("#chart-crosshair")!;
  let selectedStep = 0;
  const show = (step: number): void => {
    selectedStep = Math.max(0, Math.min(maxStep, step));
    const nearest = <T extends number[]>(rows: T[]): T =>
      rows.reduce((a, b) =>
        Math.abs(b[0] - selectedStep) < Math.abs(a[0] - selectedStep) ? b : a,
      );
    const snapshot = nearest(site.snapshots);
    const lines = [
      `${Math.round(selectedStep).toLocaleString("en-US")} steps`,
      ...Object.entries(site.curves).map(
        ([seed, points]) =>
          `Seed ${seed} return: ${nearest(points)[1].toFixed(1)}`,
      ),
      `Snapshot at ${snapshot[0].toLocaleString("en-US")} steps: ${snapshot[1].toFixed(1)}% showcase / ${snapshot[2].toFixed(1)}% held-out`,
    ];
    tooltip.textContent = lines.join("\n");
    tooltip.hidden = false;
    crosshair.setAttribute("d", `M${x(selectedStep)},35V455`);
    crosshair.classList.add("active");
  };
  svg.addEventListener("pointermove", (event) => {
    const box = svg.getBoundingClientRect();
    show(
      ((((event.clientX - box.left) * width) / box.width - left) / plotWidth) *
        maxStep,
    );
  });
  svg.addEventListener("pointerleave", () => {
    tooltip.hidden = true;
    crosshair.classList.remove("active");
  });
  svg.addEventListener("focus", () => show(selectedStep));
  svg.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      show(
        selectedStep + ((event.key === "ArrowRight" ? 1 : -1) * maxStep) / 100,
      );
    }
  });
}
