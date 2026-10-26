"""Pygame manual driving. Rendering/input are separate from fixed-step simulation."""

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from sim.config import PhysicsConfig, load_physics, load_tracks  # noqa: E402
from sim.geometry import FloatArray  # noqa: E402
from sim.physics import Action  # noqa: E402
from sim.session import DriveSession  # noqa: E402
from sim.track import Track, generate_track, showcase_track  # noqa: E402


@dataclass(frozen=True)
class View:
    center_x: float
    center_y: float
    scale: float
    width: int
    height: int

    def point(self, x: float, y: float) -> tuple[int, int]:
        return (
            round(self.width / 2.0 + (x - self.center_x) * self.scale),
            round(self.height / 2.0 - (y - self.center_y) * self.scale),
        )

    def polyline(self, points: FloatArray) -> list[tuple[int, int]]:
        return [self.point(float(x), float(y)) for x, y in points]


def make_view(track: Track, width: int, height: int) -> View:
    geometry = np.vstack((track.left, track.right))
    low, high = geometry.min(axis=0), geometry.max(axis=0)
    center = (low + high) / 2.0
    scale = min((width - 100) / (high[0] - low[0]), (height - 220) / (high[1] - low[1]))
    return View(float(center[0]), float(center[1]) + 45.0 / scale, float(scale), width, height)


def road_surface(track: Track, view: View) -> pygame.Surface:
    surface = pygame.Surface((view.width, view.height))
    surface.fill((18, 29, 28))
    left, right = view.polyline(track.left), view.polyline(track.right)
    for i in range(len(left)):
        j = (i + 1) % len(left)
        pygame.draw.polygon(surface, (51, 57, 65), [left[i], left[j], right[j], right[i]])
    pygame.draw.lines(surface, (199, 207, 203), True, left, 2)
    pygame.draw.lines(surface, (199, 207, 203), True, right, 2)
    for arc in track.checkpoints[1:]:
        point, heading = track.sample(float(arc))
        normal = np.array([-math.sin(heading), math.cos(heading)]) * track.width / 2.0
        pygame.draw.line(
            surface, (86, 112, 117), view.point(*(point - normal)), view.point(*(point + normal))
        )
    for i in range(10):
        fraction = i / 10.0
        p0 = track.right[0] + fraction * (track.left[0] - track.right[0])
        p1 = track.right[0] + (fraction + 0.1) * (track.left[0] - track.right[0])
        pygame.draw.line(
            surface,
            (241, 246, 240) if i % 2 == 0 else (19, 25, 29),
            view.point(*p0),
            view.point(*p1),
            5,
        )
    return surface


def draw_car(surface: pygame.Surface, session: DriveSession, view: View) -> None:
    state, c = session.state, session.physics
    forward = np.array([math.cos(state.psi), math.sin(state.psi)])
    left = np.array([-math.sin(state.psi), math.cos(state.psi)])
    center = np.array([state.x, state.y])
    corners = [
        center + f * c.car_length / 2.0 * forward + s * c.car_width / 2.0 * left
        for f, s in ((1, 1), (1, -1), (-1, -1), (-1, 1))
    ]
    color = (252, 113, 92) if session.crashed else (68, 235, 207)
    pygame.draw.polygon(surface, color, view.polyline(np.array(corners)))
    nose = center + c.car_length / 2.0 * forward
    pygame.draw.line(surface, (250, 253, 243), view.point(*center), view.point(*nose), 2)


def draw_frame(
    surface: pygame.Surface,
    road: pygame.Surface,
    session: DriveSession,
    view: View,
    font: pygame.font.Font,
    large_font: pygame.font.Font,
) -> None:
    surface.blit(road, (0, 0))
    draw_car(surface, session, view)
    pygame.draw.rect(surface, (13, 21, 25), (0, 0, view.width, 110))
    best = min(session.progress.lap_times, default=None)
    title = f"{session.track.name}   /   {session.state.speed * 3.6:05.1f} km/h"
    timing = (
        f"LAP {session.progress.laps + 1}    {session.lap_time:06.2f}s    "
        f"BEST {'—' if best is None else f'{best:.2f}s'}    "
        f"CHECKPOINT {session.progress.checkpoint_index}/{len(session.track.checkpoints) - 1}"
    )
    surface.blit(large_font.render(title, True, (223, 243, 237)), (24, 14))
    surface.blit(font.render(timing, True, (158, 188, 184)), (24, 55))
    surface.blit(
        font.render(
            "WASD / arrows: drive    Space: brake    R: respawn    Enter: restart    Esc: quit",
            True,
            (158, 188, 184),
        ),
        (24, 83),
    )
    status = "RESPAWNING…" if session.crashed else "WRONG WAY" if session.progress.wrong_way else ""
    if status:
        text = large_font.render(status, True, (255, 150, 115))
        surface.blit(text, text.get_rect(center=(view.width // 2, view.height - 45)))


def keyboard_action() -> Action:
    keys = pygame.key.get_pressed()
    steer = float(keys[pygame.K_LEFT] or keys[pygame.K_a]) - float(
        keys[pygame.K_RIGHT] or keys[pygame.K_d]
    )
    throttle = float(keys[pygame.K_UP] or keys[pygame.K_w]) - float(
        keys[pygame.K_DOWN] or keys[pygame.K_s]
    )
    if keys[pygame.K_SPACE]:
        throttle = -1.0
    return Action(steer, throttle)


def run(
    track: Track,
    physics: PhysicsConfig,
    *,
    headless: bool = False,
    steps: int = 600,
    screenshot: Path | None = None,
) -> DriveSession:
    tracks, _ = load_tracks()
    session = DriveSession(track, physics, tracks)
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.display.init()
    pygame.font.init()
    try:
        surface = pygame.display.set_mode((1200, 850))
        pygame.display.set_caption(f"Racing RL · M1 · {track.name} (seed {track.seed})")
        font = pygame.font.Font(None, 23)
        large_font = pygame.font.Font(None, 34)
        view = make_view(track, *surface.get_size())
        road = road_surface(track, view)
        clock = pygame.time.Clock()
        accumulator = 0.0
        action = Action()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        session.respawn()
                    elif event.key == pygame.K_RETURN:
                        session.restart()
            if headless:
                if session.ticks >= steps:
                    break
                # Deterministic scripted controls exercise rendering, motion and respawn.
                if session.ticks % physics.action_repeat == 0:
                    action = Action(0.3 * math.sin(session.time), 1.0)
                session.tick(action)
            else:
                accumulator += min(clock.tick(60) / 1000.0, 0.25)
                while accumulator >= physics.dt:
                    if session.ticks % physics.action_repeat == 0:
                        action = keyboard_action()
                    session.tick(action)
                    accumulator -= physics.dt
            draw_frame(surface, road, session, view, font, large_font)
            pygame.display.flip()
        if screenshot is not None:
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(surface, str(screenshot))
    finally:
        pygame.quit()
    return session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--seed", type=int, help="Generate a track from this seed")
    group.add_argument("--showcase", help="Showcase id from configs/tracks.yaml (default: sweep)")
    group.add_argument("--track", type=Path, help="Load an exported track.v1 JSON")
    parser.add_argument("--export-track", type=Path, help="Save the loaded/generated geometry")
    parser.add_argument(
        "--headless", action="store_true", help="Run a deterministic SDL smoke check"
    )
    parser.add_argument("--steps", type=int, default=600, help="Physics ticks for --headless")
    parser.add_argument("--screenshot", type=Path, help="Save the final frame")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    physics = load_physics()
    config, showcases = load_tracks()
    if args.track is not None:
        track = Track.load(args.track)
    elif args.seed is not None:
        track = generate_track(args.seed, config, physics)
    else:
        selected = next((item for item in showcases if item.id == (args.showcase or "sweep")), None)
        if selected is None:
            parser.error(f"--showcase must be one of: {', '.join(item.id for item in showcases)}")
        track = showcase_track(selected, config, physics)
    if args.export_track is not None:
        track.save(args.export_track)
    session = run(
        track, physics, headless=args.headless, steps=args.steps, screenshot=args.screenshot
    )
    print(
        f"seed={track.seed} ticks={session.ticks} laps={session.progress.laps} "
        f"crashes={session.crashes} speed={session.state.speed:.3f} m/s"
    )


if __name__ == "__main__":
    main()
