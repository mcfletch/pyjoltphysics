"""Vehicles on the world thread; one JSON line per run.

usage: python benchmarks/vehicles.py [--vehicles N] [--threads N ...] [--ticks T]

Each run is ``--vehicles`` cars (glisteel's car: four wheels, 1180 kg, a
single-gear electric drivetrain) on a 256 x 256 height field of 1 m cells and
a few metres of relief, driven in lockstep at 120 Hz with full throttle and a
steer that changes every 60 ticks. Reported in milliseconds per tick, from
the world's own timings after a second of warm-up: ``step_ms`` is
``PhysicsSystem::Update`` (the wheel casts, the vehicle constraints and the
chassis bodies), ``tick_ms`` the whole tick.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import types

import numpy as np

from pyjoltphysics import BodyBatch, VehicleSettings, World, WorldSettings


def heightfield(n: int = 256) -> np.ndarray:
    xs = (np.arange(n) - n / 2).astype(np.float64)
    x, z = np.meshgrid(xs, xs)
    return (2.0 * np.sin(x * 0.03) * np.cos(z * 0.04) + 0.3 * np.sin(x * 0.11 + z * 0.07)).astype(
        np.float32
    )


def car() -> VehicleSettings:
    tuning = types.SimpleNamespace(
        engine_force=20000.0,
        brake_force=16000.0,
        reverse_fraction=0.4,
        maximum_steer=0.52,
        steer_speed=3.2,
        rolling_resistance=0.02,
        downforce=6.0,
        base_speed=21.0,
        drag=0.9,
        steer_falloff_speed=10.0,
        holding=6000.0,
    )
    wheels = [
        types.SimpleNamespace(
            position=(side * 0.85, -0.02, -1.275 if front else 1.275),
            radius=0.33,
            suspension_travel=0.22,
            suspension_stiffness=26.0,
            suspension_damping=0.55,
            steering=front,
            driven=not front,
            braked=True,
            grip=1.9,
        )
        for front in (True, False)
        for side in (-1.0, 1.0)
    ]
    return VehicleSettings.from_raycast(tuning, wheels, 1180.0)


def run(vehicles: int, threads: int, ticks: int) -> dict[str, float | int | str]:
    world = World(WorldSettings(tick_hz=120.0, threads=threads))
    try:
        heights = heightfield()
        world.add_bodies(BodyBatch.heightfield(heights, origin=(-128.0, 0.0, -128.0), spacing=1.0))
        settings = car()
        cars = []
        for i in range(vehicles):
            angle = 2 * math.pi * i / vehicles
            position = (40.0 * math.cos(angle), 4.0, 40.0 * math.sin(angle))
            uids, _ = world.add_bodies(
                BodyBatch.boxes(
                    [position], size=(1.85, 0.62, 4.2), mass=1180.0, angular_damping=0.4
                )
            )
            cars.append(world.add_vehicle(int(uids[0]), settings))
        world.run_ticks(120)
        rng = np.random.default_rng(3)
        world.reset_stats()
        for _ in range(0, ticks, 60):
            ids = [c.id for c in cars]
            steer = rng.uniform(-0.6, 0.6, len(ids))
            world.vehicle_inputs(ids, [(0.6, float(s), 0.0, 0.0) for s in steer])
            world.run_ticks(60)
        stats = world.stats()
        return {
            'case': 'vehicles%d' % vehicles,
            'threads': threads,
            'ticks': ticks,
            'step_ms': round(float(stats['step_ms']), 4),
            'tick_ms': round(float(stats['tick_ms']), 4),
        }
    finally:
        world.stop()


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--vehicles', type=int, default=10)
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 8])
    parser.add_argument('--ticks', type=int, default=1200)
    args = parser.parse_args(argv)
    for threads in args.threads:
        print(json.dumps(run(args.vehicles, threads, args.ticks)), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
