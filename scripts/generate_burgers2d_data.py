#!/usr/bin/env python
r"""Generate a small 2D viscous Burgers dataset for PDEformer finetuning."""
import argparse
import os
from typing import Tuple

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/burgers2d/burgers2d_viscous.hdf5")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--nt", type=int, default=8,
                        help="Number of saved steps after the initial condition.")
    parser.add_argument("--substeps", type=int, default=10,
                        help="RK4 substeps between saved frames.")
    parser.add_argument("--viscosities", type=float, nargs="+",
                        default=[1.0e-2, 5.0e-3, 1.0e-3],
                        help="Viscosity values to cycle over across samples.")
    parser.add_argument("--amplitude", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def ddx_periodic(field: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) / (2 * dx)


def ddy_periodic(field: np.ndarray, dy: float) -> np.ndarray:
    return (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / (2 * dy)


def ddx_upwind(field: np.ndarray, velocity: np.ndarray, dx: float) -> np.ndarray:
    backward = (field - np.roll(field, 1, axis=0)) / dx
    forward = (np.roll(field, -1, axis=0) - field) / dx
    return np.where(velocity >= 0, backward, forward)


def ddy_upwind(field: np.ndarray, velocity: np.ndarray, dy: float) -> np.ndarray:
    backward = (field - np.roll(field, 1, axis=1)) / dy
    forward = (np.roll(field, -1, axis=1) - field) / dy
    return np.where(velocity >= 0, backward, forward)


def lap_periodic(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return ((np.roll(field, -1, axis=0) - 2 * field + np.roll(field, 1, axis=0)) / dx**2
            + (np.roll(field, -1, axis=1) - 2 * field + np.roll(field, 1, axis=1)) / dy**2)


def rhs(u: np.ndarray, v: np.ndarray, nu: float, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
    u_x = ddx_upwind(u, u, dx)
    u_y = ddy_upwind(u, v, dy)
    v_x = ddx_upwind(v, u, dx)
    v_y = ddy_upwind(v, v, dy)
    du = -(u * u_x + v * u_y) + nu * lap_periodic(u, dx, dy)
    dv = -(u * v_x + v * v_y) + nu * lap_periodic(v, dx, dy)
    return du, dv


def rk4_step(u: np.ndarray, v: np.ndarray, dt: float, nu: float, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
    k1u, k1v = rhs(u, v, nu, dx, dy)
    k2u, k2v = rhs(u + 0.5 * dt * k1u, v + 0.5 * dt * k1v, nu, dx, dy)
    k3u, k3v = rhs(u + 0.5 * dt * k2u, v + 0.5 * dt * k2v, nu, dx, dy)
    k4u, k4v = rhs(u + dt * k3u, v + dt * k3v, nu, dx, dy)
    u_next = u + dt * (k1u + 2 * k2u + 2 * k3u + k4u) / 6
    v_next = v + dt * (k1v + 2 * k2v + 2 * k3v + k4v) / 6
    return u_next, v_next


def stable_dt(u: np.ndarray, v: np.ndarray, nu: float, dx: float, dy: float, requested_dt: float) -> float:
    max_speed = max(float(np.max(np.abs(u))), float(np.max(np.abs(v))), 1.0e-6)
    adv_dt = 0.35 * min(dx, dy) / max_speed
    diff_dt = 0.20 / (nu * (1 / dx**2 + 1 / dy**2)) if nu > 0 else requested_dt
    return min(requested_dt, adv_dt, diff_dt)


def random_smooth_field(rng: np.random.Generator,
                        x: np.ndarray,
                        y: np.ndarray,
                        amplitude: float,
                        modes: int = 4) -> np.ndarray:
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    field = np.zeros_like(x_grid)
    for kx in range(1, modes + 1):
        for ky in range(1, modes + 1):
            scale = amplitude / (kx * kx + ky * ky)
            phase_x = rng.uniform(0, 2 * np.pi)
            phase_y = rng.uniform(0, 2 * np.pi)
            field += scale * rng.normal() * np.sin(2 * np.pi * kx * x_grid + phase_x)
            field += scale * rng.normal() * np.cos(2 * np.pi * ky * y_grid + phase_y)
    field -= field.mean()
    max_abs = np.max(np.abs(field))
    if max_abs > 0:
        field *= amplitude / max_abs
    return field


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    x = np.linspace(0, 1, args.nx, endpoint=False, dtype=np.float32)
    y = np.linspace(0, 1, args.ny, endpoint=False, dtype=np.float32)
    t = np.linspace(0, 1, args.nt + 1, dtype=np.float32)
    dx = 1.0 / args.nx
    dy = 1.0 / args.ny
    requested_dt = 1.0 / (args.nt * args.substeps)
    frame_dt = 1.0 / args.nt

    sol_u = np.empty((args.samples, args.nt + 1, args.nx, args.ny), dtype=np.float32)
    sol_v = np.empty_like(sol_u)
    viscosity = np.empty(args.samples, dtype=np.float32)

    for idx in range(args.samples):
        nu = float(args.viscosities[idx % len(args.viscosities)])
        viscosity[idx] = nu
        u = random_smooth_field(rng, x, y, args.amplitude)
        v = random_smooth_field(rng, x, y, args.amplitude)
        sol_u[idx, 0] = u
        sol_v[idx, 0] = v

        for step in range(1, args.nt + 1):
            elapsed = 0.0
            while elapsed < frame_dt:
                dt = stable_dt(u, v, nu, dx, dy,
                               min(requested_dt, frame_dt - elapsed))
                u, v = rk4_step(u, v, dt, nu, dx, dy)
                elapsed += dt
            if not np.isfinite(u).all() or not np.isfinite(v).all():
                raise FloatingPointError(
                    "Burgers solver produced non-finite values. Try reducing "
                    "--amplitude or increasing --substeps.")
            sol_u[idx, step] = u
            sol_v[idx, step] = v

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with h5py.File(args.output, "w") as h5_file:
        h5_file.create_dataset("coord/t", data=t)
        h5_file.create_dataset("coord/x", data=x)
        h5_file.create_dataset("coord/y", data=y)
        h5_file.create_dataset("sol/u", data=sol_u, compression="gzip")
        h5_file.create_dataset("sol/v", data=sol_v, compression="gzip")
        h5_file.create_dataset("coef/u_ic", data=sol_u[:, 0], compression="gzip")
        h5_file.create_dataset("coef/v_ic", data=sol_v[:, 0], compression="gzip")
        h5_file.create_dataset("coef/viscosity", data=viscosity)
        h5_file.create_dataset("args/viscosities", data=np.array(args.viscosities, dtype=np.float32))
        h5_file.attrs["equation"] = "2D viscous Burgers, periodic"

    print(f"saved {args.samples} samples to {args.output}")


if __name__ == "__main__":
    main()
