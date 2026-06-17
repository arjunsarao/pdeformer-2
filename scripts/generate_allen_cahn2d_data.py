#!/usr/bin/env python
r"""Generate a small 2D Allen-Cahn dataset for PDEformer finetuning."""
import argparse
import os
from typing import Tuple

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/allen_cahn2d/allen_cahn2d.hdf5")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--nt", type=int, default=8,
                        help="Number of saved steps after the initial condition.")
    parser.add_argument("--substeps", type=int, default=20,
                        help="RK4 substeps between saved frames.")
    parser.add_argument("--epsilons", type=float, nargs="+",
                        default=[1.0e-3, 3.0e-3, 1.0e-2],
                        help="Diffusion coefficients to cycle over samples.")
    parser.add_argument("--rhos", type=float, nargs="+",
                        default=[1.0, 2.0, 5.0],
                        help="Reaction strengths to cycle over samples.")
    parser.add_argument("--amplitude", type=float, default=0.35)
    parser.add_argument("--bias", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=4321)
    return parser.parse_args()


def spectral_laplacian(field: np.ndarray,
                       kx2_plus_ky2: np.ndarray) -> np.ndarray:
    field_hat = np.fft.fftn(field)
    return np.fft.ifftn(-kx2_plus_ky2 * field_hat).real


def rhs(u: np.ndarray,
        epsilon: float,
        rho: float,
        kx2_plus_ky2: np.ndarray) -> np.ndarray:
    return epsilon * spectral_laplacian(u, kx2_plus_ky2) + rho * (u - u**3)


def rk4_step(u: np.ndarray,
             dt: float,
             epsilon: float,
             rho: float,
             kx2_plus_ky2: np.ndarray) -> np.ndarray:
    k1 = rhs(u, epsilon, rho, kx2_plus_ky2)
    k2 = rhs(u + 0.5 * dt * k1, epsilon, rho, kx2_plus_ky2)
    k3 = rhs(u + 0.5 * dt * k2, epsilon, rho, kx2_plus_ky2)
    k4 = rhs(u + dt * k3, epsilon, rho, kx2_plus_ky2)
    return u + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def stable_dt(epsilon: float,
              rho: float,
              dx: float,
              dy: float,
              requested_dt: float) -> float:
    diff_dt = 0.20 / (epsilon * (1 / dx**2 + 1 / dy**2)) if epsilon > 0 else requested_dt
    reaction_dt = 0.10 / max(rho, 1.0e-6)
    return min(requested_dt, diff_dt, reaction_dt)


def random_smooth_field(rng: np.random.Generator,
                        x: np.ndarray,
                        y: np.ndarray,
                        amplitude: float,
                        bias: float,
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
    return np.clip(field + bias, -0.95, 0.95)


def wave_numbers(nx: int, ny: int, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    return np.meshgrid(kx, ky, indexing="ij")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    x = np.linspace(0, 1, args.nx, endpoint=False, dtype=np.float32)
    y = np.linspace(0, 1, args.ny, endpoint=False, dtype=np.float32)
    t = np.linspace(0, 1, args.nt + 1, dtype=np.float32)
    dx = 1.0 / args.nx
    dy = 1.0 / args.ny
    frame_dt = 1.0 / args.nt
    requested_dt = 1.0 / (args.nt * args.substeps)
    kx, ky = wave_numbers(args.nx, args.ny, dx, dy)
    kx2_plus_ky2 = kx**2 + ky**2

    sol_u = np.empty((args.samples, args.nt + 1, args.nx, args.ny), dtype=np.float32)
    epsilon_arr = np.empty(args.samples, dtype=np.float32)
    rho_arr = np.empty(args.samples, dtype=np.float32)

    for idx in range(args.samples):
        epsilon = float(args.epsilons[idx % len(args.epsilons)])
        rho = float(args.rhos[(idx // len(args.epsilons)) % len(args.rhos)])
        epsilon_arr[idx] = epsilon
        rho_arr[idx] = rho
        u = random_smooth_field(rng, x, y, args.amplitude, args.bias)
        sol_u[idx, 0] = u

        for step in range(1, args.nt + 1):
            elapsed = 0.0
            while elapsed < frame_dt:
                dt = stable_dt(epsilon, rho, dx, dy,
                               min(requested_dt, frame_dt - elapsed))
                u = rk4_step(u, dt, epsilon, rho, kx2_plus_ky2)
                elapsed += dt
            if not np.isfinite(u).all():
                raise FloatingPointError(
                    "Allen-Cahn solver produced non-finite values. Try reducing "
                    "--rhos, --epsilons, or --amplitude, or increasing --substeps.")
            sol_u[idx, step] = np.clip(u, -1.5, 1.5)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with h5py.File(args.output, "w") as h5_file:
        h5_file.create_dataset("coord/t", data=t)
        h5_file.create_dataset("coord/x", data=x)
        h5_file.create_dataset("coord/y", data=y)
        h5_file.create_dataset("sol/u", data=sol_u, compression="gzip")
        h5_file.create_dataset("coef/u_ic", data=sol_u[:, 0], compression="gzip")
        h5_file.create_dataset("coef/epsilon", data=epsilon_arr)
        h5_file.create_dataset("coef/rho", data=rho_arr)
        h5_file.create_dataset("args/epsilons", data=np.array(args.epsilons, dtype=np.float32))
        h5_file.create_dataset("args/rhos", data=np.array(args.rhos, dtype=np.float32))
        h5_file.attrs["equation"] = "2D Allen-Cahn reaction-diffusion, periodic"

    print(f"saved {args.samples} samples to {args.output}")


if __name__ == "__main__":
    main()
