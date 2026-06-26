#!/usr/bin/env python
r"""Generate a small 2D Gray-Scott dataset for PDEformer finetuning."""
import argparse
import os
from typing import Tuple

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/grayscott2d/grayscott2d.hdf5")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--nt", type=int, default=8,
                        help="Number of saved steps after the initial condition.")
    parser.add_argument("--substeps", type=int, default=20,
                        help="RK4 substeps between saved frames.")
    parser.add_argument("--du-values", type=float, nargs="+",
                        default=[2.0e-5, 5.0e-5, 1.0e-4],
                        help="u diffusion coefficients to cycle over samples.")
    parser.add_argument("--dv-values", type=float, nargs="+",
                        default=[1.0e-5, 2.5e-5, 5.0e-5],
                        help="v diffusion coefficients to cycle over samples.")
    parser.add_argument("--feed-values", type=float, nargs="+",
                        default=[0.020, 0.035, 0.050],
                        help="Feed rates F to cycle over samples.")
    parser.add_argument("--kill-values", type=float, nargs="+",
                        default=[0.045, 0.055, 0.065],
                        help="Kill rates k to cycle over samples.")
    parser.add_argument("--num-blobs", type=int, default=6)
    parser.add_argument("--blob-radius", type=float, default=0.055)
    parser.add_argument("--noise", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=2468)
    return parser.parse_args()


def spectral_laplacian(field: np.ndarray,
                       kx2_plus_ky2: np.ndarray) -> np.ndarray:
    field_hat = np.fft.fftn(field)
    return np.fft.ifftn(-kx2_plus_ky2 * field_hat).real


def rhs(u: np.ndarray,
        v: np.ndarray,
        du: float,
        dv: float,
        feed: float,
        kill: float,
        kx2_plus_ky2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    uv2 = u * v * v
    du_dt = du * spectral_laplacian(u, kx2_plus_ky2) - uv2 + feed * (1 - u)
    dv_dt = dv * spectral_laplacian(v, kx2_plus_ky2) + uv2 - (feed + kill) * v
    return du_dt, dv_dt


def rk4_step(u: np.ndarray,
             v: np.ndarray,
             dt: float,
             du: float,
             dv: float,
             feed: float,
             kill: float,
             kx2_plus_ky2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    k1u, k1v = rhs(u, v, du, dv, feed, kill, kx2_plus_ky2)
    k2u, k2v = rhs(u + 0.5 * dt * k1u, v + 0.5 * dt * k1v,
                   du, dv, feed, kill, kx2_plus_ky2)
    k3u, k3v = rhs(u + 0.5 * dt * k2u, v + 0.5 * dt * k2v,
                   du, dv, feed, kill, kx2_plus_ky2)
    k4u, k4v = rhs(u + dt * k3u, v + dt * k3v,
                   du, dv, feed, kill, kx2_plus_ky2)
    u_next = u + dt * (k1u + 2 * k2u + 2 * k3u + k4u) / 6
    v_next = v + dt * (k1v + 2 * k2v + 2 * k3v + k4v) / 6
    return u_next, v_next


def stable_dt(du: float,
              dv: float,
              feed: float,
              kill: float,
              dx: float,
              dy: float,
              requested_dt: float) -> float:
    max_diffusion = max(du, dv)
    if max_diffusion > 0:
        diff_dt = 0.20 / (max_diffusion * (1 / dx**2 + 1 / dy**2))
    else:
        diff_dt = requested_dt
    reaction_dt = 0.10 / max(feed + kill, 1.0e-6)
    return min(requested_dt, diff_dt, reaction_dt)


def random_initial_condition(rng: np.random.Generator,
                             x: np.ndarray,
                             y: np.ndarray,
                             num_blobs: int,
                             blob_radius: float,
                             noise: float) -> Tuple[np.ndarray, np.ndarray]:
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    u = np.ones_like(x_grid)
    v = np.zeros_like(x_grid)

    for _ in range(num_blobs):
        cx = rng.uniform(0, 1)
        cy = rng.uniform(0, 1)
        radius = blob_radius * rng.uniform(0.7, 1.3)
        dx = np.minimum(np.abs(x_grid - cx), 1 - np.abs(x_grid - cx))
        dy = np.minimum(np.abs(y_grid - cy), 1 - np.abs(y_grid - cy))
        bump = np.exp(-(dx * dx + dy * dy) / (2 * radius * radius))
        v += rng.uniform(0.18, 0.32) * bump
        u -= rng.uniform(0.35, 0.60) * bump

    if noise > 0:
        u += noise * rng.normal(size=u.shape)
        v += noise * rng.normal(size=v.shape)
    return np.clip(u, 0.0, 1.2), np.clip(v, 0.0, 1.0)


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
    sol_v = np.empty_like(sol_u)
    du_arr = np.empty(args.samples, dtype=np.float32)
    dv_arr = np.empty(args.samples, dtype=np.float32)
    feed_arr = np.empty(args.samples, dtype=np.float32)
    kill_arr = np.empty(args.samples, dtype=np.float32)

    for idx in range(args.samples):
        du = float(args.du_values[idx % len(args.du_values)])
        dv = float(args.dv_values[(idx // len(args.du_values)) % len(args.dv_values)])
        feed = float(args.feed_values[(idx // (len(args.du_values) * len(args.dv_values))) % len(args.feed_values)])
        kill = float(args.kill_values[(idx // (len(args.du_values) * len(args.dv_values) * len(args.feed_values))) % len(args.kill_values)])
        du_arr[idx] = du
        dv_arr[idx] = dv
        feed_arr[idx] = feed
        kill_arr[idx] = kill

        u, v = random_initial_condition(
            rng, x, y, args.num_blobs, args.blob_radius, args.noise)
        sol_u[idx, 0] = u
        sol_v[idx, 0] = v

        for step in range(1, args.nt + 1):
            elapsed = 0.0
            while elapsed < frame_dt:
                dt = stable_dt(du, dv, feed, kill, dx, dy,
                               min(requested_dt, frame_dt - elapsed))
                u, v = rk4_step(u, v, dt, du, dv, feed, kill, kx2_plus_ky2)
                elapsed += dt
            if not np.isfinite(u).all() or not np.isfinite(v).all():
                raise FloatingPointError(
                    "Gray-Scott solver produced non-finite values. Try reducing "
                    "the rates/noise, or increasing --substeps.")
            sol_u[idx, step] = np.clip(u, 0.0, 1.2)
            sol_v[idx, step] = np.clip(v, 0.0, 1.0)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with h5py.File(args.output, "w") as h5_file:
        h5_file.create_dataset("coord/t", data=t)
        h5_file.create_dataset("coord/x", data=x)
        h5_file.create_dataset("coord/y", data=y)
        h5_file.create_dataset("sol/u", data=sol_u, compression="gzip")
        h5_file.create_dataset("sol/v", data=sol_v, compression="gzip")
        h5_file.create_dataset("coef/u_ic", data=sol_u[:, 0], compression="gzip")
        h5_file.create_dataset("coef/v_ic", data=sol_v[:, 0], compression="gzip")
        h5_file.create_dataset("coef/du", data=du_arr)
        h5_file.create_dataset("coef/dv", data=dv_arr)
        h5_file.create_dataset("coef/feed", data=feed_arr)
        h5_file.create_dataset("coef/kill", data=kill_arr)
        h5_file.create_dataset("args/du_values", data=np.array(args.du_values, dtype=np.float32))
        h5_file.create_dataset("args/dv_values", data=np.array(args.dv_values, dtype=np.float32))
        h5_file.create_dataset("args/feed_values", data=np.array(args.feed_values, dtype=np.float32))
        h5_file.create_dataset("args/kill_values", data=np.array(args.kill_values, dtype=np.float32))
        h5_file.attrs["equation"] = "2D Gray-Scott reaction-diffusion, periodic"

    print(f"saved {args.samples} samples to {args.output}")


if __name__ == "__main__":
    main()
