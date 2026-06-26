#!/usr/bin/env python
r"""Generate a small 2D Oldroyd-B/FENE-style viscoelastic flow dataset."""
import argparse
import os
from typing import Tuple

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/viscoelastic2d/viscoelastic2d.hdf5")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--nt", type=int, default=8,
                        help="Number of saved steps after the initial condition.")
    parser.add_argument("--substeps", type=int, default=160,
                        help="RK4 substeps between saved frames.")
    parser.add_argument("--viscosities", type=float, nargs="+",
                        default=[5.0e-4, 1.0e-3, 2.0e-3])
    parser.add_argument("--polymer-couplings", type=float, nargs="+",
                        default=[5.0e-4, 1.0e-3, 2.0e-3])
    parser.add_argument("--relaxation-times", type=float, nargs="+",
                        default=[2.0, 4.0, 8.0])
    parser.add_argument("--stress-diffusivities", type=float, nargs="+",
                        default=[5.0e-4, 1.0e-3])
    parser.add_argument("--fene-strengths", type=float, nargs="+",
                        default=[0.0, 0.01],
                        help="0 gives Oldroyd-B; positive values add a FENE-style trace correction.")
    parser.add_argument("--velocity-amplitude", type=float, default=0.06)
    parser.add_argument("--stress-amplitude", type=float, default=0.02)
    parser.add_argument("--max-speed", type=float, default=0.30,
                        help="Clip velocity components after each substep.")
    parser.add_argument("--max-conformation", type=float, default=2.0,
                        help="Clip Cxx and Cyy after each substep.")
    parser.add_argument("--max-shear-conformation", type=float, default=0.75,
                        help="Clip Cxy after each substep.")
    parser.add_argument("--seed", type=int, default=97531)
    return parser.parse_args()


def wave_numbers(nx: int, ny: int, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    return np.meshgrid(kx, ky, indexing="ij")


def spectral_derivatives(field: np.ndarray,
                         kx: np.ndarray,
                         ky: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    field_hat = np.fft.fftn(field)
    dx = np.fft.ifftn(1j * kx * field_hat).real
    dy = np.fft.ifftn(1j * ky * field_hat).real
    lap = np.fft.ifftn(-(kx * kx + ky * ky) * field_hat).real
    return dx, dy, lap


def random_smooth_field(rng: np.random.Generator,
                        x_grid: np.ndarray,
                        y_grid: np.ndarray,
                        amplitude: float,
                        modes: int = 3) -> np.ndarray:
    field = np.zeros_like(x_grid)
    for kx_mode in range(1, modes + 1):
        for ky_mode in range(1, modes + 1):
            scale = amplitude / (kx_mode * kx_mode + ky_mode * ky_mode)
            phase_x = rng.uniform(0, 2 * np.pi)
            phase_y = rng.uniform(0, 2 * np.pi)
            field += scale * rng.normal() * np.sin(2 * np.pi * kx_mode * x_grid + phase_x)
            field += scale * rng.normal() * np.cos(2 * np.pi * ky_mode * y_grid + phase_y)
    field -= field.mean()
    max_abs = np.max(np.abs(field))
    if max_abs > 0:
        field *= amplitude / max_abs
    return field


def random_initial_condition(rng: np.random.Generator,
                             x: np.ndarray,
                             y: np.ndarray,
                             velocity_amplitude: float,
                             stress_amplitude: float) -> Tuple[np.ndarray, ...]:
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    stream = random_smooth_field(rng, x_grid, y_grid, velocity_amplitude, modes=4)
    psi_x, psi_y, _ = spectral_derivatives(
        stream,
        *wave_numbers(x.size, y.size, 1.0 / x.size, 1.0 / y.size),
    )
    u = psi_y
    v = -psi_x
    max_speed = max(np.max(np.abs(u)), np.max(np.abs(v)), 1.0e-8)
    u *= velocity_amplitude / max_speed
    v *= velocity_amplitude / max_speed

    cxx = 1.0 + random_smooth_field(rng, x_grid, y_grid, stress_amplitude, modes=3)
    cxy = random_smooth_field(rng, x_grid, y_grid, 0.5 * stress_amplitude, modes=3)
    cyy = 1.0 + random_smooth_field(rng, x_grid, y_grid, stress_amplitude, modes=3)
    cxx = np.clip(cxx, 0.35, 1.8)
    cyy = np.clip(cyy, 0.35, 1.8)
    return u, v, cxx, cxy, cyy


def rhs(state: Tuple[np.ndarray, ...],
        viscosity: float,
        polymer_coupling: float,
        inv_relaxation_time: float,
        stress_diffusivity: float,
        fene_strength: float,
        kx: np.ndarray,
        ky: np.ndarray) -> Tuple[np.ndarray, ...]:
    u, v, cxx, cxy, cyy = state

    ux, uy, lap_u = spectral_derivatives(u, kx, ky)
    vx, vy, lap_v = spectral_derivatives(v, kx, ky)
    cxx_x, cxx_y, lap_cxx = spectral_derivatives(cxx, kx, ky)
    cxy_x, cxy_y, lap_cxy = spectral_derivatives(cxy, kx, ky)
    cyy_x, cyy_y, lap_cyy = spectral_derivatives(cyy, kx, ky)

    trace_shift = cxx + cyy - 2.0
    fene_correction = fene_strength * trace_shift

    adv_u = u * ux + v * uy
    adv_v = u * vx + v * vy
    du_dt = -adv_u + viscosity * lap_u + polymer_coupling * (cxx_x + cxy_y)
    dv_dt = -adv_v + viscosity * lap_v + polymer_coupling * (cxy_x + cyy_y)

    adv_cxx = u * cxx_x + v * cxx_y
    adv_cxy = u * cxy_x + v * cxy_y
    adv_cyy = u * cyy_x + v * cyy_y

    stretch_cxx = 2.0 * (cxx * ux + cxy * uy)
    stretch_cxy = cxx * vx + cxy * vy + cxy * ux + cyy * uy
    stretch_cyy = 2.0 * (cxy * vx + cyy * vy)

    relax_cxx = inv_relaxation_time * (cxx - 1.0 + fene_correction * cxx)
    relax_cxy = inv_relaxation_time * (cxy + fene_correction * cxy)
    relax_cyy = inv_relaxation_time * (cyy - 1.0 + fene_correction * cyy)

    dcxx_dt = -adv_cxx + stretch_cxx - relax_cxx + stress_diffusivity * lap_cxx
    dcxy_dt = -adv_cxy + stretch_cxy - relax_cxy + stress_diffusivity * lap_cxy
    dcyy_dt = -adv_cyy + stretch_cyy - relax_cyy + stress_diffusivity * lap_cyy
    return du_dt, dv_dt, dcxx_dt, dcxy_dt, dcyy_dt


def rk4_step(state: Tuple[np.ndarray, ...],
             dt: float,
             viscosity: float,
             polymer_coupling: float,
             inv_relaxation_time: float,
             stress_diffusivity: float,
             fene_strength: float,
             kx: np.ndarray,
             ky: np.ndarray) -> Tuple[np.ndarray, ...]:
    args = (viscosity, polymer_coupling, inv_relaxation_time,
            stress_diffusivity, fene_strength, kx, ky)
    k1 = rhs(state, *args)
    k2_state = tuple(s + 0.5 * dt * k for s, k in zip(state, k1))
    k2 = rhs(k2_state, *args)
    k3_state = tuple(s + 0.5 * dt * k for s, k in zip(state, k2))
    k3 = rhs(k3_state, *args)
    k4_state = tuple(s + dt * k for s, k in zip(state, k3))
    k4 = rhs(k4_state, *args)
    return tuple(s + dt * (a + 2 * b + 2 * c + d) / 6
                 for s, a, b, c, d in zip(state, k1, k2, k3, k4))


def project_state(state: Tuple[np.ndarray, ...],
                  max_speed: float,
                  max_conformation: float,
                  max_shear_conformation: float) -> Tuple[np.ndarray, ...]:
    u, v, cxx, cxy, cyy = state
    return (
        np.clip(u, -max_speed, max_speed),
        np.clip(v, -max_speed, max_speed),
        np.clip(cxx, 0.10, max_conformation),
        np.clip(cxy, -max_shear_conformation, max_shear_conformation),
        np.clip(cyy, 0.10, max_conformation),
    )


def stable_dt(viscosity: float,
              stress_diffusivity: float,
              inv_relaxation_time: float,
              velocity_amplitude: float,
              max_velocity_gradient: float,
              max_conformation: float,
              dx: float,
              dy: float,
              requested_dt: float) -> float:
    diff = max(viscosity, stress_diffusivity)
    diff_dt = 0.20 / (diff * (1 / dx**2 + 1 / dy**2)) if diff > 0 else requested_dt
    adv_dt = 0.20 * min(dx, dy) / max(velocity_amplitude, 1.0e-6)
    relax_dt = 0.10 / max(inv_relaxation_time, 1.0e-6)
    stretch_dt = 0.05 / max(max_velocity_gradient * max_conformation, 1.0e-6)
    return min(requested_dt, diff_dt, adv_dt, relax_dt, stretch_dt)


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

    names = ["u", "v", "cxx", "cxy", "cyy"]
    sol = {
        name: np.empty((args.samples, args.nt + 1, args.nx, args.ny), dtype=np.float32)
        for name in names
    }
    viscosity_arr = np.empty(args.samples, dtype=np.float32)
    coupling_arr = np.empty(args.samples, dtype=np.float32)
    relaxation_arr = np.empty(args.samples, dtype=np.float32)
    stress_diff_arr = np.empty(args.samples, dtype=np.float32)
    fene_arr = np.empty(args.samples, dtype=np.float32)

    for idx in range(args.samples):
        viscosity = float(args.viscosities[idx % len(args.viscosities)])
        polymer_coupling = float(args.polymer_couplings[(idx // len(args.viscosities)) % len(args.polymer_couplings)])
        relaxation_time = float(args.relaxation_times[(idx // (len(args.viscosities) * len(args.polymer_couplings))) % len(args.relaxation_times)])
        stress_diffusivity = float(args.stress_diffusivities[(idx // (len(args.viscosities) * len(args.polymer_couplings) * len(args.relaxation_times))) % len(args.stress_diffusivities)])
        fene_strength = float(args.fene_strengths[(idx // (len(args.viscosities) * len(args.polymer_couplings) * len(args.relaxation_times) * len(args.stress_diffusivities))) % len(args.fene_strengths)])
        inv_relaxation_time = 1.0 / relaxation_time

        viscosity_arr[idx] = viscosity
        coupling_arr[idx] = polymer_coupling
        relaxation_arr[idx] = relaxation_time
        stress_diff_arr[idx] = stress_diffusivity
        fene_arr[idx] = fene_strength

        state = random_initial_condition(
            rng, x, y, args.velocity_amplitude, args.stress_amplitude)
        for name, field in zip(names, state):
            sol[name][idx, 0] = field

        for step in range(1, args.nt + 1):
            elapsed = 0.0
            while elapsed < frame_dt:
                ux, uy, _ = spectral_derivatives(state[0], kx, ky)
                vx, vy, _ = spectral_derivatives(state[1], kx, ky)
                max_speed = max(np.max(np.abs(state[0])), np.max(np.abs(state[1])))
                max_velocity_gradient = max(
                    np.max(np.abs(ux)), np.max(np.abs(uy)),
                    np.max(np.abs(vx)), np.max(np.abs(vy)))
                max_conformation = max(
                    np.max(np.abs(state[2])), np.max(np.abs(state[3])),
                    np.max(np.abs(state[4])))
                dt = stable_dt(viscosity, stress_diffusivity, inv_relaxation_time,
                               max_speed, max_velocity_gradient, max_conformation,
                               dx, dy, min(requested_dt, frame_dt - elapsed))
                state = rk4_step(state, dt, viscosity, polymer_coupling,
                                 inv_relaxation_time, stress_diffusivity,
                                 fene_strength, kx, ky)
                if not all(np.isfinite(field).all() for field in state):
                    raise FloatingPointError(
                        "Viscoelastic solver produced non-finite values at "
                        f"sample={idx}, saved_step={step}, elapsed={elapsed:.6g}, "
                        f"dt={dt:.6g}, nu={viscosity:g}, beta={polymer_coupling:g}, "
                        f"lambda={relaxation_time:g}, kappa={stress_diffusivity:g}, "
                        f"chi={fene_strength:g}. Try increasing --substeps or "
                        "reducing --velocity-amplitude/--stress-amplitude.")
                state = project_state(
                    state, args.max_speed, args.max_conformation,
                    args.max_shear_conformation)
                elapsed += dt
            if not all(np.isfinite(field).all() for field in state):
                raise FloatingPointError(
                    "Viscoelastic solver produced non-finite values. Try reducing "
                    "amplitudes/couplings or increasing --substeps.")
            for name, field in zip(names, state):
                sol[name][idx, step] = field

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with h5py.File(args.output, "w") as h5_file:
        h5_file.create_dataset("coord/t", data=t)
        h5_file.create_dataset("coord/x", data=x)
        h5_file.create_dataset("coord/y", data=y)
        for name in names:
            h5_file.create_dataset(f"sol/{name}", data=sol[name], compression="gzip")
            h5_file.create_dataset(f"coef/{name}_ic", data=sol[name][:, 0], compression="gzip")
        h5_file.create_dataset("coef/viscosity", data=viscosity_arr)
        h5_file.create_dataset("coef/polymer_coupling", data=coupling_arr)
        h5_file.create_dataset("coef/relaxation_time", data=relaxation_arr)
        h5_file.create_dataset("coef/stress_diffusivity", data=stress_diff_arr)
        h5_file.create_dataset("coef/fene_strength", data=fene_arr)
        h5_file.create_dataset("args/viscosities", data=np.array(args.viscosities, dtype=np.float32))
        h5_file.create_dataset("args/polymer_couplings", data=np.array(args.polymer_couplings, dtype=np.float32))
        h5_file.create_dataset("args/relaxation_times", data=np.array(args.relaxation_times, dtype=np.float32))
        h5_file.create_dataset("args/stress_diffusivities", data=np.array(args.stress_diffusivities, dtype=np.float32))
        h5_file.create_dataset("args/fene_strengths", data=np.array(args.fene_strengths, dtype=np.float32))
        h5_file.attrs["equation"] = "2D Oldroyd-B/FENE-style viscoelastic flow, periodic"

    print(f"saved {args.samples} samples to {args.output}")


if __name__ == "__main__":
    main()
