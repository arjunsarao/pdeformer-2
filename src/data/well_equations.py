r"""Dataset-specific PDEformer DAGs for Polymathic AI's The Well datasets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator

from .pde_dag import PDEAsDAG, PDENode, PDENodesCollector


Array = NDArray[np.float32]
Builder = Callable[[PDENodesCollector, List[PDENode], List[str], Dict[str, Any]], None]


@dataclass(frozen=True)
class WellEquationSpec:
    r"""Equation metadata and DAG construction recipe for one Well dataset."""

    dataset: str
    aliases: Tuple[str, ...]
    pde_latex: str
    builder: Builder
    spatial_dims: int = 2
    notes: str = ""

    @property
    def all_names(self) -> Tuple[str, ...]:
        return (self.dataset, *self.aliases)


def normalize_axis(values: NDArray[np.floating]) -> Array:
    r"""Map one coordinate axis to [0, 1], preserving constants."""
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1.0e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def normalize_space_grid(space_grid: NDArray[np.floating], enabled: bool = True) -> Array:
    r"""Normalize each spatial coordinate in a Well grid independently."""
    if not enabled:
        return space_grid.astype(np.float32)
    out = np.empty_like(space_grid, dtype=np.float32)
    for axis in range(space_grid.shape[-1]):
        out[..., axis] = normalize_axis(space_grid[..., axis])
    return out


def normalize_time_grid(time_grid: NDArray[np.floating], enabled: bool = True) -> Array:
    r"""Map a Well output time grid to PDEformer's normalized time scale."""
    time_grid = time_grid.astype(np.float32)
    if not enabled:
        return time_grid
    max_abs = float(np.max(np.abs(time_grid)))
    if max_abs < 1.0e-12:
        return np.zeros_like(time_grid, dtype=np.float32)
    return time_grid / max_abs


def coordinate_array(space_grid: Array, output_time_grid: Array) -> Array:
    r"""Build PDEformer txyz query coordinates from Well 2D grids."""
    if space_grid.ndim < 2:
        raise ValueError("space_grid must include spatial axes and a coordinate axis.")
    n_spatial = space_grid.shape[-1]
    if n_spatial != 2:
        raise ValueError(
            "PDEformer-2 fine-tuning currently expects 2D Well grids; "
            f"got {n_spatial} spatial dimensions.")

    spatial_shape = space_grid.shape[:-1]
    n_t = output_time_grid.shape[0]
    coord = np.zeros((n_t, *spatial_shape, 4), dtype=np.float32)
    coord[..., 0] = output_time_grid.reshape((n_t,) + (1,) * len(spatial_shape))
    coord[..., 1:3] = space_grid[np.newaxis, ...]
    return coord


def interpolate_to_resolution(
        field: NDArray[np.floating],
        space_grid: Array,
        resolution: int) -> Tuple[Array, Array, Array]:
    r"""Interpolate one scalar 2D Well field to PDEformer's function grid."""
    if field.ndim != 2:
        raise ValueError(f"Expected a scalar 2D field, got shape {field.shape}.")
    x_old = np.asarray(space_grid[:, 0, 0], dtype=np.float32)
    y_old = np.asarray(space_grid[0, :, 1], dtype=np.float32)
    x_new = np.linspace(float(x_old.min()), float(x_old.max()),
                        resolution, dtype=np.float32)
    y_new = np.linspace(float(y_old.min()), float(y_old.max()),
                        resolution, dtype=np.float32)
    interp = RegularGridInterpolator(
        (x_old, y_old),
        field.astype(np.float32),
        bounds_error=False,
        fill_value=None,
    )
    x_ext, y_ext = np.meshgrid(x_new, y_new, indexing="ij")
    points = np.stack([x_ext, y_ext], axis=-1)
    return interp(points).astype(np.float32), x_ext, y_ext


def interpolate_fields_to_resolution(
        fields: NDArray[np.floating],
        space_grid: Array,
        resolution: int) -> Tuple[Array, Array, Array]:
    r"""Interpolate channel-last 2D fields to PDEformer's function grid."""
    if fields.ndim != 3:
        raise ValueError(f"Expected [nx, ny, channels] fields, got {fields.shape}.")
    interp_fields = []
    x_ext = y_ext = None
    for idx in range(fields.shape[-1]):
        interp, x_cur, y_cur = interpolate_to_resolution(
            fields[..., idx], space_grid, resolution)
        interp_fields.append(interp)
        x_ext, y_ext = x_cur, y_cur
    return np.stack(interp_fields, axis=-1), x_ext, y_ext


def build_well_pde_dag(
        config,
        dataset_name: str,
        ic_fields: NDArray[np.floating],
        space_grid: Array,
        channel_names: Sequence[str],
        equation_scalars: Optional[Dict[str, float]] = None,
        coordinate_scales: Optional[Dict[str, float]] = None) -> PDEAsDAG:
    r"""Build a PDEformer DAG with numeric ICs for a Well sample."""
    resolution = int(config.model.function_encoder.get("resolution", 128))
    ic_on_fenc, x_ext, y_ext = interpolate_fields_to_resolution(
        ic_fields, space_grid, resolution)
    pde = _build_pde(
        dataset_name,
        channel_names,
        x_ext,
        y_ext,
        ic_on_fenc,
        equation_scalars=equation_scalars,
        coordinate_scales=coordinate_scales,
    )
    return pde.gen_dag(config)


def build_well_pde_template(
        dataset_name: str,
        channel_names: Sequence[str],
        x_ext: Array,
        y_ext: Array,
        coordinate_scales: Optional[Dict[str, float]] = None) -> PDENodesCollector:
    r"""Build a PDE graph with NaN IC placeholders for a Well dataset."""
    return _build_pde(
        dataset_name,
        channel_names,
        x_ext,
        y_ext,
        None,
        coordinate_scales=coordinate_scales,
    )


def get_well_equation_spec(dataset_name: str) -> WellEquationSpec:
    r"""Look up equation metadata by Well dataset name or alias."""
    try:
        return _SPEC_BY_NAME[dataset_name]
    except KeyError as exc:
        known = ", ".join(sorted(_SPEC_BY_NAME))
        raise KeyError(f"Unknown Well equation '{dataset_name}'. Known: {known}") from exc


def list_well_equation_names() -> List[str]:
    r"""Return canonical registered Well dataset equation names."""
    return sorted(spec.dataset for spec in _SPECS)


def _build_pde(
        dataset_name: str,
        channel_names: Sequence[str],
        x_ext: Array,
        y_ext: Array,
        ic_fields: Optional[Array],
        equation_scalars: Optional[Dict[str, float]] = None,
        coordinate_scales: Optional[Dict[str, float]] = None) -> PDENodesCollector:
    spec = get_well_equation_spec(dataset_name)
    if spec.spatial_dims != 2:
        raise ValueError(
            f"{spec.dataset} is {spec.spatial_dims}D in The Well. "
            "This PDEformer-2 adapter currently supports 2D datasets.")
    pde = PDENodesCollector(dim=2)
    variables = []
    for idx, _name in enumerate(channel_names):
        var = pde.new_uf()
        ic_value = np.nan if ic_fields is None else ic_fields[..., idx]
        pde.set_ic(var, ic_value, x=x_ext, y=y_ext)
        variables.append(var)
    context = {
        "equation_scalars": equation_scalars or {},
        "coordinate_scales": coordinate_scales or {},
    }
    spec.builder(pde, variables, [str(name) for name in channel_names], context)
    return pde


def _by_name(names: Sequence[str], wanted: Iterable[str]) -> Optional[int]:
    lowered = [name.lower() for name in names]
    for item in wanted:
        item = item.lower()
        if item in lowered:
            return lowered.index(item)
    return None


def _var(names: Sequence[str], variables: Sequence[PDENode], *wanted: str) -> Optional[PDENode]:
    idx = _by_name(names, wanted)
    if idx is None or idx >= len(variables):
        return None
    return variables[idx]


def _lap(u: PDENode) -> PDENode:
    return u.dx.dx + u.dy.dy


def _add_closure_eqs(
        pde: PDENodesCollector,
        variables: Sequence[PDENode],
        tag: float,
        known_terms: Optional[Dict[int, List[PDENode]]] = None) -> None:
    r"""Add per-variable equations with optional known terms and learned closure."""
    if not variables:
        return
    tag_node = pde.new_coef(tag)
    closure_inputs = [*variables, tag_node]
    closures = pde.unknown_func(closure_inputs, squeeze=False)
    known_terms = known_terms or {}
    for idx, var in enumerate(variables):
        pde.sum_eq0(var.dt, *known_terms.get(idx, []), closures[idx])


def _acoustic(pde: PDENodesCollector, v: List[PDENode], names: List[str], _context: Dict[str, Any]) -> None:
    p = _var(names, v, "pressure", "pressure_re")
    ux = _var(names, v, "velocity_x", "u")
    uy = _var(names, v, "velocity_y", "v")
    if p is not None and ux is not None and uy is not None:
        bulk_modulus = pde.new_coef(4.0)
        inv_density = pde.new_coef(1.0)
        pde.sum_eq0(p.dt, bulk_modulus * (ux.dx + uy.dy))
        pde.sum_eq0(ux.dt, inv_density * p.dx)
        pde.sum_eq0(uy.dt, inv_density * p.dy)
        return
    _add_closure_eqs(pde, v, 10.0)


def _scaled_coef(pde: PDENodesCollector, value: float, scale: float) -> PDENode:
    return pde.new_coef(float(value) * float(scale))


def _sample_or_placeholder_coef(
        pde: PDENodesCollector,
        scalars: Dict[str, float],
        name: str,
        scale: float) -> PDENode:
    if name in scalars:
        return pde.new_coef(float(scalars[name]) * float(scale))
    return pde.new_coef(np.nan)


def _gray_scott(pde: PDENodesCollector, v: List[PDENode], names: List[str], context: Dict[str, Any]) -> None:
    a = _var(names, v, "A", "concentration_A")
    b = _var(names, v, "B", "concentration_B")
    if a is not None and b is not None:
        scalars = context.get("equation_scalars", {})
        scales = context.get("coordinate_scales", {})
        time_scale = float(scales.get("time", 1.0))
        x_scale = float(scales.get("x", 1.0))
        y_scale = float(scales.get("y", 1.0))
        if abs(x_scale) < 1.0e-12:
            x_scale = 1.0
        if abs(y_scale) < 1.0e-12:
            y_scale = 1.0

        d_a_x = _scaled_coef(pde, 2.0e-5, time_scale / (x_scale * x_scale))
        d_a_y = _scaled_coef(pde, 2.0e-5, time_scale / (y_scale * y_scale))
        d_b_x = _scaled_coef(pde, 1.0e-5, time_scale / (x_scale * x_scale))
        d_b_y = _scaled_coef(pde, 1.0e-5, time_scale / (y_scale * y_scale))
        feed = _sample_or_placeholder_coef(pde, scalars, "F", time_scale)
        kill = _sample_or_placeholder_coef(pde, scalars, "k", time_scale)
        reaction_scale = pde.new_coef(time_scale)
        one = pde.new_coef(1.0)
        ab2 = a * b * b
        pde.sum_eq0(
            a.dt,
            -(d_a_x * a.dx.dx),
            -(d_a_y * a.dy.dy),
            reaction_scale * ab2,
            -(feed * (one - a)),
        )
        pde.sum_eq0(
            b.dt,
            -(d_b_x * b.dx.dx),
            -(d_b_y * b.dy.dy),
            -(reaction_scale * ab2),
            (feed + kill) * b,
        )
        return
    _add_closure_eqs(pde, v, 20.0)


def _navier_stokes_tracer(pde: PDENodesCollector, v: List[PDENode], names: List[str], _context: Dict[str, Any]) -> None:
    tracer = _var(names, v, "tracer", "buoyancy", "concentration")
    pressure = _var(names, v, "pressure")
    ux = _var(names, v, "velocity_x", "momentum_x")
    uy = _var(names, v, "velocity_y", "momentum_y")
    known: Dict[int, List[PDENode]] = {}
    nu = pde.new_coef(1.0e-3)
    diff = pde.new_coef(1.0e-3)
    if ux is not None and uy is not None:
        if tracer is not None:
            idx = v.index(tracer)
            known[idx] = [ux * tracer.dx, uy * tracer.dy, -(diff * _lap(tracer))]
        if pressure is not None:
            idx = v.index(pressure)
            known[idx] = [pressure]
        if ux in v:
            terms = [ux * ux.dx, uy * ux.dy, -(nu * _lap(ux))]
            if pressure is not None:
                terms.append(pressure.dx)
            known[v.index(ux)] = terms
        if uy in v:
            terms = [ux * uy.dx, uy * uy.dy, -(nu * _lap(uy))]
            if pressure is not None:
                terms.append(pressure.dy)
            if tracer is not None and "buoyancy" in [name.lower() for name in names]:
                terms.append(-tracer)
            known[v.index(uy)] = terms
    _add_closure_eqs(pde, v, 30.0, known)


def _shallow_water(pde: PDENodesCollector, v: List[PDENode], names: List[str], _context: Dict[str, Any]) -> None:
    h = _var(names, v, "height")
    ux = _var(names, v, "velocity_theta", "velocity_x")
    uy = _var(names, v, "velocity_phi", "velocity_y")
    if h is not None and ux is not None and uy is not None:
        grav = pde.new_coef(1.0)
        h_mean = pde.new_coef(1.0)
        pde.sum_eq0(h.dt, h_mean * (ux.dx + uy.dy), (h * ux).dx, (h * uy).dy)
        pde.sum_eq0(ux.dt, ux * ux.dx, uy * ux.dy, grav * h.dx)
        pde.sum_eq0(uy.dt, ux * uy.dx, uy * uy.dy, grav * h.dy)
        return
    _add_closure_eqs(pde, v, 40.0)


def _euler(pde: PDENodesCollector, v: List[PDENode], names: List[str], _context: Dict[str, Any]) -> None:
    rho = _var(names, v, "density")
    e = _var(names, v, "energy")
    p = _var(names, v, "pressure")
    mx = _var(names, v, "momentum_x", "velocity_x")
    my = _var(names, v, "momentum_y", "velocity_y")
    known: Dict[int, List[PDENode]] = {}
    if rho is not None and mx is not None and my is not None:
        known[v.index(rho)] = [mx.dx, my.dy]
    if mx is not None:
        terms = []
        if p is not None:
            terms.append(p.dx)
        known[v.index(mx)] = terms
    if my is not None:
        terms = []
        if p is not None:
            terms.append(p.dy)
        known[v.index(my)] = terms
    if e is not None and p is not None and mx is not None and my is not None:
        known[v.index(e)] = [((e + p) * mx).dx, ((e + p) * my).dy]
    _add_closure_eqs(pde, v, 50.0, known)


def _hydro_cooling(pde: PDENodesCollector, v: List[PDENode], names: List[str], _context: Dict[str, Any]) -> None:
    rho = _var(names, v, "density")
    p = _var(names, v, "pressure")
    temp = _var(names, v, "temperature")
    ux = _var(names, v, "velocity_x")
    uy = _var(names, v, "velocity_y")
    known: Dict[int, List[PDENode]] = {}
    gamma_minus_one = pde.new_coef(2.0 / 3.0)
    if rho is not None and ux is not None and uy is not None:
        known[v.index(rho)] = [(rho * ux).dx, (rho * uy).dy]
    if p is not None and ux is not None and uy is not None:
        known[v.index(p)] = [ux * p.dx, uy * p.dy, gamma_minus_one * p * (ux.dx + uy.dy)]
    if temp is not None and ux is not None and uy is not None:
        known[v.index(temp)] = [ux * temp.dx, uy * temp.dy]
    _add_closure_eqs(pde, v, 60.0, known)


def _helmholtz(pde: PDENodesCollector, v: List[PDENode], _names: List[str], _context: Dict[str, Any]) -> None:
    omega2 = pde.new_coef(1.0)
    for idx, u in enumerate(v):
        source = pde.unknown_func(u, pde.new_coef(float(idx)), squeeze=False)[0]
        pde.sum_eq0(-_lap(u), -(omega2 * u), source)


def _viscoelastic(pde: PDENodesCollector, v: List[PDENode], names: List[str], _context: Dict[str, Any]) -> None:
    pressure = _var(names, v, "pressure")
    ux = _var(names, v, "velocity_x")
    uy = _var(names, v, "velocity_y")
    cxx = _var(names, v, "C_xx")
    cxy = _var(names, v, "C_xy", "C_yx")
    cyy = _var(names, v, "C_yy")
    czz = _var(names, v, "c_zz")
    known: Dict[int, List[PDENode]] = {}
    nu = pde.new_coef(1.0e-3)
    eps = pde.new_coef(1.0e-3)
    if ux is not None and uy is not None:
        if ux in v:
            terms = [ux * ux.dx, uy * ux.dy, -(nu * _lap(ux))]
            if pressure is not None:
                terms.append(pressure.dx)
            if cxx is not None and cxy is not None:
                terms.append(-(cxx.dx + cxy.dy))
            known[v.index(ux)] = terms
        if uy in v:
            terms = [ux * uy.dx, uy * uy.dy, -(nu * _lap(uy))]
            if pressure is not None:
                terms.append(pressure.dy)
            if cxy is not None and cyy is not None:
                terms.append(-(cxy.dx + cyy.dy))
            known[v.index(uy)] = terms
        for c in (cxx, cxy, cyy, czz):
            if c is not None and c in v:
                known[v.index(c)] = [ux * c.dx, uy * c.dy, -(eps * _lap(c))]
    _add_closure_eqs(pde, v, 70.0, known)


def _active_matter(pde: PDENodesCollector, v: List[PDENode], names: List[str], _context: Dict[str, Any]) -> None:
    concentration = _var(names, v, "concentration")
    ux = _var(names, v, "velocity_x")
    uy = _var(names, v, "velocity_y")
    known: Dict[int, List[PDENode]] = {}
    diff = pde.new_coef(1.0e-3)
    if concentration is not None and ux is not None and uy is not None:
        known[v.index(concentration)] = [
            ux * concentration.dx,
            uy * concentration.dy,
            -(diff * _lap(concentration)),
        ]
    _add_closure_eqs(pde, v, 80.0, known)


def _generic_closure(pde: PDENodesCollector, v: List[PDENode], _names: List[str], _context: Dict[str, Any]) -> None:
    _add_closure_eqs(pde, v, 90.0)


_SPECS = (
    WellEquationSpec(
        "acoustic_scattering_discontinuous",
        aliases=("acoustic_scattering_inclusions", "acoustic_scattering_maze"),
        pde_latex=(
            r"$p_t + K(x,y)(u_x+v_y)=0$\n"
            r"$u_t + \rho(x,y)^{-1}p_x=0,\quad v_t+\rho(x,y)^{-1}p_y=0$"
        ),
        builder=_acoustic,
        notes="Variable-density acoustics; the DAG uses K=4 and a learned density closure.",
    ),
    WellEquationSpec(
        "gray_scott_reaction_diffusion",
        aliases=("gray_scott",),
        pde_latex=(
            r"$A_t=\delta_A\Delta A-AB^2+f(1-A)$\n"
            r"$B_t=\delta_B\Delta B+AB^2-(f+k)B$"
        ),
        builder=_gray_scott,
    ),
    WellEquationSpec(
        "shear_flow",
        aliases=(),
        pde_latex=(
            r"$u_t+\nabla p-\nu\Delta u=-u\cdot\nabla u$\n"
            r"$s_t-D\Delta s=-u\cdot\nabla s$"
        ),
        builder=_navier_stokes_tracer,
    ),
    WellEquationSpec(
        "rayleigh_benard",
        aliases=("rayleigh_benard_uniform",),
        pde_latex=(
            r"$b_t-\kappa\Delta b=-u\cdot\nabla b$\n"
            r"$u_t-\nu\Delta u+\nabla p-b e_y=-u\cdot\nabla u$"
        ),
        builder=_navier_stokes_tracer,
    ),
    WellEquationSpec(
        "planetswe",
        aliases=(),
        pde_latex=(
            r"$u_t=-u\cdot\nabla u-g\nabla h-\nu\nabla^4u-2\Omega\times u$\n"
            r"$h_t=-H\nabla\cdot u-\nabla\cdot(hu)-\nu\nabla^4h+F$"
        ),
        builder=_shallow_water,
    ),
    WellEquationSpec(
        "euler_multi_quadrants_openBC",
        aliases=("euler_multi_quadrants_periodicBC",),
        pde_latex=r"$U_t+F(U)_x+G(U)_y=0$ for compressible Euler gas dynamics.",
        builder=_euler,
    ),
    WellEquationSpec(
        "turbulent_radiative_layer_2D",
        aliases=(),
        pde_latex=(
            r"$\rho_t+\nabla\cdot(\rho v)=0$\n"
            r"$(\rho v)_t+\nabla\cdot(\rho vv+P)=0$\n"
            r"$E_t+\nabla\cdot((E+P)v)=-E/t_{\rm cool}$"
        ),
        builder=_hydro_cooling,
    ),
    WellEquationSpec(
        "helmholtz_staircase",
        aliases=(),
        pde_latex=(
            r"$U_{tt}-\Delta U=\delta(t)\delta(x-x_0)$; "
            r"frequency domain: $-(\Delta+\omega^2)u=\delta_{x_0}$"
        ),
        builder=_helmholtz,
    ),
    WellEquationSpec(
        "viscoelastic_instability",
        aliases=(),
        pde_latex=(
            r"$Re(u_t+u\cdot\nabla u)+\nabla p=\beta\Delta u+(1-\beta)\nabla\cdot T(C)$\n"
            r"$C_t+u\cdot\nabla C+T(C)=C\nabla u+(\nabla u)^T C+\epsilon\Delta C$"
        ),
        builder=_viscoelastic,
    ),
    WellEquationSpec(
        "active_matter",
        aliases=(),
        pde_latex="Active-matter continuum equations (associated paper equations 1-5).",
        builder=_active_matter,
        notes="The public README references paper equations; the DAG keeps advection-diffusion plus learned closure terms.",
    ),
    WellEquationSpec(
        "MHD_64",
        aliases=("MHD_256",),
        pde_latex=(
            r"$\rho_t+\nabla\cdot(\rho v)=0$\n"
            r"$(\rho v)_t+\nabla\cdot(\rho vv-BB)+\nabla p=0$\n"
            r"$B_t-\nabla\times(v\times B)=0$"
        ),
        builder=_generic_closure,
        spatial_dims=3,
    ),
    WellEquationSpec(
        "convective_envelope_rsg",
        aliases=("post_neutron_star_merger", "rayleigh_taylor_instability",
                 "supernova_explosion_64", "supernova_explosion_128",
                 "turbulence_gravity_cooling", "turbulent_radiative_layer_3D"),
        pde_latex="3D hydrodynamics/MHD/radiation equations described in the Well dataset README.",
        builder=_generic_closure,
        spatial_dims=3,
    ),
)

_SPEC_BY_NAME: Dict[str, WellEquationSpec] = {}
for _spec in _SPECS:
    for _name in _spec.all_names:
        _SPEC_BY_NAME[_name] = _spec
