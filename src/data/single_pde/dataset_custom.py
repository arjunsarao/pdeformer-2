r"""
Project-local single-PDE datasets.
"""
import os
from typing import Tuple, Dict, Any, Optional

import h5py
import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from ..pde_dag import PDENodesCollector
from .basics import register_pde_type, CartesianGridInputFileDataset


@register_pde_type("burgers2d_viscous", "burgers2d")
class ViscousBurgers2DInputDataset(CartesianGridInputFileDataset):
    r"""Load generated 2D viscous Burgers data on a periodic Cartesian grid."""
    n_vars: int = 2
    var_latex = "uv"
    pde_latex = (r"$u_t+uu_x+vu_y-\nu(u_{xx}+u_{yy})=0$" + "\n"
                 r"$v_t+uv_x+vv_y-\nu(v_{xx}+v_{yy})=0$")

    def __init__(self, config: DictConfig, pde_param: str) -> None:
        super().__init__(config, pde_param)

        filepath = pde_param
        if not filepath.endswith((".h5", ".hdf5")):
            filepath = filepath + ".hdf5"
        filepath = os.path.join(config.data.path, filepath)
        self.h5_file_u = h5py.File(filepath, "r")
        self.dataset_size = self.h5_file_u["sol/u"].shape[0]

        t_coord = self.h5_file_u["coord/t"][1:]
        x_coord = self.h5_file_u["coord/x"][()]
        y_coord = self.h5_file_u["coord/y"][()]
        if "coef/viscosity" in self.h5_file_u:
            self.viscosity = self.h5_file_u["coef/viscosity"][()]
        else:
            self.viscosity = np.full(
                self.dataset_size,
                float(self.h5_file_u["args/viscosity"][()]),
                dtype=np.float32)
        self.coef_dict = {r"\nu": "sample-dependent"}

        pde = self._gen_pde_nodes(x_coord, y_coord)
        self.pde_dag = pde.gen_dag(config)
        self.txyz_coord = self._gen_coords(t_coord, x_coord, y_coord)

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        u_label = np.stack([
            self.h5_file_u["sol/u"][idx_pde, 1:],
            self.h5_file_u["sol/v"][idx_pde, 1:],
        ], axis=-1)
        u_label = np.expand_dims(u_label, axis=-2)

        input_field = np.stack([
            self.h5_file_u["coef/u_ic"][idx_pde],
            self.h5_file_u["coef/v_ic"][idx_pde],
        ], axis=-1)
        nu_val = float(self.viscosity[idx_pde])
        nu_ext = np.full(shape=2, fill_value=nu_val)
        return input_field, nu_ext, self.txyz_coord, u_label

    def get_pde_info(self,
                     idx_pde: int,
                     idx_var: Optional[int] = None) -> Dict[str, Any]:
        data_info = super().get_pde_info(idx_pde, idx_var)
        data_info["coef_dict"] = {r"\nu": float(self.viscosity[idx_pde])}
        return data_info

    @staticmethod
    def _gen_pde_nodes(x_coord: NDArray[float],
                       y_coord: NDArray[float]) -> PDENodesCollector:
        pde = PDENodesCollector(dim=2)
        x_ext, y_ext = np.meshgrid(x_coord, y_coord, indexing="ij")

        u_ = pde.new_uf()
        v_ = pde.new_uf()
        pde.set_ic(u_, np.nan, x=x_ext, y=y_ext)
        pde.set_ic(v_, np.nan, x=x_ext, y=y_ext)
        nu_u = pde.new_coef(np.nan)
        nu_v = pde.new_coef(np.nan)

        pde.sum_eq0(u_.dt,
                    u_ * u_.dx,
                    v_ * u_.dy,
                    -(nu_u * (u_.dx.dx + u_.dy.dy)))
        pde.sum_eq0(v_.dt,
                    u_ * v_.dx,
                    v_ * v_.dy,
                    -(nu_v * (v_.dx.dx + v_.dy.dy)))
        return pde


@register_pde_type("allen_cahn2d", "allen-cahn2d", "allen_cahn")
class AllenCahn2DInputDataset(CartesianGridInputFileDataset):
    r"""Load generated 2D Allen-Cahn reaction-diffusion data."""
    n_vars: int = 1
    var_latex = "u"
    pde_latex = (
        r"$u_t-\epsilon(u_{xx}+u_{yy})-\rho(u-u^3)=0$"
    )

    def __init__(self, config: DictConfig, pde_param: str) -> None:
        super().__init__(config, pde_param)

        filepath = pde_param
        if not filepath.endswith((".h5", ".hdf5")):
            filepath = filepath + ".hdf5"
        filepath = os.path.join(config.data.path, filepath)
        self.h5_file_u = h5py.File(filepath, "r")
        self.dataset_size = self.h5_file_u["sol/u"].shape[0]

        t_coord = self.h5_file_u["coord/t"][1:]
        x_coord = self.h5_file_u["coord/x"][()]
        y_coord = self.h5_file_u["coord/y"][()]
        self.epsilon = self.h5_file_u["coef/epsilon"][()]
        self.rho = self.h5_file_u["coef/rho"][()]
        self.coef_dict = {r"\epsilon": "sample-dependent",
                          r"\rho": "sample-dependent"}

        pde = self._gen_pde_nodes(x_coord, y_coord)
        self.pde_dag = pde.gen_dag(config)
        self.txyz_coord = self._gen_coords(t_coord, x_coord, y_coord)

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        u_label = np.expand_dims(self.h5_file_u["sol/u"][idx_pde, 1:], axis=-1)
        u_label = np.expand_dims(u_label, axis=-2)

        input_field = np.expand_dims(self.h5_file_u["coef/u_ic"][idx_pde], axis=-1)
        input_scalar = np.array([
            float(self.epsilon[idx_pde]),
            float(self.rho[idx_pde]),
        ])
        return input_field, input_scalar, self.txyz_coord, u_label

    def get_pde_info(self,
                     idx_pde: int,
                     idx_var: Optional[int] = None) -> Dict[str, Any]:
        data_info = super().get_pde_info(idx_pde, idx_var)
        data_info["coef_dict"] = {
            r"\epsilon": float(self.epsilon[idx_pde]),
            r"\rho": float(self.rho[idx_pde]),
        }
        return data_info

    @staticmethod
    def _gen_pde_nodes(x_coord: NDArray[float],
                       y_coord: NDArray[float]) -> PDENodesCollector:
        pde = PDENodesCollector(dim=2)
        x_ext, y_ext = np.meshgrid(x_coord, y_coord, indexing="ij")

        u_ = pde.new_uf()
        pde.set_ic(u_, np.nan, x=x_ext, y=y_ext)
        epsilon = pde.new_coef(np.nan)
        rho = pde.new_coef(np.nan)

        pde.sum_eq0(u_.dt,
                    -(epsilon * (u_.dx.dx + u_.dy.dy)),
                    -(rho * (u_ - u_ * u_ * u_)))
        return pde


@register_pde_type("grayscott2d", "gray_scott2d", "gray-scott2d")
class GrayScott2DInputDataset(CartesianGridInputFileDataset):
    r"""Load generated 2D Gray-Scott reaction-diffusion data."""
    n_vars: int = 2
    var_latex = "uv"
    pde_latex = (
        r"$u_t-D_u(u_{xx}+u_{yy})+uv^2-F(1-u)=0$" + "\n"
        r"$v_t-D_v(v_{xx}+v_{yy})-uv^2+(F+k)v=0$"
    )

    def __init__(self, config: DictConfig, pde_param: str) -> None:
        super().__init__(config, pde_param)

        filepath = pde_param
        if not filepath.endswith((".h5", ".hdf5")):
            filepath = filepath + ".hdf5"
        filepath = os.path.join(config.data.path, filepath)
        self.h5_file_u = h5py.File(filepath, "r")
        self.dataset_size = self.h5_file_u["sol/u"].shape[0]

        t_coord = self.h5_file_u["coord/t"][1:]
        x_coord = self.h5_file_u["coord/x"][()]
        y_coord = self.h5_file_u["coord/y"][()]
        self.du = self.h5_file_u["coef/du"][()]
        self.dv = self.h5_file_u["coef/dv"][()]
        self.feed = self.h5_file_u["coef/feed"][()]
        self.kill = self.h5_file_u["coef/kill"][()]
        self.coef_dict = {
            r"D_u": "sample-dependent",
            r"D_v": "sample-dependent",
            r"F": "sample-dependent",
            r"k": "sample-dependent",
        }

        pde = self._gen_pde_nodes(x_coord, y_coord)
        self.pde_dag = pde.gen_dag(config)
        self.txyz_coord = self._gen_coords(t_coord, x_coord, y_coord)

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        u_label = np.stack([
            self.h5_file_u["sol/u"][idx_pde, 1:],
            self.h5_file_u["sol/v"][idx_pde, 1:],
        ], axis=-1)
        u_label = np.expand_dims(u_label, axis=-2)

        input_field = np.stack([
            self.h5_file_u["coef/u_ic"][idx_pde],
            self.h5_file_u["coef/v_ic"][idx_pde],
        ], axis=-1)
        input_scalar = np.array([
            float(self.du[idx_pde]),
            float(self.dv[idx_pde]),
            float(self.feed[idx_pde]),
            float(self.kill[idx_pde]),
        ])
        return input_field, input_scalar, self.txyz_coord, u_label

    def get_pde_info(self,
                     idx_pde: int,
                     idx_var: Optional[int] = None) -> Dict[str, Any]:
        data_info = super().get_pde_info(idx_pde, idx_var)
        data_info["coef_dict"] = {
            r"D_u": float(self.du[idx_pde]),
            r"D_v": float(self.dv[idx_pde]),
            r"F": float(self.feed[idx_pde]),
            r"k": float(self.kill[idx_pde]),
        }
        return data_info

    @staticmethod
    def _gen_pde_nodes(x_coord: NDArray[float],
                       y_coord: NDArray[float]) -> PDENodesCollector:
        pde = PDENodesCollector(dim=2)
        x_ext, y_ext = np.meshgrid(x_coord, y_coord, indexing="ij")

        u_ = pde.new_uf()
        v_ = pde.new_uf()
        pde.set_ic(u_, np.nan, x=x_ext, y=y_ext)
        pde.set_ic(v_, np.nan, x=x_ext, y=y_ext)
        du = pde.new_coef(np.nan)
        dv = pde.new_coef(np.nan)
        feed = pde.new_coef(np.nan)
        kill = pde.new_coef(np.nan)
        one = pde.new_coef(1.0)
        uv2 = u_ * v_ * v_

        pde.sum_eq0(u_.dt,
                    -(du * (u_.dx.dx + u_.dy.dy)),
                    uv2,
                    -(feed * (one - u_)))
        pde.sum_eq0(v_.dt,
                    -(dv * (v_.dx.dx + v_.dy.dy)),
                    -uv2,
                    (feed + kill) * v_)
        return pde


@register_pde_type("viscoelastic2d", "oldroyd_b2d", "oldroyd-b2d",
                   "fene2d", "fene-p2d")
class ViscoelasticFlow2DInputDataset(CartesianGridInputFileDataset):
    r"""Load generated 2D Oldroyd-B/FENE-style viscoelastic flow data."""
    n_vars: int = 5
    var_latex = ["u", "v", "C_{xx}", "C_{xy}", "C_{yy}"]
    pde_latex = (
        r"$u_t+uu_x+vu_y-\nu\Delta u-\beta(C_{xx,x}+C_{xy,y})=0$" + "\n"
        r"$v_t+uv_x+vv_y-\nu\Delta v-\beta(C_{xy,x}+C_{yy,y})=0$" + "\n"
        r"$C_t+\mathbf{u}\cdot\nabla C-C\nabla\mathbf{u}-(\nabla\mathbf{u})^TC"
        r"+\lambda^{-1}[(1+\chi(\mathrm{tr}C-2))C-I]-\kappa\Delta C=0$"
    )
    var_names = ("u", "v", "cxx", "cxy", "cyy")

    def __init__(self, config: DictConfig, pde_param: str) -> None:
        super().__init__(config, pde_param)

        filepath = pde_param
        if not filepath.endswith((".h5", ".hdf5")):
            filepath = filepath + ".hdf5"
        filepath = os.path.join(config.data.path, filepath)
        self.h5_file_u = h5py.File(filepath, "r")
        self.dataset_size = self.h5_file_u["sol/u"].shape[0]

        t_coord = self.h5_file_u["coord/t"][1:]
        x_coord = self.h5_file_u["coord/x"][()]
        y_coord = self.h5_file_u["coord/y"][()]
        self.viscosity = self.h5_file_u["coef/viscosity"][()]
        self.polymer_coupling = self.h5_file_u["coef/polymer_coupling"][()]
        self.relaxation_time = self.h5_file_u["coef/relaxation_time"][()]
        self.stress_diffusivity = self.h5_file_u["coef/stress_diffusivity"][()]
        self.fene_strength = self.h5_file_u["coef/fene_strength"][()]
        self.coef_dict = {
            r"\nu": "sample-dependent",
            r"\beta": "sample-dependent",
            r"\lambda": "sample-dependent",
            r"\lambda^{-1}": "sample-dependent",
            r"\kappa": "sample-dependent",
            r"\chi": "sample-dependent",
        }

        pde = self._gen_pde_nodes(x_coord, y_coord)
        self.pde_dag = pde.gen_dag(config)
        self.txyz_coord = self._gen_coords(t_coord, x_coord, y_coord)

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        u_label = np.stack([
            self.h5_file_u[f"sol/{name}"][idx_pde, 1:]
            for name in self.var_names
        ], axis=-1)
        u_label = np.expand_dims(u_label, axis=-2)

        input_field = np.stack([
            self.h5_file_u[f"coef/{name}_ic"][idx_pde]
            for name in self.var_names
        ], axis=-1)
        relaxation_time = float(self.relaxation_time[idx_pde])
        input_scalar = np.array([
            float(self.viscosity[idx_pde]),
            float(self.polymer_coupling[idx_pde]),
            1.0 / relaxation_time,
            float(self.stress_diffusivity[idx_pde]),
            float(self.fene_strength[idx_pde]),
        ])
        return input_field, input_scalar, self.txyz_coord, u_label

    def get_pde_info(self,
                     idx_pde: int,
                     idx_var: Optional[int] = None) -> Dict[str, Any]:
        data_info = super().get_pde_info(idx_pde, idx_var)
        relaxation_time = float(self.relaxation_time[idx_pde])
        data_info["coef_dict"] = {
            r"\nu": float(self.viscosity[idx_pde]),
            r"\beta": float(self.polymer_coupling[idx_pde]),
            r"\lambda": relaxation_time,
            r"\lambda^{-1}": 1.0 / relaxation_time,
            r"\kappa": float(self.stress_diffusivity[idx_pde]),
            r"\chi": float(self.fene_strength[idx_pde]),
        }
        return data_info

    @staticmethod
    def _gen_pde_nodes(x_coord: NDArray[float],
                       y_coord: NDArray[float]) -> PDENodesCollector:
        pde = PDENodesCollector(dim=2)
        x_ext, y_ext = np.meshgrid(x_coord, y_coord, indexing="ij")

        u_ = pde.new_uf()
        v_ = pde.new_uf()
        cxx = pde.new_uf()
        cxy = pde.new_uf()
        cyy = pde.new_uf()
        for var in (u_, v_, cxx, cxy, cyy):
            pde.set_ic(var, np.nan, x=x_ext, y=y_ext)

        nu = pde.new_coef(np.nan)
        beta = pde.new_coef(np.nan)
        inv_lambda = pde.new_coef(np.nan)
        kappa = pde.new_coef(np.nan)
        chi = pde.new_coef(np.nan)
        one = pde.new_coef(1.0)
        two = pde.new_coef(2.0)
        trace_shift = cxx + cyy - two
        fene_correction = chi * trace_shift

        pde.sum_eq0(u_.dt,
                    u_ * u_.dx,
                    v_ * u_.dy,
                    -(nu * (u_.dx.dx + u_.dy.dy)),
                    -(beta * (cxx.dx + cxy.dy)))
        pde.sum_eq0(v_.dt,
                    u_ * v_.dx,
                    v_ * v_.dy,
                    -(nu * (v_.dx.dx + v_.dy.dy)),
                    -(beta * (cxy.dx + cyy.dy)))

        pde.sum_eq0(cxx.dt,
                    u_ * cxx.dx,
                    v_ * cxx.dy,
                    -(two * (cxx * u_.dx + cxy * u_.dy)),
                    inv_lambda * (cxx - one + fene_correction * cxx),
                    -(kappa * (cxx.dx.dx + cxx.dy.dy)))
        pde.sum_eq0(cxy.dt,
                    u_ * cxy.dx,
                    v_ * cxy.dy,
                    -(cxx * v_.dx + cxy * v_.dy + cxy * u_.dx + cyy * u_.dy),
                    inv_lambda * (cxy + fene_correction * cxy),
                    -(kappa * (cxy.dx.dx + cxy.dy.dy)))
        pde.sum_eq0(cyy.dt,
                    u_ * cyy.dx,
                    v_ * cyy.dy,
                    -(two * (cxy * v_.dx + cyy * v_.dy)),
                    inv_lambda * (cyy - one + fene_correction * cyy),
                    -(kappa * (cyy.dx.dx + cyy.dy.dy)))
        return pde
