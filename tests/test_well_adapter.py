import unittest

import numpy as np
import torch

from scripts.evaluate_the_well import iter_sample_indices, resolve_device
from src.data.single_pde.dataset_well import (
    _coprime_stride,
    _delta_affine_from_normalizer,
    _field_affine_from_normalizer,
    _field_names_from_metadata,
    _normalized_absolute_from_residual,
    _normalized_residual_target,
)
from src.data.well_equations import forecast_time_grid, normalize_time_grid


class _ZScoreStub:
    flattened_means = {"variable": torch.tensor([0.7, 0.1])}
    flattened_stds = {"variable": torch.tensor([0.2, 0.05])}
    flattened_means_delta = {"variable": torch.tensor([-0.01, 0.02])}
    flattened_stds_delta = {"variable": torch.tensor([0.04, 0.01])}


class _MetadataStub:
    field_names = {0: ["A", "B"], 1: [], 2: []}


class WellAdapterTest(unittest.TestCase):
    def test_forecast_times_are_relative_to_last_input(self):
        lead = forecast_time_grid(
            np.array([0.0, 10.0]), np.array([20.0, 30.0]))
        np.testing.assert_array_equal(lead, np.array([10.0, 20.0]))
        np.testing.assert_allclose(
            normalize_time_grid(lead), np.array([0.5, 1.0]))

    def test_uniform_stride_is_a_permutation_and_spreads_prefix(self):
        size = 960_000
        stride = _coprime_stride(size)
        indices = [(123456 + idx * stride) % size for idx in range(64)]
        self.assertEqual(len(set(indices)), 64)
        self.assertEqual({idx // 160_000 for idx in indices}, set(range(6)))

    def test_evaluator_sampling_is_reproducible_and_not_a_prefix(self):
        first = list(iter_sample_indices(1000, 16, "uniform", 7))
        second = list(iter_sample_indices(1000, 16, "uniform", 7))
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 16)
        self.assertNotEqual(first, list(range(16)))

    def test_field_affine_uses_selected_channel_statistics(self):
        affine = _field_affine_from_normalizer(
            _ZScoreStub(), ("B",), (1,))
        self.assertAlmostEqual(affine["B"][0], 0.1, places=6)
        self.assertAlmostEqual(affine["B"][1], 0.05, places=6)

    def test_delta_affine_uses_selected_channel_statistics(self):
        affine = _delta_affine_from_normalizer(
            _ZScoreStub(), ("B",), (1,))
        self.assertAlmostEqual(affine["B"][0], 0.02, places=6)
        self.assertAlmostEqual(affine["B"][1], 0.01, places=6)

    def test_residual_target_round_trip_reconstructs_absolute_state(self):
        field_affine = _field_affine_from_normalizer(
            _ZScoreStub(), ("A", "B"), (0, 1))
        delta_affine = _delta_affine_from_normalizer(
            _ZScoreStub(), ("A", "B"), (0, 1))
        initial = np.array([[[0.5, -0.5]]], dtype=np.float32)
        absolute = np.array([
            [[[0.7, -0.3]]],
            [[[0.4, -0.8]]],
        ], dtype=np.float32)
        residual = _normalized_residual_target(
            initial, absolute, field_affine, delta_affine, ("A", "B"))

        reconstructed = _normalized_absolute_from_residual(
            initial, residual, field_affine, delta_affine, ("A", "B"))
        np.testing.assert_allclose(reconstructed, absolute, rtol=1e-6, atol=1e-6)

    def test_tensor_order_field_names_are_flattened(self):
        self.assertEqual(
            _field_names_from_metadata(_MetadataStub()), ("A", "B"))

    def test_unavailable_explicit_cuda_falls_back_to_cpu(self):
        with self.assertWarns(RuntimeWarning):
            device = resolve_device("cuda:0", "GPU", cuda_available=False)
        self.assertEqual(device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
