"""Tests for the flux module."""
# pylint: disable-msg=C0103,W0212

import unittest

import numpy as np

import katpoint


class TestFluxDensityModel(unittest.TestCase):
    """Test flux density model calculation."""
    def setUp(self):
        self.flux_model = katpoint.FluxDensityModel('(1.0 2.0 2.0 0.0 0.0 0.0 0.0 0.0)')
        self.too_many_params = katpoint.FluxDensityModel('(1.0 2.0 2.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0)')
        self.too_few_params = katpoint.FluxDensityModel('(1.0 2.0 2.0)')
        self.flux_target = katpoint.Target('radec, 0.0, 0.0, ' + self.flux_model.description)
        self.no_flux_target = katpoint.Target('radec, 0.0, 0.0')

    def test_flux_density(self):
        """Test flux density calculation."""
        unit_model = katpoint.FluxDensityModel(100., 200., [0.])
        self.assertEqual(unit_model.flux_density(110.), 1.0, 'Flux calculation wrong')
        self.assertRaises(ValueError, katpoint.FluxDensityModel, '1.0 2.0 2.0', 2.0, [2.0])
        self.assertRaises(ValueError, katpoint.FluxDensityModel, '1.0')
        self.assertEqual(self.flux_model.flux_density(1.5), 100.0, 'Flux calculation wrong')
        self.assertEqual(self.too_many_params.flux_density(1.5), 100.0, 'Flux calculation for too many params wrong')
        self.assertEqual(self.too_few_params.flux_density(1.5), 100.0, 'Flux calculation for too few params wrong')
        np.testing.assert_equal(self.flux_model.flux_density([1.5, 1.5]),
                                np.array([100.0, 100.0]), 'Flux calculation for multiple frequencies wrong')
        np.testing.assert_equal(self.flux_model.flux_density([0.5, 2.5]),
                                np.array([np.nan, np.nan]), 'Flux calculation for out-of-range frequencies wrong')
        self.assertRaises(ValueError, self.no_flux_target.flux_density)
        np.testing.assert_equal(self.no_flux_target.flux_density([1.5, 1.5]),
                                np.array([np.nan, np.nan]), 'Empty flux model leads to wrong empty flux shape')
        self.flux_target.flux_freq_MHz = 1.5
        self.assertEqual(self.flux_target.flux_density(), 100.0, 'Flux calculation for default freq wrong')
        print self.flux_target

