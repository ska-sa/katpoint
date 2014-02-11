"""Tests for the model module."""

import unittest
import StringIO

import katpoint


class TestDelayModel(unittest.TestCase):
    """Test antenna delay model."""
    def test_construct_save_load(self):
        """Test construction / save / load of delay model."""
        m = katpoint.DelayModel('1.0, -2.0, -3.0, 4.123, 5.0, 6.0')
        m.header['date'] = '2014-01-15'
        # An empty file should lead to a BadModelFile exception
        cfg_file = StringIO.StringIO()
        self.assertRaises(katpoint.BadModelFile, m.fromfile, cfg_file)
        m.tofile(cfg_file)
        cfg_str = cfg_file.getvalue()
        cfg_file.close()
        # Load the saved config file
        cfg_file = StringIO.StringIO(cfg_str)
        m2 = katpoint.DelayModel()
        m2.fromfile(cfg_file)
        self.assertEqual(m, m2, 'Saving delay model to file and loading it again failed')
        params = m.delay_params
        m3 = katpoint.DelayModel()
        m3.fromdelays(params)
        self.assertEqual(m, m3, 'Converting delay model to delay parameters and loading it again failed')

class TestCorrelatorDelays(unittest.TestCase):
    """Test correlator delay model."""
    def test_construct(self):
        pass
