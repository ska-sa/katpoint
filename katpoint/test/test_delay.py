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

class TestDelayCorrection(unittest.TestCase):
    """Test correlator delay corrections."""
    def setUp(self):
        self.target = katpoint.construct_azel_target('45:00:00.0', '75:00:00.0')
        self.ant1 = katpoint.Antenna('A1, -31.0, 18.0, 0.0, 12.0, 0.0 0.0 0.0')
        self.ant2 = katpoint.Antenna('A2, -31.0, 18.0, 0.0, 12.0, 10.0 -10.0 0.0')
        self.ant3 = katpoint.Antenna('A3, -31.0, 18.0, 0.0, 12.0, 5.0 10.0 3.0')
        self.ts = katpoint.Timestamp('2013-08-14 08:25')

    def test_correction(self):
        delays = katpoint.DelayCorrection([self.ant2, self.ant3], self.ant1, 1.285e9)
        delay0, phase0 = delays.corrections(self.target, self.ts)
        delay1, phase1 = delays.corrections(self.target, self.ts, self.ts + 1.0)
