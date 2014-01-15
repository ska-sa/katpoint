"""Tests for the model module."""

import unittest
import StringIO

import katpoint


class TestModel(unittest.TestCase):
    """Test generic model."""
    def new_params(self):
        """Generate fresh set of parameters (otherwise models share the same ones)."""
        params = []
        params.append(katpoint.Parameter('POS_E', 'm', 'East', value=10.0))
        params.append(katpoint.Parameter('POS_N', 'm', 'North', value=-9.0))
        params.append(katpoint.Parameter('POS_U', 'm', 'Up', value=3.0))
        params.append(katpoint.Parameter('NIAO', 'm', 'non-inter', value=0.88))
        params.append(katpoint.Parameter('CAB_H', '', 'horizontal', value=20.2))
        params.append(katpoint.Parameter('CAB_V', 'deg', 'vertical', value=20.3))
        return params

    def test_model_save_load(self):
        """Test construction / save / load of generic model."""
        m = katpoint.Model(self.new_params())
        m.header['date'] = '2014-01-15'
        # Exercise all string representations for coverage purposes
        print repr(m), m, repr(m.params[0])
        # An empty file should lead to a BadModelFile exception
        cfg_file = StringIO.StringIO()
        self.assertRaises(katpoint.BadModelFile, m.load, cfg_file)
        m.save(cfg_file)
        cfg_str = cfg_file.getvalue()
        cfg_file.close()
        # Load the saved config file
        cfg_file = StringIO.StringIO(cfg_str)
        m2 = katpoint.Model(self.new_params())
        m2.load(cfg_file)
        self.assertEqual(m, m2, 'Saving model to file and loading it again failed')
        # Build model from description string
        m3 = katpoint.Model(self.new_params())
        m3.loads(m.description)
        self.assertEqual(m, m3, 'Saving model to string and loading it again failed')
        # Empty model
        cfg_file = StringIO.StringIO('[header]\n[params]\n')
        m4 = katpoint.Model(self.new_params())
        m4.load(cfg_file)
        print m4
        self.assertNotEqual(m, m4, 'Model should not be equal to an empty one')
