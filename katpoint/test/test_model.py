################################################################################
# Copyright (c) 2009-2016, National Research Foundation (Square Kilometre Array)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

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

    def test_construct_save_load(self):
        """Test construction / save / load of generic model."""
        m = katpoint.Model(self.new_params())
        m.header['date'] = '2014-01-15'
        # Exercise all string representations for coverage purposes
        print repr(m), m, repr(m.params['POS_E'])
        # An empty file should lead to a BadModelFile exception
        cfg_file = StringIO.StringIO()
        self.assertRaises(katpoint.BadModelFile, m.fromfile, cfg_file)
        m.tofile(cfg_file)
        cfg_str = cfg_file.getvalue()
        cfg_file.close()
        # Load the saved config file
        cfg_file = StringIO.StringIO(cfg_str)
        m2 = katpoint.Model(self.new_params())
        m2.fromfile(cfg_file)
        self.assertEqual(m, m2, 'Saving model to file and loading it again failed')
        cfg_file = StringIO.StringIO(cfg_str)
        m2.set(cfg_file)
        self.assertEqual(m, m2, 'Saving model to file and loading it again failed')
        # Build model from description string
        m3 = katpoint.Model(self.new_params())
        m3.fromstring(m.description)
        self.assertEqual(m, m3, 'Saving model to string and loading it again failed')
        m3.set(m.description)
        self.assertEqual(m, m3, 'Saving model to string and loading it again failed')
        # Build model from sequence of floats
        m4 = katpoint.Model(self.new_params())
        m4.fromlist(m.values())
        self.assertEqual(m, m4, 'Saving model to list and loading it again failed')
        m4.set(m.values())
        self.assertEqual(m, m4, 'Saving model to list and loading it again failed')
        # Empty model
        cfg_file = StringIO.StringIO('[header]\n[params]\n')
        m5 = katpoint.Model(self.new_params())
        m5.fromfile(cfg_file)
        print m5
        self.assertNotEqual(m, m5, 'Model should not be equal to an empty one')
        m6 = katpoint.Model(self.new_params())
        m6.set()
        self.assertEqual(m6, m5, 'Setting empty model failed')
        m7 = katpoint.Model(self.new_params())
        m7.set(m)
        self.assertEqual(m, m7, 'Construction from model object failed')
        class OtherModel(katpoint.Model):
            pass
        m8 = OtherModel(self.new_params())
        self.assertRaises(katpoint.BadModelFile, m8.set, m)
        try:
            self.assertEqual(hash(m), hash(m4), 'Model hashes not equal')
        except TypeError:
            self.fail('Model object not hashable')

    def test_dict_interface(self):
        """Test dict interface of generic model."""
        params = self.new_params()
        names = [p.name for p in params]
        values = [p.value for p in params]
        m = katpoint.Model(params)
        self.assertEqual(len(m), 6, 'Unexpected model length')
        self.assertEqual(m.keys(), names, 'Parameter names do not match')
        self.assertEqual(m.values(), values, 'Parameter values do not match')
        m['NIAO'] = 6789.0
        self.assertEqual(m['NIAO'], 6789.0, 'Parameter setting via dict interface failed')

    def test_writable_docstring(self):
        """Check that model class docstring is writable."""
        params = self.new_params()
        m = katpoint.Model(params)
        katpoint.Model.__doc__ = 'Hooray'
        self.assertEqual(katpoint.Model.__doc__, 'Hooray', 'Class docstring not writable')
