"""Model base class.

This provides a base class for pointing and delay models, handling the loading,
saving and display of parameters.

"""

import ConfigParser
try:
    # Python 2.7 and above have builtin support
    from collections import OrderedDict
except ImportError:
    # Alternative for Python 2.4, 2.5, 2.6, pypy from http://code.activestate.com/recipes/576693/
    from .ordereddict import OrderedDict

import numpy as np

from .ephem_extra import is_iterable


class Parameter(object):
    """Generic model parameter."""
    def __init__(self, name, units, doc, from_str=float, to_str=str,
                 value=None, default_value=0.0):
        self.name = name
        self.units = units
        self.__doc__ = doc
        self._from_str = from_str
        self._to_str = to_str
        self.value = value if value is not None else default_value
        self.default_value = default_value

    def __nonzero__(self):
        """True if parameter is active, i.e. its value differs from default."""
        # Do explicit cast to bool as value can be a NumPy type, resulting in
        # an np.bool_ type for the expression (not allowed for __nonzero__)
        return bool(self.value != self.default_value)

    @property
    def value_str(self):
        """String form of parameter value."""
        return self._to_str(self.value)
    @value_str.setter
    def value_str(self, valstr):
        self.value = self._from_str(valstr)

    def __repr__(self):
        """Short human-friendly string representation of parameter object."""
        return "<katpoint.Parameter %s = %s %s at 0x%x>" % \
               (self.name, self.value_str, self.units, id(self))


class BadModelFile(Exception):
    """Unable to load model from config file (unrecognised format)."""
    pass


# Use metaclass trick to make Model class docstrings writable.
# This is unnecessary in Python 3.3 and above.
# For more info, see Python issue 12773 (http://bugs.python.org/issue12773)
# and discussion on python-dev:
# https://mail.python.org/pipermail/python-dev/2012-January/115656.html
class WritableDocstring(type):
    """Metaclass with the sole purpose of enabling writable class docstrings."""


class Model(object):

    __metaclass__ = WritableDocstring

    __doc__ = """Base class for models (e.g. pointing and delay models).

    Parameters
    ----------
    params : sequence of :class:`Parameter` objects
        Full set of model parameters in the expected order

    """
    def __init__(self, params):
        self.header = {}
        self.params = OrderedDict((p.name, p) for p in params)

    def __len__(self):
        """Number of parameters in full model."""
        return len(self.params)

    def __nonzero__(self):
        """True if model contains any active parameters."""
        return any(p for p in self)

    def __iter__(self):
        """Iterate over parameter objects."""
        return self.params.itervalues()

    def param_strs(self):
        """Justified (name, value, units, doc) strings for active parameters."""
        name_len = max(len(p.name) for p in self)
        value_len = max(len(p.value_str) for p in self.params.itervalues())
        units_len = max(len(p.units) for p in self.params.itervalues())
        return [(p.name.ljust(name_len), p.value_str.ljust(value_len),
                 p.units.ljust(units_len), p.__doc__)
                for p in self.params.itervalues() if p]

    def __repr__(self):
        """Short human-friendly string representation of model object."""
        num_active = len([p for p in self if p])
        return "<katpoint.%s active_params=%d/%d at 0x%x>" % \
               (self.__class__.__name__, num_active, len(self), id(self))

    def __str__(self):
        """Verbose human-friendly string representation of model object."""
        num_active = len([p for p in self if p])
        summary = "%s has %d parameters with %d active (non-default)" % \
                  (self.__class__.__name__, len(self), num_active)
        if num_active == 0:
            return summary
        return summary + ':\n' + '\n'.join(('%s = %s %s (%s)' % ps)
                                           for ps in self.param_strs())

    def __eq__(self, other):
        """Equality comparison operator."""
        return self.description == \
               (other.description if isinstance(other, self.__class__) else other)

    def __ne__(self, other):
        """Inequality comparison operator."""
        return not (self == other)

    def __getitem__(self, key):
        """Access parameter value by name."""
        return self.params[key].value

    def __setitem__(self, key, value):
        """Modify parameter value by name."""
        self.params[key].value = value

    def keys(self):
        """List of parameter names in the expected order."""
        return self.params.keys()

    def values(self):
        """List of parameter values in the expected order ('tolist')."""
        return [p.value for p in self]

    def fromlist(self, floats):
        """Load model from sequence of floats."""
        self.header = {}
        params = [p for p in self]
        min_len = min(len(params), len(floats))
        for param, value in zip(params[:min_len], floats[:min_len]):
            param.value = value
        for param in params[min_len:]:
            param.value = param.default_value

    @property
    def description(self):
        """Compact string representation, sufficient to reconstruct model ('tostring')."""
        active = np.nonzero([bool(p) for p in self])[0]
        last_active = active[-1] if len(active) else -1
        return ' '.join([p.value_str for p in self][:last_active + 1])

    def fromstring(self, description):
        """Load model from description string."""
        self.header = {}
        # Split string either on commas or whitespace, for good measure
        param_vals = [p.strip() for p in description.split(',')] \
                     if ',' in description else description.split()
        params = [p for p in self]
        min_len = min(len(params), len(param_vals))
        for param, param_val in zip(params[:min_len], param_vals[:min_len]):
            param.value_str = param_val
        for param in params[min_len:]:
            param.value = param.default_value

    def tofile(self, file_like):
        """Save model to config file."""
        cfg = ConfigParser.SafeConfigParser()
        cfg.add_section('header')
        for key, val in self.header.items():
            cfg.set('header', key, str(val))
        cfg.add_section('params')
        for param_str in self.param_strs():
            cfg.set('params', param_str[0], '%s ; %s (%s)' % param_str[1:])
        cfg.write(file_like)

    def fromfile(self, file_like):
        """Load model from config file."""
        defaults = dict((p.name, p._to_str(p.default_value)) for p in self)
        cfg = ConfigParser.SafeConfigParser(defaults)
        try:
            cfg.readfp(file_like)
            if cfg.sections() != ['header', 'params']:
                raise ConfigParser.Error('Expected sections not found in model file')
        except ConfigParser.Error, exc:
            filename = getattr(file_like, 'name', '')
            msg = 'Could not construct %s from %s\n\nOriginal exception: %s' % \
                  (self.__class__.__name__,
                   ('file %r' % (filename,)) if filename else 'file-like object',
                   str(exc))
            raise BadModelFile(msg)
        self.header = dict(cfg.items('header'))
        for param in defaults:
            self.header.pop(param.lower())
        for param in self:
            param.value_str = cfg.get('params', param.name)

    def set(self, model=None):
        """Load parameter values from the appropriate source.

        Parameters
        ----------
        model : file-like or model object, sequence of floats, or string, optional
            Model specification. If this is a file-like or model object, load
            the model from it. If this is a sequence of floats, accept it
            directly as the model parameters (defaults to sequence of zeroes).
            If it is a string, interpret it as a comma-separated (or whitespace-
            separated) sequence of parameters in their string form (i.e. a
            description string). The default is an empty model.

        """
        if isinstance(model, Model):
            if not isinstance(model, type(self)):
                raise BadModelFile('Cannot construct a %r from a %r' %
                                   (self.__class__.__name__,
                                    model.__class__.__name__))
            self.fromlist(model.values())
            self.header = dict(model.header)
        elif isinstance(model, basestring):
            self.fromstring(model)
        else:
            array = np.atleast_1d(model)
            if array.dtype.kind in 'iuf' and array.ndim == 1:
                self.fromlist(model)
            elif model is not None:
                self.fromfile(model)
            else:
                self.fromlist([])
