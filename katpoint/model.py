"""Model base class.

This provides a base class for pointing and delay models, handling the loading,
saving and display of parameters.

"""

import ConfigParser


class Parameter(object):
    """Generic model parameter."""
    def __init__(self, name, units, doc, from_str=float, to_str=str, value=0.0):
        self.name = name
        self.units = units
        self.doc = doc
        self._from_str = from_str
        self._to_str = to_str
        self.value = value

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
    pass


class Model(object):
    """Base class for models (e.g. pointing and delay models).

    Parameters
    ----------
    params : sequence of :class:`Parameter` objects
        Full set of model parameters in the expected order

    """
    def __init__(self, params):
        self.header = {}
        self.params = params

    def param_strs(self):
        name_len = max([len(param.name) for param in self.params])
        value_len = max([len(param.value_str) for param in self.params])
        units_len = max([len(param.units) for param in self.params])
        return [(p.name.ljust(name_len), p.value_str.ljust(value_len),
                 p.units.ljust(units_len), p.doc) for p in self.params if p.value]

    def __len__(self):
        """Number of parameters in full model."""
        return len(self.params)

    def __repr__(self):
        """Short human-friendly string representation of model object."""
        num_active = len([p for p in self.params if p.value])
        return "<katpoint.%s active_params=%d/%d at 0x%x>" % \
               (self.__class__.__name__, num_active, len(self), id(self))

    def __str__(self):
        """Verbose human-friendly string representation of model object."""
        num_active = len([p for p in self.params if p.value])
        summary = "%s has %d parameters with %d active (non-zero)" % \
                  (self.__class__.__name__, len(self), num_active)
        if num_active == 0:
            return summary
        return summary + ':\n' + '\n'.join(('%s = %s %s (%s)' % ps)
                                           for ps in self.param_strs())

    def __eq__(self, other):
        """Equality comparison operator."""
        return self.description == (other.description if isinstance(other, self.__class__) else other)

    def __ne__(self, other):
        """Inequality comparison operator."""
        return not (self == other)

    @property
    def description(self):
        """Compact string representation of model, sufficient to reconstruct it."""
        return ', '.join(p.value_str for p in self.params)

    def loads(self, description):
        """Load model from description string."""
        self.header = {}
        # Split string either on commas or whitespace
        param_vals = [p.strip() for p in description.split(',')] \
                     if ',' in description else description.split()
        n = min(len(param_vals), len(self))
        for param, param_val in zip(self.params[:n], param_vals[:n]):
            param.value_str = param_val

    def load(self, file_like):
        """Load model from config file."""
        defaults = dict([(param.name, '0.0') for param in self.params])
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
        for param in self.params:
            param.value_str = cfg.get('params', param.name)

    def save(self, file_like):
        """Save model to config file."""
        cfg = ConfigParser.SafeConfigParser()
        cfg.add_section('header')
        for key, val in self.header.items():
            cfg.set('header', key, str(val))
        cfg.add_section('params')
        for param_str in self.param_strs():
            cfg.set('params', param_str[0], '%s ; %s (%s)' % param_str[1:])
        cfg.write(file_like)