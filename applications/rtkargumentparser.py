import re
import argparse
from gettext import gettext as _
from itk import RTK as rtk

__all__ = [
    "rtk_argument_parser"
]

class RTKHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _format_usage(self, usage, actions, groups, prefix=None):
        if prefix is None:
            prefix = (rtk.version() + "\n\nusage: ")
        return super()._format_usage(usage, actions, groups, prefix)

class rtk_argument_parser(argparse.ArgumentParser):
    def __init__(self, description=None, **kwargs):
        super().__init__(description=description, **kwargs)
        self.formatter_class = RTKHelpFormatter
        # allow negative numbers in scientific notation
        self._negative_number_matcher = re.compile(r'^-?[\d\.,eE+\-]+$')
        self.add_argument('-V', '--version', action='version', version=rtk.version())

    @staticmethod
    def _comma_separated_args(value_type):
        def parser(value):
            if isinstance(value, str) and ',' in value:
                parts = [s.strip() for s in value.split(',') if s.strip()]
                return [value_type(s) for s in parts]
            return value_type(value)
        return parser

    def _wrap_actions(self):
        # Wrap actions added so comma-containing tokens are handled.
        for action in self._actions:
            if getattr(action, "_rtk_comma_sep", False):
                continue
            if not getattr(action, "nargs", None):
                continue

            orig_type = getattr(action, "type", None)
            action.type = self._comma_separated_args(orig_type)
            setattr(action, "_rtk_comma_sep", True)

    def add_argument(self, *args, **kwargs):
        # keep the convenience of marking nargs='+'
        if kwargs.get('nargs') == '+':
            orig_type = kwargs.get('type', str)
            kwargs['type'] = self._comma_separated_args(orig_type)
            action = super().add_argument(*args, **kwargs)
            setattr(action, '_rtk_comma_sep', True)
            return action
        return super().add_argument(*args, **kwargs)

    def parse_args(self, args=None, namespace=None):
        # ensure actions added by RTK are wrapped before actual parsing
        self._wrap_actions()
        namespace = super().parse_args(args, namespace)
        # post-process: convert tuples->lists and flatten nested lists for marked actions
        for action in self._actions:
            if getattr(action, "_rtk_comma_sep", False):
                val = getattr(namespace, action.dest, None)
                if isinstance(val, tuple):
                    val = list(val)
                if isinstance(val, (list, tuple)) and not isinstance(val, (str, bytes)):
                    # flatten one level
                    flat = []
                    for item in val:
                        flat.extend(item if isinstance(item, (list, tuple)) else [item])
                    val = flat
                setattr(namespace, action.dest, val)
        return namespace
