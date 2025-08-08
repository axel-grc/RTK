import re
import argparse
from itk import RTK as rtk
import difflib
import inspect

__all__ = [
    "rtk_argument_parser"
]

class RTKHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _format_usage(self, usage, actions, groups, prefix=None):
        # Keep standard usage line, just prepend version
        if prefix is None:
            prefix = (rtk.version() + "\n\nusage: ")
        return super()._format_usage(usage, actions, groups, prefix)


class rtk_argument_parser(argparse.ArgumentParser):
    def __init__(self, description=None, **kwargs):
        super().__init__(description=description, **kwargs)
        self.formatter_class = RTKHelpFormatter
        self.add_argument('-V', '--version', action='version', version=rtk.version())

    def build_signature(self) -> inspect.Signature:
        """Build a compact Python signature: only required kwargs + **kwargs."""
        required_params = []
        for action in self._actions:
            name = getattr(action, 'dest', None)
            if not name or name in ("help", "version"):
                continue
            if getattr(action, 'required', False):
                required_params.append(
                    inspect.Parameter(name=name, kind=inspect.Parameter.KEYWORD_ONLY)
                )
        # Add catch-all for the many optional CLI options to keep help inline
        var_kw = inspect.Parameter('kwargs', kind=inspect.Parameter.VAR_KEYWORD)
        return inspect.Signature(required_params + [var_kw])

    def build_usage_examples(self, app_name: str | None = None) -> str:
        """Return a Usage examples block for Python help()."""
        name = app_name or self.prog
        # Collect required destinations
        req = [a.dest for a in self._actions
               if getattr(a, 'required', False) and a.dest and a.dest not in ("help", "version")]
        shell = f"{name}(\"{' '.join([f'--{d} {d.upper()}' for d in req])}\")"
        py = f"{name}({', '.join([f'{d}={d.upper()}' for d in req])})"
        return (
            "Usage:\n"
            f"    • Shell-style: {shell}\n"
            f"    • Python API:  {py}\n\n"
        )

    def apply_signature(self, func):
        """Apply the built signature to a callable and return it."""
        func.__signature__ = self.build_signature()
        return func

    @staticmethod
    def _comma_separated_args(value_type):
        """Return a converter that accepts either a single value or a comma list."""
        def parser(value):
            if isinstance(value, str) and ',' in value:
                parts = [s.strip() for s in value.split(',') if s.strip()]
                return [value_type(s) for s in parts]
            return value_type(value)
        return parser

    def _wrap_actions(self):
        """At parse-time, wrap any action with nargs to accept comma tokens.
        Needed for options added via argument groups that didn't go through
        our overridden add_argument.
        """
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
        """When nargs='+', enable comma-separated tokens and mark the action."""
        if kwargs.get('nargs') == '+':
            orig_type = kwargs.get('type', str)
            kwargs['type'] = self._comma_separated_args(orig_type)
            action = super().add_argument(*args, **kwargs)
            setattr(action, '_rtk_comma_sep', True)
            return action
        return super().add_argument(*args, **kwargs)

    def parse_args(self, args=None, namespace=None):
        """Parse args with support for comma lists and flattening.
        - Ensure group-added actions are wrapped before parsing
        - After parsing, flatten nested lists once (argparse can nest)
        """
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

    def parse_kwargs(self, func_name: str | None = None, **kwargs):
        """Convert Python kwargs to argv and parse them.
        Behavior:
        - Unknown keys raise TypeError with close-match suggestions
        - Prefer long flags when available
        - For actions marked _rtk_comma_sep (i.e., nargs set), list/tuple values
          are emitted as a single comma-separated token to mirror CLI input.
          Otherwise list/tuple values are expanded as separate tokens.
        This preserves Python API parity with the CLI while keeping legacy
        RTK argument-group additions working.
        """
        # Collect actions by dest
        actions = {a.dest: a for a in self._actions if a.dest and a.dest not in ("help", "version")}

        # Early reject unknown keys
        for key in kwargs:
            if key not in actions:
                matches = difflib.get_close_matches(key, actions.keys(), n=3, cutoff=0.5)
                name = func_name or self.prog or "function"
                msg = f"{name}() got an unexpected keyword argument '{key}'"
                if matches:
                    msg += f"\nDid you mean: {', '.join(matches)}?"
                else:
                    msg += f"\nValid arguments are: {', '.join(sorted(actions.keys()))}"
                raise TypeError(msg)

        # Build argv
        argv = []
        for key, val in kwargs.items():
            action = actions[key]
            opt_strings = list(action.option_strings)
            # Prefer long flag if available
            flag = None
            for element in opt_strings:
                if element.startswith('--'):
                    flag = element
                    break
            if flag is None:
                flag = opt_strings[0]

            if isinstance(val, bool):
                if val:
                    argv.append(flag)
            elif isinstance(val, (list, tuple)):
                if getattr(action, "_rtk_comma_sep", False):
                    argv += [flag, ",".join(map(str, val))]
                else:
                    argv.append(flag)
                    argv.extend(map(str, val))
            else:
                argv += [flag, str(val)]

        return self.parse_args(argv)
