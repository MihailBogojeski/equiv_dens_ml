"""Dependency-free support for retained script and helper-module entry points."""

from importlib import import_module
import runpy
import sys
from types import ModuleType


class _ModuleForwarder(ModuleType):
    """Also forward writes from callers using importlib's explicit loaders."""

    def __getattribute__(self, name):
        if name not in {"_implementation", "__class__", "__dict__"}:
            implementation = ModuleType.__getattribute__(self, "_implementation")
            return getattr(implementation, name)
        return ModuleType.__getattribute__(self, name)

    def __setattr__(self, name, value):
        if name in {"_implementation", "__class__"}:
            ModuleType.__setattr__(self, name, value)
        else:
            setattr(ModuleType.__getattribute__(self, "_implementation"), name, value)

    def __delattr__(self, name):
        delattr(ModuleType.__getattribute__(self, "_implementation"), name)


def forward_module(namespace, target):
    """Preserve old helper imports and monkeypatches without copying code."""
    name = namespace["__name__"]
    implementation = import_module(target)
    current = sys.modules.get(name)
    if current is not None and current is not implementation:
        current._implementation = implementation
        current.__class__ = _ModuleForwarder
    sys.modules[name] = implementation
    namespace.update(vars(implementation))


def run_module(target):
    """Execute with proper __main__ registration, including spawn workers."""
    runpy.run_module(target, run_name="__main__", alter_sys=True)
