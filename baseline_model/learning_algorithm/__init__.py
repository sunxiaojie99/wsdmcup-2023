from __future__ import absolute_import
from .base_algorithm import *
from .dla import *
from .naive import *

def list_available() -> list:
    from .base_algorithm import BaseAlgorithm
    from baseline_model.utils.sys_tools import list_recursive_concrete_subclasses
    return list_recursive_concrete_subclasses(BaseAlgorithm)
