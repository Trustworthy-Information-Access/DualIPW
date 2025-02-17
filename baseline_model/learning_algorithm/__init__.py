from .base_algorithm import *
from .naive_dnn import *
from .dla_dnn import *
from .PRS_dnn import *
from .UPE_dnn import *
from .dualipw_dnn import *
from .double_drop import *
from .double_gradrev import *

def list_available() -> list:
    from .base_algorithm import BaseAlgorithm
    from baseline_model.utils.sys_tools import list_recursive_concrete_subclasses
    return list_recursive_concrete_subclasses(BaseAlgorithm)
