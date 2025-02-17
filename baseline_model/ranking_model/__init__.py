from .base_ranking_model import *
from .DNN import *


def list_available() -> list:
    from .base_ranking_model import BaseRankingModel
    from baseline_model.utils.sys_tools import list_recursive_concrete_subclasses
    return list_recursive_concrete_subclasses(BaseRankingModel)
