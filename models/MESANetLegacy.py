"""Legacy MESANet entry point for reproducing pre-v622 benchmark tables."""

from copy import copy

from models.CAGNet import Model as CAGNetModel
from utils.ExpConfigs import ExpConfigs


class Model(CAGNetModel):
    """Freeze the pre-v622 closed-admission forward path."""

    def __init__(self, configs: ExpConfigs):
        if not str(getattr(configs, "ablation_name", "") or "").strip():
            configs = copy(configs)
            configs.ablation_name = "no_q_graph_no_prob"
        super().__init__(configs)
