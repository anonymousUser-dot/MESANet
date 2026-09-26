"""MESANet: affine-anchored dual-coordinate forecasting backbone.

The fixed forward path normalizes each observed sample-variable history around
its deployment-visible affine anchor, keeps multiresolution support and ordered
waveform coordinates separate, and combines their forecasts through bounded
DCT-subspace separation.
"""

from copy import copy

from models.CAGNet import Model as CAGNetModel
from utils.ExpConfigs import ExpConfigs


class Model(CAGNetModel):
    """Fixed MESANet backbone used for benchmark and deployment runs.

    An empty ``ablation_name`` selects the frozen dual-coordinate path.
    Controlled ablations pass an explicit mode through this entry point.
    """

    def __init__(self, configs: ExpConfigs):
        mode = str(getattr(configs, "ablation_name", "") or "").strip()
        if not mode or mode == "mesa_final":
            configs = copy(configs)
            configs.ablation_name = (
                "aa_no_q_graph_no_prob_waveform_token_parallel_"
                "dct_dual_coordinate_dct_dual_rho_half_"
                "dct_dual_no_causal"
            )
        super().__init__(configs)
