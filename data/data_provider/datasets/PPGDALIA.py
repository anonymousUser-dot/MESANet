"""Native multirate PPG-DaLiA wrist-sensor forecasting provider."""

from data.data_provider.datasets.WESAD import Data as _WESADData


class Data(_WESADData):
    dataset_name = "PPGDALIA"


from data.data_provider.datasets.HumanActivity import collate_fn, collate_fn_patch, collate_fn_tpatch
