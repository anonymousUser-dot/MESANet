"""Native multirate WESAD wrist-sensor forecasting provider."""

from pathlib import Path

import torch

from data.data_provider.datasets.HumanActivity import collate_fn, collate_fn_patch, collate_fn_tpatch
from data.data_provider.datasets.wearable_cache import load_wearable_splits, save_stacked_cache
from data.dependencies.WearableActivity import (
    WearableActivity,
    Wearable_time_chunk,
    attach_subject_weights,
    split_wearable_records,
)
from utils.ExpConfigs import ExpConfigs
from utils.globals import logger


class Data:
    dataset_name = "WESAD"

    def __init__(self, configs: ExpConfigs, flag: str = "train", **kwargs):
        logger.debug("getting %s set of %s", flag, self.dataset_name)
        self.configs = configs
        if flag not in {"train", "test", "val", "test_all"}:
            raise ValueError(f"unknown {self.dataset_name} split: {flag}")
        self.flag = flag
        self.preprocess()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def preprocess(self):
        mask_protocol = str(
            getattr(self.configs, "wearable_mask_protocol", "real_missing_only_if_available")
            or "real_missing_only_if_available"
        )
        mask_seed = int(getattr(self.configs, "seed_base", 0))
        split_protocol = str(
            getattr(self.configs, "wearable_split_protocol", "subject_heldout") or "subject_heldout"
        )
        subject_fold = int(getattr(self.configs, "wearable_subject_fold", 0))
        test_subject = str(getattr(self.configs, "wearable_test_subject", "") or "")
        cache_tag = WearableActivity.cache_tag(
            mask_protocol, mask_seed, split_protocol, subject_fold, test_subject
        )
        mode = str(getattr(self.configs, "ablation_name", "") or "").lower()
        needs_subject_ids = "subject_balanced" in mode or "subject_groupdro" in mode
        subject_suffix = "_sid" if needs_subject_ids else ""
        cache_path = (
            Path(self.configs.dataset_root_path)
            / "processed"
            / f"chunks_sl{self.configs.seq_len}_pl{self.configs.pred_len}{cache_tag}{subject_suffix}.pt"
        )
        if cache_path.exists():
            train_data, val_data, test_data = load_wearable_splits(cache_path)
        else:
            dataset = WearableActivity(
                root=self.configs.dataset_root_path,
                dataset=self.dataset_name,
                mask_protocol=mask_protocol,
                mask_seed=mask_seed,
                split_protocol=split_protocol,
                subject_fold=subject_fold,
                test_subject=test_subject,
            )
            train_data, val_data, test_data = split_wearable_records(
                dataset, split_protocol, subject_fold, test_subject
            )
            train_data = Wearable_time_chunk(train_data, self.configs)
            val_data = Wearable_time_chunk(val_data, self.configs)
            test_data = Wearable_time_chunk(test_data, self.configs)
            splits = {"train": train_data, "val": val_data, "test": test_data}
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(splits, cache_path)
            save_stacked_cache(cache_path, splits)
        train_data = attach_subject_weights(train_data)
        val_data = attach_subject_weights(val_data)
        test_data = attach_subject_weights(test_data)
        all_data = train_data + val_data + test_data
        self.configs.seq_len_max_irr = max(
            (sample["x"].shape[0] for sample in all_data), default=self.configs.seq_len
        )
        self.configs.pred_len_max_irr = max(
            (sample["y"].shape[0] for sample in all_data), default=self.configs.pred_len
        )
        self.configs.patch_len_max_irr = max(
            self.configs.seq_len_max_irr, self.configs.pred_len_max_irr
        )
        split_map = {
            "train": train_data,
            "val": val_data,
            "test": test_data,
            "test_all": all_data,
        }
        self.data = split_map[self.flag]
