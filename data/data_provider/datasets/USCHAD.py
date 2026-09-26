import torch
from pathlib import Path
from utils.globals import logger
from utils.ExpConfigs import ExpConfigs
from data.dependencies.WearableActivity import (
    WearableActivity,
    Wearable_time_chunk,
    attach_subject_weights,
    split_wearable_records,
)
from data.data_provider.datasets.HumanActivity import (
    collate_fn,
    collate_fn_patch,
    collate_fn_tpatch,
)
from data.data_provider.datasets.wearable_cache import (
    load_wearable_splits,
    save_stacked_cache,
)


class Data:
    def __init__(self, configs: ExpConfigs, flag: str = "train", **kwargs):
        logger.debug(f"getting {flag} set of USCHAD")
        self.configs = configs
        assert flag in ["train", "test", "val", "test_all"]
        self.flag = flag
        self.preprocess()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def preprocess(self):
        mask_protocol = str(getattr(self.configs, "wearable_mask_protocol", "group_async_current") or "group_async_current")
        mask_seed = int(getattr(self.configs, "seed_base", 0))
        split_protocol = str(getattr(self.configs, "wearable_split_protocol", "record_chronological") or "record_chronological")
        subject_fold = int(getattr(self.configs, "wearable_subject_fold", 0))
        test_subject = str(getattr(self.configs, "wearable_test_subject", "") or "")
        cache_tag = WearableActivity.cache_tag(mask_protocol, mask_seed, split_protocol, subject_fold, test_subject)
        mode = str(getattr(self.configs, "ablation_name", "") or "").lower()
        needs_subject_ids = "subject_balanced" in mode or "subject_groupdro" in mode
        subject_suffix = "_sid" if needs_subject_ids else ""
        cache_path = Path(self.configs.dataset_root_path) / "processed" / f"chunks_sl{self.configs.seq_len}_pl{self.configs.pred_len}{cache_tag}{subject_suffix}.pt"
        if cache_path.exists():
            train_data, val_data, test_data = load_wearable_splits(cache_path)
        else:
            dataset = WearableActivity(
                root=self.configs.dataset_root_path,
                dataset="USCHAD",
                mask_protocol=mask_protocol,
                mask_seed=mask_seed,
                split_protocol=split_protocol,
                subject_fold=subject_fold,
                test_subject=test_subject,
            )
            train_data, val_data, test_data = split_wearable_records(dataset, split_protocol, subject_fold, test_subject)
            train_data = Wearable_time_chunk(train_data, self.configs)
            val_data = Wearable_time_chunk(val_data, self.configs)
            test_data = Wearable_time_chunk(test_data, self.configs)
            splits = {"train": train_data, "val": val_data, "test": test_data}
            torch.save(splits, cache_path)
            save_stacked_cache(cache_path, splits)
        train_data = attach_subject_weights(train_data)
        val_data = attach_subject_weights(val_data)
        test_data = attach_subject_weights(test_data)
        all_data = train_data + val_data + test_data
        self._set_max_lengths(all_data)
        if self.flag == "test_all":
            self.data = all_data
        elif self.flag == "train":
            self.data = train_data
        elif self.flag == "val":
            self.data = val_data
        else:
            self.data = test_data

    def _set_max_lengths(self, data):
        self.configs.seq_len_max_irr = max((sample["x"].shape[0] for sample in data), default=self.configs.seq_len)
        self.configs.pred_len_max_irr = max((sample["y"].shape[0] for sample in data), default=self.configs.pred_len)
        self.configs.patch_len_max_irr = max(self.configs.seq_len_max_irr, self.configs.pred_len_max_irr)
