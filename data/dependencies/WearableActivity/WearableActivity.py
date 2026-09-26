"""Wearable activity datasets as irregular forecasting records.

The original public datasets are activity-recognition datasets.  For the
HumanActivity-like forecasting direction we use their continuous sensor streams
and a fixed sensor-group asynchronous observation protocol.  This turns each
subject/session into records of the same shape used by the HumanActivity
provider: ``(record_id, tt, vals, mask)``.
"""

from __future__ import annotations

import io
import math
import os
import pickle
import re
import zlib
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import torch


@dataclass(frozen=True)
class WearableSpec:
    name: str
    url: str
    zip_name: str
    feature_dim: int
    downsample: int
    group_slices: tuple[tuple[int, ...], ...]


UCI_BASE = "https://archive.ics.uci.edu/static/public"


SPECS: dict[str, WearableSpec] = {
    "MHEALTH": WearableSpec(
        name="MHEALTH",
        url=f"{UCI_BASE}/319/mhealth+dataset.zip",
        zip_name="mhealth+dataset.zip",
        feature_dim=23,
        downsample=2,
        group_slices=(
            tuple(range(0, 5)),      # chest acceleration + ECG
            tuple(range(5, 14)),     # left ankle IMU
            tuple(range(14, 23)),    # right wrist IMU
        ),
    ),
    "PAMAP2": WearableSpec(
        name="PAMAP2",
        url=f"{UCI_BASE}/231/pamap2+physical+activity+monitoring.zip",
        zip_name="pamap2+physical+activity+monitoring.zip",
        feature_dim=37,
        downsample=5,
        group_slices=(
            (0,),                    # heart rate, naturally lower-rate
            tuple(range(1, 13)),     # hand IMU
            tuple(range(13, 25)),    # chest IMU
            tuple(range(25, 37)),    # ankle IMU
        ),
    ),
    "OPPORTUNITY": WearableSpec(
        name="OPPORTUNITY",
        url=f"{UCI_BASE}/226/opportunity+activity+recognition.zip",
        zip_name="opportunity+activity+recognition.zip",
        feature_dim=64,
        downsample=3,
        group_slices=(
            tuple(range(0, 16)),
            tuple(range(16, 32)),
            tuple(range(32, 48)),
            tuple(range(48, 64)),
        ),
    ),
    "USCHAD": WearableSpec(
        name="USCHAD",
        url="http://sipi.usc.edu/had/USC-HAD.zip",
        zip_name="USC-HAD.zip",
        feature_dim=6,
        downsample=2,
        group_slices=(
            (0, 1, 2),               # accelerometer
            (3, 4, 5),               # gyroscope
        ),
    ),
    "REALDISP": WearableSpec(
        name="REALDISP",
        url=f"{UCI_BASE}/305/realdisp+activity+recognition+dataset.zip",
        zip_name="realdisp+activity+recognition+dataset.zip",
        feature_dim=117,
        downsample=10,
        group_slices=tuple(tuple(range(13 * i, 13 * (i + 1))) for i in range(9)),
    ),
    "WESAD": WearableSpec(
        name="WESAD",
        # UCI's 261-byte archive only contains this official Sciebo link.
        url="https://uni-siegen.sciebo.de/s/pYjSgfOVs6Ntahr/download",
        zip_name="WESAD.zip",
        feature_dim=6,
        downsample=1,
        group_slices=((0,), (1,), (2,), (3, 4, 5)),
    ),
    "PPGDALIA": WearableSpec(
        name="PPGDALIA",
        url=f"{UCI_BASE}/495/ppg+dalia.zip",
        zip_name="PPG_DaLiA.zip",
        feature_dim=6,
        downsample=1,
        group_slices=((0,), (1,), (2,), (3, 4, 5)),
    ),
}


class WearableActivity:
    def __init__(
        self,
        root: str | os.PathLike[str],
        dataset: str,
        download: bool = True,
        mask_protocol: str = "group_async_current",
        mask_seed: int = 0,
        split_protocol: str = "record_chronological",
        subject_fold: int = 0,
        test_subject: str = "",
    ):
        self.root = Path(root)
        self.dataset = dataset
        if dataset not in SPECS:
            raise ValueError(f"unknown wearable dataset: {dataset}")
        self.spec = SPECS[dataset]
        self.mask_protocol = mask_protocol.strip().lower()
        self.mask_seed = int(mask_seed)
        self.split_protocol = split_protocol.strip().lower()
        self.subject_fold = int(subject_fold)
        self.test_subject = str(test_subject).strip()
        valid_protocols = {
            "group_async_current",
            "random_mcar_matched_density",
            "block_dropout_matched_density",
            "real_missing_only_if_available",
        }
        if self.mask_protocol not in valid_protocols:
            raise ValueError(f"unknown wearable mask protocol: {mask_protocol}")
        if self.split_protocol not in {"record_chronological", "subject_heldout"}:
            raise ValueError(f"unknown wearable split protocol: {split_protocol}")
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError(f"{dataset} not found under {self.root}")
        self.data = torch.load(self._processed_data_file(), map_location="cpu")

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> Path:
        return self.root / "raw"

    @property
    def processed_folder(self) -> Path:
        return self.root / "processed"

    def _check_exists(self) -> bool:
        return self._processed_data_file().exists()

    def _processed_data_file(self) -> Path:
        cache_tag = self.cache_tag(
            self.mask_protocol,
            self.mask_seed,
            self.split_protocol,
            self.subject_fold,
            self.test_subject,
        )
        if self.dataset == "REALDISP":
            mode = os.environ.get("REALDISP_MEMBER_MODE", "small").strip().lower()
            limit = os.environ.get("REALDISP_MEMBER_LIMIT", "0").strip()
            if mode != "small" or limit not in {"", "0"}:
                return self.processed_folder / f"data_{mode}_lim{limit or '0'}{cache_tag}.pt"
        return self.processed_folder / f"data{cache_tag}.pt"

    @staticmethod
    def cache_tag(
        mask_protocol: str,
        mask_seed: int,
        split_protocol: str,
        subject_fold: int = 0,
        test_subject: str = "",
    ) -> str:
        mask_protocol = str(mask_protocol).strip().lower()
        split_protocol = str(split_protocol).strip().lower()
        if mask_protocol == "group_async_current":
            mask_tag = "groupasync"
        elif mask_protocol == "real_missing_only_if_available":
            mask_tag = "realmissing"
        else:
            safe = mask_protocol.replace("_matched_density", "").replace("_only_if_available", "")
            mask_tag = f"{safe}_s{int(mask_seed)}"
        split_tag = "subject" if split_protocol == "subject_heldout" else "record"
        if split_protocol == "subject_heldout" and str(test_subject).strip():
            safe_subject = re.sub(r"[^A-Za-z0-9]+", "", str(test_subject))
            split_tag = f"subjectloso_{safe_subject.lower()}"
        elif split_protocol == "subject_heldout" and int(subject_fold) != 0:
            split_tag = f"subjectf{int(subject_fold)}"
        return f"_trnorm_{split_tag}_{mask_tag}"

    def download(self) -> None:
        if self._check_exists():
            return
        self.raw_folder.mkdir(parents=True, exist_ok=True)
        self.processed_folder.mkdir(parents=True, exist_ok=True)
        extract_dir = self.raw_folder / "extracted"
        extracted_ready = extract_dir.exists() and any(extract_dir.iterdir())
        zip_path = self.raw_folder / self.spec.zip_name
        if self.dataset == "REALDISP" or not extracted_ready:
            if zip_path.exists() and not self._zip_is_valid(zip_path):
                zip_path.unlink()
            if not zip_path.exists():
                response = requests.get(self.spec.url, stream=True, timeout=120, verify=False)
                response.raise_for_status()
                with zip_path.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            if self.dataset in {"WESAD", "PPGDALIA"}:
                records = self._build_wesad_archive_records(zip_path)
                torch.save(records, self._processed_data_file())
                return
            if self.dataset == "REALDISP":
                extract_dir.mkdir(parents=True, exist_ok=True)
            elif not extracted_ready:
                extract_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    if self.dataset in {"WESAD", "PPGDALIA"}:
                        self._extract_subject_pickles(zf, extract_dir)
                    else:
                        zf.extractall(extract_dir)
        if self.dataset not in {"WESAD", "PPGDALIA"}:
            for nested_zip in extract_dir.rglob("*.zip"):
                nested_dir = nested_zip.with_suffix("")
                if not nested_dir.exists():
                    nested_dir.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(nested_zip, "r") as zf:
                        zf.extractall(nested_dir)
        records = self._build_records(extract_dir)
        torch.save(records, self._processed_data_file())

    @staticmethod
    def _extract_subject_pickles(zf: zipfile.ZipFile, extract_dir: Path) -> None:
        """Extract only subject payloads needed by the native-clock pipeline."""

        members = [
            info
            for info in zf.infolist()
            if not info.is_dir() and re.search(r"(?:^|/)S\d+/S\d+\.pkl$", info.filename)
        ]
        if not members:
            members = [
                info
                for info in zf.infolist()
                if not info.is_dir() and re.search(r"(?:^|/)S\d+\.pkl$", info.filename)
            ]
        if not members:
            raise RuntimeError("wearable archive contains no S<id>.pkl subject payloads")
        for info in members:
            zf.extract(info, extract_dir)

    def _zip_is_valid(self, zip_path: Path) -> bool:
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                return zf.testzip() is None
        except zipfile.BadZipFile:
            return False

    def _build_records(self, extract_dir: Path):
        if self.dataset in {"WESAD", "PPGDALIA"}:
            return self._build_wesad_records(extract_dir)
        if self.dataset == "MHEALTH":
            arrays = self._load_mhealth(extract_dir)
        elif self.dataset == "PAMAP2":
            arrays = self._load_pamap2(extract_dir)
        elif self.dataset == "OPPORTUNITY":
            arrays = self._load_opportunity(extract_dir)
        elif self.dataset == "USCHAD":
            arrays = self._load_uschad(extract_dir)
        elif self.dataset == "REALDISP":
            arrays = self._load_realdisp(extract_dir)
        else:
            raise ValueError(self.dataset)
        deduplicated: dict[str, np.ndarray] = {}
        for record_id, array in arrays:
            previous = deduplicated.get(record_id)
            if previous is None or len(array) > len(previous):
                deduplicated[record_id] = array
        arrays = sorted(deduplicated.items(), key=lambda item: item[0])
        arrays = [(rid, arr[:: self.spec.downsample].astype(np.float32)) for rid, arr in arrays if len(arr) > 0]
        train_arrays, _, _ = split_wearable_records(
            arrays, self.split_protocol, self.subject_fold, self.test_subject
        )
        mean, std = self._global_standardizer([arr for _, arr in train_arrays])
        records = []
        for rid, arr in arrays:
            finite = np.isfinite(arr)
            vals = (np.where(finite, arr, mean) - mean) / std
            mask = finite.astype(np.float32)
            mask = self._apply_mask_protocol(mask, rid)
            vals = np.where(mask > 0, vals, 0.0).astype(np.float32)
            tt = np.arange(vals.shape[0], dtype=np.float32)
            records.append((rid, torch.from_numpy(tt), torch.from_numpy(vals), torch.from_numpy(mask)))
        return records

    @staticmethod
    def _mapping_value(mapping, key: str):
        if key in mapping:
            return mapping[key]
        byte_key = key.encode("utf-8")
        if byte_key in mapping:
            return mapping[byte_key]
        raise KeyError(key)

    def _build_wesad_records(self, extract_dir: Path):
        """Build wrist-only native multirate records without interpolation."""

        files = sorted(extract_dir.rglob("S*.pkl"))
        arrays: list[tuple[str, np.ndarray]] = []
        for path in files:
            try:
                with path.open("rb") as handle:
                    payload = pickle.load(handle, encoding="latin1")
            except (KeyError, OSError, pickle.UnpicklingError, ValueError):
                continue
            values = self._wrist_native_clock_array(payload)
            if values is not None:
                arrays.append((path.stem, values))

        return self._finalize_wesad_arrays(arrays, str(extract_dir))

    def _build_wesad_archive_records(self, zip_path: Path):
        """Stream subject pickles from the archive without extracting chest data."""

        arrays: list[tuple[str, np.ndarray]] = []
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [
                info
                for info in zf.infolist()
                if not info.is_dir() and re.search(r"(?:^|/)S\d+/S\d+\.pkl$", info.filename)
            ]
            for info in sorted(members, key=lambda item: item.filename):
                try:
                    with zf.open(info, "r") as handle:
                        payload = pickle.load(handle, encoding="latin1")
                    values = self._wrist_native_clock_array(payload)
                except (EOFError, KeyError, OSError, pickle.UnpicklingError, ValueError):
                    continue
                if values is not None:
                    arrays.append((Path(info.filename).stem, values))
        return self._finalize_wesad_arrays(arrays, str(zip_path))

    def _wrist_native_clock_array(self, payload) -> np.ndarray | None:
        signal = self._mapping_value(payload, "signal")
        wrist = self._mapping_value(signal, "wrist")
        bvp = np.asarray(self._mapping_value(wrist, "BVP"), dtype=np.float32).reshape(-1, 1)
        eda = np.asarray(self._mapping_value(wrist, "EDA"), dtype=np.float32).reshape(-1, 1)
        temp = np.asarray(self._mapping_value(wrist, "TEMP"), dtype=np.float32).reshape(-1, 1)
        acc = np.asarray(self._mapping_value(wrist, "ACC"), dtype=np.float32).reshape(-1, 3)

        duration = min(len(bvp) / 64.0, len(acc) / 32.0, len(eda) / 4.0, len(temp) / 4.0)
        ticks = int(math.floor(duration * 64.0))
        if ticks < 64:
            return None
        values = np.full((ticks, self.spec.feature_dim), np.nan, dtype=np.float32)
        n_bvp = min(len(bvp), ticks)
        values[:n_bvp, 0] = bvp[:n_bvp, 0]

        acc_idx = np.arange(min(len(acc), int(math.ceil(ticks / 2.0))), dtype=np.int64) * 2
        acc_idx = acc_idx[acc_idx < ticks]
        values[acc_idx, 3:6] = acc[: len(acc_idx)]

        slow_limit = int(math.ceil(ticks / 16.0))
        slow_idx = np.arange(min(len(eda), len(temp), slow_limit), dtype=np.int64) * 16
        slow_idx = slow_idx[slow_idx < ticks]
        values[slow_idx, 1] = eda[: len(slow_idx), 0]
        values[slow_idx, 2] = temp[: len(slow_idx), 0]
        return values

    def _finalize_wesad_arrays(self, arrays: list[tuple[str, np.ndarray]], source: str):
        if not arrays:
            raise RuntimeError(f"no {self.dataset} subject pickle payloads found in {source}")
        def subject_order(item):
            subject = wearable_subject_id(item[0])
            match = re.search(r"(\d+)$", subject)
            return (0, int(match.group(1))) if match else (1, subject)

        arrays = sorted(arrays, key=subject_order)
        train_arrays, _, _ = split_wearable_records(
            arrays, self.split_protocol, self.subject_fold, self.test_subject
        )
        mean, std = self._global_standardizer([array for _, array in train_arrays])
        records = []
        for record_id, array in arrays:
            finite = np.isfinite(array)
            values = (np.where(finite, array, mean) - mean) / std
            mask = self._apply_mask_protocol(finite.astype(np.float32), record_id)
            values = np.where(mask > 0, values, 0.0).astype(np.float32)
            tt = np.arange(len(values), dtype=np.float32)
            records.append(
                (record_id, torch.from_numpy(tt), torch.from_numpy(values), torch.from_numpy(mask))
            )
        return records

    def _global_standardizer(self, arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        sums = np.zeros(self.spec.feature_dim, dtype=np.float64)
        sqs = np.zeros(self.spec.feature_dim, dtype=np.float64)
        counts = np.zeros(self.spec.feature_dim, dtype=np.float64)
        for arr in arrays:
            finite = np.isfinite(arr)
            safe = np.where(finite, arr, 0.0)
            sums += safe.sum(axis=0)
            sqs += (safe * safe).sum(axis=0)
            counts += finite.sum(axis=0)
        counts = np.maximum(counts, 1.0)
        mean = sums / counts
        var = np.maximum(sqs / counts - mean * mean, 1e-6)
        return mean.astype(np.float32), np.sqrt(var).astype(np.float32)

    def _apply_group_async_mask(self, mask: np.ndarray) -> np.ndarray:
        if mask.size == 0:
            return mask
        async_mask = np.zeros_like(mask, dtype=np.float32)
        n_groups = len(self.spec.group_slices)
        for t in range(mask.shape[0]):
            group = t % n_groups
            cols = self.spec.group_slices[group]
            async_mask[t, list(cols)] = 1.0
        return mask.astype(np.float32) * async_mask

    def _rng(self, record_id: str) -> np.random.Generator:
        key = f"{self.dataset}:{record_id}:{self.mask_protocol}:{self.mask_seed}".encode("utf-8")
        seed = (zlib.crc32(key) + self.mask_seed) % (2**32)
        return np.random.default_rng(seed)

    def _match_density(self, finite_mask: np.ndarray, candidate_mask: np.ndarray, target_count: int, rng: np.random.Generator) -> np.ndarray:
        finite_flat = np.flatnonzero(finite_mask.reshape(-1) > 0)
        if len(finite_flat) == 0 or target_count <= 0:
            return np.zeros_like(finite_mask, dtype=np.float32)
        target_count = min(int(target_count), len(finite_flat))
        cand_flat = np.flatnonzero((candidate_mask.reshape(-1) > 0) & (finite_mask.reshape(-1) > 0))
        if len(cand_flat) >= target_count:
            keep = rng.choice(cand_flat, size=target_count, replace=False)
        else:
            extra_pool = np.setdiff1d(finite_flat, cand_flat, assume_unique=False)
            extra_need = target_count - len(cand_flat)
            if extra_need > 0 and len(extra_pool) > 0:
                extra = rng.choice(extra_pool, size=min(extra_need, len(extra_pool)), replace=False)
                keep = np.concatenate([cand_flat, extra])
            else:
                keep = cand_flat
        out = np.zeros(finite_mask.size, dtype=np.float32)
        out[keep] = 1.0
        return out.reshape(finite_mask.shape)

    def _apply_mask_protocol(self, mask: np.ndarray, record_id: str) -> np.ndarray:
        finite_mask = mask.astype(np.float32)
        if self.mask_protocol == "real_missing_only_if_available":
            return finite_mask
        baseline = self._apply_group_async_mask(finite_mask)
        if self.mask_protocol == "group_async_current":
            return baseline
        target_count = int(round(float(baseline.sum())))
        rng = self._rng(record_id)
        if self.mask_protocol == "random_mcar_matched_density":
            return self._match_density(finite_mask, finite_mask, target_count, rng)
        if self.mask_protocol == "block_dropout_matched_density":
            block_mask = np.zeros_like(finite_mask, dtype=np.float32)
            n_groups = len(self.spec.group_slices)
            block = max(2, n_groups * 4)
            for t in range(finite_mask.shape[0]):
                group = (t // block) % n_groups
                block_mask[t, list(self.spec.group_slices[group])] = 1.0
            return self._match_density(finite_mask, block_mask, target_count, rng)
        raise ValueError(f"unknown wearable mask protocol: {self.mask_protocol}")

    def _load_mhealth(self, extract_dir: Path) -> list[tuple[str, np.ndarray]]:
        files = sorted(extract_dir.rglob("mHealth_subject*.log"))
        arrays = []
        for path in files:
            raw = np.loadtxt(path, dtype=np.float32)
            if raw.ndim != 2 or raw.shape[1] < 24:
                continue
            # Last column is activity label; the first 23 are continuous sensors.
            arrays.append((path.stem, raw[:, :23]))
        return arrays

    def _load_pamap2(self, extract_dir: Path) -> list[tuple[str, np.ndarray]]:
        files = sorted(extract_dir.rglob("subject*.dat"))
        arrays = []
        feature_idx = [
            2,
            *range(4, 16),
            *range(21, 33),
            *range(38, 50),
        ]
        for path in files:
            raw = np.loadtxt(path, dtype=np.float32)
            if raw.ndim != 2 or raw.shape[1] < 54:
                continue
            # Remove transient activity id 0, following common PAMAP2 practice.
            raw = raw[raw[:, 1] > 0]
            arrays.append((path.stem, raw[:, feature_idx]))
        return arrays

    def _load_opportunity(self, extract_dir: Path) -> list[tuple[str, np.ndarray]]:
        files = sorted(extract_dir.rglob("*.dat"))
        arrays = []
        for path in files:
            try:
                raw = np.loadtxt(path, dtype=np.float32)
            except Exception:
                continue
            if raw.ndim != 2 or raw.shape[1] < self.spec.feature_dim + 8:
                continue
            # OPPORTUNITY has labels in trailing columns.  Keep the first stable
            # sensor block after the timestamp-like first column.
            sensor = raw[:, 1 : 1 + self.spec.feature_dim]
            valid_rate = np.isfinite(sensor).mean(axis=0)
            keep = np.argsort(-valid_rate)[: self.spec.feature_dim]
            sensor = sensor[:, np.sort(keep)]
            arrays.append((path.stem, sensor))
        return arrays

    def _load_uschad(self, extract_dir: Path) -> list[tuple[str, np.ndarray]]:
        try:
            import scipy.io as sio
        except ImportError as exc:
            raise RuntimeError("USCHAD requires scipy to read MATLAB files") from exc

        files = sorted(extract_dir.rglob("Subject*/a*t*.mat"))
        arrays = []
        for path in files:
            try:
                mat = sio.loadmat(path)
            except Exception:
                continue
            raw = mat.get("sensor_readings")
            if raw is None or raw.ndim != 2 or raw.shape[1] < self.spec.feature_dim:
                continue
            subject = path.parent.name
            trial = path.stem
            arrays.append((f"{subject}_{trial}", raw[:, : self.spec.feature_dim].astype(np.float32)))
        return arrays

    def _load_realdisp(self, extract_dir: Path) -> list[tuple[str, np.ndarray]]:
        # REALDISP is large. For quick scouts we read a fixed, representative
        # subject/placement subset directly from the ZIP without extracting 7GB.
        zip_path = self.raw_folder / self.spec.zip_name
        small_members = [
            "subject1_ideal.log",
            "subject1_self.log",
            "subject2_ideal.log",
            "subject2_self.log",
            "subject2_mutual4.log",
            "subject5_ideal.log",
            "subject5_self.log",
            "subject5_mutual4.log",
        ]
        mode = os.environ.get("REALDISP_MEMBER_MODE", "small").strip().lower()
        limit = int(os.environ.get("REALDISP_MEMBER_LIMIT", "0") or 0)

        def select_members(available: set[str]) -> list[str]:
            if mode == "all":
                selected = sorted(name for name in available if name.startswith("subject") and name.endswith(".log"))
            elif mode == "medium":
                preferred = []
                for subject in range(1, 10):
                    for placement in ("ideal", "self", "mutual4"):
                        preferred.append(f"subject{subject}_{placement}.log")
                selected = [name for name in preferred if name in available]
            else:
                selected = [name for name in small_members if name in available]
            if limit > 0:
                selected = selected[:limit]
            return selected

        arrays = []
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, "r") as zf:
                available = set(zf.namelist())
                members = select_members(available)
                for name in members:
                    with zf.open(name) as handle:
                        raw = np.loadtxt(handle, dtype=np.float32)
                    if raw.ndim != 2 or raw.shape[1] < 120:
                        continue
                    arrays.append((Path(name).stem, raw[:, 2:-1]))
            return arrays

        files = sorted(extract_dir.rglob("subject*.log"))
        available = {path.name for path in files}
        keep = {Path(name).stem for name in select_members(available)}
        for path in files:
            if path.stem not in keep:
                continue
            raw = np.loadtxt(path, dtype=np.float32)
            if raw.ndim != 2 or raw.shape[1] < 120:
                continue
            arrays.append((path.stem, raw[:, 2:-1]))
        return arrays


def wearable_subject_id(record_id: str) -> str:
    """Extract the released subject identity from a wearable record id."""

    text = str(record_id)
    match = re.search(r"(?i)subject[_-]?(\d+)", text)
    if match:
        return f"subject{int(match.group(1))}"
    match = re.match(r"(?i)s(\d+)(?:[-_]|$)", text)
    if match:
        return f"subject{int(match.group(1))}"
    return text


def _chronological_counts(n_items: int) -> tuple[int, int, int]:
    if n_items < 3:
        raise ValueError(f"wearable split requires at least three records/groups, got {n_items}")
    n_test = max(1, int(math.ceil(0.20 * n_items)))
    n_seen = n_items - n_test
    n_val = max(1, int(math.ceil(0.125 * n_seen)))
    n_train = n_seen - n_val
    if n_train < 1:
        raise ValueError(f"wearable split leaves no training records/groups for {n_items} items")
    return n_train, n_val, n_test


def split_wearable_records(
    data,
    split_protocol: str,
    subject_fold: int = 0,
    test_subject: str = "",
):
    """Split records chronologically or by disjoint released subject ids."""

    records = list(data)
    protocol = str(split_protocol).strip().lower()
    if protocol == "record_chronological":
        n_train, n_val, _ = _chronological_counts(len(records))
        return (
            records[:n_train],
            records[n_train : n_train + n_val],
            records[n_train + n_val :],
        )
    if protocol != "subject_heldout":
        raise ValueError(f"unknown wearable split protocol: {split_protocol}")

    grouped: dict[str, list] = {}
    for record in records:
        grouped.setdefault(wearable_subject_id(record[0]), []).append(record)

    def subject_order(key: str):
        match = re.search(r"(\d+)$", key)
        return (0, int(match.group(1))) if match else (1, key)

    subject_ids = sorted(grouped, key=subject_order)
    requested_test = str(test_subject).strip()
    if requested_test:
        canonical_test = wearable_subject_id(requested_test)
        matches = [subject_id for subject_id in subject_ids if subject_id.lower() == canonical_test.lower()]
        if len(matches) != 1:
            raise ValueError(
                f"unknown or ambiguous wearable test subject {test_subject!r}; "
                f"available subjects are {subject_ids}"
            )
        test_id = matches[0]
        test_index = subject_ids.index(test_id)
        val_id = subject_ids[(test_index - 1) % len(subject_ids)]
        train_ids = set(subject_ids) - {test_id, val_id}
        val_ids = {val_id}
        test_ids = {test_id}

        def collect(ids: set[str]):
            return [record for record in records if wearable_subject_id(record[0]) in ids]

        return collect(train_ids), collect(val_ids), collect(test_ids)

    n_train, n_val, _ = _chronological_counts(len(subject_ids))
    fold_offset = (int(subject_fold) * (len(subject_ids) - n_train - n_val)) % len(subject_ids)
    if fold_offset:
        subject_ids = subject_ids[fold_offset:] + subject_ids[:fold_offset]
    train_ids = set(subject_ids[:n_train])
    val_ids = set(subject_ids[n_train : n_train + n_val])
    test_ids = set(subject_ids[n_train + n_val :])

    def collect(ids: set[str]):
        return [record for record in records if wearable_subject_id(record[0]) in ids]

    return collect(train_ids), collect(val_ids), collect(test_ids)


def Wearable_time_chunk(data, configs):
    chunk_data = []
    history = int(configs.seq_len)
    pred_window = int(configs.pred_len)
    stride = max(history + pred_window, 1)
    sample_id = 0
    for record_id, tt, vals, mask in data:
        subject_text = wearable_subject_id(record_id)
        subject_match = re.search(r"(\d+)$", subject_text)
        subject_id = int(subject_match.group(1)) if subject_match else -1
        if len(tt) < history + pred_window + 1:
            continue
        t_max = int(tt.max().item())
        for st in range(0, max(t_max - history - pred_window, 1), stride):
            et_x = st + history
            et_y = st + history + pred_window
            idx_x = torch.where((tt >= st) & (tt < et_x))[0]
            idx_y = torch.where((tt >= et_x) & (tt < et_y))[0]
            if len(idx_x) < 2 or len(idx_y) < 1:
                continue
            if mask[idx_y].sum() <= 0:
                continue
            t_start = tt[idx_x][0]
            t_end = tt[idx_y][-1] + 1
            denom = max(float((t_end - t_start).item()), 1.0)
            chunk_data.append({
                "sample_ID": sample_id,
                "subject_ID": subject_id,
                "x_mark": (tt[idx_x] - t_start) / denom,
                "y_mark": (tt[idx_y] - t_start) / denom,
                "x": vals[idx_x],
                "y": vals[idx_y],
                "x_mask": mask[idx_x],
                "y_mask": mask[idx_y],
            })
            sample_id += 1
    return chunk_data


def attach_subject_weights(chunks):
    """Attach inverse-frequency weights without exposing identity to the model input."""
    counts: dict[int, int] = {}
    for sample in chunks:
        subject_id = int(sample.get("subject_ID", -1))
        counts[subject_id] = counts.get(subject_id, 0) + 1
    n_subjects = max(len(counts), 1)
    total = max(len(chunks), 1)
    for sample in chunks:
        subject_id = int(sample.get("subject_ID", -1))
        sample["subject_weight"] = total / float(n_subjects * max(counts.get(subject_id, 1), 1))
    return chunks
