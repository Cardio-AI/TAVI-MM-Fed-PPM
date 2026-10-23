"""Local and Fed-BioMed entry points for the multimodal PPM requirement outcome model.

Configuration is intentionally external to this publication snapshot. In
particular, dataset paths, node identifiers, site mappings, and credentials
must never be committed alongside this file.
"""

from __future__ import annotations

from ast import literal_eval
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Mapping, Optional, Sequence

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from custom_multimodal_training_plan import TP


PUBLIC_METADATA_KEYS = (
    "attention_balance_method",
    "attention_use_learnable_token",
    "batch_size",
    "bce_pos_weight",
    "both_dropout",
    "ct_norm_mode",
    "dropout",
    "epochs",
    "grad_norm",
    "lr",
    "modality_mode",
    "num_rounds",
    "random_seed",
    "reconstruction_loss_type",
    "reconstruction_loss_weight",
    "sampler",
    "sampler_seed",
    "site_dropout",
    "test_size",
    "training_mode",
    "weight_decay",
)


def _plain(value: Any) -> Any:
    """Convert OmegaConf containers to ordinary Python containers."""

    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    value = _plain(value)
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _sequence(value: Any, name: str) -> list[Any]:
    value = _plain(value)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return list(value)


def _optional_tuple(
    value: Any,
    name: str,
    length: int,
    *,
    integer: bool = False,
) -> Optional[tuple]:
    """Parse tuple-like config values without executing arbitrary code."""

    if value is None:
        return None
    value = _plain(value)
    if isinstance(value, str):
        value = literal_eval(value)
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    converter = int if integer else float
    return tuple(converter(item) for item in value)


def _safe_component(value: Any) -> str:
    """Return a path-safe experiment label that cannot escape its output root."""

    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise ValueError(
            "experiment_name must be 1-80 characters using letters, digits, "
            "periods, underscores, or hyphens"
        )
    return value


def _site_alias(value: Any) -> str:
    """Accept only deliberately non-identifying numeric site aliases."""

    value = str(value)
    if value != "unknown" and not re.fullmatch(r"site_[0-9]+", value):
        raise ValueError("site aliases must use 'unknown' or the form 'site_<number>'")
    return value


def _outside_source_path(value: Any, name: str) -> Path:
    """Resolve a private path and reject locations inside this source tree."""

    path = Path(to_absolute_path(str(value))).resolve()
    source_root = Path(__file__).resolve().parent
    if path == source_root or source_root in path.parents:
        raise ValueError(f"{name} must be outside the source checkout")
    return path


def _preprocess_values(cfg: DictConfig) -> dict[str, dict[str, Any]]:
    raw = _mapping(cfg.features.preprocess_values, "features.preprocess_values")
    prepared: dict[str, dict[str, Any]] = {}
    for feature_name, raw_spec in raw.items():
        spec = _mapping(raw_spec, f"preprocessing for {feature_name}")
        feature_type = spec.get("type")
        if feature_type not in {"binary", "continuous", "category"}:
            raise ValueError("each feature type must be binary, continuous, or category")
        if feature_type == "continuous":
            if "mean" not in spec or "std" not in spec or float(spec["std"]) <= 0:
                raise ValueError("continuous features require a mean and positive std")
            spec["mean"] = float(spec["mean"])
            spec["std"] = float(spec["std"])
        if feature_type == "category":
            spec["values"] = _sequence(spec["values"], f"{feature_name}.values")
            if not spec["values"]:
                raise ValueError("categorical feature values must not be empty")
        prepared[feature_name] = spec
    return prepared


def _model_args(cfg: DictConfig) -> dict[str, Any]:
    seed = int(cfg.get("random_seed", 42))
    model_args = {
        "features": [
            str(item) for item in _sequence(cfg.features.features, "features.features")
        ],
        "preprocess_values": _preprocess_values(cfg),
        "ecg_fm_cfg": _mapping(cfg.ecg_fm.cfg, "ecg_fm.cfg"),
        "wav2vec2_cfg": _mapping(cfg.ecg_fm.wav2vec2_cfg, "ecg_fm.wav2vec2_cfg"),
        "train_ecg_encoder": bool(cfg.get("train_ecg_encoder", False)),
        "train_img_encoder": bool(cfg.get("train_img_encoder", False)),
        "prosthesis_names": [
            str(item) for item in _sequence(cfg.get("prosthesis_names", []), "prosthesis_names")
        ],
        "dropna_subset": [
            str(item) for item in _sequence(cfg.get("dropna_subset", []), "dropna_subset")
        ],
        "modalities": [
            str(item) for item in _sequence(
                cfg.get("modalities", ["ct", "ecg", "demographics"]), "modalities"
            )
        ],
        "modality_mode": str(cfg.get("modality_mode", "all")),
        "ct_feature_extractor_ckpt": cfg.get("ct_feature_extractor_ckpt"),
        "ecg_feature_extractor_ckpt": cfg.get("ecg_feature_extractor_ckpt"),
        "dropout": float(cfg.get("dropout", 0.1)),
        "reconstruction_loss_weight": float(cfg.get("reconstruction_loss_weight", 1.0)),
        "reconstruction_loss_type": str(cfg.get("reconstruction_loss_type", "cosine")),
        "test_size": float(cfg.get("test_size", cfg.get("test_ratio", 0.2))),
        "both_dropout": float(cfg.get("both_dropout", 0.0)),
        "random_seed": seed,
        "training_mode": str(cfg.get("training_mode", TP.TrainingMode.RECONSTRUCTION)),
        "attention_use_learnable_token": bool(
            cfg.get("attention_use_learnable_token", True)
        ),
        "attention_balance_method": cfg.get("attention_balance_method"),
        "ckpt": cfg.get("ckpt"),
        "unknown_locations": [
            str(item)
            for item in _sequence(cfg.get("unknown_locations", []), "unknown_locations")
            if item is not None
        ],
        "site_name": _site_alias(cfg.get("site_name", "unknown")),
        "site_dropout": float(cfg.get("site_dropout", 0.5)),
        "text_encoder_name_or_path": str(
            cfg.get("text_encoder_name_or_path", "distilbert-base-uncased")
        ),
        "text_encoder_cache_dir": cfg.get("text_encoder_cache_dir"),
        "allow_model_download": bool(cfg.get("allow_model_download", False)),
        "ct_type": str(cfg.get("ct_type", "aortic_root_roi")),
        "ct_norm_mode": cfg.get("ct_norm_mode"),
        "ct_global_stats": _optional_tuple(
            cfg.get("ct_global_stats"), "ct_global_stats", 2
        ),
        "ct_target_size": _optional_tuple(
            cfg.get("ct_target_size", (64, 64, 64)),
            "ct_target_size",
            3,
            integer=True,
        ),
        "sampler": cfg.get("sampler"),
        "sampler_seed": int(cfg.get("sampler_seed", seed)),
        "bce_pos_weight": float(cfg.get("bce_pos_weight", 1.0)),
    }
    if not model_args["features"]:
        raise ValueError("features.features must not be empty")
    if any(
        feature not in model_args["preprocess_values"]
        for feature in model_args["features"]
    ):
        raise ValueError("every selected feature requires a preprocessing specification")
    if model_args["ct_target_size"] is None or any(
        side <= 0 for side in model_args["ct_target_size"]
    ):
        raise ValueError("ct_target_size must contain three positive integers")
    if model_args["ct_norm_mode"] == "global" and model_args["ct_global_stats"] is None:
        raise ValueError("ct_global_stats is required when ct_norm_mode='global'")
    if not 0.0 < model_args["test_size"] < 1.0:
        raise ValueError("test_size must be strictly between 0 and 1")
    if model_args["bce_pos_weight"] <= 0:
        raise ValueError("bce_pos_weight must be positive")
    if model_args["training_mode"] not in {
        TP.TrainingMode.RECONSTRUCTION,
        TP.TrainingMode.SIMPLE_MISSING_TOKEN,
        TP.TrainingMode.CROSS_ATTENTION,
    }:
        raise ValueError("training_mode is not supported")
    if model_args["modality_mode"] not in {"all", "only_ct", "only_ecg", "both"}:
        raise ValueError("modality_mode is not supported")
    if not model_args["modalities"] or set(model_args["modalities"]) - {
        "ct",
        "ecg",
        "demographics",
    }:
        raise ValueError("modalities may contain only ct, ecg, and demographics")
    if model_args["ct_norm_mode"] not in {None, "global", "clip_range", "local"}:
        raise ValueError("ct_norm_mode is not supported")
    if model_args["sampler"] not in {None, "none", "over"}:
        raise ValueError("sampler must be null, none, or over")
    if not 0.0 <= model_args["site_dropout"] <= 1.0:
        raise ValueError("site_dropout must be between 0 and 1")
    model_args["unknown_locations"] = [
        _site_alias(alias) for alias in model_args["unknown_locations"]
    ]
    return model_args


def _artifact_dir(cfg: DictConfig) -> Path:
    configured_root = cfg.get("output_dir")
    if not configured_root:
        raise ValueError("output_dir is required and must be outside the source checkout")
    root = _outside_source_path(configured_root, "output_dir")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = _safe_component(cfg.get("experiment_name", "experiment"))
    output = root / f"{name}_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def _public_metadata(cfg: DictConfig, model_args: Mapping[str, Any]) -> dict[str, Any]:
    """Whitelist reproducibility metadata and exclude private configuration."""

    metadata = {
        key: _plain(cfg.get(key))
        for key in PUBLIC_METADATA_KEYS
        if cfg.get(key) is not None
    }
    metadata.update(
        {
            "num_tabular_features": len(model_args["features"]),
            "modalities": list(model_args["modalities"]),
            "ct_target_size": list(model_args["ct_target_size"]),
        }
    )
    return metadata


def _write_public_metadata(
    output_dir: Path,
    cfg: DictConfig,
    model_args: Mapping[str, Any],
) -> None:
    metadata = _public_metadata(cfg, model_args)
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def federated_training(cfg: DictConfig) -> None:
    """Run a Fed-BioMed experiment using private runtime configuration."""

    from fedbiomed.common.logger import logger
    from fedbiomed.researcher.aggregators.fedavg import FedAverage
    from fedbiomed.researcher.federated_workflows import Experiment
    from fedbiomed.researcher.federated_workflows.jobs import TrainingJob

    model_args = _model_args(cfg)
    base_sampler_seed = int(model_args["sampler_seed"])
    output_dir = _artifact_dir(cfg)
    _write_public_metadata(output_dir, cfg, model_args)

    training_args = {
        "loader_args": {"batch_size": int(cfg.batch_size)},
        "optimizer_args": {
            "lr": float(cfg.lr),
            "weight_decay": float(cfg.get("weight_decay", 1e-4)),
        },
        "test_ratio": float(cfg.get("test_ratio", cfg.get("test_size", 0.2))),
        "test_batch_size": int(cfg.get("test_batch_size", cfg.batch_size)),
        "test_on_global_updates": bool(cfg.get("test_on_global_updates", True)),
        "test_on_local_updates": bool(cfg.get("test_on_local_updates", True)),
        "use_gpu": bool(cfg.get("use_gpu", True)),
        "epochs": int(cfg.get("epochs", 100)),
        "dry_run": False,
        "log_interval": int(cfg.get("log_interval", 1)),
    }

    private_nodes = _mapping(cfg.fedbiomed.nodes, "fedbiomed.nodes")
    aliases = [_site_alias(item) for item in _sequence(cfg.locations, "locations")]
    missing_aliases = [alias for alias in aliases if alias not in private_nodes]
    if missing_aliases:
        raise KeyError("Every location alias must exist in private fedbiomed.nodes")
    node_ids = [str(private_nodes[alias]) for alias in aliases]

    experiment = Experiment(
        tags=[
            str(item) for item in _sequence(cfg.get("tags", []), "tags")
        ],
        nodes=node_ids,
        model_args=model_args,
        experimentation_folder=output_dir.name,
        training_plan_class=TP,
        training_args=training_args,
        round_limit=int(cfg.get("num_rounds", 1)),
        aggregator=FedAverage(),
        tensorboard=bool(cfg.get("tensorboard", True)),
        node_selection_strategy=None,
    )

    mode = str(cfg.get("mode", "train"))
    if mode == "train":
        while experiment.round_current() < experiment.round_limit():
            round_seed = base_sampler_seed + experiment.round_current()
            round_model_args = dict(experiment.model_args())
            round_model_args["sampler_seed"] = round_seed
            experiment.set_model_args(round_model_args, keep_weights=True)
            experiment.run(rounds=1)
        experiment.training_plan().export_model(output_dir / "final_model.pt")
    elif mode == "test":
        # Fed-BioMed 6.x has no stable high-level evaluate-only API. This mirrors
        # the original workflow and may need adjustment for another release.
        training_args["test_ratio"] = 1.0
        training_args["test_on_local_updates"] = False
        training_args["test_on_global_updates"] = True

        training_nodes = experiment._node_selection_strategy.sample_nodes(
            from_nodes=experiment.filtered_federation_nodes(),
            round_i=experiment._round_current,
        )
        experiment._aggregator.set_training_plan_type(experiment.training_plan().type())
        experiment._aggregator.check_values(
            n_updates=experiment._training_args.get("num_updates"),
            training_plan=experiment.training_plan(),
            secagg=experiment.secagg.active,
        )
        params = experiment.training_plan().after_training_params()
        aggregator_args = experiment._aggregator.create_aggregator_args(
            params, training_nodes
        )
        experiment._update_nodes_states_agent(before_training=True)
        node_state_ids = experiment._node_state_agent.get_last_node_states()
        job = TrainingJob(
            researcher_id=experiment._researcher_id,
            requests=experiment._reqs,
            nodes=training_nodes,
            keep_files_dir=experiment.experimentation_path(),
            experiment_id=experiment._experiment_id,
            round_=experiment._round_current,
            training_plan=experiment.training_plan(),
            training_args=experiment._training_args,
            model_args=experiment.model_args(),
            data=experiment._fds,
            nodes_state_ids=node_state_ids,
            aggregator_args=aggregator_args,
            do_training=False,
            secagg_arguments=experiment.secagg_setup(training_nodes),
            optim_aux_var=experiment._collect_optim_aux_var(),
        )
        logger.info("Running evaluation on %d configured node(s)", len(training_nodes))
        job.execute()
    else:
        raise ValueError("mode must be either 'train' or 'test'")


def local_training(cfg: DictConfig) -> None:
    """Train on one locally mounted, pseudonymized dataset."""

    model_args = _model_args(cfg)
    dataset_path = _outside_source_path(cfg.dataset_path, "dataset_path")
    if not (dataset_path / "demographics.csv").is_file():
        raise FileNotFoundError(
            "dataset_path must contain a demographics.csv file"
        )

    output_dir = _artifact_dir(cfg)
    _write_public_metadata(output_dir, cfg, model_args)

    loader_args = {
        "batch_size": int(cfg.batch_size),
        "num_workers": int(cfg.get("num_workers", 0)),
        "pin_memory": bool(cfg.get("pin_memory", False)),
    }

    plan = TP()
    plan._loader_args = loader_args
    plan.model_args = lambda: model_args
    plan.dataset_path = dataset_path

    _ = plan.training_data()
    trainset = plan.trainset
    testset = plan.custom_testset

    train_sampler = (
        TP.make_balanced_sampler(trainset, model_args["sampler_seed"])
        if model_args["sampler"] == "over"
        else None
    )
    plan.training_data_loader = DataLoader(
        trainset,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        **loader_args,
    )
    plan.testing_data_loader = DataLoader(
        testset,
        shuffle=False,
        **loader_args,
    )
    plan.bce_pos_weight = (
        1.0 if train_sampler is not None else model_args["bce_pos_weight"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plan._device = device
    model = plan.init_model(model_args).to(device)
    plan.model = lambda: model
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    grad_norm = float(cfg.get("grad_norm", 0.0))
    epochs = int(cfg.get("epochs", 100))
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    best_test_auc = float("-inf")
    best_epoch = 0
    history: dict[str, list] = {"train_losses": [], "test_metrics": []}

    for epoch in range(epochs):
        plan.train_results = []
        plan.test_results = []
        for tracker in (plan.train_losses, plan.test_losses):
            for key in tracker:
                tracker[key] = []

        model.train()
        train_losses = []
        for data, target in tqdm(plan.training_data_loader, leave=False):
            optimizer.zero_grad()
            loss = plan.training_step(data, target)
            loss.backward()
            if grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            for data, target in tqdm(plan.testing_data_loader, leave=False):
                plan.testing_step(data, target)

        test_metrics = plan.evaluate_by_modality(mode="test")
        test_metrics["loss"] = float(np.mean(plan.test_losses["total"]))
        mean_train_loss = float(np.mean(train_losses))
        history["train_losses"].append({"avg_loss": mean_train_loss})
        history["test_metrics"].append(test_metrics)

        current_auc = float(test_metrics["roc_auc_test"])
        print(
            f"Epoch {epoch}: train_loss={mean_train_loss:.4f}, "
            f"test_loss={test_metrics['loss']:.4f}, "
            f"test_auroc={current_auc:.4f}"
        )
        is_better = epoch == 0 or (
            np.isfinite(current_auc)
            and (not np.isfinite(best_test_auc) or current_auc > best_test_auc)
        )
        if is_better:
            best_test_auc = current_auc
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    torch.save(
        {
            "best_test_auc": best_test_auc,
            "best_epoch": best_epoch,
            "training_history": history,
        },
        output_dir / "results.pt",
    )
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    print(f"Training artifacts written to {output_dir}")


@hydra.main(version_base="1.2", config_path=None, config_name=None)
def main(cfg: DictConfig) -> None:
    if bool(cfg.get("local", False)):
        local_training(cfg)
    else:
        federated_training(cfg)


if __name__ == "__main__":
    main()
