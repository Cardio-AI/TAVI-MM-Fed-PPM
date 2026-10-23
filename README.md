# Multimodal Federated Learning for Predicting Permanent Pacemaker Requirement Following Transcatheter Aortic Valve Implantation

This directory is a publication-oriented snapshot of the multimodal training
code used for binary PPM requirement prediction from cardiac CT, 12-lead ECG, and
tabular clinical variables. It supports local training and federated training
with Fed-BioMed.

This is research code, not a clinical device. It must not be used for patient
care or prospective decision-making.

## Scope

The snapshot contains:

- `custom_multimodal_training_plan.py`: dataset loading, preprocessing,
  modality encoders, fusion models, loss functions, and the Fed-BioMed training
  plan.
- `run_exp.py`: local and federated entry points.
- `runs.sh`: a privacy-safe launcher for the three seeded all-modality runs.
- `.gitignore`: denylist for data, credentials, configs, logs, and learned
  artifacts.
- this README.

No clinical records, patient identifiers, institutional node identifiers,
credentials, certificates, private paths, raw logs, trained weights, cohort
statistics, or result tables are included.

The private data, exact preprocessing statistics, model checkpoints, and
runtime configuration are required to reproduce paper numbers. This snapshot
therefore documents the interface to those inputs.

## Model summary

The training plan predicts a binary PPM requirement target and supports three fusion modes:

- `simple`: pooled CT/ECG features, tabular features, and a text embedding,
  with learned representations for missing modalities.
- `cross_attention`: cross-attention between the four modality
  representations.
- `reconstruction`: cross-modal imputation with an auxiliary reconstruction
  loss.

CT features are produced by a 3D MONAI Swin Transformer. ECG features are
produced by an ECG-FM/fairseq-signals transformer. A frozen DistilBERT encoder
represents prosthesis metadata and a neutral site alias. Missing CT or ECG
inputs are represented by NaN tensors and handled by the selected fusion mode.

## Private data contract

Use only properly governed, pseudonymized data. The expected layout is:

```text
<dataset-root>/
├── demographics.csv
└── <opaque-sample-id>/
    ├── pre_ecg/
    │   └── <scan>.nii.gz
    └── <ct_type>/
        └── <scan>.nii.gz
```

`<opaque-sample-id>` must be a single directory name, not a path. Use random
study identifiers that cannot be reversed without a separately protected
linkage table. Do not use names, medical-record numbers, accession numbers,
dates of birth, or unmodified source-system IDs. Remove identifying image
metadata before this pipeline sees the files.

`demographics.csv` must contain:

- `PatientID`: the same opaque identifier used for the sample directory.
- `discharge_pm_ops`: binary target encoded as `0` or `1`.
- every column listed in `features.features`.
- `prosthesis_name` and `prosthesis_size`, unless their documented defaults
  are acceptable.

Optional filtering columns are `admission_pre_pm` and those listed in
`dropna_subset`. When `admission_pre_pm` exists, the code retains only rows
encoded as `0`. At least one CT or ECG file must exist for every retained
sample.

The loader assumes:

- ECG arrays become shape `[12, samples]` after SimpleITK loading and
  transposition. Recordings are treated as 10 seconds long, reordered to the
  standard 12-lead order, resampled to 500 Hz, standardized, and cropped into
  non-overlapping 5-second segments.
- CT arrays are clipped to `[-200, 800]`, optionally normalized, resized to
  `ct_target_size`, and presented as one-channel volumes.
- one NIfTI file is selected from each modality directory. Ensure each
  directory contains exactly the intended input.

Only required modelling columns should be present in the CSV supplied to this
code. Keep linkage tables and raw exports outside the training environment.

## Feature preprocessing

`features.preprocess_values` is a mapping for the selected tabular features:

```yaml
features:
  features: [age, sex]
  preprocess_values:
    age:
      type: continuous
      mean: <training-partition mean>
      std: <training-partition standard deviation>
    sex:
      type: binary
```

Supported types are:

- `binary`: passed through unchanged.
- `continuous`: standardized using the configured mean and standard
  deviation.
- `category`: ordinal-coded using the configured `values` list.

Compute preprocessing statistics from training data only. For a federated
study, agree on the approved aggregation procedure before calculating shared
statistics. Do not copy confidential cohort statistics from exploratory files
into the public repository.

## Environment

Use Python 3.10 and a mutually compatible research environment containing:

- PyTorch, NumPy, pandas, scikit-learn, SimpleITK, tqdm, and torchmetrics;
- MONAI with `SwinTransformer`;
- Hydra and OmegaConf;
- Transformers with DistilBERT;
- fairseq-signals and the ECG-FM code exposing the imported
  `fairseq_signals` and `ecg_transform` modules;
- Fed-BioMed 6.x with the APIs imported by the training plan.

Import and source-level checks were run in the available compatible environment
with these versions:

```text
Python 3.10
fedbiomed==6.2.0
torch==2.12.0
torchvision==0.27.0
monai==1.5.2
transformers==4.40.0
fairseq-signals==1.0.0a0+571a124
ecg-transform==0.1.3
hydra-core==1.3.2
omegaconf==2.3.0
numpy==2.2.6
pandas==2.3.3
scikit-learn==1.5.2
SimpleITK==2.5.3
torchmetrics==1.9.0
```

Suggested setup order:

1. Create an isolated Python 3.10 environment.
2. Install the compatible Fed-BioMed researcher/node stack.
3. Install the matching fairseq-signals build and `ecg-transform` package,
   then the remaining packages listed above.
4. Verify that both Python files compile and that
   `python run_exp.py --help` starts without an import error.
5. Stage approved CT, ECG, and DistilBERT assets outside the checkout.
6. Prepare the pseudonymized dataset and private Hydra config described below.

The CT and ECG checkpoint architectures must match the constructors in
`custom_multimodal_training_plan.py`. Treat checkpoint files as trusted input:
PyTorch checkpoint loading can be unsafe for files from untrusted sources.

By default, DistilBERT network downloads are disabled. Pre-stage an approved
local copy and set `text_encoder_name_or_path` to that directory. Set
`allow_model_download: true` only in an environment where outbound downloads
are permitted.

## Configuration

Runtime configuration is deliberately external. The following is an outline, not a complete ready-to-run config. The two
ECG-FM mappings must contain the full architecture configuration matching the
chosen checkpoint.

```yaml
local: true
experiment_name: paper_debug
dataset_path: ${oc.env:FLOTO_DATASET_DIR}
output_dir: ${oc.env:FLOTO_OUTPUT_DIR}

random_seed: 42
sampler_seed: 42
epochs: 10
batch_size: 4
num_workers: 0
lr: 0.0001
weight_decay: 0.0001
grad_norm: 1.0

training_mode: simple
modalities: [ct, ecg, demographics]
modality_mode: all
dropout: 0.3
sampler: null
bce_pos_weight: 1.0
test_size: 0.2
both_dropout: 0.0

ct_type: aortic_root_roi
ct_target_size: [64, 64, 64]
ct_norm_mode: local
ct_global_stats: null
ct_feature_extractor_ckpt: ${oc.env:FLOTO_CT_CHECKPOINT}
ecg_feature_extractor_ckpt: ${oc.env:FLOTO_ECG_CHECKPOINT}
train_img_encoder: false
train_ecg_encoder: false

text_encoder_name_or_path: ${oc.env:FLOTO_TEXT_ENCODER}
text_encoder_cache_dir: ${oc.env:FLOTO_MODEL_CACHE}
allow_model_download: false

prosthesis_names: []
dropna_subset: [discharge_pm_ops]
unknown_locations: []
site_name: site_01
site_dropout: 0.5

features:
  features: [age, sex]
  preprocess_values:
    age: {type: continuous, mean: <training mean>, std: <training std>}
    sex: {type: binary}

ecg_fm:
  cfg:
    # Complete ECG classifier architecture mapping goes here.
  wav2vec2_cfg:
    # Complete ECG encoder architecture mapping goes here.

# Keep all Hydra output outside the source checkout. Disabling output_subdir
# prevents Hydra from writing a second copy of the private composed config.
hydra:
  run:
    dir: ${oc.env:FLOTO_OUTPUT_DIR}/hydra
  output_subdir: null
  job:
    chdir: false
```

Tuple-valued settings accept YAML lists or tuple strings. They are parsed with
`ast.literal_eval`; arbitrary Python expressions are not executed.

## Local training

Set the environment variables referenced by the private config, then run from
this directory:

```bash
python run_exp.py \
  --config-path "$PRIVATE_CONFIG_DIR" \
  --config-name floto
```

The output directory contains a best and final state dictionary, aggregate
metrics, and `run_metadata.json`. The JSON file is built from an explicit
allowlist and excludes dataset/checkpoint paths, node IDs, site identity,
feature names, preprocessing statistics, and the full Hydra config.

Model weights and metrics can still reveal information about a small or
sensitive cohort. They are ignored by Git and must undergo the study's
disclosure review before release.

## Repeated seeded runs

`runs.sh` reproduces the active experiment loop from the original research
launcher without embedding institution names, node IDs, or private paths. By
default it runs seeds `0`, `1`, and `2`, uses all three modalities, sets
`modality_mode=all`, and assigns neutral experiment names such as
`all_modalities_seed_0`. Federation locations remain in the private config.

After setting the data, model, and output environment variables referenced by
that config, run:

```bash
export PRIVATE_CONFIG_DIR=/secure/path/to/private-config
./runs.sh
```

The launcher accepts additional Hydra overrides, for example:

```bash
./runs.sh local=true epochs=2
```

The run-defining fields are intentionally enforced after those extra
overrides. Customize the public part of the run plan with environment
variables:

```bash
FLOTO_SEEDS="10,11,12" \
FLOTO_RUN_PREFIX=replication \
FLOTO_CONFIG_NAME=floto \
FLOTO_PYTHON=python \
./runs.sh
```

Use `FLOTO_DRY_RUN=1 ./runs.sh` to validate the config location and seed list
without starting training. The script stops at the first failed run so a
partial sweep is not mistaken for a complete one. Seeds control the split,
model initialization, stochastic training operations, and balanced sampler;
federated sampler seeds advance by round. Bitwise equality can still depend on
hardware, CUDA kernels, and exact dependency versions.

## Federated training

For federated execution, add the following only to the private runtime config:

```yaml
local: false
mode: train
num_rounds: 10
tags: [<private Fed-BioMed dataset tags>]
locations: [site_01, site_02]

fedbiomed:
  nodes:
    site_01: ${oc.env:FLOTO_NODE_SITE_01}
    site_02: ${oc.env:FLOTO_NODE_SITE_02}
```

Use numeric neutral aliases such as `site_01`; the code rejects other site
label formats. Do not put institution names into configs or experiment names.
Set `FLOTO_SITE_ALIAS=site_01` (or the corresponding alias) in each node's
private runtime environment. The training plan reads that alias locally and
never receives the node-ID-to-site mapping as a model argument.
`site_dropout` is the probability of replacing the alias with `unknown`
during embedding; aliases listed in `unknown_locations` are always replaced.

Each node must register a compatible dataset and have the same approved code
dependencies and pretrained assets. Dataset paths are resolved by Fed-BioMed
on the node and are not sent from the researcher script.

## Missing-modality and split behaviour

Samples with only CT, only ECG, or both are supported. `modality_mode` can be
`all`, `only_ct`, `only_ecg`, or `both`. The train/test split is
stratified jointly by target and modality availability. `both_dropout` can
randomly mask one modality in training samples that have both.

The split can fail when a target/modality stratum has too few samples. In that
case, revise the pre-specified split strategy.
