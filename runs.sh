#!/usr/bin/env bash

# Repeat the publication experiment with independent random seeds.
# Private paths and federation details stay in the external Hydra config.

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage:
  PRIVATE_CONFIG_DIR=/path/to/private/config ./runs.sh [HYDRA_OVERRIDE ...]

Environment variables:
  PRIVATE_CONFIG_DIR  Directory containing the private Hydra config (required)
  FLOTO_CONFIG_NAME   Hydra config name without .yaml (default: floto)
  FLOTO_PYTHON        Python executable (default: python)
  FLOTO_RUN_PREFIX    Non-identifying experiment prefix (default: all_modalities)
  FLOTO_SEEDS         Space- or comma-separated seeds (default: 0 1 2)
  FLOTO_DRY_RUN       Set to 1 to validate and show the run plan only (default: 0)

The script fixes each run to all available modalities and sets both
random_seed and sampler_seed. Additional Hydra overrides are applied first;
the seed, experiment name, modality list, and modality mode are enforced last.
EOF
}

die() {
    printf 'runs.sh: %s\n' "$*" >&2
    exit 2
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly RUNNER="$SCRIPT_DIR/run_exp.py"
readonly CONFIG_NAME="${FLOTO_CONFIG_NAME:-floto}"
readonly PYTHON_BIN="${FLOTO_PYTHON:-python}"
readonly RUN_PREFIX="${FLOTO_RUN_PREFIX:-all_modalities}"
readonly DRY_RUN="${FLOTO_DRY_RUN:-0}"

[[ -n ${PRIVATE_CONFIG_DIR:-} ]] || die "PRIVATE_CONFIG_DIR is required"
[[ -d $PRIVATE_CONFIG_DIR ]] || die "PRIVATE_CONFIG_DIR is not a directory"
readonly CONFIG_DIR="$(cd -- "$PRIVATE_CONFIG_DIR" && pwd -P)"

[[ -f $RUNNER ]] || die "run_exp.py was not found next to this script"
if [[ $CONFIG_DIR == "$SCRIPT_DIR" || $CONFIG_DIR == "$SCRIPT_DIR/"* ]]; then
    die "PRIVATE_CONFIG_DIR must be outside the publication source directory"
fi
[[ $CONFIG_NAME =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] \
    || die "FLOTO_CONFIG_NAME contains unsupported characters"
if [[ ! -r "$CONFIG_DIR/$CONFIG_NAME.yaml" && ! -r "$CONFIG_DIR/$CONFIG_NAME.yml" ]]; then
    die "the requested Hydra config was not found"
fi

[[ $RUN_PREFIX =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] \
    || die "FLOTO_RUN_PREFIX must contain only letters, digits, '.', '_' or '-'"
(( ${#RUN_PREFIX} <= 64 )) \
    || die "FLOTO_RUN_PREFIX must be at most 64 characters"
[[ $DRY_RUN == "0" || $DRY_RUN == "1" ]] \
    || die "FLOTO_DRY_RUN must be 0 or 1"
if [[ $DRY_RUN == "0" ]] && ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    die "FLOTO_PYTHON is not an executable on PATH"
fi

seed_spec="${FLOTO_SEEDS:-0 1 2}"
seed_spec="${seed_spec//,/ }"
read -r -a requested_seeds <<< "$seed_spec"
(( ${#requested_seeds[@]} > 0 )) || die "FLOTO_SEEDS must contain at least one seed"

declare -a seeds=()
declare -A seen_seeds=()
for raw_seed in "${requested_seeds[@]}"; do
    [[ $raw_seed =~ ^[0-9]+$ ]] || die "every seed must be a non-negative integer"
    (( ${#raw_seed} <= 10 )) || die "seed is outside the supported 32-bit range"
    seed=$((10#$raw_seed))
    (( seed <= 4294967295 )) || die "seed is outside the supported 32-bit range"
    [[ -z ${seen_seeds[$seed]+present} ]] || die "duplicate seed: $seed"
    seen_seeds[$seed]=1
    seeds+=("$seed")
done

printf 'Prepared %d all-modality seeded run(s).\n' "${#seeds[@]}"

for index in "${!seeds[@]}"; do
    seed="${seeds[$index]}"
    experiment_name="${RUN_PREFIX}_seed_${seed}"
    printf '[%d/%d] seed=%s, experiment=%s\n' \
        "$((index + 1))" "${#seeds[@]}" "$seed" "$experiment_name"

    if [[ $DRY_RUN == "1" ]]; then
        continue
    fi

    "$PYTHON_BIN" "$RUNNER" \
        --config-path "$CONFIG_DIR" \
        --config-name "$CONFIG_NAME" \
        "$@" \
        "experiment_name=$experiment_name" \
        "random_seed=$seed" \
        "sampler_seed=$seed" \
        'modality_mode=all' \
        'modalities=[ct,ecg,demographics]'
done

if [[ $DRY_RUN == "1" ]]; then
    printf 'Dry run complete; no training was started.\n'
else
    printf 'All seeded runs completed successfully.\n'
fi
