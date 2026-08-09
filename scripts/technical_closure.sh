#!/usr/bin/env bash
# =============================================================================
# technical_closure.sh
#
# Robust, resumable, audit-ready technical closure orchestrator for ORBITA.
#
# Features:
#   - Checkpointing: Saves completed steps to data/.technical_closure_state.json
#   - Resumable: Can resume execution from the exact step where it was paused
#   - Safe Interruption: Traps SIGINT/SIGTERM, allowing the user to either:
#       1) Finish creating the current output file before exiting, OR
#       2) Discard the partial output file and save progress for clean re-execution
#   - Output Modes: Interactive prompt or CLI flags (--quiet / -q, --verbose / -v)
#
# Usage:
#   ./scripts/technical_closure.sh [--quiet | --verbose] [--reset]
# =============================================================================
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON=".venv/bin/python -u"
STATE_FILE="data/.technical_closure_state.json"
STATUS_LOG="/tmp/orbita_technical_closure_status.txt"
LOG_DIR="/tmp/orbita_technical_closure_logs"
mkdir -p "$LOG_DIR" "data"

MODE="prompt"  # prompt, quiet, or verbose
RESET=false
CURRENT_STEP=""
CURRENT_PID=""
CURRENT_OUTPUT=""

# Parse Command Line Arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -q|--quiet)
            MODE="quiet"
            shift
            ;;
        -v|--verbose)
            MODE="verbose"
            shift
            ;;
        -r|--reset)
            RESET=true
            shift
            ;;
        -h|--help)
            echo "Usage: ./scripts/technical_closure.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -q, --quiet    Execute in quiet mode (minimal output, logs saved to /tmp)"
            echo "  -v, --verbose  Execute in verbose mode (stream progress to console)"
            echo "  -r, --reset    Reset saved state checkpoint and start from step 1"
            echo "  -h, --help     Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Initialize or reset state JSON
init_state() {
    if [[ "$RESET" == "true" ]] || [[ ! -f "$STATE_FILE" ]]; then
        cat <<EOF > "$STATE_FILE"
{
  "completed_steps": [],
  "current_step": null,
  "current_output": null
}
EOF
    fi
}

init_state

# Helper to check if step is completed
is_step_completed() {
    local step_name="$1"
    $PYTHON -c "
import json, sys
with open('$STATE_FILE') as f:
    data = json.load(f)
sys.exit(0 if '$step_name' in data.get('completed_steps', []) else 1)
" 2>/dev/null
}

# Helper to mark step in-progress
set_step_in_progress() {
    local step_name="$1"
    local output_file="$2"
    $PYTHON -c "
import json
with open('$STATE_FILE', 'r') as f:
    data = json.load(f)
data['current_step'] = '$step_name'
data['current_output'] = '$output_file' if '$output_file' else None
with open('$STATE_FILE', 'w') as f:
    json.dump(data, f, indent=2)
" 2>/dev/null
}

# Helper to mark step completed
mark_step_completed() {
    local step_name="$1"
    $PYTHON -c "
import json
with open('$STATE_FILE', 'r') as f:
    data = json.load(f)
if '$step_name' not in data.get('completed_steps', []):
    data['completed_steps'].append('$step_name')
data['current_step'] = None
data['current_output'] = None
with open('$STATE_FILE', 'w') as f:
    json.dump(data, f, indent=2)
" 2>/dev/null
}

# Status writer
write_status() {
    local msg="$(date --iso-8601=seconds) | $*"
    echo "$msg" >> "$STATUS_LOG"
    if [[ "$MODE" == "verbose" ]]; then
        echo "$msg"
    fi
}

# Safe Interrupt Handler
on_interrupt() {
    echo ""
    echo "======================================================================"
    echo " [INTERRUPTED] Technical closure execution was paused."
    echo " Step in progress: ${CURRENT_STEP:-None}"
    if [[ -n "$CURRENT_OUTPUT" ]]; then
        echo " Target file being created: $CURRENT_OUTPUT"
    fi
    echo "======================================================================"

    if [[ -n "$CURRENT_PID" ]] && kill -0 "$CURRENT_PID" 2>/dev/null; then
        if [ -t 0 ]; then
            # Interactive prompt
            echo "Choose how to handle the active step:"
            echo "  [1] Finish creating the current file/step before exiting"
            echo "  [2] Stop immediately and discard incomplete file (will regenerate on resume)"
            read -p "Select option [1/2] (default: 2): " user_choice < /dev/tty
            user_choice="${user_choice:-2}"

            if [[ "$user_choice" == "1" ]]; then
                echo " [info] Waiting for step '$CURRENT_STEP' to complete..."
                wait "$CURRENT_PID" || true
                if [[ -n "$CURRENT_STEP" ]]; then
                    mark_step_completed "$CURRENT_STEP"
                fi
                echo " [info] Step completed successfully. State saved."
                exit 0
            fi
        fi

        # Option 2 or non-interactive default: kill process and discard partial file
        echo " [info] Terminating active step process (PID: $CURRENT_PID)..."
        kill -9 "$CURRENT_PID" 2>/dev/null || true
        wait "$CURRENT_PID" 2>/dev/null || true

        if [[ -n "$CURRENT_OUTPUT" ]] && [[ -f "$CURRENT_OUTPUT" ]]; then
            echo " [clean] Deleting incomplete file: $CURRENT_OUTPUT"
            rm -f "$CURRENT_OUTPUT"
        fi
    fi

    echo " [info] Progress saved up to last completed step."
    echo " [info] Run './scripts/technical_closure.sh' again to resume."
    exit 130
}

trap 'on_interrupt' SIGINT SIGTERM

# Prompt user for mode if running interactively without flags
if [[ "$MODE" == "prompt" ]]; then
    if [ -t 0 ]; then
        echo "======================================================================"
        echo " ORBITA Technical Closure Pipeline"
        echo "======================================================================"
        if [[ -f "$STATE_FILE" ]] && is_step_completed "generate_global_dataset"; then
            echo " Found previous state checkpoint."
            echo "   [1] Resume from last step"
            echo "   [2] Reset and restart from step 1"
            read -p "Select choice [1/2] (default: 1): " res_choice < /dev/tty
            if [[ "$res_choice" == "2" ]]; then
                RESET=true
                init_state
            fi
            echo ""
        fi

        echo "Select Output Mode:"
        echo "   [1] Quiet (Minimal console logs, detailed output in /tmp)"
        echo "   [2] Verbose (Stream step progress directly to console)"
        read -p "Select mode [1/2] (default: 1): " mode_choice < /dev/tty
        if [[ "$mode_choice" == "2" ]]; then
            MODE="verbose"
        else
            MODE="quiet"
        fi
    else
        MODE="quiet"
    fi
fi

echo " [start] Running technical closure in '$MODE' mode..."

run_step() {
    local name="$1"
    local expected_output="$2"
    shift 2

    if is_step_completed "$name"; then
        echo " [skip] Already completed: $name"
        return 0
    fi

    CURRENT_STEP="$name"
    CURRENT_OUTPUT="$expected_output"
    set_step_in_progress "$name" "$expected_output"

    write_status "START $name"
    local log_file="$LOG_DIR/${name}.log"

    if [[ "$MODE" == "quiet" ]]; then
        echo " [exec] $name..."
        "$@" >"$log_file" 2>&1 &
        CURRENT_PID=$!
        wait "$CURRENT_PID"
    else
        echo " [exec] $name..."
        "$@" 2>&1 | tee "$log_file" &
        CURRENT_PID=$!
        wait "$CURRENT_PID"
    fi

    CURRENT_PID=""
    mark_step_completed "$name"
    write_status "DONE $name"
    CURRENT_STEP=""
    CURRENT_OUTPUT=""
}

# =====================================================================
# 1. DATASET GENERATION
# =====================================================================
run_step \
    "generate_global_dataset" \
    "data/datasets/training/orbita_dataset_300-2000_0.0000-0.1000_0.00-90.00.csv" \
    $PYTHON src/generate_base_dataset.py

# =====================================================================
# 2. GLOBAL MODEL TRAINING (5 architectures)
# =====================================================================
for arch in resnet mlp lstm linear tree; do
    if [[ "$arch" == "tree" ]]; then
        out_model="models/tree/orbita_predictor_tree_300-2000_0.0000-0.1000_0.00-90.00.joblib"
    else
        out_model="models/${arch}/base/orbita_predictor_${arch}_300-2000_0.0000-0.1000_0.00-90.00.pth"
    fi
    run_step \
        "train_${arch}" \
        "$out_model" \
        $PYTHON src/train_base.py --model_type "$arch"
done

# =====================================================================
# 3. K-FOLD CROSS-VALIDATION (5 architectures)
# =====================================================================
for arch in resnet mlp lstm linear tree; do
    run_step \
        "cv_${arch}" \
        "data/metrics/metrics_cv.csv" \
        $PYTHON src/train_cv.py --model_type "$arch" --folds 5
done

# =====================================================================
# 4. BENCHMARKS (5 architectures)
# =====================================================================
run_step \
    "benchmark_resnet_space" \
    "data/benchmarks/ablation_resnet/benchmark_space_domain_resnet.csv" \
    $PYTHON src/benchmark.py --model_type resnet --mode_choice 2 --quiet --seed 42

for arch in mlp lstm linear tree; do
    run_step \
        "benchmark_${arch}_full" \
        "data/benchmarks/ablation_global/benchmark_space_domain_${arch}.csv" \
        $PYTHON src/benchmark.py --model_type "$arch" --mode_choice 3 --quiet --seed 42
done

# =====================================================================
# 5. SIMULATE MISSION
# =====================================================================
run_step \
    "simulate_mission" \
    "" \
    $PYTHON src/simulate_mission.py

# =====================================================================
# 6. FIGURES
# =====================================================================
for arch in resnet mlp lstm linear tree; do
    run_step \
        "figures_${arch}" \
        "figures/benchmarks/${arch}/benchmark_space_domain_${arch}_violin.svg" \
        $PYTHON src/visualize_benchmark.py --mode single --model_type "$arch"
done

run_step \
    "figures_ablation" \
    "figures/ablation_global/ablation_space_domain_cdf.svg" \
    $PYTHON src/visualize_benchmark.py --mode ablation

run_step \
    "figures_metrics" \
    "figures/metrics/metrics_cross_validation.svg" \
    $PYTHON src/visualize_benchmark.py --mode metrics

run_step \
    "figures_resnet_ablation" \
    "figures/ablation_resnet/resnet_ablation_time_domain.svg" \
    $PYTHON src/plot_resnet_ablation_time.py --only-figures

run_step \
    "figures_regional_ablation" \
    "figures/orbita_expert_mesh.svg" \
    $PYTHON src/plot_expert_mesh.py

# =====================================================================
# 7. QUALITY GATES
# =====================================================================
run_step \
    "flake8" \
    "" \
    $PYTHON -m flake8 src tests

run_step \
    "pytest" \
    "" \
    $PYTHON -m pytest tests/ -v

run_step \
    "audit_results" \
    "" \
    $PYTHON src/audit_results.py

# =====================================================================
# 8. CLEANUP
# =====================================================================
find . -path './.venv' -prune -o -type d \
    \( -name '__pycache__' -o -name '.pytest_cache' \) \
    -prune -exec rm -rf {} +

write_status "DONE technical closure"
echo " [complete] Technical closure pipeline finished successfully!"
