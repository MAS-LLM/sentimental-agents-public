#!/usr/bin/env bash
set -euo pipefail

# 0) Tune these as needed
export TOKENIZERS_PARALLELISM=false # disable HuggingFace tokenizers threads
export NUM_PROCS=4 # how many worker processes to allow

# CPU core assignment - use cores 8-15 for multiagent (leaving 0-7 for RL)
MULTIAGENT_CORES="8-15"

# Path to your simulation setup JSON
SIM_SETUP="data/input/simulation_setup_data.json"

# Directory containing all of your sample CSVs
INPUT_DIR="data/input"

# How many times to retry a failing run
MAX_ATTEMPTS=3

echo "🔬 Starting batch runs with up to ${NUM_PROCS} parallel workers"
echo "🎯 Using CPU cores: ${MULTIAGENT_CORES}"

for i in $(seq 27 30); do
    CANDIDATE_CSV="${INPUT_DIR}/sample_${i}.csv"

    echo "=========================================="
    echo "🚀 Processing ${CANDIDATE_CSV}"
    echo "=========================================="

    if [ ! -f "${CANDIDATE_CSV}" ]; then
        echo "  ⚠️  File not found, skipping."
        echo
        continue
    fi

    attempt=0
    success=false

    until [ $attempt -ge $MAX_ATTEMPTS ]; do
        attempt=$((attempt+1))
        echo "  → Attempt ${attempt}/${MAX_ATTEMPTS}"

        # THIS IS WHERE PYTHON3 IS CALLED:
        taskset -c ${MULTIAGENT_CORES} nice -n 10 python3 main.py \
            --simulation_setup_data "${SIM_SETUP}" \
            --candidate_csv "${CANDIDATE_CSV}" \
            --num_processes "${NUM_PROCS}" &

        PID=$!
        wait $PID
        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
            echo "  ✅ Success on attempt ${attempt}"
            success=true
            break
        else
            echo "  ❌ Failed (exit code ${EXIT_CODE}), retrying in 10s…"
            sleep 10
        fi
    done

    if ! $success; then
        echo "  ⚠️  Giving up on ${CANDIDATE_CSV} after ${MAX_ATTEMPTS} attempts"
    fi

    echo
done

echo "🏁 All runs complete!"