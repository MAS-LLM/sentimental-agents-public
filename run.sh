#!/usr/bin/env bash
set -euo pipefail

# Cross-platform batch execution script for simulation project

# ===============================================
# CONFIGURATION - Adjust these for your machine
# ===============================================

export TOKENIZERS_PARALLELISM=false  # disable HuggingFace tokenizers threads
export NUM_PROCS=4                    # how many worker processes to allow
MULTIAGENT_CORES="8-15"               # CPU cores to use (Linux only - ignored on macOS/Windows)
MAX_ATTEMPTS=3                        # how many times to retry a failing run

# File paths
SIM_SETUP="data/input/simulation_setup_data.json"
INPUT_DIR="data/input"

# ===============================================
# PLATFORM DETECTION
# ===============================================

detect_os() {
    case "$(uname -s)" in
        Linux*)     echo "linux";;
        Darwin*)    echo "macos";;
        CYGWIN*|MINGW*|MSYS*) echo "windows";;
        *)          echo "unknown";;
    esac
}

OS=$(detect_os)

# Platform-specific settings
case "${OS}" in
    "linux")
        PYTHON_CMD="python3"
        USE_CPU_AFFINITY=true
        ;;
    "macos")
        # Prefer conda env python if active
        if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
            PYTHON_CMD="${CONDA_PREFIX}/bin/python"
        else
            PYTHON_CMD="$(command -v python3)"
        fi
        USE_CPU_AFFINITY=false
        ;;
    "windows")
        PYTHON_CMD="python"
        USE_CPU_AFFINITY=false
        ;;
    *)
        PYTHON_CMD="python"
        USE_CPU_AFFINITY=false
        ;;
esac

# ===============================================
# HELPER FUNCTIONS
# ===============================================

check_python() {
    if ! command -v "${PYTHON_CMD}" &> /dev/null; then
        echo "❌ ${PYTHON_CMD} not found"
        case "${OS}" in
            "windows")
                echo "   Try: 'py' or 'python3'"
                if command -v py &> /dev/null; then
                    PYTHON_CMD="py"
                    echo "   ✅ Using 'py' instead"
                fi
                ;;
            *)
                echo "   Try: 'python' or 'python3'"
                if command -v python &> /dev/null; then
                    PYTHON_CMD="python"
                    echo "   ✅ Using 'python' instead"
                fi
                ;;
        esac
    fi
}

cross_platform_sleep() {
    local seconds="$1"
    if [ "${OS}" = "windows" ]; then
        timeout /t "${seconds}" /nobreak > /dev/null 2>&1 || sleep "${seconds}"
    else
        sleep "${seconds}"
    fi
}

# ===============================================
# MAIN EXECUTION
# ===============================================

# Pre-flight checks
check_python

echo "🔬 Starting batch runs with the following configuration:"
echo "   Platform: ${OS}"
echo "   Python: ${PYTHON_CMD}"
echo "   Workers: ${NUM_PROCS}"
if [ "${USE_CPU_AFFINITY}" = true ]; then
    echo "   CPU cores: ${MULTIAGENT_CORES}"
else
    echo "   CPU affinity: Not available on ${OS}"
fi
echo "   Max attempts: ${MAX_ATTEMPTS}"
echo ""

# Check if required files exist
if [ ! -f "${SIM_SETUP}" ]; then
    echo "❌ Simulation setup file not found: ${SIM_SETUP}"
    echo "   Please ensure the file exists"
    exit 1
fi

if [ ! -d "${INPUT_DIR}" ]; then
    echo "❌ Input directory not found: ${INPUT_DIR}"
    echo "   Please create the directory and add CSV files"
    exit 1
fi

# Process CSV files
for i in $(seq 1 10); do  # Increased range to check more files
    CANDIDATE_CSV="${INPUT_DIR}/default_sample_${i}.csv"

    # Skip if file doesn't exist
    if [ ! -f "${CANDIDATE_CSV}" ]; then
        continue
    fi

    echo "=========================================="
    echo "🚀 Processing ${CANDIDATE_CSV}"
    echo "=========================================="

    attempt=0
    success=false

    until [ $attempt -ge $MAX_ATTEMPTS ]; do
        attempt=$((attempt+1))
        echo "   → Attempt ${attempt}/${MAX_ATTEMPTS}"

        # Build and execute Python command
        if [ "${USE_CPU_AFFINITY}" = true ] && command -v taskset &> /dev/null; then
            echo "   🎯 Using CPU cores: ${MULTIAGENT_CORES}"
            taskset -c "${MULTIAGENT_CORES}" nice -n 10 ${PYTHON_CMD} main.py \
                --simulation_setup_data "${SIM_SETUP}" \
                --candidate_csv "${CANDIDATE_CSV}" \
                --num_processes ${NUM_PROCS}
        else
            # Fallback without CPU affinity
            if command -v nice &> /dev/null; then
                nice -n 10 ${PYTHON_CMD} main.py \
                    --simulation_setup_data "${SIM_SETUP}" \
                    --candidate_csv "${CANDIDATE_CSV}" \
                    --num_processes ${NUM_PROCS}
            else
                ${PYTHON_CMD} main.py \
                    --simulation_setup_data "${SIM_SETUP}" \
                    --candidate_csv "${CANDIDATE_CSV}" \
                    --num_processes ${NUM_PROCS}
            fi
        fi

        # Check exit code and handle success/failure
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            echo "   ✅ Success on attempt ${attempt}"
            success=true
            break
        else
            echo "   ❌ Failed (exit code ${EXIT_CODE}), retrying in 10s…"
            cross_platform_sleep 10
        fi
    done

    if ! $success; then
        echo "   ⚠️ Giving up on ${CANDIDATE_CSV} after ${MAX_ATTEMPTS} attempts"
    fi

    echo ""
done

echo "=========================================="
echo "🏁 All batch runs complete!"
echo "=========================================="

# Platform-specific usage notes
case "${OS}" in
    "windows")
        echo ""
        echo "📝 Windows Notes:"
        echo "   • For better performance, consider using WSL"
        echo "   • CPU affinity requires specialized tools"
        echo "   • Monitor performance with Task Manager"
        ;;
    "macos")
        echo ""
        echo "📝 macOS Notes:"
        echo "   • CPU affinity is not available"
        echo "   • Monitor performance with Activity Monitor"
        ;;
    "linux")
        echo ""
        echo "📝 Linux Notes:"
        echo "   • CPU affinity enabled for optimal performance"
        echo "   • Monitor with 'htop' or 'top'"
        ;;
esac

echo ""
echo "📁 Check output_files/ directory for results"