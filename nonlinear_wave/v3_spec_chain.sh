#!/usr/bin/env bash
# Chapter-5 production chain under EKI_algorithm_spec_2026-08-23.
#
# Origin: 3. KDV_nonlinear_case/v3_spec_chain.sh
# Changes vs origin: paths adapted to code_rp/nonlinear_wave; the
# server-specific ~/h3env python autodetect replaced by a plain
# PYTHON override; comments in English (they already were).
#
# Spec configuration this chain pins (the drivers hard-set the rest):
#   T_y = T_G = 6000 s analysis window (600 s burn-in -> 6600 s runs),
#   one closure-noise path per realisation, N_Gamma = 50 reference
#   records (each with its own incident sea state, own baseline, own
#   noise), Gamma = var_ref only (no forward term, no floor, no n_eff),
#   G(theta) = mean of N_G realisations with common random numbers,
#   J = 100, n_iter = 20, stop = relative change of Phi < 1 % for three
#   consecutive iterations, phi in log coordinates, report = final
#   ensemble mean +/- sd.
#
#   [1] truth record | GPR reference | N_Gamma reference records
#   [2] N_G calibration (K = 20 at two probes)  -> NG_calibration.json
#   [3] branch "full"  : S1 -> S2   (Gamma = shrunk full Cov_ref)
#   [4] branch "diag"  : S1 -> S2   (Gamma = diag(var_ref), the spec)
#
# IMPORTANT: every driver reads the SW_*/SDE_* environment AT IMPORT
# TIME (see v3_world.py).  Run the stages through this script, or export
# the same variables before calling a driver by hand.
#
# Usage:  bash v3_spec_chain.sh [processes]      (run from this folder)
#   PYTHON=/path/to/python  interpreter override (default: python)
#   BRANCHES="full diag"    which branches, in order
#   PAUSE_AFTER_S1=1        stop after each branch's S1 (relaunch to go on)
#   N_G=<int>               override the calibrated N_G
set -euo pipefail
cd "$(dirname "$0")"
P="${1:-100}"
PY="${PYTHON:-python}"
export SW_FINE=1 OMP_NUM_THREADS=1 PYTHONIOENCODING=utf-8
export SW_VERSION_DIR="${SW_VERSION_DIR:-v3spec}"
export SW_DURATION_S="${SW_DURATION_S:-6600}"
export SW_SYNTH_PERIOD_S="${SW_SYNTH_PERIOD_S:-6600}"
export SW_FORWARD_PATHS=1 SW_REF_NEW_SEA_STATE=1
# The shipped drivers are spec-only; these two exports are kept for
# documentation value (sw_gamma_unified reads SW_GAMMA_TERMS at call
# time, and the drivers hard-set it anyway).
export SW_GAMMA_TERMS=var_ref_only
export SW_S1_VARIANT=S1a SW_S1_SUFFIX=_fine
N_GAMMA="${N_GAMMA:-50}"; ITER="${ITER:-20}"; K_CAL="${K_CAL:-20}"
BRANCHES="${BRANCHES-full diag}"; PAUSE_AFTER_S1="${PAUSE_AFTER_S1:-0}"
PROCS="${PROCS:-$P}"   # worker processes (J*N_G tasks per iteration)
V3="results/stepwise/$SW_VERSION_DIR"
mkdir -p "$V3"
LOG="$V3/v3_spec_chain.log"
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
trap 'log "!!! CHAIN FAILED at line $LINENO !!!"' ERR

log "=== V3 SPEC CHAIN start (P=$P, T=${SW_DURATION_S}s, N_Gamma=$N_GAMMA, branches: $BRANCHES) ==="
"$PY" -c "import v3_world; print('[world]', v3_world.describe())" | tee -a "$LOG"

# ---------------------------------------------------------------- [1]
log "[1/4] truth record | GPR reference | $N_GAMMA reference records (parallel)"
TRUTH="$V3/truth_S1a_fine"
mkdir -p "$TRUTH"
PIDS=""
if [ ! -f "$TRUTH/truth_bundle.npz" ]; then
  "$PY" -u sw_truth.py --variant S1a --suffix _fine --paths 1 --overwrite \
      > "$V3/stage1_truth.log" 2>&1 & PIDS="$PIDS $!"
else log "      truth present -- skipped"; fi
if [ ! -f "$V3/GPR_reference/calibration.json" ]; then
  "$PY" -u sw_gpr_reference.py --paths 1 > "$V3/stage1_gpr.log" 2>&1 & PIDS="$PIDS $!"
else log "      GPR reference present -- skipped"; fi
"$PY" -u sw_ref_records.py --n-ref "$N_GAMMA" --paths 1 \
    --processes "$((P < 60 ? P : 60))" --resume > "$V3/stage1_refs.log" 2>&1 & PIDS="$PIDS $!"
for pid in $PIDS; do wait "$pid" || { log "!!! stage-1 job $pid failed !!!"; exit 1; }; done
grep -h -v -i warn "$V3"/stage1_*.log 2>/dev/null | tail -20 >> "$LOG" || true
log "[1/4] done"

# ---------------------------------------------------------------- [2]
log "[2/4] N_G calibration (K=$K_CAL at two probes)"
if [ -f "$V3/NG_calibration.json" ]; then
  log "      calibration present -- skipped"
else
  "$PY" -u v3_calibrate_ng.py --k "$K_CAL" --processes "$P" >> "$LOG" 2>&1
fi
if [ -z "${N_G:-}" ]; then
  N_G=$("$PY" -c "import json;print(json.load(open('$V3/NG_calibration.json'))['N_G_chosen'])")
fi
export SW_N_G="$N_G"
log "      N_G = $SW_N_G realisations averaged per evaluation"

# ---------------------------------------------------------------- [3,4]
for BR in $BRANCHES; do
  case "$BR" in
    diag) GT="diag" ;;
    full) GT="full" ;;
    *) log "!!! unknown branch $BR !!!"; exit 1 ;;
  esac
  S1DIR="$V3/S1a_eki_dense_fine_$BR"; S2DIR="$V3/H2_eki_$BR"
  log "[S1:$BR] Step 1 (J=$P, q=111, Gamma type $GT, N_G=$SW_N_G)"
  if [ -f "$S1DIR/summary.json" ]; then
    log "      S1 $BR present -- skipped"
  else
    SW_GAMMA_TYPE="$GT" SW_S1_OUTTAG="_$BR" \
      "$PY" -u sw_eki_s1.py --config S1a --members "$P" --processes "$PROCS" \
      --iterations "$ITER" --overwrite >> "$LOG" 2>&1
  fi
  if [ "$PAUSE_AFTER_S1" = 1 ] && [ ! -f "$S2DIR/summary.json" ]; then
    log "[PAUSE] S1 $BR finished -- relaunch with PAUSE_AFTER_S1=0 to run S2"
    exit 0
  fi
  log "[S2:$BR] Step 2 (J=$P, q=151, Gamma type $GT, prior = S1 $BR posterior)"
  SW_GAMMA_TYPE="$GT" SW_H_PHIQ_PRIOR=s1 SW_H_S1_SUMMARY="$S1DIR/summary.json" \
    SW_H_OUTTAG="_$BR" \
    "$PY" -u sw_eki_h.py --variant H2 --members "$P" --processes "$PROCS" \
    --iterations "$ITER" --resume >> "$LOG" 2>&1
done

log "=== V3 SPEC CHAIN done ==="
