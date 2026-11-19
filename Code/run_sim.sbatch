
set -eo pipefail
module load cuda12.6/toolkit/12.6.3          # matches torch 2.6.0+cu126; gsplat compiles with it
PY=${PY:-$HOME/.conda/envs/aerial_robotics/bin/python}
POLICY=${POLICY:-policy_template:GapFlyer}
export PATH="$(dirname "$PY"):$PATH"
cd "${SLURM_SUBMIT_DIR:-$PWD}"

echo "node: $(hostname)   job: ${SLURM_JOB_ID:-local}   policy: $POLICY"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

OUT="runs/${SLURM_JOB_ID:-local_$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"
set +e
"$PY" Code/Wrapper.py --pack . --worker-python "$PY" --policy "$POLICY" --log "$OUT/run.json" "$@" 2>&1 | tee "$OUT/run_log.txt"
STATUS=${PIPESTATUS[0]}
set -e

# the policy writes fpv/<timestamp>/ and fpv/debug/ in the working directory: move them into the run
FPV=$(ls -td fpv/2*/ 2>/dev/null | head -1 || true)
[ -n "$FPV" ] && mv "$FPV" "$OUT/fpv"
[ -d fpv/debug ] && mv fpv/debug "$OUT/debug"

if [ -d "$OUT/fpv" ]; then
    module load ffmpeg >/dev/null 2>&1 || true
    if command -v ffmpeg >/dev/null 2>&1; then
        ffmpeg -loglevel error -y -framerate 15 -i "$OUT/fpv/frame_%05d.png" -pix_fmt yuv420p "$OUT/FPVVideo.mp4"
    else
        "$PY" - "$OUT" <<'EOF'
import glob, sys, cv2
out = sys.argv[1]
fr = sorted(glob.glob(out + "/fpv/frame_*.png"))
h, w = cv2.imread(fr[0]).shape[:2]
vw = cv2.VideoWriter(out + "/FPVVideo.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
for f in fr: vw.write(cv2.imread(f))
vw.release()
EOF
    fi
fi
echo "results in $OUT"
exit "$STATUS"