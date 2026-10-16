#!/bin/bash
# Failure-recovery collection smoke test.
# L0: supervisor unit tests (no GPU).
# L1: one serial episode compared with the pre-change golden run, plus a no-injection run.
# L2/L3: 2 GPUs x 2 workers, then kill -9 one worker; all 8 episodes must still land.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate RoboTwin
fi

PY="${PYTHON:-python}"
GOLDEN="${ROOT}/error_recovery/data/_golden/move_can_pot"
OUT="${SMOKE_OUT:-${ROOT}/error_recovery/data/_smoke_upgrade}"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1

STAGE="${SMOKE_STAGE:-all}"
GPUS="${GPUS:-0,1}"

if [ "$STAGE" = "all" ] || [ "$STAGE" = "unit" ]; then
echo "===== L0 unit tests ====="
"$PY" error_recovery/tests/test_parallel.py
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "serial" ]; then
echo "===== L1 serial regression ====="
"$PY" error_recovery/collect_universal.py \
  --task move_can_pot \
  --episodes 1 \
  --gpu 0 \
  --error_types premature_close \
  --seed_start 0 \
  --save_path "${OUT}/l1"
"$PY" error_recovery/tools/verify_dataset.py \
  --path "${OUT}/l1" \
  --expect 1 \
  --golden "$GOLDEN"

echo "===== L1 no injection ====="
"$PY" error_recovery/collect_universal.py \
  --task move_can_pot \
  --episodes 1 \
  --gpu 0 \
  --error_types none \
  --seed_start 1000 \
  --save_path "${OUT}/l1_none"
"$PY" error_recovery/tools/verify_dataset.py --path "${OUT}/l1_none"
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "parallel" ]; then
echo "===== L2/L3 parallel run with a killed worker ====="
rm -rf "${OUT}/l2"
nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader > "${OUT}/gpu_before.txt" || true
"$PY" error_recovery/collect_parallel.py \
  --task move_can_pot \
  --episodes 8 \
  --gpus "$GPUS" \
  --workers_per_gpu 2 \
  --seed_start 2000 \
  --max_episodes_per_process 50 \
  --episode_timeout 1200 \
  --save_path "${OUT}/l2" \
  --error_types premature_close \
  > "${OUT}/l2_supervisor.log" 2>&1 &
SUP_PID=$!

cleanup() {
  if kill -0 "$SUP_PID" 2>/dev/null; then
    kill "$SUP_PID" 2>/dev/null || true
    sleep 1
    kill -9 "$SUP_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

python3 - "$OUT/l2" << 'PY'
import json, os, sys, time
out = sys.argv[1]
deadline = time.time() + 240
status_path = os.path.join(out, "status.json")
seen = None
while time.time() < deadline:
    if os.path.exists(status_path):
        status = json.load(open(status_path))
        running = [w for w in status.get("workers", []) if w.get("pid")]
        if running:
            seen = running
            break
    time.sleep(2)
if not seen:
    sys.exit("no running worker pid in status.json")
print("running", [(w["id"], w["gpu"], w["pid"]) for w in seen])
json.dump(seen, open(os.path.join(out, "running_snapshot.json"), "w"))
PY

python3 - "$OUT/l2" "$OUT/gpu_before.txt" << 'PY'
import json, os, subprocess, sys, time
out = sys.argv[1]
before_path = sys.argv[2]
nproc = os.cpu_count() or 1
running = json.load(open(os.path.join(out, "running_snapshot.json")))
before = set()
if os.path.exists(before_path):
    for line in open(before_path, encoding="utf-8"):
        pid = line.split(",", 1)[0].strip()
        if pid.isdigit():
            before.add(int(pid))

def gpu_by_pid():
    uuid_to_index = {}
    listing = subprocess.check_output(["nvidia-smi", "-L"], text=True)
    for line in listing.splitlines():
        if not line.startswith("GPU ") or "UUID:" not in line:
            continue
        index = line.split(":", 1)[0].replace("GPU", "").strip()
        uuid = line.split("UUID:", 1)[1].strip().rstrip(")")
        uuid_to_index[uuid] = index
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
        text=True,
    )
    found = {}
    for line in apps.splitlines():
        if "," not in line:
            continue
        pid, uuid = [part.strip() for part in line.split(",", 1)]
        try:
            found[int(pid)] = uuid_to_index.get(uuid, uuid)
        except ValueError:
            continue
    return found

# nvidia-smi inside this container reports host PIDs, which do not match
# /proc. Pinning is checked by the new compute apps that appeared, grouped
# by physical GPU, against the GPUs the workers were assigned.
expected = {}
for worker in running:
    expected[str(worker["gpu"])] = expected.get(str(worker["gpu"]), 0) + 1

deadline = time.time() + 240
new_counts = {}
while time.time() < deadline:
    pid_gpu = gpu_by_pid()
    fresh = {pid: gpu for pid, gpu in pid_gpu.items() if pid not in before}
    new_counts = {}
    for gpu in fresh.values():
        new_counts[str(gpu)] = new_counts.get(str(gpu), 0) + 1
    if all(new_counts.get(gpu, 0) >= count for gpu, count in expected.items()):
        break
    time.sleep(3)
else:
    sys.exit(f"workers did not show up on the assigned GPUs: expected={expected} new={new_counts}")
extra = [gpu for gpu in new_counts if gpu not in expected]
if extra:
    sys.exit(f"new compute apps appeared on unassigned GPUs {extra}: {new_counts}")
print("gpu pin ok", new_counts)

# Affinity must be a private slice, and the machine load must stay runnable.
time.sleep(15)
load1 = os.getloadavg()[0]
print("load1", load1, "nproc", nproc)
if load1 > nproc:
    sys.exit(f"load {load1} is above nproc {nproc}")
for worker in running:
    pid = int(worker["pid"])
    affinity = worker.get("cores") or []
    if affinity and max(affinity) >= nproc - 1 and len(affinity) > nproc * 0.8:
        sys.exit(f"worker {worker['id']} was given almost every core: {len(affinity)}")
    running_threads = 0
    try:
        for tid in os.listdir(f"/proc/{pid}/task"):
            state = open(f"/proc/{pid}/task/{tid}/stat", encoding="utf-8").read().split(")", 1)[-1].split()[0]
            if state == "R":
                running_threads += 1
    except OSError:
        continue
    print(f"worker {worker['id']} cores={len(affinity)} running_threads={running_threads}")
    if running_threads > max(16, len(affinity)):
        sys.exit(f"worker {worker['id']} has {running_threads} runnable threads")
print("thread cap ok")
PY

KILL_PID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[0]["pid"])' "$OUT/l2/running_snapshot.json")"
echo "kill -9 worker pid ${KILL_PID}"
kill -9 "$KILL_PID" || true

SUP_CODE=0
wait "$SUP_PID" || SUP_CODE=$?
trap - EXIT
if [ "$SUP_CODE" -ne 0 ]; then
  echo "supervisor failed, log follows"
  tail -n 80 "${OUT}/l2_supervisor.log" || true
  exit "$SUP_CODE"
fi

"$PY" error_recovery/tools/verify_dataset.py --path "${OUT}/l2" --expect 8
fi
echo "smoke test passed"
