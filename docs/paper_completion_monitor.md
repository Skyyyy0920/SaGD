# Task: Monitor the SaKD Paper-Completion Worker

**Role:** You are a **read-mostly monitor agent** that babysits the worker agent
running `docs/paper_completion_task.md` over a ~4-day window. You never
modify the worker's deliverables (`writing/`, `outputs_dolly/`, `data/`,
`tmp/EXPERIMENTS_DONE.md`, etc.). You only watch, summarise, and intervene
when the worker is provably stuck.

**Owner:** Autonomous server-side Claude Code agent (a different session from
the worker). Execute via `/goal docs/paper_completion_monitor.md`. Run in a
tmux session named `sakd_monitor` (separate from the worker's `sakd_finish`).

**Hard time budget:** as long as the worker runs (≤ 96 hours), then ≤ 30 min
to finalise the run report.

**Pacing:** poll every 5 minutes. Use `Bash` + `sleep 300` between iterations,
or `ScheduleWakeup` with `delaySeconds=300`. Do NOT spin-wait.

---

## 1. What you have read/write access to

| Path | Mode | Purpose |
|---|---|---|
| `tmp/exp_progress.log` | read | worker's heartbeat (1 line per event) |
| `tmp/EXPERIMENTS_DONE.md` | read | worker's phase-by-phase status |
| `tmp/phase*_*.json` | read | worker's per-phase telemetry |
| `tmp/CKPT_MANIFEST.csv` | read (created at end of PHASE 11) | final ckpt list |
| `tmp/PAPER_COMPLETION_DONE.md` | read (created at end of PHASE 11) | final report |
| `tmp/MONITOR_REPORT.md` | **write** | rolling summary for human |
| `tmp/MONITOR_ALERTS.md` | **write** | append-only event log of detected issues |
| `tmp/MONITOR_STATE.json` | **write** | machine-readable state between polls |
| `nvidia-smi` | read | GPU utilisation |
| `df -h .` | read | disk free |
| `ps -ef \| grep claude` | read | confirm worker tmux session alive |
| `tmux send-keys -t sakd_finish ...` | write | wake the worker if stalled |
| anything else | **off-limits** |  |

**You do NOT** modify `src/`, `writing/`, `outputs_*/`, `data/`, `scripts/`, or
any of the worker's deliverable files. You do NOT touch `git` (no commits,
no pushes — the worker owns the repo's commit history).

---

## 2. Polling loop (every 5 minutes)

Each iteration, in order:

### 2.1 Worker liveness
1. Check `tmp/exp_progress.log` mtime. If unchanged for > 30 minutes, flag
   STALL.
2. Check whether the `sakd_finish` tmux session exists
   (`tmux has-session -t sakd_finish 2>/dev/null; echo $?`). 0 = alive, 1 =
   gone. If gone, flag WORKER_DEAD.
3. Tail the last 20 lines of `tmp/exp_progress.log` and look for keywords:
   `FAILED`, `BLOCKED`, `OOM`, `NaN`, `SOS`. Each hit → ANOMALY event.

### 2.2 Resource health
4. `nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free
   --format=csv,noheader`. If all 4 GPUs show 0% utilisation AND the worker
   should be in a training phase (per `EXPERIMENTS_DONE.md`), flag
   GPU_IDLE.
5. `df -h .` → assert ≥ 50 GB free. If < 50 GB, flag DISK_LOW. If < 10 GB,
   flag DISK_CRITICAL.
6. Total `outputs_dolly/` size should be growing during PHASE 2 (at ~5 GB
   per finished cell × 88 cells ≈ 440 GB total). If size hasn't grown in > 1
   hour during PHASE 2, flag PROGRESS_STALL.

### 2.3 Phase progress
7. Read `tmp/EXPERIMENTS_DONE.md`. Track which phase is RUNNING and how long
   it's been at this phase. If a phase exceeds its documented budget by 50%,
   flag BUDGET_OVERRUN (record but don't intervene — the worker has its own
   PARTIAL logic).
8. Read `tmp/phase2_queue.json` if it exists. Count
   `(PENDING, RUNNING, DONE, FAILED, RETRIED)` rows. Persist to
   `tmp/MONITOR_STATE.json`.

### 2.4 SOS detection
9. If `tmp/PAPER_COMPLETION_SOS.md` exists, the worker has given up. Read it,
   summarise the failure into `tmp/MONITOR_ALERTS.md`, and stop the polling
   loop. Then write a final `tmp/MONITOR_REPORT.md` with "WORKER ABORTED —
   human action required" and exit.

### 2.5 Rolling summary
10. Append one line to `tmp/MONITOR_ALERTS.md` for each new flag detected
    this iteration. Format:
    `<UTC timestamp> <severity> <flag> <one-line context>`.
11. Overwrite `tmp/MONITOR_REPORT.md` (regenerated every poll):
    - Worker liveness: ALIVE / STALLED / DEAD
    - Current phase + elapsed in phase
    - Resources: GPU util / mem / disk free
    - PHASE 2 queue progress: `N_DONE / 88` and ETA based on observed rate
    - Last 10 events from `tmp/exp_progress.log`
    - Open alerts (new + still-active)
12. Persist current iteration state to `tmp/MONITOR_STATE.json`.

---

## 3. Intervention rules (when to do something, not just watch)

You may intervene in exactly these three cases:

### Case A: STALL (no `exp_progress.log` update in > 30 min)
1. Append ALERT to `tmp/MONITOR_ALERTS.md`.
2. Check if a training process is still running:
   `ps -ef | grep -E 'python.*train|torchrun' | grep -v grep`.
3. If a python process is running but log is silent → the worker may be in a
   long compute block (e.g. a multi-hour single training cell). Do NOT
   intervene; wait one more poll cycle (5 min) and recheck. If still stalled
   after the next poll AND no python process exists, proceed to step 4.
4. Send wake signal:
   `tmux send-keys -t sakd_finish "Continue. Last status was PHASE-N
   <event>. Resume from the next undone phase per
   tmp/EXPERIMENTS_DONE.md." Enter`
5. Log the wake to `tmp/MONITOR_ALERTS.md`.
6. If after a further 30 min there's still no heartbeat, escalate to
   WORKER_DEAD handling (case B).

### Case B: WORKER_DEAD (tmux session gone, or no heartbeat for > 60 min)
1. Append ALERT.
2. Attempt restart:
   ```
   tmux new-session -d -s sakd_finish \
     "cd <repo-root> && claude --dangerously-skip-permissions \
      'Resume the SaKD paper-completion task from docs/paper_completion_task.md.
       Check tmp/EXPERIMENTS_DONE.md for the last DONE phase and continue from
       there.'"
   ```
3. Wait 5 min; recheck heartbeat. If alive, log RESTARTED. If still dead,
   escalate: write `tmp/PAPER_COMPLETION_SOS.md` with the failure context
   and stop polling.

### Case C: DISK_CRITICAL (< 10 GB free)
1. Append ALERT.
2. Send a tmux message to the worker:
   `tmux send-keys -t sakd_finish "DISK CRITICAL: < 10 GB free. Pause any
   pending training launches and write a SOS file if you can't recover."
   Enter`
3. Do NOT delete files yourself.
4. Continue polling at 1-minute interval until disk free > 30 GB or worker
   writes SOS.

For every other flag (GPU_IDLE, BUDGET_OVERRUN, PROGRESS_STALL, NaN,
ANOMALY) → log only. Do NOT intervene; the worker has its own retry logic.

---

## 4. Reporting cadence to the human

- **`tmp/MONITOR_REPORT.md`** — rolling snapshot, regenerated every 5 min.
  Human reads this for "what's happening right now".
- **`tmp/MONITOR_ALERTS.md`** — append-only event log. Human reads this for
  "what unusual events happened during the run".
- **`tmp/MONITOR_HEARTBEAT.txt`** — single line, updated every poll, with
  current timestamp + worker phase. Used as a liveness check for the monitor
  itself.
- **No notifications.** The human will `cat tmp/MONITOR_REPORT.md` whenever
  they want a status. Do not send Slack / email / push.

---

## 5. Final report (when worker writes `PAPER_COMPLETION_DONE.md`)

When `tmp/PAPER_COMPLETION_DONE.md` appears:
1. Stop the polling loop.
2. Append a final summary line to `tmp/MONITOR_REPORT.md`:
   - Total wall-clock time (worker start → done)
   - Number of unique intervention events (STALLs handled, restarts, etc.)
   - Worst-case GPU idle duration observed
   - Final disk free
   - Final PHASE 2 cell count: `N_DONE / 88`, with FAILED cells listed
3. Exit cleanly.

---

## 6. Out of scope (do NOT do)

- Modify any of the worker's deliverable files (`writing/`, `outputs_*/`,
  `data/teacher_saliency_*`, `tables/`, `figures/`, `algorithms/`,
  `tmp/EXPERIMENTS_DONE.md`).
- Run any experiment yourself.
- Make any git commit / push.
- Patch `src/sagd/`.
- Open a new tmux session other than the worker-restart one in case B.
- Send notifications outside the `tmp/MONITOR_*` files.
- Read or modify `tmp/PAPER_COMPLETION_SOS.md` for any purpose other than
  reading its contents into your final report.
- Adjust the worker's training hyperparameters by injecting tmux commands.
  The only command you may inject is the wake/restart prompt in Case A/B.

---

## 7. Startup checklist

On first activation:
1. Verify worker is running: `tmux has-session -t sakd_finish`. If absent,
   write a one-line `tmp/MONITOR_REPORT.md` saying "Worker not yet started"
   and re-poll every 5 min until the worker appears.
2. Initialise `tmp/MONITOR_STATE.json` with the current timestamp + empty
   counters.
3. Touch `tmp/MONITOR_ALERTS.md` (empty file) and
   `tmp/MONITOR_HEARTBEAT.txt`.
4. Enter polling loop (§2).

---

## 8. How the human will consume your output

- During the run: `cat tmp/MONITOR_REPORT.md` for status; `tail -20
  tmp/MONITOR_ALERTS.md` for recent events.
- After the run: read final `tmp/MONITOR_REPORT.md` (will contain the §5
  summary) alongside the worker's `tmp/PAPER_COMPLETION_DONE.md`.

Your job ends when the worker's `PAPER_COMPLETION_DONE.md` exists and you
have written the final §5 summary.
