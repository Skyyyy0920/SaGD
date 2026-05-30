# Task: Monitor the SaKD Paper-Completion Worker

**Role:** You are a **read-mostly monitor agent** that babysits the worker
agent running `docs/paper_completion_task.md` over a ~4-day window. You
never modify the worker's deliverables. You only watch, summarise, and
intervene when the worker is provably stuck.

**Owner:** Autonomous server-side Claude Code agent (a different session
from the worker). Execute via `/goal docs/paper_completion_monitor.md`.
Run in a tmux session named `sakd_monitor` (separate from the worker's
`sakd_finish`).

**Hard time budget:** as long as the worker runs (≤ 96 hours), then
≤ 30 min to finalise the run report.

---

## 0. Pacing model (important — read first)

**One Claude turn = one polling iteration. Never block with `sleep` for
the full 5-minute interval.**

Each iteration is structured as:

1. **Load state** from `tmp/MONITOR_STATE.json` (restore counters from
   previous wake; if file absent, initialise).
2. **Do one pass** of the polling loop in §2 (≤ 30 seconds of work).
3. **Persist state** atomically to `tmp/MONITOR_STATE.json` (`.tmp` +
   `os.replace`).
4. **Re-schedule** the next wake by calling
   `ScheduleWakeup(delaySeconds=300, prompt='Continue monitor loop per
   docs/paper_completion_monitor.md')`. Then exit the turn.

If `ScheduleWakeup` is unavailable, fall back to a single long-lived
background shell wrapping a `while true; do <one iteration>; sleep 300;
done` — **but only as a fallback**; the schedule-and-exit pattern is
preferred because it does not consume an agent slot for 4 days.

---

## 1. What you have read/write access to

**Writable (monitor-owned):**
| Path | Notes |
|---|---|
| `tmp/MONITOR_REPORT.md` | rolling summary; **overwritten** every iteration |
| `tmp/MONITOR_ALERTS.md` | append-only event log |
| `tmp/MONITOR_STATE.json` | persisted state between turns; schema below |
| `tmp/MONITOR_HEARTBEAT.txt` | single line; updated every iteration |
| `tmp/MONITOR_FINAL_REPORT.md` | written ONCE at end of run (see §5) |

**Read-only (worker-owned, monitor must not write):**
| Path | Notes |
|---|---|
| `tmp/exp_progress.log` | worker heartbeat; one event per line |
| `tmp/EXPERIMENTS_DONE.md` | phase status table |
| `tmp/phase*_*.json` | per-phase telemetry; atomic via `os.replace` |
| `tmp/gpu_assignment.json` | worker's current GPU choice |
| `tmp/PAPER_COMPLETION_SOS.md` | worker's give-up signal |
| `tmp/PAPER_COMPLETION_DONE.md` | worker's final report |
| `tmp/PAPER_COMPLETION_START.txt` | commit SHA + start time |
| `tmp/CKPT_MANIFEST.csv` | final ckpt list (PHASE 11) |

**System reads (read-only):**
- `nvidia-smi --query-gpu=...` for GPU state
- `df -h .` for disk free
- `ps -ef` for process lookup
- `tmux list-sessions` / `tmux has-session`

**Intervention writes (only in cases A/B/C of §3):**
- `tmux send-keys -t <session> <msg> Enter` — wake worker
- `tmux new-session -d -s sakd_finish -c <repo_root> ...` — restart worker

**Off-limits absolutely (do not touch):**
- `src/`, `writing/`, `outputs_dolly/`, `outputs_squad/`, `data/`,
  `scripts/`, `tests/`, `CLAUDE.md`, `README.md`, `.git/`, `.agents/`
- All git commands (no `git add`, `git commit`, `git push`, `git pull`,
  `git checkout`)

---

## 2. Polling loop (each iteration, ≤ 30 s of work)

Order matters; later steps depend on earlier ones.

### 2.0 Load state, set CWD

1. `cd $(jq -r '.repo_root' tmp/MONITOR_STATE.json)` if state file
   exists; else `cd $(pwd)` and remember the value (see §7 startup).
2. Read `tmp/MONITOR_STATE.json` into local variable `state`:
   ```json
   {
     "schema_version": "monitor_state_v1",
     "repo_root": "/abs/path/to/repo",
     "monitor_started_at_utc": "2026-05-30T20:00:00Z",
     "last_poll_at_utc": "...",
     "iter_count": 42,
     "worker_session_name": "sakd_finish",
     "last_known_phase": "PHASE-2",
     "last_known_phase_started_at_utc": "...",
     "stall_first_detected_at_utc": null,
     "last_wake_at_utc": null,
     "wake_count": 0,
     "restart_count": 0,
     "open_alerts": [{"flag": "...", "first_seen_utc": "...", "context": "..."}]
   }
   ```

### 2.1 Worker liveness

3. **Heartbeat freshness:** stat `tmp/exp_progress.log`. If mtime
   unchanged for **> 30 min**, flag `STALL` (record into `state` if not
   already present).
4. **tmux session check:** auto-discover the worker session — try
   `tmux has-session -t sakd_finish 2>/dev/null` first; if fail, try
   `tmux list-sessions -F '#{session_name}' | grep -i sakd | head -1`
   and use what's found, persisting the discovered name to
   `state.worker_session_name`. If no SaKD-related session exists, flag
   `WORKER_DEAD`.
5. **Zombie heartbeat detection:** if the last 30 min of
   `tmp/exp_progress.log` contains "alive" lines (background heartbeat
   pattern) but
   `ps -ef | grep -E 'python.*train|torchrun|python.*evaluate|lm_eval' | grep -v grep`
   returns nothing, flag `ZOMBIE_HEARTBEAT`. Treat as STALL.
6. **Anomaly tail:** tail last 20 lines of `tmp/exp_progress.log`; new
   occurrences of `FAILED`, `BLOCKED`, `OOM`, `NaN`, `SOS` → log as
   `ANOMALY` events (these are not interventions, just records).

### 2.2 Resource health

7. **GPU utilisation (uses worker's gpu_assignment):** read
   `tmp/gpu_assignment.json.train_ready_ids` (the worker's current
   "ours and big enough to train on" set). If that set is non-empty AND
   the worker is in a training phase (PHASE 1/2/4/5/6 per `state.last_known_phase`)
   AND `nvidia-smi -i <comma-joined ids> --query-gpu=utilization.gpu --format=csv,noheader,nounits`
   shows ALL of those GPUs at 0% for the last 5 min → flag
   `GPU_IDLE`. (Do NOT use raw all-4-GPU utilisation — other tenants'
   GPUs are irrelevant to our liveness.)
8. **Disk free** (aligned with worker v3 §Disk policy):
   - `df -h .` parsed to GB.
   - `>= 200 GB` → no flag
   - `100-200 GB` → flag `DISK_WARN` (log only)
   - `50-100 GB` → flag `DISK_LOW` (log only; worker will pause itself)
   - `< 50 GB` → flag `DISK_CRITICAL` (intervene per §3 Case C; worker
     may have already SOS'd at this point)
9. **`outputs_dolly/` growth:** record total size (`du -sb outputs_dolly`).
   Compare with same value recorded N polls ago. If `state.last_known_phase
   == PHASE-2`, threshold is:
   - `N_train_ready >= 2` → growth expected within 1 hour
   - `N_train_ready == 1` → growth expected within 2 hours
   If growth threshold not met → flag `PROGRESS_STALL`.

### 2.3 Phase progress

10. **Parse current phase** from `tmp/exp_progress.log`: grep the last
    `PHASE-N start` line; if a subsequent `PHASE-N done` exists, the
    phase finished — set `state.last_known_phase` to that N's done
    state; otherwise set to `PHASE-N` (RUNNING). Record
    `state.last_known_phase_started_at_utc`.
11. **Budget overrun:** if a phase has been RUNNING longer than 1.5× its
    documented budget (per worker doc §5 phase headers), flag
    `BUDGET_OVERRUN`. Log only; do not intervene (the worker handles
    its own PARTIAL transition).
12. **Queue progress (PHASE 2):** if `tmp/phase2_queue.json` exists,
    load it (atomic; reads are safe because the worker uses
    `os.replace` per the worker doc). Count
    `(PENDING, RUNNING, DONE, FAILED, RETRIED, PERMANENTLY_FAILED,
    WORKER_RESTART)`. Compute total cell count from
    `len(phase2_queue.cells)` (do NOT hard-code 88; the worker may have
    reduced the matrix). Compute completion rate over the last 3 hours
    of poll history and project an ETA.

### 2.4 SOS detection

13. If `tmp/PAPER_COMPLETION_SOS.md` exists, the worker has given up.
    Read it, summarise into `tmp/MONITOR_ALERTS.md`, write final
    `tmp/MONITOR_FINAL_REPORT.md` (see §5) with status
    "WORKER ABORTED — human action required", set
    `state.stop_polling = true`, persist state, do NOT schedule next
    wake, exit.

### 2.5 Rolling write

14. For each newly-detected flag this iteration, append one line to
    `tmp/MONITOR_ALERTS.md`:
    ```
    <UTC iso8601> <severity: WARN|ERROR> <flag> <one-line context>
    ```
    Severity: STALL/WORKER_DEAD/DISK_CRITICAL/ZOMBIE_HEARTBEAT/SOS = ERROR;
    everything else = WARN.
15. **Overwrite** `tmp/MONITOR_REPORT.md` with:
    - Worker liveness: ALIVE / STALLED / DEAD / ABORTED
    - Current phase + elapsed in phase
    - GPU: per-train_ready-id `{util%, mem_used_gb, mem_free_gb}`
    - Disk free in GB
    - PHASE 2 queue: `N_DONE / N_TOTAL` (from `phase2_queue.json`,
      with breakdown of PERMANENTLY_FAILED) + ETA
    - Open alerts (those still active, i.e. not auto-cleared by
      condition reversing)
    - Last 10 lines of `tmp/exp_progress.log`
16. Update `tmp/MONITOR_HEARTBEAT.txt` with one line:
    `<UTC iso8601> iter=<N> phase=<X> alerts_open=<K>`.

### 2.6 Re-schedule and exit

17. Increment `state.iter_count`; update `state.last_poll_at_utc`;
    persist `tmp/MONITOR_STATE.json` atomically (`.tmp` + `os.replace`).
18. `ScheduleWakeup(delaySeconds=300, prompt='Continue monitor loop per
    docs/paper_completion_monitor.md')`. Exit turn.

---

## 3. Intervention rules (cases A/B/C only)

You may intervene in **exactly** these three cases. All other flags →
log only.

### Cooldowns and limits (global)
- **Wake cooldown:** at least **30 min** (1800 s) between consecutive
  wake signals. Track `state.last_wake_at_utc`.
- **Restart cap:** at most **3 total** restarts over the run. Track
  `state.restart_count`. 4th restart attempt → SOS instead.

### Case A: STALL (heartbeat silent > 30 min OR zombie heartbeat detected)

1. If `now - state.last_wake_at_utc < 1800`, do nothing more this
   iteration (cooldown still active). Re-evaluate next poll.
2. Append ALERT.
3. `ps -ef | grep -E 'python.*(train|evaluate)|torchrun|lm_eval' |
   grep -v grep` — if a python process is running AND not a zombie
   heartbeat → the worker may be in a long compute block; do not
   intervene; recheck next poll.
4. Otherwise send wake signal using **proper tmux quoting** (single
   quotes around the message, `Enter` as a separate keystroke token):
   ```bash
   PHASE_TAG=$(jq -r '.last_known_phase // "PHASE-?"' tmp/MONITOR_STATE.json)
   MSG="Continue. Last status was ${PHASE_TAG}. Resume from the next undone phase per tmp/EXPERIMENTS_DONE.md."
   tmux send-keys -t "$(jq -r '.worker_session_name' tmp/MONITOR_STATE.json)" "$MSG" Enter
   ```
5. Update `state.last_wake_at_utc = now`, increment `state.wake_count`.
6. Log RESTARTED message to `tmp/MONITOR_ALERTS.md`.
7. If a further 30 min later there's still no heartbeat AND no python
   process, escalate to Case B.

### Case B: WORKER_DEAD (tmux session gone, OR no heartbeat ≥ 60 min and no python process)

1. Append ALERT.
2. If `state.restart_count >= 3`, do NOT attempt restart — write
   `tmp/PAPER_COMPLETION_SOS.md` with the failure context and stop
   polling. The monitor's role then collapses to "report and exit".
3. Attempt restart (use the captured repo root from `state.repo_root`):
   ```bash
   REPO_ROOT=$(jq -r '.repo_root' tmp/MONITOR_STATE.json)
   tmux new-session -d -s sakd_finish -c "$REPO_ROOT" \
     "claude --dangerously-skip-permissions"
   sleep 5
   tmux send-keys -t sakd_finish '/goal docs/paper_completion_task.md' Enter
   ```
   **Caveat:** the exact CLI invocation depends on the local Claude
   Code build. If the above fails (verifiable by checking `tmux
   capture-pane -p -t sakd_finish` doesn't show a Claude prompt within
   30 s), fall back to step 4 immediately.
4. Wait 5 min (next iteration); recheck heartbeat. If alive, log
   RESTARTED, increment `state.restart_count`. If still dead:
   - If `state.restart_count >= 3`, write SOS, stop polling.
   - Else retry restart (step 3) next iteration.

### Case C: DISK_CRITICAL (< 50 GB free)

1. Append ALERT.
2. Send a tmux warning to the worker (it may already be in the act of
   SOS'ing but redundancy is cheap):
   ```bash
   tmux send-keys -t "$(jq -r '.worker_session_name' tmp/MONITOR_STATE.json)" \
     'DISK CRITICAL: < 50 GB free. Pause any pending training launches and write a SOS file if you cannot recover.' Enter
   ```
3. Do NOT delete files yourself.
4. Reduce polling interval to **60 s** until disk free > 100 GB or
   worker writes SOS. Reset to 300 s once recovered.

**For every other flag** (GPU_IDLE, BUDGET_OVERRUN, PROGRESS_STALL,
DISK_WARN, DISK_LOW, ANOMALY) → log only; the worker has its own
retry/PARTIAL logic.

---

## 4. Reporting cadence to the human

- **`tmp/MONITOR_REPORT.md`** — rolling snapshot, overwritten every
  iteration. Human reads this for "what's happening right now".
- **`tmp/MONITOR_ALERTS.md`** — append-only event log. Human reads this
  for "what unusual events happened during the run".
- **`tmp/MONITOR_HEARTBEAT.txt`** — single line, refreshed every
  iteration. Format: `<UTC iso8601> iter=<N> phase=<X>
  alerts_open=<K>`. **Human-side liveness check on the monitor itself:**
  if this file's mtime is > 10 min old, the monitor has crashed; human
  action needed (kill the `sakd_monitor` tmux and re-launch).
- **`tmp/MONITOR_FINAL_REPORT.md`** — written ONCE at end of run; see §5.
- **No notifications.** The human will `cat tmp/MONITOR_REPORT.md`
  whenever they want a status. Do not send Slack / email / push.

---

## 5. Final report (when worker writes `PAPER_COMPLETION_DONE.md` OR SOS)

When either `tmp/PAPER_COMPLETION_DONE.md` or
`tmp/PAPER_COMPLETION_SOS.md` appears:

1. Stop the polling loop (do NOT call `ScheduleWakeup` again).
2. **Write** `tmp/MONITOR_FINAL_REPORT.md` (do not append to
   `MONITOR_REPORT.md`, which gets overwritten anyway):
   ```markdown
   # Monitor final report

   - End state: {WORKER_DONE | WORKER_ABORTED_SOS}
   - Total wall-clock: <duration>
   - Iteration count: <N>
   - Intervention events:
     - Wakes: <count> (last at <utc>)
     - Restarts: <count> (last at <utc>)
   - Worst-case observed:
     - Heartbeat gap: <minutes>
     - GPU idle duration in train phase: <minutes>
     - Min disk free: <GB>
   - Final PHASE 2 cell count: N_DONE / N_TOTAL (from phase2_queue.json)
     PERMANENTLY_FAILED cells: [<id>, ...]
   - Open alerts at end of run: [...]
   - SOS payload (if any): <copy of PAPER_COMPLETION_SOS.md>
   ```
3. Update `tmp/MONITOR_STATE.json` with `stop_polling: true`.
4. Update `tmp/MONITOR_HEARTBEAT.txt` one last time with
   `<UTC> MONITOR_EXIT <reason>`.
5. Exit cleanly. Do not schedule next wake.

---

## 6. Out of scope (do NOT do)

- Modify any of the worker's deliverable files
  (`writing/`, `outputs_*/`, `data/teacher_saliency_*`, `tables/`,
  `figures/`, `algorithms/`, `tmp/EXPERIMENTS_DONE.md`,
  `tmp/phase*_*.json`, `tmp/gpu_assignment.json`,
  `tmp/exp_progress.log`, `tmp/CKPT_MANIFEST.csv`,
  `tmp/PAPER_COMPLETION_DONE.md`).
- Run any experiment yourself.
- Make any git commit / push / pull.
- Patch `src/sagd/` or `scripts/`.
- Open new tmux sessions beyond the Case B restart of `sakd_finish`.
- Send notifications outside the `tmp/MONITOR_*` files.
- Read `tmp/PAPER_COMPLETION_SOS.md` for any purpose other than
  copying its contents into `MONITOR_FINAL_REPORT.md`.
- Adjust the worker's training hyperparameters by injecting tmux
  commands. The only commands you may inject are the wake/restart
  prompts in Cases A/B/C.
- Spin-wait or sleep > 30 seconds in a single Claude turn.
- Send more than 1 wake per 30 min (cooldown).
- Attempt more than 3 restarts (cap).

---

## 7. Startup checklist (first activation only)

On first `/goal docs/paper_completion_monitor.md`:

1. Capture `REPO_ROOT=$(pwd)` (the directory the monitor was launched
   from — assumed to be the repo root). Persist to
   `tmp/MONITOR_STATE.json` immediately.
2. Verify worker is running: `tmux has-session -t sakd_finish
   2>/dev/null; echo $?`. If absent:
   - Try `tmux list-sessions -F '#{session_name}' | grep -i sakd | head -1`.
   - If still nothing, write a one-line `tmp/MONITOR_REPORT.md` saying
     "Worker not yet started" and proceed to step 3 anyway (next
     iteration will recheck).
3. Initialise `tmp/MONITOR_STATE.json` with schema:
   ```json
   {
     "schema_version": "monitor_state_v1",
     "repo_root": "<REPO_ROOT>",
     "monitor_started_at_utc": "<now>",
     "last_poll_at_utc": null,
     "iter_count": 0,
     "worker_session_name": "sakd_finish",
     "last_known_phase": null,
     "last_known_phase_started_at_utc": null,
     "stall_first_detected_at_utc": null,
     "last_wake_at_utc": null,
     "wake_count": 0,
     "restart_count": 0,
     "open_alerts": [],
     "stop_polling": false
   }
   ```
   Write atomically (`.tmp` + `os.replace`).
4. Touch (create-if-absent) `tmp/MONITOR_ALERTS.md` and
   `tmp/MONITOR_HEARTBEAT.txt`.
5. Do one polling iteration immediately (§2).
6. `ScheduleWakeup(delaySeconds=300, prompt='Continue monitor loop per
   docs/paper_completion_monitor.md')`. Exit.

On subsequent wakes: skip steps 1–4; jump directly to §2.0 (load state).

---

## 8. How the human will consume your output

- During the run:
  - `cat tmp/MONITOR_REPORT.md` for current status
  - `tail -20 tmp/MONITOR_ALERTS.md` for recent unusual events
  - `cat tmp/MONITOR_HEARTBEAT.txt` to confirm monitor itself is alive
    (if mtime > 10 min, the monitor has crashed)
- After the run:
  - `cat tmp/MONITOR_FINAL_REPORT.md` alongside the worker's
    `tmp/PAPER_COMPLETION_DONE.md`

Your job ends when `tmp/MONITOR_FINAL_REPORT.md` is written and the
polling loop is stopped.

---

## 9. State schema reference

`tmp/MONITOR_STATE.json` always conforms to:

```json
{
  "schema_version": "monitor_state_v1",
  "repo_root": "string (abs path)",
  "monitor_started_at_utc": "iso8601",
  "last_poll_at_utc": "iso8601 or null",
  "iter_count": "int >= 0",
  "worker_session_name": "string (default 'sakd_finish')",
  "last_known_phase": "string ('PHASE-0' .. 'PHASE-11') or null",
  "last_known_phase_started_at_utc": "iso8601 or null",
  "stall_first_detected_at_utc": "iso8601 or null (reset when heartbeat resumes)",
  "last_wake_at_utc": "iso8601 or null",
  "wake_count": "int >= 0",
  "restart_count": "int >= 0, hard cap 3",
  "open_alerts": [
    {"flag": "STALL|WORKER_DEAD|GPU_IDLE|...",
     "first_seen_utc": "iso8601",
     "context": "string"}
  ],
  "stop_polling": "bool (true after final report written)"
}
```

If a future schema bump becomes needed, increment the `_v` suffix and
write a migration block at the top of §7 startup.
