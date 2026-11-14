# RME-1 real-model evaluation harness

This harness runs the three RME-1 engineering scenarios against a real model through the existing LM Studio
provider. It is separate from deterministic runtime acceptance: scripted-provider tests validate the harness only
and are never evidence of model capability.

## What it contains
- `harness.py`: scenario fixtures (RME-01 utility, RME-02 repair, RME-03 extension), trusted criteria as
  `AcceptanceCriterion` objects with origin `user`, hidden evaluator checks kept outside every workspace,
  protected-file hashing, result classification, and bounded JSON records.
- `tests/test_rme1_harness.py`: deterministic harness validation (no LLM).

## Running a trial (manual, explicit)
A real trial is never started by the tests. To run one yourself:

1. Confirm the endpoint and model. Configured values are in `src/config.py`: `LM_STUDIO_URL` and `MODEL_NAME`.
   Do not change temperature or token limits between scenarios.
2. Confirm the model is loaded in LM Studio. A listed model is not necessarily loaded.
3. Create a directory outside the repository for trial output, for example `~/rme1-trials`.
4. From the repository root, call `run_trial("RME-01", trial_root, "trial-001", evaluator_dir, execute=True, endpoint_ok=True)`
   with `trial_root` and `evaluator_dir` outside the repository, then `write_record(...)`.
5. Read the record. Do not edit the model's workspace during a trial. Any intervention makes the trial assisted.

## Limitations (not enforced by this harness)
- **Isolation is not a sandbox.** The trial allowlist is `python3`, `ls`, and `cat`. Those can read or write outside
  the workspace under the user's account, because the shell is confined only by its working directory. Do not run
  trials on a machine holding data you cannot afford to expose to model-chosen commands.
- The wall-clock budget (300 s) is recorded but not enforced. Enforcing it would need an external runner.
- `MAX_TURNS` is enforced through `config.MAX_ITERATIONS`, which the loop reads at run time.
- Verification-command timeouts come from the criterion (`timeout_seconds`), not a global limit.

## RME-1.5 infrastructure: deadline drill, snapshots, single-model protocol

**Deadline drill** (`drill.py`, host only, no LLM request). `python evaluation/rme1/drill.py [graceful|forced|both] --image <ref>`.
- Graceful: the runner's request goes to a silent loopback server inside the container. At the deadline SIGTERM reaches
  the runner through tini; the runtime cancels the in-flight request, records INTERRUPTED, saves, and the runner
  publishes RESULT and SESSION frames. Expected classification: `DEADLINE_INTERRUPTED`.
- Forced: a stand-in saves a valid format-5 session and ignores SIGTERM. The container is removed and the process
  group is killed. Expected classification: `SUPERVISOR_TIMEOUT_NO_OUTCOME`, no RESULT frame, and no INTERRUPTED
  (the snapshot's run list is empty).
- Each drill writes `tmp/rme1/drill-<kind>-*/results/drill.json`, `container.log` and `session_snapshot.json`.
  `kill_error` records a refused process-group kill (EPERM was observed on macOS); the direct child is then killed.

**Snapshots.** The host reads the session with `docker exec cat` (not `docker cp`, which did not see the tmpfs-backed
`/tmp/sessions`). Each copy is validated with `validate_session_data`; an invalid, partial or failed copy never replaces
the last valid snapshot (atomic `os.replace`). A removed container is counted as `source_gone`, not as a rejection.
`last_lag_seconds` is the age of the snapshot's `saved_at`, which is naive UTC from the UTC container, so it is
interpreted as UTC on the host.

**Single-model protocol for real-model trials** (no unloading is performed by the harness or by any step of RME-1.5):
1. Run one model at a time. GLM (`zai-org/glm-4.7-flash`) first; only after its trials are finished and its
   results are archived, unload GLM through LM Studio, then load Qwen (`qwen/qwen3.8-27b`). Never load both.
2. Before each trial record: model identity from `/api/v0/models` (id, arch, quantization, state), the loaded
   context length, `memory_pressure -Q` and `sysctl -n vm.swapusage`.
3. Keep task, acceptance criteria, authorization policy, temperature and iteration budget (`MAX_TURNS`) unchanged.
4. Results from this protocol are new single-model results. They are reported separately from the original
   benchmark results and never merged into them.
5. Token measurements come from existing response metadata only (`metrics.py`). Time to first token and reasoning
   tokens are not available without a production change and are reported as unavailable.

## RME-01 policy versions (evaluation infrastructure only)

Contract revision `RME-01-CONTRACT-R2` (`policy.CONTRACT_REVISION`). Results recorded under earlier contracts keep their
original fields and are not relabelled. R2 replaces the earlier arrangement, in which `RME-01-POLICY-V1` advertised the
full historical registry (including the managed-job tools) and `RME-01-POLICY-V2` advertised a fixed list of four.

`--policy` is required by `supervisor.py` and sent to the runner as `RME_POLICY`. Nothing is implied.
- `RME-01-POLICY-V1`: the production system prompt, unmodified.
- `RME-01-POLICY-V2`: the production prompt with nine exact, count-checked replacements that remove the examples of
  unauthorized shell commands (`find`, `grep`, `rg`, pipelines, `make`, background jobs), plus an appended
  authorization block that lists the only authorized basenames (`python3`, `ls`, `cat`) and directs discovery to
  `list_directory` and `read_file`. Its advertised tools also have corrected descriptions.

Tools are a property of the scenario, not the policy. Each scenario declares the tools it allows
(`scenarios.Scenario.tools`, currently `run_command`, `write_file`, `read_file`, `list_directory` for every RME-01
scenario). Under either policy:
- only the declared tools are advertised, in declared order, with their production schemas;
- any other name, including managed-job, MCP and workspace tools that a scenario did not declare, is refused with the
  permission envelope before it runs;
- each declared tool's production schema must match its reviewed hash in `policy.REVIEWED_TOOL_SHA256`. A change to a
  declared schema, or a declaration with no reviewed entry, refuses the policy. Changing a declared schema is therefore
  a contract change: review it, update that table, and advance `CONTRACT_REVISION`. Production tools that no scenario
  declares are not pinned, so unrelated changes to them do not block evaluation.

The runner records the applied policy id, contract revision, source revision (`policy.source_fingerprint()`),
production and effective prompt hashes, the shell allowlist, the advertised tool names, and the tool-schema hash of
every request. The host recomputes the expected values from the scenario and the contract; a mismatch of any of them
is `INVALID_POLICY_MISMATCH`. Policy v2 trials require `--expected-revision`; the image label and immutable image ID are
recorded (traceability, not build provenance). `python evaluation/rme1/drill.py policy --image <ref> --expected-revision
<rev>` is the non-LLM check of this path.

The shell allowlist (`scenarios.TRIAL_TOOL_ALLOWLIST`) is not changed by any policy.
