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
