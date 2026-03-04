# Model Calibration Recovery Log (2026-03-04)

## Summary
This document records the notebook/code recovery actions taken after calibration files appeared missing on `dev_levers`.

The loss was caused by branch history rewrite (rebase), not permanent deletion.

## Root Cause
- A snapshot commit existed on `dev_levers`: `9c1dafd` (`chore: snapshot dev_levers work including docs backup`).
- `dev_levers` was later rebased onto a different commit chain (`d585428 -> 4e151f7 -> ed2e2d1`).
- Files that only existed in `9c1dafd` disappeared from branch tip after the rewrite.
- Git reflog still referenced `9c1dafd`, so recovery was possible.

## Recovery Actions Performed
1. Verified missing assets via git history and working tree.
2. Created safety branch:
   - `backup/notebooks_9c1dafd` -> `9c1dafd`
3. Restored missing notebooks from backup commit.
4. Committed restored notebooks:
   - `d201db3` (`restore: recover calibration notebooks from backup snapshot`)
5. Push initially failed due to GitHub 100MB limit (`notebooks/scenario_testing_v2_1.ipynb`).
6. Cleared outputs in the oversized notebook and amended recovery commit.
7. Pushed branch successfully using explicit non-thin push:
   - `origin/dev_levers` updated to `d201db3`
8. Restored calibration engine code from backup snapshot for the coupled runtime files:
   - `household_energy/agent.py`
   - `household_energy/analyze.py`
   - `household_energy/climate.py`
   - `household_energy/config_defaults.yaml`
   - `household_energy/make_animation.py`
   - `household_energy/model.py`
   - `household_energy/run.py`
   - `household_energy/run_lsoa_batch.py`
   - `household_energy/server.py`
9. Committed restored engine files on `codex/model_calibration`:
   - `d727ad7` (`restore: recover calibration engine files from backup snapshot`)

## Branch State
- `dev_levers` and `origin/dev_levers` share recovered notebook history at `d201db3`.
- `codex/model_calibration` is based on the recovered state and adds:
  - `d727ad7` (calibration engine file restore)

## Notes
- Local uncommitted/untracked items not included in the recovery commit were intentionally left untouched:
  - `README.md` (modified)
  - `household_energy/config.py` (modified)
  - `household_energy/serl_calibration_pipeline.py` (untracked)
  - `notebooks/recovery/` (untracked)

## Recommended Safeguards
- Before rebases on working branches, create a backup branch (`backup/<name>`).
- For large notebooks, clear outputs before commit or use Git LFS.
- Keep recovery checkpoints as small, focused commits (not mixed with unrelated edits).
