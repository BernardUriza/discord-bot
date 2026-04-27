# Workflow Rules

## CI Pipeline
4 layers in GitHub Actions, all must pass on every push/PR:
1. **Ruff Lint & Format** — fastest, runs first, blocks everything else
2. **Tests + Coverage** — pytest with 80% minimum coverage gate
3. **Dependency Audit** — pip-audit scans for CVEs in dependencies
4. **Code Security** — bandit SAST for Python security issues

### Running CI Locally
```bash
ruff check .                                    # lint
ruff format --check .                           # format check
pytest -v --cov --cov-fail-under=80             # tests + coverage
bandit -r insult/ -c pyproject.toml             # security code
pip-audit                                       # security deps
```

### When CI Fails
- Ruff lint: run `ruff check --fix .` to auto-fix, then `ruff format .`
- Coverage below 80%: add tests for uncovered code, or add module to omit list in pyproject.toml if it's pure I/O
- pip-audit: update the vulnerable dependency
- bandit: fix the security issue or add to skips in pyproject.toml with justification

## Git Workflow
- Main branch: `main`
- Always run tests locally before pushing
- Commit messages: imperative mood, explain what and why
- One logical change per commit

## Version Bumping
- On every commit, bump the patch version (micro point) in BOTH:
  - `pyproject.toml` → `version = "X.Y.Z"`
  - `insult/cogs/chat.py` → `VERSION_TAG = "ᵇᵉᵗᵃ ᵛX·Y·Z"` (superscript unicode)
- The version tag appears at the bottom of every bot response so we can track which deploy is responding
- Bump patch (Z) for fixes/small changes, minor (Y) for features, major (X) for breaking changes

## Development Flow
1. Make changes
2. Run `ruff check . && ruff format .` (lint + format)
3. Run `pytest -v --cov` (tests + coverage)
4. Commit and push
5. Watch CI: `gh run watch --exit-status`

## Python Version
- Runtime: Python 3.14 (pinned in `.python-version`)
- CI tests only on 3.14

## Session Continuity — Don't Re-Litigate Priority
- When the prior session has already identified a critical bug, regression, or load-bearing issue, the next turn STARTS work on it. Do not open a "what should we attack first?" menu, do not enumerate alternatives like `1. fix the critical / 2. commit WIP / 3. other`, and do not ask for permission to begin.
- Surfacing options is bureaucratic friction when the priority is unambiguous. It signals lack of judgment and wastes the user's time.
- Correct behavior: open with the plan, then execute. Show the plan AS you start the fix, not before. Use TaskCreate if multi-step.
- The "what's next?" question is only legitimate when there is genuine priority ambiguity — multiple equally weighted critical items, no recent context, or an explicit user pivot. Otherwise, work.
- This rule was registered after a `/work` invocation re-asked priority on a session that had just diagnosed a grave blob-download race condition losing longitudinal facts. The user's response was unambiguous: stop asking, start working.
