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

## Development Flow
1. Make changes
2. Run `ruff check . && ruff format .` (lint + format)
3. Run `pytest -v --cov` (tests + coverage)
4. Commit and push
5. Watch CI: `gh run watch --exit-status`

## Python Version
- Runtime: Python 3.14 (pinned in `.python-version`)
- CI tests only on 3.14
