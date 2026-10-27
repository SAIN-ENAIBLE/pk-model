# Contributing

1. Open (or pick) an issue describing the change.
2. Create a branch and make your changes, with tests.
3. Check locally:

   ```bash
   uv sync
   uv run pytest --cov
   uv run ruff check
   uv run ruff format --check
   ```

4. Open a pull request that references the issue. CI must pass (lint, tests,
   docs build) before merging.
