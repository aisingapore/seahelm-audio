This repo consists of 2 packages, seahelm-audio and seahelm-models

seahelm-models is a thin pipeline to show how you may want to structure your pipeline to achieve the necessary inputs for seahelm-audio and is optional to install
seahelm-audio is the main evaluation pipeline, taking in the model outputs in the required format and performing the evaluation runs.


pre-commit run --all-files
That runs both ruff hooks (check with autofix, then format) on every tracked file. Without --all-files it only checks files currently staged in git.

Two related ones you may want:

pre-commit install sets up the git hook so it runs automatically on every git commit. Needed once per clone.
pre-commit autoupdate bumps the rev in .pre-commit-config.yaml to the latest ruff release. If you use it, update the ruff==0.16.10 pin in your root pyproject.toml to match.
If pre-commit is not found, install it first with uv tool install pre-commit.