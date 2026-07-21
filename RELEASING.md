# Releasing MetalTreeShap

The release process is designed so that the published files are the same artifacts tested
by GitHub Actions. Do not build a second set of files for upload.

## One-time repository setup

1. Create GitHub environments named `testpypi` and `pypi`.
2. Add this repository as a trusted publisher for the `metal-treeshap` project on
   TestPyPI and PyPI. Use the matching environment name and the release workflow file.
3. Protect the production `pypi` environment with a required reviewer.
4. Require the portable, XGBoost compatibility, macOS Metal, and package jobs on the
   default branch.

The normalized names `metal-treeshap` and `metaltreeshap` returned no project from the
PyPI JSON API when 0.1.0 was prepared on 2026-07-12. That is an availability observation,
not a reservation; configure the trusted publisher before the first upload.

## Prepare and verify

1. Update `CHANGELOG.md` and the single package version source.
2. Run the complete local validation: the full CTest suite (including the Metal
   differential fixtures on Apple Silicon), the golden suite against every supported
   XGBoost version, and the wheel/sdist checks in `ci/verify_dist.py` — the same
   gates the CI workflow runs.
3. Push the release branch — name it `release/<version>` so the push-triggered CI gate
   runs (CI only triggers on pushes to `main` and `release/**`; feature branches get
   CI through their pull request) — and wait for every required CI job to pass.
4. Build the workflow artifacts and inspect the wheel and source distribution contents.
5. Dispatch the release workflow with target `testpypi`. Its dependent Apple Silicon job
   installs the immutable TestPyPI wheel with `--no-deps` and runs the public API tests on
   Metal; do not proceed unless that job passes.

## Publish

1. Create an annotated `vX.Y.Z` tag on the verified commit and push it.
2. Create the corresponding GitHub release. The release workflow downloads the already
   built artifacts and publishes them through the protected `pypi` environment using OIDC.
3. Wait for the dependent PyPI Apple-Silicon install/API smoke job, then verify the PyPI
   metadata and published hashes before announcing the release.

Publishing requires GitHub and PyPI authorization and is intentionally impossible from a
source checkout that has neither a configured remote nor a trusted publisher.
