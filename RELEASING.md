# Releasing imcombiners

Pushing a matching `vX.Y.Z` tag runs `.github/workflows/release.yml`. Rust tests,
crate packaging, five platform wheels, and a source installation must pass
before the workflow publishes to PyPI and crates.io. It creates a GitHub release
with the Python distributions and the corresponding changelog entry after both
registries succeed. Wheels use the CPython 3.10+ stable ABI: Linux and macOS
x86-64/ARM64, plus Windows x86-64. Wheel tests use Python 3.10; source installation
tests use Python 3.13. Performance and external IRAF tests remain opt-in.

## One-time publisher setup

Create GitHub environments named `pypi` and `crates-io`, restricted to `v*` tags.
Configure these GitHub trusted publishers in the existing registry projects:

| Setting | PyPI | crates.io |
| --- | --- | --- |
| Project/crate | `imcombiners` | `imcombiners` |
| Repository owner | `ysBach` | `ysBach` |
| Repository name | `imcombiners` | `imcombiners` |
| Workflow filename | `release.yml` | `release.yml` |
| Environment | `pypi` | `crates-io` |

Use [PyPI's project Publishing settings](https://pypi.org/manage/project/imcombiners/settings/publishing/)
and the crate's Settings → Trusted Publishing. Both registries authenticate the
workflow through short-lived OIDC credentials; no persistent publishing token
is needed in GitHub secrets. See the official
[PyPI setup instructions](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
and [crates.io instructions](https://crates.io/docs/trusted-publishing).

## Each release

1. Set the same version in `Cargo.toml`, `pyproject.toml`, and their lockfiles;
   add its `CHANGELOG.md` entry. Commit and push the release changes to `main`.
2. Run `gh workflow run release.yml --ref main` and check the resulting run with
   `gh run list --workflow release.yml` / `gh run view RUN_ID`. Manual runs build
   and test packages without publishing, including when dispatched on a tag.
3. After verification succeeds, create and push the matching annotated tag:

   ```sh
   git tag -a v0.1.2 -m "Release 0.1.2"
   git push origin v0.1.2
   ```

4. Check the tag-triggered workflow and both registry version pages. Publishing
   is not atomic across registries: if one succeeds and another fails, fix the
   external problem and rerun **failed jobs**, preserving successful jobs and
   the original artifacts. A published version cannot be replaced; source
   corrections require a new version. Do not move a published tag.

CI installs built artifacts using published dependencies. The local editable
`reducers` override in `tool.uv.sources` is disabled during release validation.
