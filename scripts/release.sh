#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/release.sh prepare <version> [--base <branch>] [--dry-run]
  scripts/release.sh finalize <version> [--base <branch>] [--dry-run]

Examples:
  scripts/release.sh prepare 1.3.3
  scripts/release.sh finalize 1.3.3

What it does:
  prepare:
    - create release branch from origin/<base>
    - bump pyproject.toml version
    - ensure publish workflow has skip-existing=true
    - run tests + build
    - commit, push, create PR

  finalize:
    - create/push tag v<version> at origin/<base>
    - create GitHub release from that tag (latest)
USAGE
}

die() {
  echo "[release.sh] $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

run() {
  echo "+ $*"
  if [[ "$DRY_RUN" == "false" ]]; then
    "$@"
  fi
}

ensure_clean() {
  local status
  status="$(git status --porcelain)"
  [[ -z "$status" ]] || die "working tree is not clean"
}

ensure_version() {
  [[ "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "version must match X.Y.Z"
}

ensure_remote_tag_absent() {
  local tag="$1"
  if [[ -n "$(git tag --list "$tag")" ]]; then
    die "local tag already exists: $tag"
  fi
  if git ls-remote --tags origin "$tag" | grep -q "$tag"; then
    die "remote tag already exists: $tag"
  fi
}

bump_version() {
  local version="$1"
  # perl -i behaves the same on GNU and BSD (macOS); the first match is the project version.
  run perl -0pi -e "s/^version = \"[^\"]+\"/version = \"${version}\"/m" pyproject.toml
  run perl -0pi -e "s/(\nname = \"econ-viz\"\nversion = )\"[^\"]+\"/\$1\"${version}\"/" uv.lock
  if [[ "$DRY_RUN" == "false" ]]; then
    grep -q "^version = \"${version}\"" pyproject.toml || die "failed to bump version in pyproject.toml"
  fi
}

ensure_skip_existing() {
  if grep -Eq "skip-existing:[[:space:]]*true" .github/workflows/publish.yml >/dev/null 2>&1; then
    return
  fi

  run perl -0777 -i -pe 's|(\n\s*- name: Publish to PyPI\n\s*uses:\s*pypa/gh-action-pypi-publish@release/v1\n)|$1        with:\n          skip-existing: true\n|s' .github/workflows/publish.yml
  grep -Eq "skip-existing:[[:space:]]*true" .github/workflows/publish.yml >/dev/null 2>&1 || die "failed to inject skip-existing: true"
}

create_pr_body() {
  local version="$1"
  cat <<EOF_BODY
## Summary
- bump version to ${version} in pyproject.toml
- ensure PyPI publish workflow skips existing files

## Validation
- uv sync --frozen --all-extras
- uv run pytest
- uv build

## Before merging
- [ ] Add a v${version} section to CHANGELOG.md
- [ ] Add a v${version} row to the econ-viz-docs changelog in all three languages (skip documentation-only changes)
- [ ] Update econ-viz-docs guides for new features (publish after the release is on PyPI)
EOF_BODY
}

prepare_release() {
  local version="$1"
  local base="$2"
  local tag="v${version}"
  local rel_branch="release/${tag}"

  ensure_clean
  ensure_remote_tag_absent "$tag"

  run git fetch origin
  run git checkout -b "$rel_branch" "origin/${base}"

  bump_version "$version"
  ensure_skip_existing

  run uv lock
  run uv sync --frozen --all-extras
  run uv run pytest
  run uv build

  run git add .github/workflows/publish.yml pyproject.toml uv.lock
  run git commit -m "chore(release): prepare ${tag}"
  run git push -u origin "$rel_branch"

  local body
  body="$(create_pr_body "$version")"
  run gh pr create --base "$base" --head "$rel_branch" --title "chore(release): prepare ${tag}" --body "$body"

  echo
  echo "[release.sh] prepare done. Before merging the PR:"
  echo "[release.sh]   - add a v${version} section to CHANGELOG.md"
  echo "[release.sh]   - add a v${version} row to the econ-viz-docs changelog (en, zh-TW, zh-CN)"
  echo "[release.sh] Then merge the PR and run:"
  echo "[release.sh]   scripts/release.sh finalize ${version}"
}

finalize_release() {
  local version="$1"
  local base="$2"
  local tag="v${version}"

  ensure_clean
  ensure_remote_tag_absent "$tag"

  run git fetch origin
  local target_sha
  target_sha="$(git rev-parse "origin/${base}")"

  run git tag -a "$tag" "$target_sha" -m "release: ${tag}"
  run git push origin "$tag"

  run gh release create "$tag" --title "$tag" --generate-notes --latest

  echo "[release.sh] finalize done: ${tag}"
}

main() {
  need_cmd git
  need_cmd gh
  need_cmd uv
  need_cmd rg

  if [[ $# -eq 0 ]]; then
    usage
    exit 1
  fi
  if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
  fi

  [[ $# -ge 2 ]] || {
    usage
    exit 1
  }

  local mode="$1"
  local version="$2"
  shift 2

  ensure_version "$version"

  BASE="main"
  DRY_RUN="false"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --base)
        [[ $# -ge 2 ]] || die "--base requires a value"
        BASE="$2"
        shift 2
        ;;
      --dry-run)
        DRY_RUN="true"
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "unknown option: $1"
        ;;
    esac
  done

  case "$mode" in
    prepare)
      prepare_release "$version" "$BASE"
      ;;
    finalize)
      finalize_release "$version" "$BASE"
      ;;
    *)
      die "unknown mode: $mode"
      ;;
  esac
}

main "$@"
