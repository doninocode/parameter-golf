#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
REPO_URL="${REPO_URL:-https://github.com/doninocode/parameter-golf.git}"
UPSTREAM_URL="${UPSTREAM_URL:-https://github.com/openai/parameter-golf.git}"
REPO_DIR="${REPO_DIR:-$WORKDIR/parameter-golf}"
BRANCH="${BRANCH:-main}"
VARIANT="${VARIANT:-sp1024}"
TRAIN_SHARDS="${TRAIN_SHARDS:-80}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-1}"

mkdir -p "$WORKDIR"

if [[ -d "$REPO_DIR/.git" ]]; then
  git -C "$REPO_DIR" fetch --all --prune
  git -C "$REPO_DIR" remote set-url origin "$REPO_URL"
else
  git clone "$REPO_URL" "$REPO_DIR"
fi

git -C "$REPO_DIR" checkout "$BRANCH"

if git -C "$REPO_DIR" remote get-url upstream >/dev/null 2>&1; then
  git -C "$REPO_DIR" remote set-url upstream "$UPSTREAM_URL"
else
  git -C "$REPO_DIR" remote add upstream "$UPSTREAM_URL"
fi

cd "$REPO_DIR"

if [[ "$DOWNLOAD_DATA" == "1" ]]; then
  python3 data/cached_challenge_fineweb.py --variant "$VARIANT" --train-shards "$TRAIN_SHARDS"
fi

echo "Repo ready at: $REPO_DIR"
echo "Origin: $(git remote get-url origin)"
echo "Upstream: $(git remote get-url upstream)"
echo "Dataset variant: $VARIANT"
echo "Train shards requested: $TRAIN_SHARDS"
