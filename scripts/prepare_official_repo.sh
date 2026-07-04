#!/usr/bin/env bash
set -euo pipefail

mkdir -p third_party
if [ ! -d third_party/AgentDoG/.git ]; then
  git clone --depth 1 https://github.com/AI45Lab/AgentDoG.git third_party/AgentDoG
fi
git -C third_party/AgentDoG rev-parse HEAD > third_party/AgentDoG_COMMIT.txt
cat third_party/AgentDoG_COMMIT.txt

