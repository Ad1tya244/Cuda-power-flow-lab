#!/usr/bin/env bash
# Builds and runs the full experiment detached from the terminal (WSL).
cd "$(dirname "$0")"
rm -f results/DONE
bash build_wsl.sh >/dev/null && ./pf_cuda demo > results/demo_log.txt 2>&1 \
  && ./pf_cuda bench > results/bench_log.txt 2>&1
echo "exit=$?" > results/DONE
