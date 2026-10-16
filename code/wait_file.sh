#!/usr/bin/env bash
# wait_file.sh <path> [max_minutes]: block until a file exists
n=0; while [ ! -s "$1" ] && [ $n -lt $(( ${2:-600} * 2 )) ]; do sleep 30; n=$((n+1)); done; ls -la "$1" 2>&1
