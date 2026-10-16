#!/usr/bin/env bash
# block until the dataset download process has exited
while pgrep -f '[f]etch_data.py all' >/dev/null; do sleep 30; done
tr '\r' '\n' < /home/snakehand/Documents/mdpi_papers/AIIA/logs/fetch_data.log | tail -2
