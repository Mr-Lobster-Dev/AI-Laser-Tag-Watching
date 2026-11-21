#!/usr/bin/env bash
# Keep training running: restart it (from the latest checkpoint) whenever it exits.
# Extra arguments are passed to laser_tag.train. Logs go to ~/train_stdout.log.
cd "$(dirname "$0")/.."
while true; do
  python3 -m laser_tag.train --out /mnt/project-files/laser-tag --envs 4 --threads 2 \
    --aim-from 701 --hints-from 2309 "$@" >> "$HOME/train_stdout.log" 2>&1
  echo "$(date -u +%FT%TZ) training exited ($?), restarting in 10 s" >> "$HOME/train_stdout.log"
  sleep 10
done
