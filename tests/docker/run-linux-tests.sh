#!/bin/sh
set -eu

IMAGE="${AGENT_LOOP_LINUX_IMAGE:-agent-loop-linux-tests}"

docker build --file tests/docker/Dockerfile --tag "$IMAGE" .
docker run --rm \
  --init \
  --network none \
  --pids-limit 128 \
  --memory 1g \
  --cpus 2 \
  "$IMAGE" "$@"
