#!/usr/bin/env bash
set -euo pipefail

service_dir=/opt/apps/roboticsservice
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$service_dir:$service_dir/lib:$service_dir/SDK/x64"
export QT_PLUGIN_PATH="$service_dir/plugins/:${QT_PLUGIN_PATH:-}"
export QT_QML_PATH="$service_dir/qml/:${QT_QML_PATH:-}"

cd "$service_dir"
./RoboticsServiceProcess &
service_pid=$!
trap 'kill "$service_pid" 2>/dev/null || true' EXIT
sleep 2
kill -0 "$service_pid"

exec python3 /opt/pico/probe.py
