#!/usr/bin/env bash
# Rubric R2.3: "Running python chat.py starts the system without import errors or
# crashes." Starts the real CLI against live Azure, feeds it "exit", and asserts a
# clean shutdown with all four agents loaded.
#
# This exists because the offline conformance suite CANNOT cover it: those tests
# patch the agent constructors, so they pass whether or not the real ones can
# reach their data stores. A stale PostgreSQL firewall rule once made `python
# chat.py` die on startup while all 61 unit tests stayed green.
set -euo pipefail

SOLUTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$SOLUTION_DIR/.venv/bin/python"
[[ -x "$PY" ]] || PY=python3
LOG="$SOLUTION_DIR/docs/evaluation/cli_startup.log"

cd "$SOLUTION_DIR"
printf 'exit\n' | "$PY" -u chat.py > "$LOG" 2>&1
status=$?

fail() { printf '\033[31mFAIL: %s\033[0m\n' "$*" >&2; exit 1; }

[[ $status -eq 0 ]]                              || fail "chat.py exited $status"
grep -q "Agent Chat Ready"       "$LOG"          || fail "no ready banner"
[[ $(grep -cE "^Loading .* Data Agent" "$LOG") -eq 4 ]] || fail "not all four agents loaded"
! grep -q "Traceback"            "$LOG"          || fail "traceback during startup"

printf '\033[32mPASS: chat.py started, loaded 4 agents, and exited cleanly\033[0m\n'
echo "transcript: ${LOG#"$SOLUTION_DIR"/}"
