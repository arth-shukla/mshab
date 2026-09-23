#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible preset implemented by the general chain runner.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXPECTED_TYPES="navigate,open,navigate,pick,navigate,place"
exec "$SCRIPT_DIR/evaluate_skill_chain.sh" \
    set_table apple_nav_open_pick_place 8:14
