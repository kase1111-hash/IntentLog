#!/bin/bash
# IntentLog Quickstart — 5 minute walkthrough
#
# This script demonstrates the core IntentLog workflow.
# Run it inside a git repository, or it will create a temporary one.
#
# Usage: bash examples/quickstart.sh

set -e

echo "=== IntentLog Quickstart ==="
echo ""

# Create a temporary workspace
WORKSPACE=$(mktemp -d)
cd "$WORKSPACE"
git init .
git config user.email "demo@example.com"
git config user.name "Demo User"
echo "# My Project" > README.md
git add . && git commit -m "Initial commit"

echo ""
echo "--- Step 1: Initialize IntentLog ---"
ilog init my-project
echo ""

echo "--- Step 2: Record your first intent ---"
ilog commit "Starting with a monolithic architecture because the team is 3 people and we need fast iteration. We'll revisit splitting services once we hit scaling pain."
echo ""

echo "--- Step 3: Make a code change and record reasoning ---"
echo "import psycopg2" > db.py
git add db.py
ilog commit "Switching from SQLite to PostgreSQL because we need concurrent writes across 3 services. Considered MongoDB but relational constraints matter more than schema flexibility here."
git commit -m "Add PostgreSQL support"
echo ""

echo "--- Step 4: View your reasoning history ---"
ilog log
echo ""

echo "--- Step 5: Search past decisions ---"
echo "Searching for 'database'..."
ilog search "database"
echo ""

echo "--- Step 6: Tag an intent for organization ---"
# Get the hash of the latest intent
LATEST=$(ilog log --json | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['intent_id'][:8])")
ilog tag "$LATEST" architecture database
echo ""

echo "--- Step 7: Check project status ---"
ilog status
echo ""

echo "--- Step 8: Show full intent details ---"
ilog show "$LATEST"
echo ""

echo "=== Done! ==="
echo "Workspace: $WORKSPACE"
echo ""
echo "Try these next:"
echo "  ilog branch experiment     # Create a reasoning branch"
echo "  ilog blame README.md       # See intent for a file's history"
echo "  ilog hooks install         # Add git commit hook"
