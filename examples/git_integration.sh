#!/bin/bash
# IntentLog + Git Integration — Side-by-side workflow
#
# Demonstrates how IntentLog captures git context (branch, commit, staged files)
# and how to use blame and git-commit filtering.
#
# Usage: bash examples/git_integration.sh

set -e

echo "=== IntentLog + Git Integration ==="
echo ""

# Create workspace
WORKSPACE=$(mktemp -d)
cd "$WORKSPACE"
git init .
git config user.email "demo@example.com"
git config user.name "Demo User"
echo "# API Service" > README.md
git add . && git commit -m "Initial commit"

# Initialize IntentLog
ilog init api-service
echo ""

echo "--- Recording intent alongside git work ---"
echo ""

# First feature: auth module
echo "from flask import Flask" > app.py
echo "def authenticate(): pass" > auth.py
git add app.py auth.py

echo "Step 1: Record intent BEFORE git commit"
ilog commit "Adding authentication module. Using JWT tokens because we need stateless auth for the API gateway. Session-based auth was considered but doesn't scale across services."
git commit -m "Add auth module"
echo ""

# Second feature: database
echo "import psycopg2" > db.py
git add db.py

echo "Step 2: Another intent with different staged files"
ilog commit "Adding database layer with PostgreSQL. Chose connection pooling via pgbouncer pattern. SQLAlchemy was considered too heavy for our simple query patterns."
git commit -m "Add database layer"
echo ""

echo "--- Viewing intents with git context ---"
echo ""
ilog log
echo ""

echo "--- Filtering by git commit ---"
echo ""
HEAD=$(git rev-parse HEAD)
echo "Finding intents for HEAD commit (${HEAD:0:12}):"
ilog log --git-commit "${HEAD:0:8}"
echo ""

echo "--- Intent blame for a file ---"
echo ""
echo "Blame for auth.py:"
ilog blame auth.py
echo ""

echo "--- JSON output for scripting ---"
echo ""
echo "ilog status --json:"
ilog status --json
echo ""

echo "=== Done! ==="
echo "Workspace: $WORKSPACE"
echo ""
echo "Key takeaway: every ilog commit automatically captures"
echo "the git branch, HEAD hash, and staged files — creating"
echo "a bidirectional link between reasoning and code."
