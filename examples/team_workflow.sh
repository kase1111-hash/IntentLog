#!/bin/bash
# IntentLog Team Workflow — Branching and merging reasoning
#
# Demonstrates how a team can use IntentLog branches to explore
# alternative approaches and merge the winning one.
#
# Usage: bash examples/team_workflow.sh

set -e

echo "=== IntentLog Team Workflow ==="
echo ""

# Create workspace
WORKSPACE=$(mktemp -d)
cd "$WORKSPACE"
git init .
git config user.email "demo@example.com"
git config user.name "Demo User"
echo "# Project" > README.md
git add . && git commit -m "Initial commit"

ilog init team-project
echo ""

echo "--- 1. Record the initial architecture decision ---"
ilog commit "Starting with a REST API because the team is familiar with it and our frontend needs simple CRUD operations. GraphQL is tempting but adds complexity we don't need yet."
echo ""

echo "--- 2. Create a branch to explore an alternative ---"
ilog branch graphql-exploration
echo ""
ilog commit "Exploring GraphQL because the mobile team says they're over-fetching data. REST requires 3 round-trips for the dashboard; GraphQL could do it in 1."
ilog commit "Prototyped GraphQL schema — it works but adds ~2000 LOC of resolvers and the learning curve is steep for 2 of 4 team members."
echo ""

echo "--- 3. Tag the exploration intents ---"
LATEST=$(ilog log --json | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['intent_id'][:8])")
ilog tag "$LATEST" exploration graphql
echo ""

echo "--- 4. Switch back to main and record the decision ---"
ilog branch main
ilog commit "Decision: staying with REST for v1. The GraphQL exploration showed benefits for mobile but the team velocity cost is too high right now. Revisit after hiring. See graphql-exploration branch for the full analysis."
echo ""

echo "--- 5. Link the decision to the exploration ---"
DECISION=$(ilog log --json | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['intent_id'][:8])")
ilog branch graphql-exploration
EXPLORATION=$(ilog log --json | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['intent_id'][:8])")
ilog branch main
ilog link "$DECISION" "$EXPLORATION"
echo ""

echo "--- 6. View the full decision trail ---"
echo ""
echo "Main branch:"
ilog log --branch main
echo ""
echo "Exploration branch:"
ilog log --branch graphql-exploration
echo ""

echo "--- 7. Show the linked decision ---"
ilog show "$DECISION"
echo ""

echo "=== Done! ==="
echo "Workspace: $WORKSPACE"
echo ""
echo "Key takeaway: IntentLog branches let you explore alternatives"
echo "and keep the reasoning history even when you reject an approach."
