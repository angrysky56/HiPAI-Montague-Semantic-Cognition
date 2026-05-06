#!/bin/bash
# Pre-commit hook to verify Isabelle logic foundation

# Try to find isabelle in PATH
ISABELLE_BIN=$(which isabelle)

# Fallback to standard HiPAI auto-install location in home directory
if [ -z "$ISABELLE_BIN" ]; then
    if [ -f "$HOME/Isabelle2025-2/bin/isabelle" ]; then
        ISABELLE_BIN="$HOME/Isabelle2025-2/bin/isabelle"
    fi
fi

if [ -z "$ISABELLE_BIN" ]; then
    echo "⚠️  Isabelle not found. Skipping logic verification."
    exit 0
fi

echo "🛡️  Verifying HiPAI Logic Foundation..."
$ISABELLE_BIN build -D docs/logic
RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo "✅ Logic verification passed."
else
    echo "❌ Logic verification failed! Commit aborted."
    exit 1
fi
