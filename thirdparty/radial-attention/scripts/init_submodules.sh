#!/bin/bash

# Initialize and update git submodules
echo "Initializing git submodules..."
git submodule update --init --recursive

# Apply patches to sparse_sageattn_2
echo "Applying patches to sparse_sageattn_2..."
PATCH_FILE="patches/sparse_sageattn_2_setup.patch"

if [ -f "$PATCH_FILE" ]; then
    cd third_party/sparse_sageattn_2

    # Check if patch has already been applied
    if git diff --quiet setup.py; then
        echo "Applying patch: $PATCH_FILE"
        git apply ../../"$PATCH_FILE"

        if [ $? -eq 0 ]; then
            echo "✓ Patch applied successfully"
        else
            echo "✗ Failed to apply patch"
            exit 1
        fi
    else
        echo "✓ Patch already applied or setup.py already modified"
    fi

    cd ../..
else
    echo "✗ Patch file not found: $PATCH_FILE"
    exit 1
fi

echo "✓ Submodules initialized and patched successfully"
