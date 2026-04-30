#!/bin/bash

set -e

for pdb in pdbs/*.pdb; do
    base=$(basename "$pdb")                 # seq1_rep1.pdb
    name=$(basename "$pdb" .pdb)           # seq1_rep1
    dir=$(echo "$name" | tr 'a-z' 'A-Z')   # SEQ1_REP1

    if [ -d "$dir" ]; then
        echo "Processing $dir with $base"

        # Copy PDB (keep original filename)
        cp -f "$pdb" "$dir/$base"

        # Link script and models
        ln -sf ../mcce2ml_pka.py "$dir/"
        ln -sfn ../MCCE_ML-models "$dir/"

        # Run inside directory
        (
            cd "$dir" || exit
            ./mcce2ml_pka.py "$base"
        )

    else
        echo "Skipping $name (no directory)"
    fi
done

