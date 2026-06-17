#!/bin/bash
# Download all 324 PDBs from RCSB (asymmetric unit) for MCCE4 benchmark
# Usage: bash download_pdbs.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PDB_LIST="$SCRIPT_DIR/pdb_list.txt"
OUT_DIR="$SCRIPT_DIR/pdb_downloads"

mkdir -p "$OUT_DIR"

total=$(wc -l < "$PDB_LIST")
ok=0
fail=0
skip=0

while read -r pdb; do
    lower=$(echo "$pdb" | tr 'A-Z' 'a-z')
    outfile="$OUT_DIR/${lower}.pdb"

    if [ -s "$outfile" ]; then
        skip=$((skip + 1))
        continue
    fi

    if wget -q -O "$outfile" "https://files.rcsb.org/download/${pdb}.pdb" 2>/dev/null && [ -s "$outfile" ]; then
        ok=$((ok + 1))
    else
        rm -f "$outfile"
        echo "FAIL: $pdb"
        fail=$((fail + 1))
    fi
done < "$PDB_LIST"

echo ""
echo "Done: $ok downloaded, $skip already existed, $fail failed (out of $total)"
