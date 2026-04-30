#!/bin/bash

set -e

BASE=$(pwd)

rm -rf pdbs
mkdir pdbs

for i in {1..17}; do
    for f in "$BASE"/../to_zara_17_samples/seq${i}/seq${i}_rep*; do
        [ -e "$f" ] || continue
        ln -s "$f" pdbs/
    done
done

