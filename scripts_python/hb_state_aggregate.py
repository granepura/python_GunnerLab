"""
hb_state_aggregate.py — Aggregate conformational microstates into residue microstates.

Overview
--------
Each row in the raw MCCE hb_states CSV is a unique conformational microstate:
a specific set of H-bond pairs where each residue is in a specific conformer.
The conformer-level names look like SER01A0085_002 or HOH01W0041_009, where
the _002 / _009 suffixes are conformer IDs, and prefixes like 01, -1, +1, BK
encode charge state and backbone/sidechain identity.

This script strips all of that down to a residue-level name (e.g. SERA0085,
HOHW0041) by removing the conformer suffix and extracting just the 3-letter
residue code + chain letter + 4-digit residue number. It then sorts and
deduplicates all H-bond pairs to produce a normalized residue-level state.

Multiple conformational microstates that have the same residues bonded — just
in different conformers — normalize to the same residue-level state and get
merged, with their counts summed.

Example
-------
If the raw file has two conformational microstates:

  (LYS+1A0508_002,GLU-1A0605_003),(GLU-1A0605_003,ARG+1A0670_004),(LYS+1A0508_002,ARG+1A0670_004)  count 10
  (LYS+1A0508_005,GLU-1A0605_007),(GLU-1A0605_007,ARG+1A0670_001),(LYS+1A0508_005,ARG+1A0670_001)  count 15

Both normalize to the same residue state:

  (ARGA0670,GLUA0605),(ARGA0670,LYSA0508),(GLUA0605,LYSA0508)

and get merged into a single output row with count 25.

Input
-----
Raw MCCE hb_states CSV with a comment header line and columns:
  state_id, averE, count, occ

Output
------
CSV of unique residue microstates (_resi-states.csv) with columns:
  state_normalized  — residue-level H-bond pair set
  hb_count          — number of H-bond pairs in that state
  count             — total microstates across all conformational variants
  occ               — occupancy (count / total state space)

Two comment header lines record input statistics (conformational microstates,
total state space, coverage %) and output statistics (unique residue microstates).
"""

import argparse
import re
import sys
from collections import defaultdict

parser = argparse.ArgumentParser(
    description='Aggregate unique conformational microstates into unique residue '
                'microstates. Each row in the input CSV is a unique conformational '
                'microstate defined by H-bond pairs between specific conformers '
                '(e.g. SER01A0085_002). This script strips conformer IDs to produce '
                'residue-level pairs (e.g. SERA0085), then merges conformational '
                'microstates that share the same residue-level H-bond topology and '
                'sums their counts. Output: a CSV of unique residue microstates with '
                'hb_count, count, and occupancy over the total state space.')
parser.add_argument('input_csv', metavar='input.csv', nargs='?',
                    help='Input CSV from MCCE hb_states (with comment header line, '
                         'columns: state_id, averE, count, occ)')
parser.add_argument('--info', action='store_true',
                    help='Print detailed description of what this script does and exit')
args = parser.parse_args()

if args.info:
    print(__doc__)
    sys.exit(0)

if not args.input_csv:
    parser.error('input.csv is required (use --info for detailed description)')

input_csv = args.input_csv

RE_CONFORMER_NUM = re.compile(r'_\d+')
RE_RESIDUE_CORE = re.compile(r'([A-Z]{3}).*?([A-Z]\d{4})')
RE_PAIRS = re.compile(r'\(([^)]+)\)')
RE_NUMS = re.compile(r'[\d,]+')


def normalize_residue(name):
    name = RE_CONFORMER_NUM.sub('', name)
    m = RE_RESIDUE_CORE.search(name)
    return m.group(1) + m.group(2) if m else name


def normalize_state(state_id):
    pairs = set()
    for raw in RE_PAIRS.findall(state_id):
        residues = sorted(normalize_residue(r.strip()) for r in raw.split(','))
        pairs.add((residues[0], residues[1]))
    return ','.join(f'({a},{b})' for a, b in sorted(pairs))


print(f'Reading {input_csv} ...')

counts = defaultdict(int)
input_unique_conf = 0

with open(input_csv) as f:
    comment_line = next(f).strip()
    next(f)
    for line in f:
        parts = line.rsplit(',', 3)
        state_id = parts[0].strip('"')
        count = int(parts[-2])
        counts[normalize_state(state_id)] += count
        input_unique_conf += 1

total = sum(counts.values())
nums = [int(n.replace(',', '')) for n in RE_NUMS.findall(comment_line)]
total_state_space = nums[-1] if len(nums) >= 3 else total
pct = total / total_state_space * 100 if total_state_space else 0
reduction = input_unique_conf - len(counts)

print(f'  Conformational microstates (input rows):  {input_unique_conf:,}')
print(f'  Total microstates (sum of counts):         {total:,}')
print(f'  Total state space:                         {total_state_space:,}')
print(f'  Coverage of state space:                   {pct:.2f}%')
print(f'Normalizing conformer -> residue pairs ...')
print(f'  Unique residue microstates:                {len(counts):,}')
print(f'  Conformational states merged:              {reduction:,}')

output_file = input_csv.replace('.csv', '_resi-states.csv')
print(f'Writing {output_file} ...')

with open(output_file, 'w') as out:
    out.write(f'# Input: {input_unique_conf:,} unique conformational microstates, '
              f'sum count {total:,}/{total_state_space:,} ({pct:.2f}%) of the state space\n')
    out.write(f'# Output: {len(counts):,} unique residue microstates '
              f'(aggregated from {input_unique_conf:,} conformational microstates)\n')
    out.write('state_normalized,hb_count,count,occ\n')
    for state, count in sorted(counts.items(), key=lambda x: -x[1]):
        hb_count = state.count('(')
        out.write(f'"{state}",{hb_count},{count},{count / total_state_space:.2e}\n')

print(f'Done. Output saved to: {output_file}')
