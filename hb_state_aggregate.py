import argparse
import re
from collections import defaultdict

parser = argparse.ArgumentParser(
    description='Aggregate hydrogen bond states from a CSV file. '
                'Normalizes residue names and state identifiers, then '
                'outputs unique states with their counts and HB counts.')
parser.add_argument('input_csv', metavar='input.csv',
                    help='Path to the input CSV file containing HB states')
args = parser.parse_args()

input_csv = args.input_csv

RE_CONFORMER_NUM = re.compile(r'_\d+')
RE_RESIDUE_CORE = re.compile(r'([A-Z]{3}).*?([A-Z]\d{4})')
RE_PAIRS = re.compile(r'\(([^)]+)\)')


def normalize_residue(name):
    name = RE_CONFORMER_NUM.sub('', name)
    m = RE_RESIDUE_CORE.search(name)
    return m.group(1) + m.group(2) if m else name


def normalize_state(state_id):
    pairs = set()
    for raw in RE_PAIRS.findall(state_id):
        residues = sorted(normalize_residue(r.strip()) for r in raw.split(','))
        pairs.add((residues[0], residues[1]))
    sorted_pairs = sorted(pairs)
    return ','.join(f'({a},{b})' for a, b in sorted_pairs)


counts = defaultdict(int)

with open(input_csv) as f:
    next(f)  # skip comment line
    next(f)  # skip header line
    for line in f:
        parts = line.rsplit(',', 3)
        state_id = parts[0].strip('"')
        count = int(parts[-2])
        key = normalize_state(state_id)
        counts[key] += count

output_file = input_csv.replace('.csv', '_output.csv')
with open(output_file, 'w') as out:
    out.write('state_normalized,count,hb_count\n')
    for state, count in sorted(counts.items(), key=lambda x: -x[1]):
        hb_count = state.count('(')
        out.write(f'"{state}",{count},{hb_count}\n')

total = sum(counts.values())
print(f"Input file:      {input_csv}")
print(f"Unique states:   {len(counts)}")
print(f"Total count:     {total}")
print(f"Output saved to: {output_file}")
