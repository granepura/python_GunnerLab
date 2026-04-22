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
input_unique_conf = 0

with open(input_csv) as f:
    comment_line = next(f).strip()
    next(f)  # skip header line
    for line in f:
        parts = line.rsplit(',', 3)
        state_id = parts[0].strip('"')
        count = int(parts[-2])
        key = normalize_state(state_id)
        counts[key] += count
        input_unique_conf += 1

total = sum(counts.values())

RE_NUMS = re.compile(r'[\d,]+')
nums = [int(n.replace(',', '')) for n in RE_NUMS.findall(comment_line)]
total_state_space = nums[-1] if len(nums) >= 3 else total

output_file = input_csv.replace('.csv', '_resi-states.csv')
with open(output_file, 'w') as out:
    pct = total / total_state_space * 100 if total_state_space else 0
    out.write(f'# Input: {input_unique_conf:,} unique conformational microstates, '
              f'sum count {total:,}/{total_state_space:,} ({pct:.2f}%) of the state space\n')
    out.write(f'# Output: {len(counts):,} unique residue microstates '
              f'(aggregated from {input_unique_conf:,} conformational microstates)\n')
    out.write('state_normalized,hb_count,count,occ\n')
    for state, count in sorted(counts.items(), key=lambda x: -x[1]):
        hb_count = state.count('(')
        occ = count / total_state_space
        out.write(f'"{state}",{hb_count},{count},{occ:.2e}\n')

print(f"Input file:                        {input_csv}")
print(f"Unique conformational microstates: {input_unique_conf:,}")
print(f"Total microstates (sum count):     {total:,}")
print(f"Total state space:                 {total_state_space:,}")
print(f"Unique residue microstates:        {len(counts):,}")
print(f"Output saved to:                   {output_file}")
