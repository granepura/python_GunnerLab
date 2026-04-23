import argparse
import csv
import math
import re
import sys
from collections import Counter, defaultdict

import networkx as nx

parser = argparse.ArgumentParser(
    description='Find H-bond networks across unique residue microstates. '
                'Accepts either the raw MCCE hb_states CSV (state_id column, '
                'conformer-level pairs) or the aggregated residue-level CSV '
                '(state_normalized column from hb_state_aggregate.py). Builds '
                'an H-bond graph per microstate and searches for networks using '
                'one of two modes: -resi_list reports connected subnetworks '
                'containing any listed residue; -resi_start_stop finds shortest '
                'paths from entry to exit residues and ranks them by Arrhenius '
                'rate and energy with pairwise and uncorrelated statistics.')
parser.add_argument('input_csv', metavar='input.csv',
                    help='Input CSV: either raw hb_states (state_id,averE,count,occ) '
                         'or aggregated (state_normalized,hb_count,count,occ)')
parser.add_argument('-resi_list', metavar='FILE',
                    help='Text file with residues of interest (one per line). '
                         'Reports connected subnetworks containing any listed residue.')
parser.add_argument('-resi_start_stop', metavar='FILE',
                    help='Tab-separated file with ENTRY/EXIT columns. '
                         'Header: ENTRY<tab>EXIT. Each line: ENTRY_RES<tab>EXIT_RES '
                         'or just ENTRY_RES (entry-only). Finds paths from any entry '
                         'residue to any exit residue.')
parser.add_argument('-max_path_length', type=int, default=None, metavar='N',
                    help='Maximum path length (edges) for start/stop search. '
                         'Default: shortest paths only. Set N to find all simple '
                         'paths up to N edges (can be slow for large N).')
parser.add_argument('-topnets', type=int, default=10, metavar='N',
                    help='Number of top networks to display (default: 10)')
parser.add_argument('-node_min', type=int, default=5, metavar='N',
                    help='Minimum number of nodes in a network path (default: 5)')
parser.add_argument('-A', type=float, default=1e13,
                    help='Arrhenius pre-exponential factor in /sec (default: 1e13)')
args = parser.parse_args()

if not args.resi_list and not args.resi_start_stop:
    parser.error('at least one of -resi_list or -resi_start_stop is required')

RE_PAIRS = re.compile(r'\(([^,]+),([^)]+)\)')
RE_CONFORMER_NUM = re.compile(r'_\d+')
RE_RESIDUE_CORE = re.compile(r'([A-Z]{3}).*?([A-Z]\d{4})')


def normalize_residue(name):
    name = RE_CONFORMER_NUM.sub('', name)
    m = RE_RESIDUE_CORE.search(name)
    return m.group(1) + m.group(2) if m else name


def normalize_state(state_id):
    pairs = set()
    for m in RE_PAIRS.finditer(state_id):
        a, b = m.group(1).strip(), m.group(2).strip()
        a, b = sorted([normalize_residue(a), normalize_residue(b)])
        pairs.add((a, b))
    return ','.join(f'({a},{b})' for a, b in sorted(pairs))


def parse_state_graph(state_str):
    G = nx.Graph()
    for m in RE_PAIRS.finditer(state_str):
        G.add_edge(m.group(1).strip(), m.group(2).strip())
    return G


def read_resi_list(path):
    residues = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                residues.add(line)
    return residues


def read_resi_start_stop(path):
    entry, exit_ = set(), set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if parts[0].upper() == 'ENTRY':
                continue
            entry.add(parts[0])
            if len(parts) >= 2:
                exit_.add(parts[1])
    if not entry:
        sys.exit(f'Error: no entry residues found in {path}')
    if not exit_:
        sys.exit(f'Error: no exit residues found in {path}')
    return entry, exit_


def format_edge_list(edges):
    return ','.join(sorted(f'({min(a,b)},{max(a,b)})' for a, b in edges))


def print_and_write(fh, text):
    print(text)
    fh.write(text + '\n')


def read_input_csv(path):
    rows = []
    with open(path) as f:
        header_line = ''
        for raw_line in f:
            raw_line = raw_line.strip()
            if raw_line.startswith('#'):
                continue
            header_line = raw_line
            break
        fields = [h.strip() for h in header_line.split(',')]
        is_raw = 'state_id' in fields

        if is_raw:
            print(f'  Format: raw MCCE hb_states (conformer-level). Normalizing ...')
            for line in f:
                parts = line.rsplit(',', 3)
                count = int(parts[-2])
                rows.append({
                    'state_normalized': normalize_state(parts[0].strip('"')),
                    'count': str(count),
                })
        else:
            print(f'  Format: aggregated residue-level CSV')
            remaining = f.read()
            for r in csv.DictReader([header_line + '\n'] + remaining.splitlines(True)):
                rows.append(r)
    return rows


print(f'Reading {args.input_csv} ...')
states = read_input_csv(args.input_csv)
total_microstates = sum(int(row['count']) for row in states)
print(f'  Unique states: {len(states):,}')
print(f'  Total microstates: {total_microstates:,}')

if args.resi_list:
    resi_set = read_resi_list(args.resi_list)
    print(f'\n--- resi_list mode ---')
    print(f'Residues of interest ({len(resi_set)}): {", ".join(sorted(resi_set))}')

    out_path = args.input_csv.replace('.csv', '_networks_resi_list.csv')
    matched = 0
    with open(out_path, 'w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['state_rank', 'count', 'hb_count',
                         'residues_matched', 'network_edges', 'network_size',
                         'full_state'])
        for rank, row in enumerate(states, 1):
            state_str = row['state_normalized']
            G = parse_state_graph(state_str)
            for comp in nx.connected_components(G):
                if len(comp) < args.node_min:
                    continue
                overlap = comp & resi_set
                if not overlap:
                    continue
                sub = G.subgraph(comp)
                writer.writerow([
                    rank, row['count'], row.get('hb_count', state_str.count('(')),
                    ' '.join(sorted(overlap)),
                    format_edge_list(sub.edges()), len(sub.edges()),
                    state_str,
                ])
                matched += 1

    print(f'Networks found: {matched:,}')
    print(f'Output saved to: {out_path}')

if args.resi_start_stop:
    entry_set, exit_set = read_resi_start_stop(args.resi_start_stop)
    print(f'\n--- resi_start_stop mode ---')
    print(f'Entry residues ({len(entry_set)}): {", ".join(sorted(entry_set))}')
    print(f'Exit residues  ({len(exit_set)}): {", ".join(sorted(exit_set))}')

    use_shortest = args.max_path_length is None
    if use_shortest:
        print('Path mode: shortest paths only (use -max_path_length N for all paths up to N edges)')
    else:
        print(f'Path mode: all simple paths up to {args.max_path_length} edges')
    print(f'Minimum path nodes: {args.node_min}')

    print(f'Scanning {len(states):,} states for entry->exit networks ...')

    network_counts = Counter()
    pairwise_counts = defaultdict(int)
    state_edge_sets = []
    states_with_paths = 0

    for i, row in enumerate(states):
        state_str = row['state_normalized']
        count = int(row['count'])
        G = parse_state_graph(state_str)
        nodes = set(G.nodes())

        edge_set = set()
        for a, b in G.edges():
            edge = (min(a, b), max(a, b))
            pairwise_counts[edge] += count
            edge_set.add(edge)
        state_edge_sets.append((edge_set, count))

        entries_in = entry_set & nodes
        exits_in = exit_set & nodes
        if not entries_in or not exits_in:
            continue

        found_any = False
        for e_start in sorted(entries_in):
            for e_end in sorted(exits_in):
                if e_start == e_end:
                    continue
                if not nx.has_path(G, e_start, e_end):
                    continue
                if use_shortest:
                    paths = nx.all_shortest_paths(G, e_start, e_end)
                else:
                    paths = nx.all_simple_paths(G, e_start, e_end,
                                                cutoff=args.max_path_length)
                for path in paths:
                    if len(path) < args.node_min:
                        continue
                    network_counts[' -> '.join(path)] += count
                    found_any = True
        if found_any:
            states_with_paths += 1

    print(f'  States with at least one path: {states_with_paths:,}')
    print(f'  Unique networks found: {len(network_counts):,}')

    top_networks = network_counts.most_common(args.topnets)
    A = args.A
    out_txt = args.input_csv.replace('.csv', '_networks_start_stop.txt')
    out_csv = args.input_csv.replace('.csv', '_networks_start_stop.csv')

    print(f'Writing results ...')

    with open(out_txt, 'w') as fh_txt, \
         open(out_csv, 'w', newline='') as fh_csv:

        csv_writer = csv.writer(fh_csv)
        csv_writer.writerow(['network_rank', 'count', 'percentage',
                             'rate_k_per_sec', 'energy_kcal_mol',
                             'network', 'path_length',
                             'subnet_percentages', 'pw_percentages',
                             'uncorr_percentage'])

        print_and_write(fh_txt, f'Total microstates: {total_microstates:,}')
        print_and_write(fh_txt, f'Total unique states: {len(states):,}')
        print_and_write(fh_txt, f'States with entry->exit paths: {states_with_paths:,}')
        print_and_write(fh_txt, f'Unique networks found: {len(network_counts):,}')
        print_and_write(fh_txt, f'Arrhenius pre-exponential factor A = {A:.2e} /sec')
        print_and_write(fh_txt, f'Minimum path nodes: {args.node_min}')
        print_and_write(fh_txt, f'Top {min(args.topnets, len(top_networks))} networks:\n')
        print_and_write(fh_txt, '=' * 120)

        for idx, (network_str, count) in enumerate(top_networks, 1):
            nodes = network_str.split(' -> ')
            ratio = count / total_microstates
            if ratio >= 1.0:
                E, k = 0.0, A
            else:
                E = -1.364 * math.log10(ratio)
                k = A * 10 ** (-E / 1.364)

            print_and_write(fh_txt, '')
            print_and_write(fh_txt,
                            f'Network {idx}: {count:,} microstates '
                            f'({ratio:.2%}) share this network.')
            print_and_write(fh_txt,
                            f'Network Rate & Energy: k = {k:.2e} /sec, '
                            f'E = {E:.2e} kcal/mol')
            print_and_write(fh_txt, 'Network structure:')
            print_and_write(fh_txt, network_str)

            subnet_percents = []
            for si in range(2, len(nodes) + 1):
                prefix_edges = set()
                for j in range(si - 1):
                    prefix_edges.add((min(nodes[j], nodes[j + 1]),
                                      max(nodes[j], nodes[j + 1])))
                subnet_count = sum(c for es, c in state_edge_sets
                                   if prefix_edges <= es)
                subnet_percents.append(subnet_count / total_microstates)

            subnet_line = 'Subnet %:' + ' ' * max(1, len(nodes[0]) - 8)
            for i in range(len(subnet_percents)):
                sn_str = f'{subnet_percents[i]:.2%}'
                padding = len(f' -> {nodes[i + 1]}') - len(sn_str)
                subnet_line += ' ' * max(padding, 1) + sn_str
            print_and_write(fh_txt, subnet_line)

            pw_percents = []
            pw_ratios = []
            for i in range(len(nodes) - 1):
                edge_key = (min(nodes[i], nodes[i + 1]),
                            max(nodes[i], nodes[i + 1]))
                pw_ratio = pairwise_counts.get(edge_key, 0) / total_microstates
                pw_percents.append(f'{pw_ratio:.2%}')
                pw_ratios.append(pw_ratio)

            uncorr_pct = math.prod(pw_ratios) * 100

            pw_line = 'PW %:' + ' ' * max(1, len(nodes[0]) - 4)
            for i in range(len(nodes) - 1):
                pw_str = pw_percents[i]
                padding = len(f' -> {nodes[i + 1]}') - len(pw_str)
                pw_line += ' ' * max(padding, 1) + pw_str
            print_and_write(fh_txt, pw_line)

            print_and_write(fh_txt,
                            f'Uncorr %: {uncorr_pct:.2e}% '
                            f'(PW is just the pairwise interaction %)')

            csv_writer.writerow([
                idx, count, f'{ratio:.6f}',
                f'{k:.2e}', f'{E:.2e}',
                network_str, len(nodes) - 1,
                ';'.join(f'{sp:.2%}' for sp in subnet_percents),
                ';'.join(pw_percents), f'{uncorr_pct:.2e}%',
            ])

            print_and_write(fh_txt, '-' * 120)

    print(f'\nStatistics saved to: {out_txt}')
    print(f'CSV saved to:        {out_csv}')
