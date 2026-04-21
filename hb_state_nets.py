import argparse
import csv
import math
import re
import sys
from collections import Counter, defaultdict

import networkx as nx

parser = argparse.ArgumentParser(
    description='Find hydrogen bond networks for each state vector. '
                'Reads the aggregated output CSV and builds an HB graph '
                'per state, then filters by residue list or entry/exit residues.')
parser.add_argument('input_csv', metavar='input.csv',
                    help='Path to the aggregated output CSV file '
                         '(e.g. hb_states_pH7.00eH0.00_output.csv)')
parser.add_argument('-resi_list', metavar='FILE',
                    help='Text file with residues of interest (one per line). '
                         'Reports connected subnetworks containing any listed residue.')
parser.add_argument('-resi_start_stop', metavar='FILE',
                    help='Tab-separated file with ENTRY/EXIT columns. '
                         'Header line: ENTRY<tab>EXIT. '
                         'Each subsequent line: ENTRY_RES<tab>EXIT_RES or just ENTRY_RES.')
parser.add_argument('-max_path_length', type=int, default=None, metavar='N',
                    help='Maximum path length (number of edges) for start/stop '
                         'network search. Default: shortest paths only. '
                         'Set to find all paths up to N edges.')
parser.add_argument('-topnets', type=int, default=10, metavar='N',
                    help='Number of top networks to display in statistics '
                         '(default: 10)')
parser.add_argument('-A', type=float, default=1e13,
                    help='Arrhenius pre-exponential factor in /sec '
                         '(default: 1e13)')
args = parser.parse_args()

if not args.resi_list and not args.resi_start_stop:
    parser.error('At least one of -resi_list or -resi_start_stop is required')

RE_PAIRS = re.compile(r'\(([^,]+),([^)]+)\)')


def parse_state_graph(state_str):
    G = nx.Graph()
    for m in RE_PAIRS.finditer(state_str):
        a, b = m.group(1).strip(), m.group(2).strip()
        G.add_edge(a, b)
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
    entry = set()
    exit_ = set()
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
        print('Error: no entry residues found in', path, file=sys.stderr)
        sys.exit(1)
    if not exit_:
        print('Error: no exit residues found in', path, file=sys.stderr)
        sys.exit(1)
    return entry, exit_


def format_edge_list(edges):
    sorted_edges = sorted(('(%s,%s)' % (min(a, b), max(a, b))) for a, b in edges)
    return ','.join(sorted_edges)


def format_path(path):
    return ' -> '.join(path)


def print_and_write(out_file, text):
    print(text)
    out_file.write(text + '\n')


states = []
with open(args.input_csv) as f:
    reader = csv.DictReader(f)
    for row in reader:
        states.append(row)

total_microstates = sum(int(row['count']) for row in states)

if args.resi_list:
    resi_set = read_resi_list(args.resi_list)
    print(f'Residues of interest ({len(resi_set)}): {", ".join(sorted(resi_set))}')
    print()

    out_file = args.input_csv.replace('.csv', '_networks_resi_list.csv')
    with open(out_file, 'w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['state_rank', 'count', 'hb_count',
                         'residues_matched', 'network_edges', 'network_size',
                         'full_state'])
        rank = 0
        matched = 0
        for row in states:
            rank += 1
            state_str = row['state_normalized']
            G = parse_state_graph(state_str)
            components = list(nx.connected_components(G))
            for comp in components:
                overlap = comp & resi_set
                if not overlap:
                    continue
                sub = G.subgraph(comp)
                writer.writerow([
                    rank,
                    row['count'],
                    row['hb_count'],
                    ' '.join(sorted(overlap)),
                    format_edge_list(sub.edges()),
                    len(sub.edges()),
                    state_str,
                ])
                matched += 1

    print(f'States processed:    {rank}')
    print(f'Networks found:      {matched}')
    print(f'Output saved to:     {out_file}')

if args.resi_start_stop:
    entry_set, exit_set = read_resi_start_stop(args.resi_start_stop)
    print(f'Entry residues ({len(entry_set)}): {", ".join(sorted(entry_set))}')
    print(f'Exit residues  ({len(exit_set)}): {", ".join(sorted(exit_set))}')
    print()

    use_shortest = args.max_path_length is None
    if use_shortest:
        print('Mode: shortest paths only (use -max_path_length N for all paths up to N edges)')
    else:
        print(f'Mode: all simple paths up to {args.max_path_length} edges')
    print()

    network_counts = Counter()
    pairwise_counts = defaultdict(int)

    for row in states:
        state_str = row['state_normalized']
        count = int(row['count'])
        G = parse_state_graph(state_str)

        for a, b in G.edges():
            edge_key = (min(a, b), max(a, b))
            pairwise_counts[edge_key] += count

        entries_in_graph = entry_set & set(G.nodes())
        exits_in_graph = exit_set & set(G.nodes())
        if not entries_in_graph or not exits_in_graph:
            continue

        for e_start in sorted(entries_in_graph):
            for e_end in sorted(exits_in_graph):
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
                    network_str = ' -> '.join(path)
                    network_counts[network_str] += count

    top_networks = network_counts.most_common(args.topnets)
    A = args.A

    out_txt = args.input_csv.replace('.csv', '_networks_start_stop.txt')
    out_csv = args.input_csv.replace('.csv', '_networks_start_stop.csv')

    with open(out_txt, 'w') as out_file, \
         open(out_csv, 'w', newline='') as csv_file:

        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['network_rank', 'count', 'percentage',
                             'rate_k_per_sec', 'energy_kcal_mol',
                             'network', 'path_length',
                             'pw_percentages', 'uncorr_percentage'])

        print_and_write(out_file,
                        f'Total microstates: {total_microstates}')
        print_and_write(out_file,
                        f'Total unique states: {len(states)}')
        print_and_write(out_file,
                        f'Unique networks found: {len(network_counts)}')
        print_and_write(out_file,
                        f'Arrhenius pre-exponential factor A = {A:.2e} /sec')
        print_and_write(out_file,
                        f'Top {min(args.topnets, len(top_networks))} networks:\n')
        print_and_write(out_file, '=' * 120)

        for idx, (network_str, count) in enumerate(top_networks, start=1):
            nodes = network_str.split(' -> ')
            ratio = count / total_microstates
            if ratio >= 1.0:
                E = 0.0
                k = A
            else:
                E = -1.364 * math.log10(ratio)
                k = A * 10 ** (-E / 1.364)

            print_and_write(out_file, '')
            print_and_write(out_file,
                            f'Network {idx}: {count} microstates '
                            f'({ratio:.2%}) share this network.')
            print_and_write(out_file,
                            f'Network Rate & Energy: k = {k:.2e} /sec, '
                            f'E = {E:.2e} kcal/mol')
            print_and_write(out_file, 'Network structure:')
            print_and_write(out_file, network_str)

            pw_percents = []
            pw_ratios = []
            for i in range(len(nodes) - 1):
                edge_key = (min(nodes[i], nodes[i + 1]),
                            max(nodes[i], nodes[i + 1]))
                edge_count = pairwise_counts.get(edge_key, 0)
                pw_ratio = edge_count / total_microstates
                pw_percents.append(f'{pw_ratio:.2%}')
                pw_ratios.append(pw_ratio)

            uncorr = 1.0
            for r in pw_ratios:
                uncorr *= r
            uncorr_pct = uncorr * 100

            pw_line = 'PW %:' + ' ' * (len(nodes[0]) - 4)
            for i in range(len(nodes) - 1):
                arrow_and_node = f' -> {nodes[i + 1]}'
                pw_str = pw_percents[i]
                padding = len(arrow_and_node) - len(pw_str)
                pw_line += ' ' * max(padding, 1) + pw_str
            print_and_write(out_file, pw_line)

            print_and_write(out_file,
                            f'Uncorr %: {uncorr_pct:.2e}% '
                            f'(PW is just the pairwise interaction %)')

            csv_writer.writerow([
                idx, count, f'{ratio:.6f}',
                f'{k:.2e}', f'{E:.2e}',
                network_str, len(nodes) - 1,
                ';'.join(pw_percents), f'{uncorr_pct:.2e}%',
            ])

            print_and_write(out_file, '-' * 120)

    print()
    print(f'Statistics saved to:  {out_txt}')
    print(f'CSV saved to:         {out_csv}')
