import argparse
import csv
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

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
parser.add_argument('-net_length', type=int, nargs=2, default=[3, 10],
                    metavar=('MIN', 'MAX'),
                    help='Minimum and maximum network path length in nodes '
                         '(default: 3 10)')
parser.add_argument('-topnets', type=int, default=10, metavar='N',
                    help='Number of top networks to display (default: 10)')
parser.add_argument('--include_bk', action='store_true',
                    help='Include backbone (BK) H-bond connections (excluded by default)')
parser.add_argument('--confs', action='store_true',
                    help='Build networks at the conformer level instead of residue level '
                         '(only effective with raw MCCE hb_states input)')
parser.add_argument('-A', type=float, default=1e13,
                    help='Arrhenius pre-exponential factor in /sec (default: 1e13)')
args = parser.parse_args()

if not args.resi_list and not args.resi_start_stop:
    parser.error('at least one of -resi_list or -resi_start_stop is required')

RE_PAIRS = re.compile(r'\(([^,]+),([^)]+)\)')
RE_CONFORMER_NUM = re.compile(r'_\d+')
RE_RESIDUE_CORE = re.compile(r'([A-Z]{3}).*?([A-Z]\d{4})')


def is_backbone(name):
    stripped = RE_CONFORMER_NUM.sub('', name)
    return len(stripped) >= 5 and stripped[3:5] == 'BK'


def normalize_residue(name):
    name = RE_CONFORMER_NUM.sub('', name)
    m = RE_RESIDUE_CORE.search(name)
    return m.group(1) + m.group(2) if m else name


def normalize_state(state_id, skip_bk=False, keep_confs=False):
    pairs = set()
    for m in RE_PAIRS.finditer(state_id):
        a_raw, b_raw = m.group(1).strip(), m.group(2).strip()
        if skip_bk and (is_backbone(a_raw) or is_backbone(b_raw)):
            continue
        if keep_confs:
            a, b = sorted([a_raw, b_raw])
        else:
            a, b = sorted([normalize_residue(a_raw), normalize_residue(b_raw)])
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


def parse_resi_for_pymol(name):
    stripped = RE_CONFORMER_NUM.sub('', name)
    m = RE_RESIDUE_CORE.search(stripped)
    if m:
        return m.group(1), m.group(2)[0], int(m.group(2)[1:])
    return None


def read_hah_file(path):
    """Read hah (H-bond atom-atom) file into a lookup keyed by (donor_conf, acceptor_conf)."""
    hah = {}
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.split()
            if len(parts) < 8:
                continue
            donor_conf = parts[0]
            acceptor_conf = parts[1]
            hb_atoms = parts[2]
            dist = float(parts[3])
            angle = float(parts[4])
            xyz_str = parts[5]
            d_occ = float(parts[6])
            a_occ = float(parts[7])
            key = (donor_conf, acceptor_conf)
            entry = {'hb_atoms': hb_atoms, 'dist': dist, 'angle': angle,
                     'xyz': xyz_str, 'd_occ': d_occ, 'a_occ': a_occ}
            if key not in hah or (d_occ * a_occ) > (hah[key]['d_occ'] * hah[key]['a_occ']):
                hah[key] = entry
    return hah


def find_best_hah_entry(hah, node_a, node_b, keep_confs=False):
    """Find the best hah entry for a pair of network nodes.

    If keep_confs is True, node names are conformer-level (e.g. SER01A0085_002)
    and we do an exact lookup. Otherwise, nodes are residue-level (e.g. SERA0085)
    and we search all conformer pairs matching these two residues, picking the
    one with highest occupancy product.
    """
    if keep_confs:
        for key in [(node_a, node_b), (node_b, node_a)]:
            if key in hah:
                donor, acceptor = key
                e = hah[key]
                return donor, acceptor, e
        return None

    resi_a = normalize_residue(node_a) if RE_RESIDUE_CORE.search(node_a) else node_a
    resi_b = normalize_residue(node_b) if RE_RESIDUE_CORE.search(node_b) else node_b

    best = None
    for (donor, acceptor), entry in hah.items():
        d_resi = normalize_residue(donor)
        a_resi = normalize_residue(acceptor)
        if {d_resi, a_resi} == {resi_a, resi_b}:
            score = entry['d_occ'] * entry['a_occ']
            if best is None or score > best[3]:
                best = (donor, acceptor, entry, score)
    if best:
        return best[0], best[1], best[2]
    return None


def extract_conformer_pdb_lines(step2_path, conformer_ids):
    """Extract PDB lines from step2_out.pdb for a set of MCCE conformer IDs.

    A conformer_id like 'SER01A0085_002' maps to lines where:
      columns 17-20 (resname) = 'SER'
      columns 21-30 (chain+resnum_conf) matches ' A0085_002'
    Also includes the backbone (_000) atoms for each residue.
    """
    needed = set()
    for cid in conformer_ids:
        stripped = RE_CONFORMER_NUM.sub('', cid)
        m = RE_RESIDUE_CORE.search(stripped)
        if m:
            chain = m.group(2)[0]
            resnum = m.group(2)[1:]
            conf_m = re.search(r'_(\d+)$', cid)
            conf_num = conf_m.group(1) if conf_m else None
            needed.add((chain, resnum, conf_num))
            needed.add((chain, resnum, '000'))

    lines = []
    with open(step2_path) as f:
        for line in f:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            pdb_chain = line[21].strip()
            pdb_resnum = line[22:26].strip()
            pdb_conf = line[27:30].strip()
            for chain, resnum, conf_num in needed:
                if pdb_chain == chain and pdb_resnum == resnum:
                    if conf_num is None or pdb_conf == conf_num:
                        lines.append(line)
                        break
    return lines


def extract_residue_pdb_lines(step2_path, residue_names):
    """Extract PDB lines from step2_out.pdb for residue-level names (e.g. SERA0085).

    Includes all conformers of each residue.
    """
    needed = set()
    for name in residue_names:
        m = RE_RESIDUE_CORE.search(name)
        if m:
            needed.add((m.group(2)[0], m.group(2)[1:], m.group(1)))

    lines = []
    with open(step2_path) as f:
        for line in f:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            pdb_chain = line[21].strip()
            pdb_resnum = line[22:26].strip()
            pdb_resname = line[17:20].strip()
            for chain, resnum, resname in needed:
                if pdb_chain == chain and pdb_resnum == resnum and pdb_resname == resname:
                    lines.append(line)
                    break
    return lines


def print_and_write(fh, text):
    print(text)
    fh.write(text + '\n')


def read_input_csv(path, skip_bk=False, keep_confs=False):
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
            level = 'conformer' if keep_confs else 'residue'
            bk_status = 'including BK' if not skip_bk else 'excluding BK'
            print(f'  Format: raw MCCE hb_states. Normalizing to {level} level ({bk_status}) ...')
            for line in f:
                parts = line.rsplit(',', 3)
                count = int(parts[-2])
                rows.append({
                    'state_normalized': normalize_state(parts[0].strip('"'),
                                                        skip_bk=skip_bk,
                                                        keep_confs=keep_confs),
                    'count': str(count),
                })
        else:
            print(f'  Format: aggregated residue-level CSV')
            if keep_confs:
                print(f'  Note: --confs has no effect on aggregated CSV (conformer info already stripped)')
            remaining = f.read()
            for r in csv.DictReader([header_line + '\n'] + remaining.splitlines(True)):
                rows.append(r)
    return rows


print(f'Reading {args.input_csv} ...')
skip_bk = not args.include_bk
states = read_input_csv(args.input_csv, skip_bk=skip_bk, keep_confs=args.confs)
total_microstates = sum(int(row['count']) for row in states)
print(f'  Unique states: {len(states):,}')
print(f'  Total microstates: {total_microstates:,}')

suffix = '_nets_confs' if args.confs else '_nets'
out_dir = Path(args.input_csv.replace('.csv', suffix))
out_dir.mkdir(exist_ok=True)
print(f'  Output directory: {out_dir}')

if args.resi_list:
    resi_set = read_resi_list(args.resi_list)
    print(f'\n--- resi_list mode ---')
    print(f'Residues of interest ({len(resi_set)}): {", ".join(sorted(resi_set))}')

    out_path = str(out_dir / 'networks_resi_list.csv')
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
                if len(comp) < args.net_length[0]:
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

    net_min, net_max = args.net_length
    print(f'Network path length: {net_min} to {net_max} nodes')

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
                paths = nx.all_simple_paths(G, e_start, e_end,
                                            cutoff=net_max - 1)
                for path in paths:
                    if len(path) < net_min:
                        continue
                    network_counts[' -> '.join(path)] += count
                    found_any = True
        if found_any:
            states_with_paths += 1

    print(f'  States with at least one path: {states_with_paths:,}')
    print(f'  Unique networks found: {len(network_counts):,}')

    top_networks = network_counts.most_common(args.topnets)
    A = args.A
    out_txt = str(out_dir / 'networks_start_stop.txt')
    out_csv = str(out_dir / 'networks_start_stop.csv')

    print(f'Writing results ...')

    with open(out_txt, 'w') as fh_txt, \
         open(out_csv, 'w', newline='') as fh_csv:

        csv_writer = csv.writer(fh_csv)
        csv_writer.writerow(['network_rank', 'count', 'percentage',
                             'rate_k_per_sec', 'energy_kcal_mol',
                             'network', 'path_length',
                             'subnet_percentages', 'pair_percentages',
                             'uncorr_percentage'])

        print_and_write(fh_txt, f'Total microstates: {total_microstates:,}')
        print_and_write(fh_txt, f'Total unique states: {len(states):,}')
        print_and_write(fh_txt, f'States with entry->exit paths: {states_with_paths:,}')
        print_and_write(fh_txt, f'Unique networks found: {len(network_counts):,}')
        print_and_write(fh_txt, f'Arrhenius pre-exponential factor A = {A:.2e} /sec')
        print_and_write(fh_txt, f'Network path length: {net_min} to {net_max} nodes')
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

            pair_line = 'Pair %:' + ' ' * max(1, len(nodes[0]) - 6)
            for i in range(len(nodes) - 1):
                pw_str = pw_percents[i]
                padding = len(f' -> {nodes[i + 1]}') - len(pw_str)
                pair_line += ' ' * max(padding, 1) + pw_str
            print_and_write(fh_txt, pair_line)

            print_and_write(fh_txt,
                            f'Uncorr %: {uncorr_pct:.2e}% '
                            f'(product of individual pair %)')

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

    for idx, (network_str, net_count) in enumerate(top_networks, 1):
        cyto_path = str(out_dir / f'Network{idx}_cytoscape.txt')
        nodes = network_str.split(' -> ')
        net_pct = net_count / total_microstates
        with open(cyto_path, 'w') as cyto:
            cyto.write('donor\tacceptor\tnetwork\tnet_count\tnet_pct\t'
                        'pair_count\tpair_pct\n')
            for i in range(len(nodes) - 1):
                a, b = nodes[i], nodes[i + 1]
                edge_key = (min(a, b), max(a, b))
                pair_count = pairwise_counts.get(edge_key, 0)
                pair_pct = pair_count / total_microstates
                cyto.write(f'{a}\t{b}\tNetwork{idx}\t{net_count}\t'
                           f'{net_pct:.6f}\t{pair_count}\t{pair_pct:.6f}\n')
    print(f'Cytoscape edge lists: {out_dir}/Network*_cytoscape.txt')

    step2_path = Path('step2_out.pdb')
    if not step2_path.exists():
        print(f'\n  Note: step2_out.pdb not found in {Path.cwd()}, skipping PyMOL visualization')
    else:
        prot_path = Path('prot_center.pdb') if Path('prot_center.pdb').exists() else Path('prot.pdb')
        if not prot_path.exists():
            print(f'\n  Note: neither prot_center.pdb nor prot.pdb found, skipping PyMOL visualization')
        else:
            hah_candidates = sorted(Path('.').glob('hah_*.txt'))
            hah = {}
            if hah_candidates:
                hah_path = hah_candidates[0]
                print(f'\nReading H-bond atoms from {hah_path} ...')
                hah = read_hah_file(str(hah_path))
                print(f'  {len(hah):,} unique conformer H-bond pairs loaded')
            else:
                print(f'\n  Note: no hah_*.txt file found, measurements will use closest-atom distance')

            print(f'Generating PyMOL sessions for top {len(top_networks)} networks ...')
            print(f'  Protein: {prot_path}')
            print(f'  Conformers: {step2_path}')
            pymol_ok = True

            for idx, (network_str, count) in enumerate(top_networks, 1):
                nodes = network_str.split(' -> ')
                pml_path = out_dir / f'Network{idx}.pml'
                pse_path = out_dir / f'Network{idx}.pse'
                net_pdb_path = out_dir / f'Network{idx}_conformers.pdb'

                conf_ids = set()
                hb_entries = []
                for i in range(len(nodes) - 1):
                    result = find_best_hah_entry(hah, nodes[i], nodes[i + 1],
                                                  keep_confs=args.confs) if hah else None
                    if result:
                        donor, acceptor, entry = result
                        conf_ids.add(donor)
                        conf_ids.add(acceptor)
                        hb_entries.append((donor, acceptor, entry))
                    else:
                        hb_entries.append(None)

                if args.confs:
                    pdb_lines = extract_conformer_pdb_lines(str(step2_path), conf_ids if conf_ids else set(nodes))
                else:
                    if conf_ids:
                        pdb_lines = extract_conformer_pdb_lines(str(step2_path), conf_ids)
                    else:
                        pdb_lines = extract_residue_pdb_lines(str(step2_path), nodes)

                with open(net_pdb_path, 'w') as pdb_out:
                    pdb_out.writelines(pdb_lines)
                    pdb_out.write('END\n')

                with open(pml_path, 'w') as pml:
                    pml.write(f'load {prot_path.resolve()}, protein\n')
                    pml.write('bg_color white\n')
                    pml.write('show cartoon, protein\n')
                    pml.write('color gray80, protein\n')
                    pml.write('set cartoon_transparency, 0.7, protein\n\n')

                    net_obj = f'Network{idx}'
                    pml.write(f'load {net_pdb_path.resolve()}, {net_obj}\n')
                    pml.write(f'show sticks, {net_obj}\n')
                    pml.write(f'util.cbay {net_obj}\n')
                    pml.write(f'show spheres, {net_obj} and resn HOH+WAT\n')
                    pml.write(f'set sphere_scale, 0.3, {net_obj} and resn HOH+WAT\n')
                    pml.write(f'label {net_obj} and name CA+O and not resn HOH+WAT, '
                              f'"%s %s%s" % (resn, chain, resi)\n')
                    pml.write(f'label {net_obj} and name O and resn HOH+WAT, '
                              f'"%s %s%s" % (resn, chain, resi)\n')
                    pml.write(f'set label_size, 12\n\n')

                    for i in range(len(nodes) - 1):
                        entry = hb_entries[i]
                        hb_label = f'hb{idx}_{i+1}'
                        if entry:
                            donor, acceptor, e = entry
                            hb_atoms_str = e['hb_atoms']
                            xyz_str = e['xyz']
                            xyz_parts = xyz_str.split(';')
                            c1 = [float(x) for x in xyz_parts[0].strip('()').split(',')]
                            c2 = [float(x) for x in xyz_parts[1].strip('()').split(',')]

                            donor_h = hb_atoms_str.split('~')[1].split('...')[0]
                            acceptor_heavy = hb_atoms_str.split('...')[1]

                            pml.write(f'# {nodes[i]} -- {nodes[i+1]}: '
                                      f'{donor}({donor_h}) -> {acceptor}({acceptor_heavy}) '
                                      f'{e["dist"]:.2f} A\n')
                            pml.write(f'pseudoatom pt_{hb_label}_d, '
                                      f'pos=[{c1[0]:.3f}, {c1[1]:.3f}, {c1[2]:.3f}]\n')
                            pml.write(f'pseudoatom pt_{hb_label}_a, '
                                      f'pos=[{c2[0]:.3f}, {c2[1]:.3f}, {c2[2]:.3f}]\n')
                            pml.write(f'distance {hb_label}, pt_{hb_label}_d, pt_{hb_label}_a\n')
                            pml.write(f'hide everything, pt_{hb_label}_d or pt_{hb_label}_a\n')
                        else:
                            a_info = parse_resi_for_pymol(nodes[i])
                            b_info = parse_resi_for_pymol(nodes[i + 1])
                            if a_info and b_info:
                                pml.write(f'# {nodes[i]} -- {nodes[i+1]}: closest atom distance\n')
                                pml.write(f'distance {hb_label}, '
                                          f'{net_obj} and chain {a_info[1]} and resi {a_info[2]}, '
                                          f'{net_obj} and chain {b_info[1]} and resi {b_info[2]}\n')
                        pml.write('\n')

                    pml.write(f'set dash_color, cyan, hb{idx}_*\n')
                    pml.write(f'set dash_gap, 0.3, hb{idx}_*\n')
                    pml.write(f'set dash_length, 0.15, hb{idx}_*\n')
                    pml.write(f'set label_color, black, hb{idx}_*\n')
                    pml.write(f'set dash_width, 3.0, hb{idx}_*\n\n')

                    pml.write(f'zoom {net_obj}, 8\n')
                    pml.write('deselect\n')
                    pml.write(f'save {pse_path.resolve()}\n')
                    pml.write('quit\n')

                if pymol_ok:
                    try:
                        subprocess.run(['pymol', '-cq', str(pml_path)],
                                       stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, timeout=120)
                        print(f'  Network {idx}: {pse_path}')
                    except FileNotFoundError:
                        pymol_ok = False
                        print(f'  PyMOL not found — .pml scripts saved, run manually to generate .pse')
                    except subprocess.TimeoutExpired:
                        print(f'  Network {idx}: PyMOL timed out, .pml script saved: {pml_path}')

            print(f'PyMOL scripts/sessions saved in: {out_dir}')
