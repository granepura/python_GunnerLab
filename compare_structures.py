#!/usr/bin/env python3
"""Compare protein structures using sequence-matched C-alpha atoms."""
import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np
from Bio.Align import PairwiseAligner
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB import MMCIFParser, PDBParser, Superimposer

EXTENSIONS = {'.pdb', '.cif', '.mmcif'}


def read_chain(path, chain_id=None):
    parser = PDBParser(QUIET=True) if path.suffix.lower() == '.pdb' else MMCIFParser(QUIET=True)
    model = next(parser.get_structure(path.stem, str(path)).get_models())
    chains = {}
    for chain in model:
        residues = [r for r in chain if r.resname in protein_letters_3to1_extended and 'CA' in r]
        if residues:
            chains[chain.id] = residues
    if chain_id is None:
        if len(chains) != 1:
            raise ValueError(f'{path}: select a chain explicitly; protein chains: {list(chains)}')
        chain_id = next(iter(chains))
    if chain_id not in chains:
        raise ValueError(f'{path}: no C-alpha protein residues in chain {chain_id!r}')
    residues = chains[chain_id]
    sequence = ''.join(protein_letters_3to1_extended[r.resname] for r in residues)
    return chain_id, residues, sequence


def residue_label(residue):
    return f'{residue.id[1]}{residue.id[2].strip()}'


def compare(left, right, chain1=None, chain2=None, cutoff=8.0):
    c1, r1, s1 = read_chain(left, chain1)
    c2, r2, s2 = read_chain(right, chain2)
    aligner = PairwiseAligner(mode='global', match_score=2, mismatch_score=-1,
                             open_gap_score=-10, extend_gap_score=-0.5)
    alignment = aligner.align(s1, s2)[0]
    pairs = [(i, j) for a, b in zip(*alignment.aligned)
             for i, j in zip(range(*a), range(*b))]
    if len(pairs) < 3:
        raise ValueError('At least three aligned C-alpha atoms are required')
    atoms1 = [r1[i]['CA'] for i, j in pairs]
    atoms2 = [r2[j]['CA'] for i, j in pairs]
    x = np.array([a.coord for a in atoms1], dtype=float)
    y = np.array([a.coord for a in atoms2], dtype=float)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError('Non-finite coordinates')
    if min(np.linalg.matrix_rank(x - x.mean(0)), np.linalg.matrix_rank(y - y.mean(0))) < 2:
        raise ValueError('Aligned coordinates are collinear; superposition is underdetermined')
    fit = Superimposer()
    fit.set_atoms(atoms1, atoms2)
    rotation, translation = fit.rotran
    distances = np.linalg.norm(x - (y @ rotation + translation), axis=1)
    # Calculate contact sets in blocks to avoid an N x N x 3 allocation.
    shared = union = contacts1 = contacts2 = 0
    for k, (i, j) in enumerate(pairs):
        eligible = np.array([abs(i - ii) > 2 and abs(j - jj) > 2
                             for ii, jj in pairs[k + 1:]], dtype=bool)
        a = (np.linalg.norm(x[k + 1:] - x[k], axis=1) < cutoff) & eligible
        b = (np.linalg.norm(y[k + 1:] - y[k], axis=1) < cutoff) & eligible
        shared += int((a & b).sum())
        union += int((a | b).sum())
        contacts1 += int(a.sum())
        contacts2 += int(b.sum())
    summary = dict(file1=str(left), file2=str(right), chain1=c1, chain2=c2,
                   residues1=len(r1), residues2=len(r2), aligned_residues=len(pairs),
                   identity_pct=100 * sum(s1[i] == s2[j] for i, j in pairs) / len(pairs),
                   coverage1_pct=100 * len(pairs) / len(r1), coverage2_pct=100 * len(pairs) / len(r2),
                   ca_rmsd_angstrom=float(fit.rms),
                   median_displacement_angstrom=float(np.median(distances)),
                   max_displacement_angstrom=float(distances.max()),
                   contacts1=contacts1, contacts2=contacts2, shared_contacts=shared,
                   contact_jaccard=shared / union if union else '')
    details = [dict(file1=str(left), file2=str(right), chain1=c1, chain2=c2,
                    residue1=residue_label(r1[i]), residue2=residue_label(r2[j]),
                    amino_acid1=s1[i], amino_acid2=s2[j], displacement_angstrom=float(d))
               for (i, j), d in zip(pairs, distances)]
    return summary, details


def files(directory):
    result = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            if path.stem in result:
                raise ValueError(f'Duplicate filename stem: {path.stem!r} in {directory}')
            result[path.stem] = path
    if not result:
        raise ValueError(f'No PDB/CIF files in {directory}')
    return result


def get_pairs(left, right, pairing):
    if left.is_file() and right.is_file():
        if any(p.suffix.lower() not in EXTENSIONS for p in (left, right)):
            raise ValueError('Files must have .pdb, .cif, or .mmcif extensions')
        return [(left, right)]
    if not (left.is_dir() and right.is_dir()):
        raise ValueError('Supply two existing files or two existing directories')
    a, b = files(left), files(right)
    if pairing == 'all':
        return itertools.product(a.values(), b.values())
    common = sorted(a.keys() & b.keys())
    for name in sorted(a.keys() ^ b.keys()):
        print(f'Warning: unmatched filename stem skipped: {name}', file=sys.stderr)
    if not common:
        raise ValueError('No matching filename stems; consider --pairing all')
    return [(a[name], b[name]) for name in common]


def write_csv(stream, rows):
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input1', type=Path)
    parser.add_argument('input2', type=Path)
    parser.add_argument('--pairing', choices=['name', 'all'], default='name')
    parser.add_argument('--chain1', help='Chain in input1 (required for multi-chain structures)')
    parser.add_argument('--chain2', help='Chain in input2 (required for multi-chain structures)')
    parser.add_argument('--contact-cutoff', type=float, default=8.0, metavar='ANGSTROM')
    parser.add_argument('-o', '--output', type=Path, default=Path('structure_compare_results.csv'),
                        help='Summary CSV; default: structure_compare_results.csv')
    parser.add_argument('--per-residue', type=Path, help='Optional residue displacement CSV')
    args = parser.parse_args()
    if not np.isfinite(args.contact_cutoff) or args.contact_cutoff <= 0:
        parser.error('--contact-cutoff must be positive and finite')
    outputs = [p.resolve() for p in (args.output, args.per_residue) if p]
    if len(outputs) != len(set(outputs)):
        parser.error('Summary and per-residue outputs must differ')
    try:
        pairs = list(get_pairs(args.input1, args.input2, args.pairing))
        if set(outputs) & {p.resolve() for pair in pairs for p in pair}:
            raise ValueError('Output would overwrite an input structure')
        summaries, details = [], []
        for left, right in pairs:
            try:
                summary, residues = compare(left, right, args.chain1, args.chain2, args.contact_cutoff)
            except Exception as exc:
                raise ValueError(f'{left} vs {right}: {exc}') from exc
            summaries.append(summary)
            details.extend(residues)
        if args.output:
            with args.output.open('w', newline='') as stream:
                write_csv(stream, summaries)
        else:
            write_csv(sys.stdout, summaries)
        if args.per_residue:
            with args.per_residue.open('w', newline='') as stream:
                write_csv(stream, details)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
