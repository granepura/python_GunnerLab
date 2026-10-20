#!/usr/bin/env python3
"""Compare protein structures using sequence-matched C-alpha atoms.

Signed delta columns use structure2 minus structure1. B-factor columns also
represent pLDDT when a structure predictor stores confidence in that field.
"""
import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from Bio.Align import PairwiseAligner
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB import MMCIFParser, NeighborSearch, PDBParser, ShrakeRupley, Superimposer
from Bio.PDB.vectors import calc_dihedral
from Bio.SVDSuperimposer import SVDSuperimposer

EXTENSIONS = {'.pdb', '.cif', '.mmcif'}

# Maximum solvent-accessible areas (A^2) from Tien et al. (2013). These make
# exposure comparable across residue types instead of using one absolute area.
MAX_RESIDUE_SASA = {
    'A': 129, 'R': 274, 'N': 195, 'D': 193, 'C': 167,
    'Q': 225, 'E': 223, 'G': 104, 'H': 224, 'I': 197,
    'L': 201, 'K': 236, 'M': 224, 'F': 240, 'P': 159,
    'S': 155, 'T': 172, 'W': 285, 'Y': 263, 'V': 174,
}


def read_chain(path, chain_id=None):
    parser = PDBParser(QUIET=True) if path.suffix.lower() == '.pdb' else MMCIFParser(QUIET=True)
    model = next(parser.get_structure(path.stem, str(path)).get_models())
    ShrakeRupley().compute(model, level='R')
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
    complex_sasa = {residue: float(residue.sasa) for residue in residues}
    ShrakeRupley().compute(model[chain_id], level='R')
    for residue in residues:
        residue.isolated_sasa = float(residue.sasa)
        residue.complex_sasa = complex_sasa[residue]
        residue.sasa = complex_sasa[residue]
    sequence = ''.join(protein_letters_3to1_extended[r.resname] for r in residues)
    return chain_id, residues, sequence


def residue_label(residue):
    return f'{residue.id[1]}{residue.id[2].strip()}'


def radius_of_gyration(coordinates):
    """Return the unweighted C-alpha radius of gyration in angstroms."""
    centered = coordinates - coordinates.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))


def relative_sasa(residue, amino_acid):
    """Return residue SASA as a percentage of its amino-acid maximum."""
    maximum = MAX_RESIDUE_SASA.get(amino_acid)
    return 100 * residue.sasa / maximum if maximum else np.nan


def tm_score(distances, length):
    """Calculate a TM-style score from the sequence-aligned superposition."""
    d0 = max(0.5, 1.24 * np.cbrt(max(length - 15, 1)) - 1.8)
    return float(np.sum(1 / (1 + (distances / d0) ** 2)) / length)


def secondary_structure(residues):
    """Assign coarse H/E/C states using backbone phi/psi angles."""
    states = ['C'] * len(residues)
    for i in range(1, len(residues) - 1):
        previous, residue, following = residues[i - 1:i + 2]
        if not all(name in obj for obj, name in ((previous, 'C'), (residue, 'N'),
                                                  (residue, 'CA'), (residue, 'C'),
                                                  (following, 'N'))):
            continue
        phi = np.degrees(calc_dihedral(previous['C'].get_vector(), residue['N'].get_vector(),
                                       residue['CA'].get_vector(), residue['C'].get_vector()))
        psi = np.degrees(calc_dihedral(residue['N'].get_vector(), residue['CA'].get_vector(),
                                       residue['C'].get_vector(), following['N'].get_vector()))
        if -100 <= phi <= -30 and -90 <= psi <= 10:
            states[i] = 'H'
        elif -180 <= phi <= -60 and (50 <= psi <= 180 or -180 <= psi <= -120):
            states[i] = 'E'
    return states


def interaction_counts(residues):
    """Return simple geometry-based clash, H-bond, and salt-bridge counts."""
    positions = {residue: i for i, residue in enumerate(residues)}
    heavy_atoms = [atom for residue in residues for atom in residue
                   if atom.element != 'H' and not atom.is_disordered()]
    neighbors = NeighborSearch(heavy_atoms)
    clashes = set()
    for atom1, atom2 in neighbors.search_all(2.0, level='A'):
        res1, res2 = atom1.get_parent(), atom2.get_parent()
        if abs(positions[res1] - positions[res2]) > 1:
            clashes.add((min(id(atom1), id(atom2)), max(id(atom1), id(atom2))))

    donors = [atom for residue in residues for atom in residue
              if atom.element == 'N']
    acceptors = [atom for residue in residues for atom in residue
                 if atom.element == 'O']
    hydrogen_bonds = set()
    for donor in donors:
        for acceptor in neighbors.search(donor.coord, 3.5, level='A'):
            if acceptor.element != 'O':
                continue
            res1, res2 = donor.get_parent(), acceptor.get_parent()
            if abs(positions[res1] - positions[res2]) > 1:
                hydrogen_bonds.add((positions[res1], positions[res2]))

    acidic = {'ASP': {'OD1', 'OD2'}, 'GLU': {'OE1', 'OE2'}}
    basic = {'LYS': {'NZ'}, 'ARG': {'NE', 'NH1', 'NH2'}, 'HIS': {'ND1', 'NE2'}}
    salt_bridges = set()
    acidic_atoms = [(i, atom) for i, residue in enumerate(residues)
                    for atom in residue if atom.name in acidic.get(residue.resname, set())]
    for i, residue in enumerate(residues):
        for atom in residue:
            if atom.name not in basic.get(residue.resname, set()):
                continue
            for acidic_index, acidic_atom in acidic_atoms:
                if i != acidic_index and np.linalg.norm(atom.coord - acidic_atom.coord) <= 4.0:
                    salt_bridges.add(tuple(sorted((i, acidic_index))))
    return len(clashes), len(hydrogen_bonds), len(salt_bridges)


def changed_regions(pairs, residues1, residues2, distances, threshold=2.0):
    """Describe contiguous aligned regions over a displacement threshold."""
    regions, start = [], None
    for k, distance in enumerate(distances):
        if distance >= threshold and start is None:
            start = k
        if start is not None and (distance < threshold or k == len(distances) - 1):
            end = k if distance >= threshold and k == len(distances) - 1 else k - 1
            i1, j1 = pairs[start]
            i2, j2 = pairs[end]
            regions.append(f'{residue_label(residues1[i1])}-{residue_label(residues1[i2])}'
                           f'/{residue_label(residues2[j1])}-{residue_label(residues2[j2])}')
            start = None
    return ';'.join(regions)


def load_pae(path):
    """Load an AlphaFold/ColabFold predicted-aligned-error JSON matrix."""
    data = json.loads(path.read_text())
    if isinstance(data, list) and data and isinstance(data[0], dict):
        data = data[0]
    if isinstance(data, dict):
        data = data.get('predicted_aligned_error', data.get('pae'))
    matrix = np.asarray(data, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError(f'{path}: expected a finite square PAE matrix')
    return matrix


def domain_rmsds(domain_text, pairs, x, y):
    """Calculate RMSDs for 1-based aligned-sequence domain ranges."""
    if not domain_text:
        return ''
    values = []
    for item in domain_text.split(','):
        try:
            start, end = (int(value) for value in item.split('-', 1))
        except ValueError as exc:
            raise ValueError(f'Invalid domain range {item!r}; use START-END') from exc
        selected = [k for k, (i, _j) in enumerate(pairs) if start <= i + 1 <= end]
        if len(selected) < 3:
            raise ValueError(f'Domain {item!r} needs at least three aligned residues')
        fit = SVDSuperimposer()
        fit.set(x[selected], y[selected])
        fit.run()
        values.append(f'{item}:{fit.get_rms():.3f}')
    return ';'.join(values)


def compare(left, right, chain1=None, chain2=None, cutoff=8.0,
            surface_threshold=25.0, pae1=None, pae2=None, domains=None):
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
    transformed_y = y @ rotation + translation
    distances = np.linalg.norm(x - transformed_y, axis=1)
    # Calculate contact sets in blocks to avoid an N x N x 3 allocation.
    shared = union = contacts1 = contacts2 = 0
    contact_degree1 = np.zeros(len(pairs), dtype=int)
    contact_degree2 = np.zeros(len(pairs), dtype=int)
    gained_degree = np.zeros(len(pairs), dtype=int)
    lost_degree = np.zeros(len(pairs), dtype=int)
    gained_contacts = lost_contacts = 0
    gained_contact_pairs, lost_contact_pairs = [], []
    distance_delta_abs_sum = distance_delta_squared_sum = 0.0
    max_distance_delta = 0.0
    distance_pair_count = 0
    lddt_totals = [0, 0]
    lddt_matches = [0, 0]
    for k, (i, j) in enumerate(pairs):
        eligible = np.array([abs(i - ii) > 2 and abs(j - jj) > 2
                             for ii, jj in pairs[k + 1:]], dtype=bool)
        distances1 = np.linalg.norm(x[k + 1:] - x[k], axis=1)
        distances2 = np.linalg.norm(y[k + 1:] - y[k], axis=1)
        pair_deltas = np.abs(distances2 - distances1)
        for reference, reference_distances in enumerate((distances1, distances2)):
            local = reference_distances < 15.0
            lddt_totals[reference] += int(local.sum()) * 4
            lddt_matches[reference] += sum(int(((pair_deltas < threshold) & local).sum())
                                             for threshold in (0.5, 1.0, 2.0, 4.0))
        a = (distances1 < cutoff) & eligible
        b = (distances2 < cutoff) & eligible
        shared += int((a & b).sum())
        union += int((a | b).sum())
        contacts1 += int(a.sum())
        contacts2 += int(b.sum())
        gained = b & ~a
        lost = a & ~b
        gained_contacts += int(gained.sum())
        lost_contacts += int(lost.sum())
        for offset in np.flatnonzero(gained):
            other = k + 1 + int(offset)
            i2, j2 = pairs[other]
            gained_contact_pairs.append(
                f'{residue_label(r1[i])}-{residue_label(r1[i2])}/'
                f'{residue_label(r2[j])}-{residue_label(r2[j2])}')
        for offset in np.flatnonzero(lost):
            other = k + 1 + int(offset)
            i2, j2 = pairs[other]
            lost_contact_pairs.append(
                f'{residue_label(r1[i])}-{residue_label(r1[i2])}/'
                f'{residue_label(r2[j])}-{residue_label(r2[j2])}')
        contact_degree1[k] += int(a.sum())
        contact_degree2[k] += int(b.sum())
        contact_degree1[k + 1:] += a
        contact_degree2[k + 1:] += b
        gained_degree[k] += int(gained.sum())
        lost_degree[k] += int(lost.sum())
        gained_degree[k + 1:] += gained
        lost_degree[k + 1:] += lost
        distance_deltas = np.abs(distances2[eligible] - distances1[eligible])
        if len(distance_deltas):
            distance_delta_abs_sum += float(distance_deltas.sum())
            distance_delta_squared_sum += float(np.dot(distance_deltas, distance_deltas))
            max_distance_delta = max(max_distance_delta, float(distance_deltas.max()))
            distance_pair_count += len(distance_deltas)

    confidence1 = np.array([atom.bfactor for atom in atoms1], dtype=float)
    confidence2 = np.array([atom.bfactor for atom in atoms2], dtype=float)
    relative_sasa1 = np.array([relative_sasa(residue, amino_acid)
                               for residue, amino_acid in zip(r1, s1)])
    relative_sasa2 = np.array([relative_sasa(residue, amino_acid)
                               for residue, amino_acid in zip(r2, s2)])
    surface1 = relative_sasa1 >= surface_threshold
    surface2 = relative_sasa2 >= surface_threshold
    surface_count1 = int(surface1.sum())
    surface_count2 = int(surface2.sum())
    all_confidence1 = np.array([residue['CA'].bfactor for residue in r1], dtype=float)
    all_confidence2 = np.array([residue['CA'].bfactor for residue in r2], dtype=float)
    mean_confidence1 = float(confidence1.mean())
    mean_confidence2 = float(confidence2.mean())
    rg1, rg2 = radius_of_gyration(x), radius_of_gyration(y)
    ss1, ss2 = secondary_structure(r1), secondary_structure(r2)
    ss_matches = sum(ss1[i] == ss2[j] for i, j in pairs)
    clashes1, hbonds1, salts1 = interaction_counts(r1)
    clashes2, hbonds2, salts2 = interaction_counts(r2)
    contact_recall = shared / contacts1 if contacts1 else ''
    contact_precision = shared / contacts2 if contacts2 else ''
    contact_f1 = (2 * shared / (contacts1 + contacts2)
                  if contacts1 + contacts2 else '')
    pae_metrics = {}
    if pae1 or pae2:
        if not (pae1 and pae2):
            raise ValueError('Supply both --pae1 and --pae2')
        matrix1, matrix2 = load_pae(pae1), load_pae(pae2)
        if matrix1.shape != matrix2.shape:
            raise ValueError(f'PAE matrix shapes differ: {matrix1.shape} vs {matrix2.shape}')
        pae_metrics = dict(mean_pae_angstrom_s1=float(matrix1.mean()),
                           mean_pae_angstrom_s2=float(matrix2.mean()),
                           delta_mean_pae_angstrom=float(matrix2.mean() - matrix1.mean()),
                           mean_abs_pae_delta_angstrom=float(np.mean(np.abs(matrix2 - matrix1))),
                           max_abs_pae_delta_angstrom=float(np.max(np.abs(matrix2 - matrix1))))
    gdt = {threshold: 100 * float(np.mean(distances <= threshold))
           for threshold in (1, 2, 4, 8)}
    displacement_counts = {threshold: int(np.sum(distances >= threshold))
                           for threshold in (1, 2, 5, 10)}
    summary = dict(file_s1=str(left), file_s2=str(right),
                   name_s1=left.stem, name_s2=right.stem,
                   delta_convention='s2 - s1',
                   chain_s1=c1, chain_s2=c2,
                   residues_s1=len(r1), residues_s2=len(r2),
                   delta_residues=len(r2) - len(r1),
                   aligned_residues=len(pairs),
                   identity_pct=100 * sum(s1[i] == s2[j] for i, j in pairs) / len(pairs),
                   coverage_pct_s1=100 * len(pairs) / len(r1),
                   coverage_pct_s2=100 * len(pairs) / len(r2),
                   delta_coverage_pct=(100 * len(pairs) / len(r2)
                                       - 100 * len(pairs) / len(r1)),
                   mean_ca_bfactor_or_plddt_s1=mean_confidence1,
                   mean_ca_bfactor_or_plddt_s2=mean_confidence2,
                   delta_mean_ca_bfactor_or_plddt=(
                       mean_confidence2 - mean_confidence1),
                   median_ca_bfactor_or_plddt_s1=float(np.median(all_confidence1)),
                   median_ca_bfactor_or_plddt_s2=float(np.median(all_confidence2)),
                   min_ca_bfactor_or_plddt_s1=float(all_confidence1.min()),
                   min_ca_bfactor_or_plddt_s2=float(all_confidence2.min()),
                   residues_plddt_below_70_s1=int(np.sum(all_confidence1 < 70)),
                   residues_plddt_below_70_s2=int(np.sum(all_confidence2 < 70)),
                   residues_plddt_below_50_s1=int(np.sum(all_confidence1 < 50)),
                   residues_plddt_below_50_s2=int(np.sum(all_confidence2 < 50)),
                   radius_of_gyration_angstrom_s1=rg1,
                   radius_of_gyration_angstrom_s2=rg2,
                   delta_radius_of_gyration_angstrom=rg2 - rg1,
                   surface_threshold_pct=surface_threshold,
                   surface_residues_s1=surface_count1,
                   surface_residues_s2=surface_count2,
                   delta_surface_residues=surface_count2 - surface_count1,
                   surface_residues_pct_s1=100 * surface_count1 / len(r1),
                   surface_residues_pct_s2=100 * surface_count2 / len(r2),
                   delta_surface_residues_pct=(100 * surface_count2 / len(r2)
                                               - 100 * surface_count1 / len(r1)),
                   total_sasa_angstrom2_s1=float(sum(residue.sasa for residue in r1)),
                   total_sasa_angstrom2_s2=float(sum(residue.sasa for residue in r2)),
                   delta_total_sasa_angstrom2=float(sum(residue.sasa for residue in r2)
                                                     - sum(residue.sasa for residue in r1)),
                   interface_buried_sasa_angstrom2_s1=float(sum(
                       residue.isolated_sasa - residue.complex_sasa for residue in r1)),
                   interface_buried_sasa_angstrom2_s2=float(sum(
                       residue.isolated_sasa - residue.complex_sasa for residue in r2)),
                   ca_rmsd_angstrom=float(fit.rms),
                   tm_score_s1_ref=tm_score(distances, len(r1)),
                   tm_score_s2_ref=tm_score(distances, len(r2)),
                   ca_lddt_s1_ref=(lddt_matches[0] / lddt_totals[0]
                                   if lddt_totals[0] else ''),
                   ca_lddt_s2_ref=(lddt_matches[1] / lddt_totals[1]
                                   if lddt_totals[1] else ''),
                   gdt_1_pct=gdt[1], gdt_2_pct=gdt[2],
                   gdt_4_pct=gdt[4], gdt_8_pct=gdt[8],
                   gdt_ts_pct=float(np.mean(list(gdt.values()))),
                   median_displacement_angstrom=float(np.median(distances)),
                   max_displacement_angstrom=float(distances.max()),
                   residues_displaced_ge_1_angstrom=displacement_counts[1],
                   residues_displaced_ge_2_angstrom=displacement_counts[2],
                   residues_displaced_ge_5_angstrom=displacement_counts[5],
                   residues_displaced_ge_10_angstrom=displacement_counts[10],
                   changed_regions_ge_2_angstrom=changed_regions(
                       pairs, r1, r2, distances),
                   domain_rmsd_angstrom=domain_rmsds(
                       domains, pairs, x, y),
                   secondary_structure_method='phi_psi',
                   helix_residues_s1=ss1.count('H'), helix_residues_s2=ss2.count('H'),
                   sheet_residues_s1=ss1.count('E'), sheet_residues_s2=ss2.count('E'),
                   coil_residues_s1=ss1.count('C'), coil_residues_s2=ss2.count('C'),
                   secondary_structure_agreement_pct=100 * ss_matches / len(pairs),
                   contacts_s1=contacts1, contacts_s2=contacts2,
                   delta_contacts=contacts2 - contacts1,
                   gained_contacts=gained_contacts, lost_contacts=lost_contacts,
                   gained_contact_pairs=';'.join(gained_contact_pairs),
                   lost_contact_pairs=';'.join(lost_contact_pairs),
                   shared_contacts=shared,
                   contact_precision_s1_ref=contact_precision,
                   contact_recall_s1_ref=contact_recall,
                   contact_f1=contact_f1,
                   contact_jaccard=shared / union if union else '',
                   mean_abs_distance_map_delta_angstrom=(distance_delta_abs_sum / distance_pair_count
                                                          if distance_pair_count else ''),
                   rms_distance_map_delta_angstrom=(np.sqrt(distance_delta_squared_sum /
                                                             distance_pair_count)
                                                     if distance_pair_count else ''),
                   max_abs_distance_map_delta_angstrom=(max_distance_delta
                                                         if distance_pair_count else ''),
                   steric_clashes_s1=clashes1, steric_clashes_s2=clashes2,
                   delta_steric_clashes=clashes2 - clashes1,
                   geometry_hbonds_s1=hbonds1, geometry_hbonds_s2=hbonds2,
                   delta_geometry_hbonds=hbonds2 - hbonds1,
                   salt_bridges_s1=salts1, salt_bridges_s2=salts2,
                   delta_salt_bridges=salts2 - salts1,
                   **pae_metrics)
    details = [dict(file_s1=str(left), file_s2=str(right),
                    chain_s1=c1, chain_s2=c2,
                    residue_s1=residue_label(r1[i]),
                    residue_s2=residue_label(r2[j]),
                    amino_acid_s1=s1[i], amino_acid_s2=s2[j],
                    ca_bfactor_or_plddt_s1=float(confidence1[k]),
                    ca_bfactor_or_plddt_s2=float(confidence2[k]),
                    delta_ca_bfactor_or_plddt=float(
                        confidence2[k] - confidence1[k]),
                    relative_sasa_pct_s1=float(relative_sasa1[i]),
                    relative_sasa_pct_s2=float(relative_sasa2[j]),
                    delta_relative_sasa_pct=float(relative_sasa2[j]
                                                  - relative_sasa1[i]),
                    surface_exposed_s1=bool(surface1[i]),
                    surface_exposed_s2=bool(surface2[j]),
                    secondary_structure_s1=ss1[i],
                    secondary_structure_s2=ss2[j],
                    secondary_structure_match=ss1[i] == ss2[j],
                    local_contacts_s1=int(contact_degree1[k]),
                    local_contacts_s2=int(contact_degree2[k]),
                    delta_local_contacts=int(
                        contact_degree2[k] - contact_degree1[k]),
                    gained_contacts=int(gained_degree[k]),
                    lost_contacts=int(lost_degree[k]),
                    displacement_angstrom=float(d))
               for k, ((i, j), d) in enumerate(zip(pairs, distances))]
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


def display_value(value):
    if isinstance(value, float):
        return f'{value:.4f}'
    return str(value) if value != '' else 'N/A'


def console_report(summaries):
    """Return a compact, human-readable console summary."""
    blocks = []
    for number, row in enumerate(summaries, 1):
        lines = [f'Comparison {number}: {row["name_s1"]} (s1) vs {row["name_s2"]} (s2)',
                 f'  Inputs: {row["file_s1"]} | {row["file_s2"]}',
                 f'  Aligned: {row["aligned_residues"]} residues, '
                 f'{display_value(row["identity_pct"])}% identity',
                 f'  C-alpha RMSD: {display_value(row["ca_rmsd_angstrom"])} A',
                 f'  TM-score: {display_value(row["tm_score_s1_ref"])} (s1 ref), '
                 f'{display_value(row["tm_score_s2_ref"])} (s2 ref)',
                 f'  C-alpha lDDT: {display_value(row["ca_lddt_s1_ref"])} (s1 ref), '
                 f'{display_value(row["ca_lddt_s2_ref"])} (s2 ref)',
                 f'  GDT-TS: {display_value(row["gdt_ts_pct"])}%',
                 f'  Surface residues: {row["surface_residues_s1"]} (s1), '
                 f'{row["surface_residues_s2"]} (s2), delta {row["delta_surface_residues"]:+d}',
                 f'  Contacts: {row["contacts_s1"]} (s1), {row["contacts_s2"]} (s2), '
                 f'{row["gained_contacts"]} gained, {row["lost_contacts"]} lost',
                 f'  Secondary-structure agreement: '
                 f'{display_value(row["secondary_structure_agreement_pct"])}%',
                 f'  Confidence mean: {display_value(row["mean_ca_bfactor_or_plddt_s1"])} (s1), '
                 f'{display_value(row["mean_ca_bfactor_or_plddt_s2"])} (s2)']
        blocks.append('\n'.join(lines))
    return '\n\n'.join(blocks)


def detailed_report(args, summaries, summary_path, residue_path):
    """Return a complete text report, including inputs and settings."""
    lines = ['STRUCTURE COMPARISON REPORT', '', 'Inputs and settings',
             f'  input1: {args.input1}', f'  input2: {args.input2}',
             f'  chain1: {args.chain1 or "auto"}', f'  chain2: {args.chain2 or "auto"}',
             f'  pairing: {args.pairing}', f'  contact cutoff: {args.contact_cutoff} A',
             f'  surface threshold: {args.surface_threshold}%',
             f'  PAE 1: {args.pae1 or "not supplied"}',
             f'  PAE 2: {args.pae2 or "not supplied"}',
             f'  domains: {args.domains or "not supplied"}',
             f'  summary CSV: {summary_path}',
             f'  per-residue CSV: {residue_path or "not requested"}', '']
    for number, row in enumerate(summaries, 1):
        lines.extend([f'Comparison {number}', '-' * 40])
        lines.extend(f'  {key}: {display_value(value)}' for key, value in row.items())
        lines.append('')
    lines.extend(['Notes',
                  '  All delta values use s2 - s1.',
                  '  TM-scores use the sequence-aligned C-alpha superposition.',
                  '  Secondary structure uses coarse phi/psi assignments (H/E/C).',
                  '  Clashes, hydrogen bonds, and salt bridges are geometry-based counts.',
                  '  B-factor values represent pLDDT only when the predictor stores it there.'])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input1', type=Path)
    parser.add_argument('input2', type=Path)
    parser.add_argument('--pairing', choices=['name', 'all'], default='name')
    parser.add_argument('--chain1', help='Chain in input1 (required for multi-chain structures)')
    parser.add_argument('--chain2', help='Chain in input2 (required for multi-chain structures)')
    parser.add_argument('--contact-cutoff', type=float, default=8.0, metavar='ANGSTROM')
    parser.add_argument('--surface-threshold', type=float, default=25.0, metavar='PERCENT',
                        help='Minimum relative SASA for a surface residue; default: 25')
    parser.add_argument('--pae1', type=Path, help='Optional AlphaFold PAE JSON for input1')
    parser.add_argument('--pae2', type=Path, help='Optional AlphaFold PAE JSON for input2')
    parser.add_argument('--domains', help='Optional s1 residue ranges, e.g. 1-100,101-200')
    parser.add_argument('--output-dir', type=Path, default=Path('structure_compare_results'),
                        help='Directory for CSV and log outputs; default: structure_compare_results')
    parser.add_argument('-o', '--output', type=Path, default=Path('summary.csv'),
                        help='Summary CSV filename inside --output-dir; default: summary.csv')
    parser.add_argument('--per-residue', nargs='?', const=Path('per_residue.csv'), type=Path,
                        help='Write residue CSV (optional filename; default: per_residue.csv)')
    parser.add_argument('--log', type=Path, default=Path('comparison.log'),
                        help='Log filename inside --output-dir; default: comparison.log')
    parser.add_argument('--no-log', action='store_true', help='Do not write the default log file')
    args = parser.parse_args()
    if not np.isfinite(args.contact_cutoff) or args.contact_cutoff <= 0:
        parser.error('--contact-cutoff must be positive and finite')
    if not np.isfinite(args.surface_threshold) or not 0 <= args.surface_threshold <= 100:
        parser.error('--surface-threshold must be between 0 and 100')
    def output_path(path):
        return path if path.is_absolute() else args.output_dir / path

    summary_path = output_path(args.output)
    residue_path = output_path(args.per_residue) if args.per_residue else None
    log_path = None if args.no_log else output_path(args.log)
    outputs = [p.resolve() for p in (summary_path, residue_path, log_path) if p]
    if len(outputs) != len(set(outputs)):
        parser.error('Summary and per-residue outputs must differ')
    try:
        pairs = list(get_pairs(args.input1, args.input2, args.pairing))
        if (args.pae1 or args.pae2) and len(pairs) != 1:
            raise ValueError('PAE inputs can only be used for one structure pair')
        if set(outputs) & {p.resolve() for pair in pairs for p in pair}:
            raise ValueError('Output would overwrite an input structure')
        summaries, details = [], []
        for left, right in pairs:
            try:
                summary, residues = compare(left, right, args.chain1, args.chain2,
                                            args.contact_cutoff, args.surface_threshold,
                                            args.pae1, args.pae2, args.domains)
            except Exception as exc:
                raise ValueError(f'{left} vs {right}: {exc}') from exc
            summaries.append(summary)
            details.extend(residues)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open('w', newline='') as stream:
            write_csv(stream, summaries)
        if residue_path:
            residue_path.parent.mkdir(parents=True, exist_ok=True)
            with residue_path.open('w', newline='') as stream:
                write_csv(stream, details)
        report = console_report(summaries)
        print(report)
        print(f'\nSummary CSV: {summary_path}')
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(detailed_report(args, summaries, summary_path, residue_path))
            print(f'Log: {log_path}')
        if residue_path:
            print(f'Per-residue CSV: {residue_path}')
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
