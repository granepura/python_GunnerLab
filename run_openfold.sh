#!/usr/bin/env bash

set -uo pipefail

# Resolve paths relative to this script, so it can be launched from anywhere.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
script_started_epoch="$(date +%s)"
script_started_utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

usage() {
    cat <<EOF
Usage: $(basename -- "$0") [OPTIONS] CSV_FILE [JSON_DIR [OUTPUT_DIR [PDB_DIR]]]
       $(basename -- "$0") --fold-only [OPTIONS] JSON_DIR [OUTPUT_DIR [PDB_DIR]]

Convert the sequence CSV into one JSON file per row, then run run_of3.sh once
for every JSON file in JSON_DIR. Each run gets its own output directory, and
generated replica PDBs are copied into PDB_DIR using names such as
seq01/seq01_rep1.pdb.

Options:
  -j, --json-only      Generate JSON files and stop; do not run folding
  -f, --fold-only      Run folding from existing JSON files; do not read CSV
  -s, --sequence NAME  Process one sequence (such as Seq05); repeatable
      --skip-existing  Skip sequences with all expected replica PDBs
      --force          Rerun selected sequences even if marked complete
      --retry N        Retry a failed folding command up to N times
      --json-dir DIR    Override the JSON input/output directory
      --results-dir DIR Override the full folding-results directory
      --pdb-dir DIR     Override the collected replica-PDB directory
  -n, --dry-run        Print planned actions without changing files
  -p, --jobs N         Run up to N folds concurrently (default: 1)
  -h, --help           Show this help message and exit

With neither --json-only nor --fold-only, both steps are performed.

Arguments:
  CSV_FILE    Input sequence CSV; required unless --fold-only is used
  JSON_DIR    Directory containing sequence JSON files
              (default: derived from CSV_FILE by replacing "sequences.csv"
              with "jsons")
  OUTPUT_DIR  Root directory for full run outputs
              (default: DATASET_OpenFold_results beside the input)
  PDB_DIR     Root directory for copied replica PDB files
              (default: DATASET_OpenFold_pdbs beside the input)

For example, cd1_round2_20_sequences.csv derives cd1_round2_20_jsons,
cd1_round2_OpenFold_results, and cd1_round2_OpenFold_pdbs.

Environment:
  RUN_OF3     run_of3.sh executable or path to it (default: run_of3.sh)
  EXPECTED_REPLICAS  Count used by --skip-existing (default: 5)

Logs are saved in OUTPUT_DIR/logs and progress in OUTPUT_DIR/progress.tsv.
Run metadata and a final summary are also saved in OUTPUT_DIR. Successful
sequences receive a verified completion marker in their PDB directory.
Existing replica PDBs with the same destination names are replaced.
EOF
}

mode="both"
skip_existing=0
force=0
retries=0
dry_run=0
jobs=1
selections=()
json_dir_option=""
results_dir_option=""
pdb_dir_option=""
positional=()
while (($# > 0)); do
    case "$1" in
        -j|--json-only)
            if [[ "$mode" == "fold-only" ]]; then
                echo "Error: --json-only and --fold-only cannot be used together." >&2
                exit 2
            fi
            mode="json-only"
            ;;
        -f|--fold-only)
            if [[ "$mode" == "json-only" ]]; then
                echo "Error: --json-only and --fold-only cannot be used together." >&2
                exit 2
            fi
            mode="fold-only"
            ;;
        -s|--sequence)
            (($# >= 2)) || { echo "Error: $1 requires a value" >&2; exit 2; }
            selections+=("$2")
            shift
            ;;
        --sequence=*) selections+=("${1#*=}") ;;
        --skip-existing) skip_existing=1 ;;
        --force) force=1 ;;
        --retry)
            (($# >= 2)) || { echo "Error: $1 requires a value" >&2; exit 2; }
            retries="$2"
            shift
            ;;
        --retry=*) retries="${1#*=}" ;;
        --json-dir|--results-dir|--pdb-dir)
            (($# >= 2)) || { echo "Error: $1 requires a directory" >&2; exit 2; }
            case "$1" in
                --json-dir) json_dir_option="$2" ;;
                --results-dir) results_dir_option="$2" ;;
                --pdb-dir) pdb_dir_option="$2" ;;
            esac
            shift
            ;;
        --json-dir=*) json_dir_option="${1#*=}" ;;
        --results-dir=*) results_dir_option="${1#*=}" ;;
        --pdb-dir=*) pdb_dir_option="${1#*=}" ;;
        -n|--dry-run) dry_run=1 ;;
        -p|--jobs)
            (($# >= 2)) || { echo "Error: $1 requires a value" >&2; exit 2; }
            jobs="$2"
            shift
            ;;
        --jobs=*) jobs="${1#*=}" ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            positional+=("$@")
            break
            ;;
        -*)
            echo "Error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            positional+=("$1")
            ;;
    esac
    shift
done
set -- "${positional[@]}"

if [[ "$mode" == "fold-only" ]]; then
    if (($# == 0)) && [[ -z "$json_dir_option" ]]; then
        echo "Error: JSON_DIR is required with --fold-only." >&2
        usage >&2
        exit 2
    fi
    max_fold_args=3
    [[ -z "$json_dir_option" ]] || max_fold_args=2
    if (($# > max_fold_args)); then
        echo "Error: --fold-only accepts at most three positional arguments." >&2
        usage >&2
        exit 2
    fi
    csv_file=""
    if [[ -n "$json_dir_option" ]]; then
        json_dir="$json_dir_option"
        fold_output_arg="${1:-}"
        fold_pdb_arg="${2:-}"
    else
        json_dir="$1"
        fold_output_arg="${2:-}"
        fold_pdb_arg="${3:-}"
    fi
    json_name="$(basename -- "$json_dir")"
    json_directory="$(dirname -- "$json_dir")"
    if [[ "$json_name" =~ ^(.+)_[0-9]+_jsons$ ]]; then
        dataset_name="${BASH_REMATCH[1]}"
    elif [[ "$json_name" == *_jsons ]]; then
        dataset_name="${json_name%_jsons}"
    else
        dataset_name="$json_name"
    fi
    output_root="${results_dir_option:-${fold_output_arg:-${json_directory}/${dataset_name}_OpenFold_results}}"
    pdb_root="${pdb_dir_option:-${fold_pdb_arg:-${json_directory}/${dataset_name}_OpenFold_pdbs}}"
else
    if (($# == 0)); then
        echo "Error: CSV_FILE is required unless --fold-only is used." >&2
        usage >&2
        exit 2
    fi
    if (($# > 4)); then
        echo "Error: too many positional arguments." >&2
        usage >&2
        exit 2
    fi
    csv_file="$1"
    csv_name="$(basename -- "$csv_file" .csv)"
    csv_directory="$(dirname -- "$csv_file")"
    if [[ "$csv_name" =~ ^(.+)_[0-9]+_sequences$ ]]; then
        dataset_name="${BASH_REMATCH[1]}"
    elif [[ "$csv_name" == *_sequences ]]; then
        dataset_name="${csv_name%_sequences}"
    else
        dataset_name="$csv_name"
    fi
    if [[ "$csv_name" == *_sequences ]]; then
        json_name="${csv_name%_sequences}_jsons"
    else
        json_name="${csv_name}_jsons"
    fi
    json_dir="${json_dir_option:-${2:-${csv_directory}/${json_name}}}"
    output_root="${results_dir_option:-${3:-${csv_directory}/${dataset_name}_OpenFold_results}}"
    pdb_root="${pdb_dir_option:-${4:-${csv_directory}/${dataset_name}_OpenFold_pdbs}}"
fi
runner="${RUN_OF3:-run_of3.sh}"
expected_replicas="${EXPECTED_REPLICAS:-5}"

[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { echo "Error: --jobs must be a positive integer." >&2; exit 2; }
[[ "$expected_replicas" =~ ^[1-9][0-9]*$ ]] || { echo "Error: EXPECTED_REPLICAS must be a positive integer." >&2; exit 2; }
[[ "$retries" =~ ^[0-9]+$ ]] || { echo "Error: --retry must be a nonnegative integer." >&2; exit 2; }
((force == 0 || skip_existing == 0)) || { echo "Error: --force and --skip-existing cannot be combined." >&2; exit 2; }

for index in "${!selections[@]}"; do
    value="${selections[$index]}"
    if [[ "$value" =~ ^([Ss][Ee][Qq])?0*([0-9]+)$ ]]; then
        selections[$index]="$(printf 'Seq%02d' "$((10#${BASH_REMATCH[2]}))")"
    else
        echo "Error: invalid sequence '$value' (expected Seq05 or 5)." >&2
        exit 2
    fi
done

if [[ "$mode" != "fold-only" ]]; then
    if [[ ! -f "$csv_file" ]]; then
        echo "Error: CSV file not found: $csv_file" >&2
        exit 1
    fi

    if ! command -v python3 >/dev/null 2>&1; then
        echo "Error: python3 is required to convert the CSV into JSON files." >&2
        exit 127
    fi

    if ((dry_run)); then
        echo "[dry-run] Generate JSON files from $csv_file in $json_dir"
    else
    mkdir -p "$json_dir"
    echo "Generating JSON files from $csv_file"
    if ! python3 - "$csv_file" "$json_dir" "${selections[@]}" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
json_dir = Path(sys.argv[2])
selected = set(sys.argv[3:])
generated = 0
generated_files = []
seen = set()
allowed = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")

with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
    reader = csv.DictReader(source)
    if reader.fieldnames != ["Seq#", "Sequence"]:
        raise SystemExit(
            f"Expected CSV columns 'Seq#,Sequence'; found {reader.fieldnames!r}"
        )

    for line_number, row in enumerate(reader, start=2):
        name = row["Seq#"].strip()
        sequence = "".join(row["Sequence"].split()).upper()
        match = re.fullmatch(r"Seq(\d+)", name, flags=re.IGNORECASE)
        if not match:
            raise SystemExit(f"Invalid sequence name on CSV line {line_number}: {name!r}")
        if not sequence:
            raise SystemExit(f"Empty protein sequence on CSV line {line_number}")

        number = int(match.group(1))
        file_stem = f"Seq{number:02d}"
        if file_stem in seen:
            raise SystemExit(f"Duplicate sequence name on CSV line {line_number}: {name!r}")
        seen.add(file_stem)
        if selected and file_stem not in selected:
            continue
        invalid = sorted(set(sequence) - allowed)
        if invalid:
            raise SystemExit(
                f"Invalid protein character(s) on CSV line {line_number}: {''.join(invalid)}"
            )
        query_name = f"Seq{number}"
        output_path = json_dir / f"{file_stem}.json"
        document = {
            "queries": {
                query_name: {
                    "chains": [
                        {
                            "molecule_type": "protein",
                            "chain_ids": ["A"],
                            "sequence": sequence,
                        }
                    ]
                }
            }
        }
        temporary_path = output_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(output_path)
        generated_files.append(output_path.name)
        generated += 1

if generated == 0:
    raise SystemExit("The CSV contains no sequence rows")

missing = selected - seen
if missing:
    raise SystemExit(f"Selected sequence(s) not found in CSV: {', '.join(sorted(missing))}")

manifest = json_dir / ".openfold_manifest"
temporary_manifest = manifest.with_suffix(".tmp")
temporary_manifest.write_text("\n".join(generated_files) + "\n", encoding="utf-8")
temporary_manifest.replace(manifest)

print(f"Generated {generated} JSON files in {json_dir}")
PY
    then
        echo "Error: failed to generate JSON files from $csv_file" >&2
        exit 1
    fi
    fi
fi

if [[ "$mode" == "json-only" ]]; then
    json_only_elapsed=$(($(date +%s) - script_started_epoch))
    printf '\nJSON generation complete — total runtime %02d:%02d:%02d\n' \
        "$((json_only_elapsed / 3600))" "$(((json_only_elapsed % 3600) / 60))" "$((json_only_elapsed % 60))"
    exit 0
fi

if ((dry_run == 0)) && ! command -v "$runner" >/dev/null 2>&1 && [[ ! -x "$runner" ]]; then
    echo "Error: cannot find executable '$runner'." >&2
    echo "Activate the environment that provides run_of3.sh, or set RUN_OF3=/path/to/run_of3.sh." >&2
    exit 127
fi

shopt -s nullglob
json_files=()
if ((dry_run)) && [[ "$mode" != "fold-only" ]]; then
    mapfile -t planned_names < <(python3 - "$csv_file" "${selections[@]}" <<'PY'
import csv, re, sys
selected = set(sys.argv[2:])
found = set()
with open(sys.argv[1], encoding="utf-8-sig", newline="") as source:
    reader = csv.DictReader(source)
    if reader.fieldnames != ["Seq#", "Sequence"]:
        raise SystemExit("CSV must have Seq#,Sequence columns")
    for line, row in enumerate(reader, 2):
        match = re.fullmatch(r"Seq(\d+)", row["Seq#"].strip(), re.I)
        if not match:
            raise SystemExit(f"Invalid sequence name on CSV line {line}")
        name = f"Seq{int(match.group(1)):02d}"
        if name in found:
            raise SystemExit(f"Duplicate sequence name: {name}")
        found.add(name)
        if not selected or name in selected:
            print(name)
missing = selected - found
if missing:
    raise SystemExit(f"Selected sequence(s) not found: {', '.join(sorted(missing))}")
PY
    )
    for sequence in "${planned_names[@]}"; do
        json_files+=("$json_dir/$sequence.json")
    done
elif [[ "$mode" == "both" && -f "$json_dir/.openfold_manifest" ]]; then
    while IFS= read -r filename; do
        [[ -n "$filename" ]] && json_files+=("$json_dir/$filename")
    done < "$json_dir/.openfold_manifest"
else
    json_files=("$json_dir"/*.json)
fi

if ((${#selections[@]} > 0)); then
    selected_files=()
    for json_file in "${json_files[@]}"; do
        sequence="$(basename -- "$json_file" .json)"
        for selected in "${selections[@]}"; do
            [[ "$sequence" == "$selected" ]] && selected_files+=("$json_file")
        done
    done
    json_files=("${selected_files[@]}")
fi

if ((${#json_files[@]} == 0)); then
    echo "Error: no JSON files found in $json_dir" >&2
    exit 1
fi

if ((dry_run)); then
    for json_file in "${json_files[@]}"; do
        sequence="$(basename -- "$json_file" .json)"
        destination="$pdb_root/${sequence,,}"
        if ((skip_existing)) && [[ -f "$destination/.openfold_complete" ]]; then
            count="$(find "$destination" -maxdepth 1 -type f -name "${sequence,,}_rep*.pdb" -size +0c | wc -l)"
            if ((count >= expected_replicas)); then
                echo "[dry-run][$sequence] Skip (verified completion marker and $count replicas)"
                continue
            fi
        fi
        printf '[dry-run][%s] ' "$sequence"
        printf '%q ' "$runner" --json "$json_file" --output-dir "$output_root/$sequence"
        printf '\n[dry-run][%s] Copy replicas to %s\n' "$sequence" "$destination"
    done
    exit 0
fi

mkdir -p "$output_root/logs" "$output_root/.openfold-status" "$pdb_root"
status_dir="$output_root/.openfold-status"
progress_file="$output_root/progress.tsv"
summary_file="$output_root/summary.tsv"
metadata_file="$output_root/run_metadata.txt"
counter_file="$status_dir/.completed_count"

for directory in "$json_dir" "$output_root" "$pdb_root"; do
    [[ -d "$directory" && -w "$directory" ]] || {
        echo "Error: directory is not writable: $directory" >&2
        exit 1
    }
done

if command -v flock >/dev/null 2>&1; then
    exec 9>"$output_root/.run_openfold.lock"
    flock -n 9 || { echo "Error: another run is using $output_root" >&2; exit 1; }
fi

printf '0\n' > "$counter_file"
run_started="$script_started_utc"
batch_started_epoch="$script_started_epoch"
csv_checksum="not-applicable"
[[ -z "$csv_file" ]] || csv_checksum="$(sha256sum "$csv_file" | cut -d' ' -f1)"
{
    printf 'started_utc=%s\n' "$run_started"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'csv_file=%s\n' "${csv_file:-not-applicable}"
    printf 'csv_sha256=%s\n' "$csv_checksum"
    printf 'json_dir=%s\nresults_dir=%s\npdb_dir=%s\n' "$json_dir" "$output_root" "$pdb_root"
    printf 'runner=%s\njobs=%s\nretries=%s\nexpected_replicas=%s\n' "$runner" "$jobs" "$retries" "$expected_replicas"
    if command -v nvidia-smi >/dev/null 2>&1; then
        printf 'gpus=%s\n' "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd ';' -)"
    else
        printf 'gpus=not-detected\n'
    fi
    df -Pk "$output_root" | awk 'NR==2 {print "free_disk_kb=" $4}'
} > "$metadata_file"

is_complete() {
    local sequence="$1" destination="$2" replica
    [[ -f "$destination/.openfold_complete" ]] || return 1
    for ((replica=1; replica<=expected_replicas; replica++)); do
        [[ -s "$destination/${sequence,,}_rep${replica}.pdb" ]] || return 1
    done
}

show_progress() {
    local sequence="$1" state="$2" completed total width filled empty bar
    total="${#json_files[@]}"
    if command -v flock >/dev/null 2>&1; then
        exec 8<>"$counter_file"
        flock 8
        read -r completed <&8
        ((completed+=1))
        printf '%d\n' "$completed" > "$counter_file"
        flock -u 8
        exec 8>&-
    else
        completed="$(<"$counter_file")"
        ((completed+=1))
        printf '%d\n' "$completed" > "$counter_file"
    fi
    width=30
    filled=$((completed * width / total))
    empty=$((width - filled))
    printf -v bar '%*s' "$filled" ''
    bar="${bar// /#}"
    printf -v padding '%*s' "$empty" ''
    printf '[%s%s] %d/%d %s: %s\n' "$bar" "$padding" "$completed" "$total" "$sequence" "$state"
}

write_status() {
    printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${3:-0}" "${4:-}" \
        > "$status_dir/$1.status"
}

format_duration() {
    local total="$1"
    printf '%02d:%02d:%02d' "$((total / 3600))" "$(((total % 3600) / 60))" "$((total % 60))"
}

fold_banner() {
    local sequence="$1" message="$2"
    printf '\n======================================================================\n'
    printf '  %-10s %s\n' "[$sequence]" "$message"
    printf '======================================================================\n\n'
}

run_one() {
    local json_file="$1" sequence output_dir destination count status attempt
    local started_epoch elapsed duration
    sequence="$(basename -- "$json_file" .json)"
    output_dir="$output_root/$sequence"
    destination="$pdb_root/${sequence,,}"
    started_epoch="$(date +%s)"

    if ((skip_existing)) && is_complete "$sequence" "$destination"; then
        fold_banner "$sequence" "SKIPPED"
        echo "Reason: completion marker and $expected_replicas replicas verified"
        write_status "$sequence" skipped 0 "verified complete"
        return 0
    fi

    rm -f -- "$destination/.openfold_complete"
    write_status "$sequence" running 0 "folding"
    fold_banner "$sequence" "STARTING FOLD"
    echo "Input:   $json_file"
    echo "Output:  $output_dir"
    echo "Log:     $output_root/logs/$sequence.log"
    echo
    status=1
    for ((attempt=1; attempt<=retries+1; attempt++)); do
        if "$runner" --json "$json_file" --output-dir "$output_dir"; then
            status=0
            break
        else
            status=$?
            if ((attempt <= retries)); then
                echo "[$sequence] Attempt $attempt failed (exit $status); retrying..." >&2
            fi
        fi
    done

    if ((status == 0)); then
        mkdir -p "$destination"

        mapfile -d '' pdb_files < <(find "$output_dir" -type f \
            -name '*_sample_*_model.pdb' -print0 | sort -z)

        if ((${#pdb_files[@]} == 0)); then
            elapsed=$(($(date +%s) - started_epoch))
            echo "[$sequence] Failed: no replica PDB files found" >&2
            write_status "$sequence" failed "$elapsed" "no replica PDB files"
            return 1
        fi

        copy_failed=0
        copied=0
        declare -A seen_replicas=()
        for pdb_file in "${pdb_files[@]}"; do
            filename="$(basename -- "$pdb_file")"
            if [[ "$filename" =~ _sample_([0-9]+)_model\.pdb$ ]]; then
                replica="${BASH_REMATCH[1]}"
                if [[ -n "${seen_replicas[$replica]:-}" ]]; then
                    echo "[$sequence] Duplicate model found for replica $replica" >&2
                    copy_failed=1
                    continue
                fi
                seen_replicas[$replica]=1
                if ! cp -- "$pdb_file" "$destination/${sequence,,}_rep${replica}.pdb"; then
                    copy_failed=1
                else
                    ((copied+=1))
                fi
            fi
        done

        if ((copied != expected_replicas)); then
            echo "[$sequence] Expected $expected_replicas replicas but found $copied" >&2
            copy_failed=1
        fi

        if ((copy_failed)); then
            elapsed=$(($(date +%s) - started_epoch))
            echo "[$sequence] Failed while copying replica PDB files" >&2
            write_status "$sequence" failed "$elapsed" "copy failed"
            return 1
        else
            elapsed=$(($(date +%s) - started_epoch))
            duration="$(format_duration "$elapsed")"
            printf 'sequence=%s\ncompleted_utc=%s\nreplicas=%s\n' \
                "$sequence" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$copied" \
                > "$destination/.openfold_complete"
            fold_banner "$sequence" "COMPLETED in $duration"
            echo "Replicas: $copied"
            echo "Copied to: $destination"
            echo
            write_status "$sequence" success "$elapsed" "$copied replicas copied"
        fi
    else
        elapsed=$(($(date +%s) - started_epoch))
        duration="$(format_duration "$elapsed")"
        fold_banner "$sequence" "FAILED after $duration"
        echo "[$sequence] Failed (exit code $status)" >&2
        write_status "$sequence" failed "$elapsed" "runner exit $status"
        return 1
    fi
}

worker() {
    local json_file="$1" sequence result state
    sequence="$(basename -- "$json_file" .json)"
    run_one "$json_file" 2>&1 | tee "$output_root/logs/$sequence.log"
    result=${PIPESTATUS[0]}
    state="$(cut -f2 "$status_dir/$sequence.status" 2>/dev/null || printf unknown)"
    show_progress "$sequence" "$state"
    return "$result"
}

interrupted=0
on_interrupt() {
    interrupted=1
    echo "Interrupted; stopping active folds..." >&2
    jobs -pr | xargs -r kill 2>/dev/null || true
}
trap on_interrupt INT TERM

pids=()
sequences=()
for json_file in "${json_files[@]}"; do
    sequence="$(basename -- "$json_file" .json)"
    worker "$json_file" &
    pids+=("$!")
    sequences+=("$sequence")
    while (($(jobs -rp | wc -l) >= jobs)); do
        wait -n || true
        ((interrupted == 0)) || break 2
    done
done

for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done

if ((interrupted)); then
    for sequence in "${sequences[@]}"; do
        if [[ -f "$status_dir/$sequence.status" ]] && grep -q $'\trunning\t' "$status_dir/$sequence.status"; then
            write_status "$sequence" interrupted 0 "user interruption"
        fi
    done
fi

{
    printf 'sequence\tstate\tupdated_utc\tduration_seconds\tdetail\n'
    find "$status_dir" -maxdepth 1 -type f -name '*.status' -print0 | sort -z | xargs -0 -r cat
} > "$progress_file.tmp"
mv -- "$progress_file.tmp" "$progress_file"

((interrupted == 0)) || exit 130

failed=()
successful=0
skipped=0
failed_count=0
batch_elapsed=$(($(date +%s) - batch_started_epoch))
batch_duration="$(format_duration "$batch_elapsed")"
{
    printf 'sequence\tstate\tduration_seconds\tduration\tdetail\n'
for sequence in "${sequences[@]}"; do
    state="$(cut -f2 "$status_dir/$sequence.status" 2>/dev/null || printf unknown)"
    elapsed="$(cut -f4 "$status_dir/$sequence.status" 2>/dev/null || printf 0)"
    detail="$(cut -f5- "$status_dir/$sequence.status" 2>/dev/null || true)"
    printf '%s\t%s\t%s\t%s\t%s\n' "$sequence" "$state" "$elapsed" "$(format_duration "$elapsed")" "$detail"
    case "$state" in
        success) ((successful+=1)) ;;
        skipped) ((skipped+=1)) ;;
        *) failed+=("$sequence:$state"); ((failed_count+=1)) ;;
    esac
done
    printf 'TOTAL\tcomplete\t%d\t%s\tsuccess=%d skipped=%d failed=%d selected=%d\n' \
        "$batch_elapsed" "$batch_duration" "$successful" "$skipped" "$failed_count" "${#sequences[@]}"
} > "$summary_file"

{
    printf 'finished_utc=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'total_runtime_seconds=%d\ntotal_runtime=%s\n' "$batch_elapsed" "$batch_duration"
    printf 'successful=%d\nskipped=%d\nfailed=%d\n' "$successful" "$skipped" "$failed_count"
} >> "$metadata_file"

if ((${#failed[@]} > 0)); then
    printf '\n======================================================================\n' >&2
    printf '  BATCH FINISHED WITH FAILURES — total runtime %s\n' "$batch_duration" >&2
    printf '======================================================================\n\n' >&2
    echo "Failed runs: ${failed[*]}" >&2
    echo "See $output_root/logs, $progress_file, and $summary_file" >&2
    exit 1
fi

sequence_word="sequences"
((${#json_files[@]} == 1)) && sequence_word="sequence"
printf '\n======================================================================\n'
printf '  BATCH COMPLETE — %d %s — total runtime %s\n' "${#json_files[@]}" "$sequence_word" "$batch_duration"
printf '======================================================================\n\n'
echo "Progress: $progress_file"
echo "Summary: $summary_file"
