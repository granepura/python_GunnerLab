#!/bin/bash

# Parameter/Options for SLURM (Simple Linux Utility for Resource Management)
#SBATCH --job-name=stepM
#SBATCH --output=stepM.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4         # Adjust number of cores if needed
#SBATCH --mem=12G                 # Adjust memory if needed
#SBATCH --time=24:00:00
#SBATCH --export=ALL

# =============================================================================
# Script Name  : stepM.sh
# Purpose      : Automate extraction and MCCE preprocessing for membrane protein systems with specified chain subsets.
# Description  :
#   - Reads a `master_chains.txt` file to identify the PDB ID and relevant chain selections
#   - Extracts specified protein-only (PROT) and protein+membrane (PROT_MEM) chains from a PDB structure
#   - Prepares and organizes input files for MCCE simulation in a new directory (PROT_MEM/)
#   - Executes MCCE Step 1 and Step 2 with user-defined parameters for membrane generation
#   - Tracks timing for each step and generates detailed logs
#   - Validates output presence and alerts on failures or missing files
#
# Input Files  :
#   - master_chains.txt (format: PDBID PROT_CHAIN_IDS PROT_MEM_CHAIN_IDS)
#    - A 3-column file with PDBID, protein chain IDs of interest (PROT), and protein chain IDs of interest of membrane generation (PROT_MEM)
#   - <PDBID>.pdb files in the current working directory
#   - Checks for successful output files and logs any warnings or failures
#
# Output Files :
#   - prot.pdb (protein-only chains)
#   - PROT_MEM/prot_mem.pdb (protein + membrane chains)
#   - step1.log, step2.log (MCCE logs)
#   - step1_out.pdb, step2_out.pdb (MCCE outputs)
#   - mcce_timing.log (runtime summary)
#
# Notes:
#   - Only the first matching PDB file is used
#   - The script is SLURM-compatible for HPC environments
#   - Chain extraction uses grep and positional filtering
#
# Author       : Gehan A. Ranepura
# Created      : July 2025
# =============================================================================

#=============================================================================
#-----------------------------------------------------------------------------
# Input and Output:
# A 3-column file with PDBID, protein chain IDs of interest (PROT), and protein chain IDs of interest of membrane generation (PROT_MEM)
master_chains="/path/to/master_chains.txt"                                    # Replace with actual file path   

# Set MCCE4 Parameters
MCCE_HOME="/path/to/MCCE4"
USER_PARAM="./user_param"
EXTRA="./user_param/extra.tpl"

# MCCE Simulation
STEP1="step1.py \$input_prot_mem -d 4 --dry"
STEP2="step2.py -l 1 -d 4"

# Set IPECE Membrane Parameters
MEM="-u IPECE_ADD_MEM=t,IPECE_MEM_THICKNESS=33,IPECE_MEM_CHAINID=M"

#------------------------------------------------------------------------------
#==============================================================================

# Clean old file if exists
rm -f "prot.pdb" "chains.txt"

# Process each entry in master_chains.txt starting from line 2
printf "%-6s %-10s %-10s\n" "PDBID" "PROT" "PROT_MEM" > chains.txt
while read -r pdbid prot prot_mem; do
    for variant in "${pdbid,,}" "${pdbid^^}"; do
        pdb_file="${variant}.pdb"
        if [[ -f "$pdb_file" ]]; then
            printf "%-6s %-10s %-10s\n" "$pdbid" "$prot" "$prot_mem" >> chains.txt
            output=$(orientation.py "$pdb_file")
            master_pdb=$(echo "$output" | grep "Structure moved to origin" | awk '{print $NF}')
            if [[ ! -f "$master_pdb" ]]; then
               echo "Error: Centered PDB file '$master_pdb' was not created. Exiting."
               exit 1
            else
               echo "Running centered PDB file: $master_pdb"
            fi
            break  # Only exit the for-loop, continue with next line
        fi
    done
done < <(tail -n +2 "$master_chains")


# Parse chain IDs from chains.txt (excluding header)
read -r pdbid prot_chains prot_mem_chains < <(tail -n 1 chains.txt)

echo "Parsed from chains.txt:"
echo "  PDBID          : $pdbid"
echo "  PROT chains    : $prot_chains"
echo "  PROT_MEM chains: $prot_mem_chains"
echo

# Create or recreate the PROT_MEM directory (literal name)
rm -rf PROT_MEM
mkdir -p PROT_MEM
echo "Created directory: PROT_MEM"

# Extract PROT chains → prot.pdb
echo "Extracting PROT chains into:     prot.pdb ..."
grep -E "^ATOM|^HETATM" "$master_pdb" \
  | grep -E "^.{21}[$prot_chains]" \
  | grep -v " OXT" > prot.pdb

# Extract PROT_MEM chains → PROT_MEM/prot_mem.pdb
echo "Extracting PROT_MEM chains into: PROT_MEM/prot_mem.pdb ..."
grep -E "^ATOM|^HETATM" "$master_pdb" \
  | grep -E "^.{21}[$prot_mem_chains]" \
  | grep -v " OXT" > PROT_MEM/prot_mem.pdb

echo
echo "✅ Extraction complete for $pdbid"
echo "  - prot.pdb:              Chains [$prot_chains]"
echo "  - PROT_MEM/prot_mem.pdb: Chains [$prot_mem_chains]"
echo

#-----------------------------------------------------------------------------------------
#=========================================================================================
# Prepare MCCE simulation in PROT_MEM directory
cd PROT_MEM
input_prot_mem="prot_mem.pdb"
if [[ -f "$input_prot_mem" ]]; then
    echo "✅ Found $input_prot_mem in PROT_MEM/"
else
    echo "❌ $input_prot_mem not found in PROT_MEM/"
    echo "Aborting run."
    exit 1
fi

# Inititiate MCCE_HOME PATH, timing log and set to exit on errors for critical parts
echo "Preparing MCCE Simulation..."
export PATH="$MCCE_HOME/bin:$PATH"  # Add MCCE_HOME/bin to PATH for mcce step executables
TIMING_FILE="mcce_timing.log"
echo "MCCE Timing Report" > $TIMING_FILE
echo "====================================" >> $TIMING_FILE
set -e

# Print MCCE Parameters used
# Check if EXTRA exists; if not, use fallback
# Check if USER_PARAM exists; if not, print N/A
echo "MCCE_HOME: $MCCE_HOME" >> $TIMING_FILE

if [ -f "$EXTRA" ]; then
    EXTRA="$EXTRA"
else
    EXTRA="$MCCE_HOME/extra.tpl"
fi
echo "EXTRA: $EXTRA" >> $TIMING_FILE

if [ -d "$USER_PARAM" ]; then
    echo "USER_PARAM: $USER_PARAM" >> $TIMING_FILE
else
    echo "USER_PARAM: N/A" >> $TIMING_FILE
fi
echo -e "====================================\n" >> $TIMING_FILE

# Finalize MCCE step commands to run
if [ -d "$USER_PARAM" ]; then
    PARAM="-u MCCE_HOME=$MCCE_HOME,EXTRA=$EXTRA,USER_PARAM=$USER_PARAM"
else
    PARAM="-u MCCE_HOME=$MCCE_HOME,EXTRA=$EXTRA"
fi

STEP1_CMD="$STEP1 $PARAM $MEM > step1.log"
STEP2_CMD="$STEP2 $PARAM $MEM > step2.log"
#-----------------------------------------------------------------------------------------
#=========================================================================================

# Function to check if a file was just modified within the last 1 minute
function file_just_made {
    file="$1"
    if [[ -f "$file" ]] && [[ $(find "$file" -mmin -5) ]]; then
        return 0
    else
        return 1
    fi
}

format_time() {
    local elapsed=$1
    local hours=$((elapsed / 3600))
    local minutes=$(( (elapsed % 3600) / 60 ))
    local seconds=$((elapsed % 60))
    printf "%02dh:%02dm:%02ds" "$hours" "$minutes" "$seconds"
}


# Helper to time and record a step
function time_step {
    step_name="$1"
    step_cmd="$2"
    success_output="$3"
    success_msg="$4"

    echo "Running $step_name ..."
    start_time=$(date +%s)

    if eval "$step_cmd"; then
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        formatted_time=$(format_time "$elapsed")

        if file_just_made "$success_output"; then
            echo "$step_name completed SUCCESSFULLY in $formatted_time."
            printf "%-6s: %s   - Success: %s\n" "$step_name" "$formatted_time" "$success_msg" >> "$TIMING_FILE"
        else
            echo "$step_name completed, but expected output $success_output was NOT updated!"
            printf "%-6s: %s   - Failed: expected output $success_output not updated.\n" "$step_name" "$formatted_time" >> "$TIMING_FILE"
        fi
    else
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        formatted_time=$(format_time "$elapsed")

        echo "$step_name FAILED after $formatted_time!"
        printf "%-6s: %s   - Failed: see console output.\n" "$step_name" "$formatted_time" >> "$TIMING_FILE"
    fi
}

#--------------------------------------------------------------------------------------------------------------
# START: RUN MCCE4 Simulations
#==============================================================================================================

script_start_time=$(date +%s)

# Run MCCE STEP 1
time_step "STEP1" "$STEP1_CMD" "step1_out.pdb" "step1_out.pdb updated."

# Run MCCE STEP 2 only if STEP1 output was updated
if file_just_made "step1_out.pdb"; then
   time_step "STEP2" "$STEP2_CMD" "step2_out.pdb" "step2_out.pdb updated."

   grep MEM step2_out.pdb > MEM_step2_out.pdb
   if [[ -s MEM_step2_out.pdb ]]; then
        echo "✅ MEM chains successfully extracted to MEM_step2_out.pdb"
    else
        echo "⚠️  No MEM lines found in step2_out.pdb"
    fi
fi


#--------------------------------------------------------------------------------------------------------------
# END: Run MCCE4 Simulations
#==============================================================================================================

# Finish and record total runtime
script_end_time=$(date +%s)
total_elapsed=$((script_end_time - script_start_time))
formatted_total=$(format_time "$total_elapsed")

echo -e "\nRun ended at: $(date)" >> "$TIMING_FILE"
printf "Total script runtime: %s\n" "$formatted_total" >> "$TIMING_FILE"

sleep 5
echo "Script complete."
echo "Timing report written to $TIMING_FILE"



