"""
CLI entry point for the MCCE4 Topology Agent.
"""

import argparse
import os
import sys
import textwrap
import logging
import subprocess
from datetime import datetime

from .config import (SUPPORTED_CHARGE_METHODS, DEFAULT_CHARGE_METHOD, DEFAULT_DIELECTRICS,
                      SUPPORTED_LLM_PROVIDERS, DEFAULT_LLM_PROVIDER,
                      DIMORPHITE_PH_MIN, DIMORPHITE_PH_MAX, DIMORPHITE_PRECISION,
                      DIMORPHITE_MAX_VARIANTS, DIMORPHITE_LABEL_STATES)


def setup_logging(log_file: str, verbose: bool = False):
    """Configure logging."""
    log_level = logging.DEBUG if verbose else logging.INFO
    file_fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    console_fmt = logging.Formatter("%(levelname)-8s | %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG); fh.setFormatter(file_fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level); ch.setFormatter(console_fmt)
    logger.addHandler(ch)


def main():
    parser = argparse.ArgumentParser(
        description="🤖 MCCE4 Topology File AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Input (use exactly one):
          %(prog)s EMH.pdb                                         # PDB file
          %(prog)s EMH.cif                                         # CIF file (auto-converts)
          %(prog)s -lig_code EMH                                   # 3-letter RCSB ligand code
          %(prog)s -smiles "c1ccccc1" -lig_id BEN                    # SMILES string + code
          %(prog)s -smiles EMH.smi                                  # SMILES from .smi file

        Options:
          %(prog)s EMH.pdb --dry-run --no-llm                      # Quick test
          %(prog)s EMH.pdb --gui                                   # Web GUI (Streamlit)
          %(prog)s EMH.pdb -state_pdbs EMH_01.pdb EMH_+1.pdb      # User state PDBs
          %(prog)s EMH.pdb -charge_method am1bcc                   # Override charges
          %(prog)s -lig_code EMH -llm_provider claude              # Use Claude LLM

        Install:
          conda install -c conda-forge google-genai langgraph dimorphite_dl rdkit streamlit
          conda install -c conda-forge anthropic openai  # optional, for Claude/ChatGPT LLM providers
          export GEMINI_API_KEY="your_free_key"   # from https://ai.google.dev
          export ANTHROPIC_API_KEY="your_key"     # for Claude provider
          export OPENAI_API_KEY="your_key"        # for ChatGPT provider
        """))

    parser.add_argument("input_file", nargs="?", default=None,
                        help="Ligand PDB or CIF file (e.g., EMH.pdb or EMH.cif).")
    parser.add_argument("-lig_code", default=None,
                        help="3-letter RCSB ligand code (e.g., EMH). SMILES and "
                             "metadata are fetched from RCSB, 3D structure is "
                             "built from SMILES using RDKit.")
    parser.add_argument("-smiles", default=None,
                        help="SMILES string or .smi file (e.g., \"c1ccccc1\" or "
                             "molecule.smi). 3D structure is built using RDKit. "
                             "Requires -lig_id for raw strings; for .smi files, "
                             "the 3-letter code is derived from the filename.")
    parser.add_argument("-lig_id", default=None,
                        help="3-letter ligand identifier for ftpl naming (e.g., TMA). "
                             "Required with -smiles when a raw SMILES string is given. "
                             "For .smi files, defaults to the first 3 chars of the filename.")
    parser.add_argument("-state_pdbs", nargs="+", default=None,
                        help="PDB files for specific states (e.g., EMH_01.pdb EMH_+1.pdb)")
    parser.add_argument("--gui", action="store_true",
                        help="Launch web GUI (Streamlit) for interactive review")
    parser.add_argument("-charge_method", default=DEFAULT_CHARGE_METHOD,
                        choices=SUPPORTED_CHARGE_METHODS, help="Charge method")
    parser.add_argument("-d", "-dielectric", dest="dielectric",
                        nargs="+", type=int, default=DEFAULT_DIELECTRICS,
                        help="Dielectric constants (default: 2 4 8)")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM reasoning")
    parser.add_argument("--dry-run", action="store_true", help="Skip RXN calibration")

    # Dimorphite-DL options
    dimorphite_group = parser.add_argument_group("Dimorphite-DL options",
        "Control protonation state enumeration (Phase 2)")
    dimorphite_group.add_argument("-ph_min", type=float,
                                  default=DIMORPHITE_PH_MIN,
                                  help=f"Minimum pH for protonation enumeration "
                                       f"(default: {DIMORPHITE_PH_MIN})")
    dimorphite_group.add_argument("-ph_max", type=float,
                                  default=DIMORPHITE_PH_MAX,
                                  help=f"Maximum pH for protonation enumeration "
                                       f"(default: {DIMORPHITE_PH_MAX})")
    dimorphite_group.add_argument("-precision", type=float,
                                  default=DIMORPHITE_PRECISION,
                                  help=f"pKa precision factor: number of std deviations "
                                       f"from mean pKa to consider (default: {DIMORPHITE_PRECISION})")
    dimorphite_group.add_argument("-max_variants", type=int,
                                  default=DIMORPHITE_MAX_VARIANTS,
                                  help=f"Max protonation variants per compound "
                                       f"(default: {DIMORPHITE_MAX_VARIANTS})")
    dimorphite_group.add_argument("--label-states", action="store_true",
                                  default=DIMORPHITE_LABEL_STATES,
                                  help="Label output SMILES as PROTONATED/DEPROTONATED/BOTH")
    parser.add_argument("-work_dir", default=".", help="Working directory")
    parser.add_argument("-o", "-output", dest="output",
                        default=None, help="Output .ftpl filename")
    parser.add_argument("-v", "--verbose", action="store_true")

    # LLM provider options
    parser.add_argument("-llm_provider", default=DEFAULT_LLM_PROVIDER,
                        choices=SUPPORTED_LLM_PROVIDERS,
                        help=f"LLM provider (default: {DEFAULT_LLM_PROVIDER})")
    parser.add_argument("-api_key", default=None,
                        help="API key for the chosen LLM provider (overrides env vars)")

    args = parser.parse_args()

    # ── Validate inputs: exactly one of input_file, -lig_code, or -smiles ──
    n_inputs = sum([
        args.input_file is not None,
        args.lig_code is not None,
        args.smiles is not None,
    ])
    if n_inputs == 0:
        parser.error("Provide one of: PDB/CIF file, -lig_code, or -smiles.")
    if n_inputs > 1:
        parser.error("Use only one input mode: PDB/CIF file, -lig_code, or -smiles.")

    lig_code = args.lig_code.upper() if args.lig_code else None
    pdb_path = None
    smiles_input = None

    if args.input_file is not None:
        if not os.path.exists(args.input_file):
            print(f"ERROR: Input file not found: {args.input_file}")
            sys.exit(1)
        pdb_path = args.input_file
        if args.input_file.lower().endswith(".cif"):
            pdb_path = _convert_cif_to_pdb(args.input_file)
        elif not args.input_file.lower().endswith(".pdb"):
            print(f"ERROR: Input file must be .pdb or .cif: {args.input_file}")
            sys.exit(1)

    if args.smiles is not None:
        if args.smiles.endswith(".smi"):
            if not os.path.exists(args.smiles):
                print(f"ERROR: SMILES file not found: {args.smiles}")
                sys.exit(1)
            with open(args.smiles) as f:
                lines = [l.strip() for l in f if l.strip()]
            if not lines:
                print(f"ERROR: SMILES file is empty: {args.smiles}")
                sys.exit(1)
            if len(lines) > 1:
                print(f"ERROR: .smi file must contain exactly one molecule, "
                      f"found {len(lines)}: {args.smiles}")
                sys.exit(1)
            smiles_input = lines[0].split()[0]
        else:
            if not args.lig_id:
                parser.error("-smiles with a raw SMILES string requires -lig_id "
                             "(e.g., -smiles \"c1ccccc1\" -lig_id BEN).")
            smiles_input = args.smiles

    # Determine lig_id
    if args.lig_id:
        lig_id = args.lig_id.upper()[:3]
        if len(lig_id) != 3:
            parser.error(f"-lig_id must be exactly 3 characters, got: '{args.lig_id}'")
    elif lig_code:
        lig_id = lig_code
    elif pdb_path:
        from .tools.mcce_tools import extract_lig_id_from_pdb
        lig_id = extract_lig_id_from_pdb(pdb_path)
    elif args.smiles and args.smiles.endswith(".smi"):
        name = os.path.splitext(os.path.basename(args.smiles))[0].upper()[:3]
        if len(name) != 3:
            parser.error(f".smi filename must be at least 3 characters for ligand code, "
                         f"got: '{os.path.basename(args.smiles)}'. Use -lig_id instead.")
        lig_id = name
    else:
        lig_id = "LIG"

    # ── GUI mode: launch Streamlit ──
    if args.gui:
        if pdb_path is None:
            print("ERROR: GUI mode requires an input PDB/CIF file.")
            sys.exit(1)
        args.pdb = pdb_path  # GUI expects args.pdb
        launch_gui(args)
        return

    # ── CLI mode: run agent directly ──
    setup_logging(f"mcce_ftpl_{lig_id}.log", args.verbose)

    logging.info(f"{'='*60}")
    logging.info(f"  🤖 MCCE4 Topology Agent v5.1 (LangGraph + per-state PDBs)")
    logging.info(f"{'='*60}")
    if pdb_path:
        logging.info(f"  Input:  {os.path.abspath(pdb_path)}")
        if args.input_file and args.input_file.lower().endswith(".cif"):
            logging.info(f"  (converted from {os.path.abspath(args.input_file)})")
    elif smiles_input:
        logging.info(f"  Input:  -smiles {smiles_input[:80]}")
        logging.info(f"  (3D structure will be built from SMILES using RDKit)")
    else:
        logging.info(f"  Input:  -lig_code {lig_id} (fetching from RCSB)")
    ph = (args.ph_min + args.ph_max) / 2
    logging.info(f"  Ligand: {lig_id}   Method: {args.charge_method}")
    logging.info(f"  Dimorphite-DL: ph_min={args.ph_min}, ph_max={args.ph_max} (pH={ph:.1f}), "
                 f"precision={args.precision}, max_variants={args.max_variants}"
                 f"{', label_states=True' if args.label_states else ''}")
    logging.info(f"  LLM:    {args.llm_provider}")
    logging.info(f"  Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"{'='*60}\n")

    # Set env to disable LLM if requested
    if args.no_llm:
        os.environ.pop("GEMINI_API_KEY", None)
        os.environ.pop("GOOGLE_API_KEY", None)
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.pop("OPENAI_API_KEY", None)

    from .agent import run_agent

    final_state = run_agent(
        pdb_path=pdb_path,
        use_gui=False,
        charge_method=args.charge_method,
        dielectrics=args.dielectric,
        ph=ph,
        work_dir=args.work_dir,
        dry_run=args.dry_run,
        user_state_pdbs=args.state_pdbs,
        output=args.output,
        llm_provider=args.llm_provider,
        api_key=args.api_key,
        lig_code=lig_code,
        smiles_input=smiles_input,
        lig_id=lig_id,
        ph_min=args.ph_min,
        ph_max=args.ph_max,
        precision=args.precision,
        max_variants=args.max_variants,
        label_states=args.label_states,
    )

    # Exit code based on errors
    if final_state.get("errors"):
        sys.exit(1)


def _convert_cif_to_pdb(cif_path: str) -> str:
    """Convert a .cif file to .pdb using cif2pdb_PyMOL.

    Returns the path to the generated PDB file.
    """
    # Locate cif2pdb_PyMOL in the same bin/ directory
    bin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    converter = os.path.join(bin_dir, "cif2pdb_PyMOL")

    if not os.path.exists(converter):
        print(f"ERROR: cif2pdb_PyMOL not found at {converter}")
        print("  This script is required to convert .cif files to .pdb format.")
        sys.exit(1)

    pdb_path = os.path.splitext(cif_path)[0] + ".pdb"
    print(f"🔄 Converting {cif_path} → {pdb_path} using cif2pdb_PyMOL ...")

    try:
        result = subprocess.run(
            [sys.executable, converter, cif_path, pdb_path],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"ERROR: CIF to PDB conversion failed:")
            print(result.stderr or result.stdout)
            sys.exit(1)
        if not os.path.exists(pdb_path):
            print(f"ERROR: Conversion completed but {pdb_path} was not created.")
            sys.exit(1)
        print(f"✅ Converted successfully: {pdb_path}")
    except subprocess.TimeoutExpired:
        print(f"ERROR: CIF to PDB conversion timed out.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"ERROR: Could not run cif2pdb_PyMOL. Ensure PyMOL is installed.")
        sys.exit(1)

    return pdb_path


def launch_gui(args):
    """Launch the Streamlit web GUI."""
    gui_path = os.path.join(os.path.dirname(__file__), "gui", "app.py")

    if not os.path.exists(gui_path):
        print(f"ERROR: GUI app not found at {gui_path}")
        sys.exit(1)

    port = 8501

    # Check if port is already in use and handle it
    if _is_port_in_use(port):
        print(f"⚠  Port {port} is already in use (previous Streamlit session?).")
        print()

        # Try to find the PID
        pid = _find_pid_on_port(port)
        if pid:
            print(f"   Found process PID {pid} on port {port}.")
            try:
                response = input(f"   Kill it and restart? [Y/n] ").strip().lower()
            except EOFError:
                response = "y"

            if response in ("", "y", "yes"):
                import signal
                try:
                    os.kill(pid, signal.SIGTERM)
                    print(f"   Killed PID {pid}. Waiting for port to free...")
                    import time
                    for _ in range(10):
                        time.sleep(0.5)
                        if not _is_port_in_use(port):
                            break
                    if _is_port_in_use(port):
                        os.kill(pid, signal.SIGKILL)
                        time.sleep(1)
                except ProcessLookupError:
                    pass  # Already dead
                except PermissionError:
                    print(f"   Permission denied. Run manually:")
                    print(f"     kill {pid}")
                    print(f"   Then rerun this command.")
                    sys.exit(1)
            else:
                print(f"\n   To free the port manually, run:")
                print(f"     kill {pid}")
                print(f"   Or use a different port:")
                print(f"     streamlit run {gui_path} --server.port 8502 -- {os.path.abspath(args.pdb)}")
                sys.exit(1)
        else:
            print(f"   Could not identify the process. To free port {port}, run:")
            print(f"     pkill -f 'streamlit run'")
            print(f"     # or: kill $(lsof -ti:{port})")
            print(f"   Then rerun this command.")
            sys.exit(1)

    # Pass PDB path via environment so Streamlit can access it
    os.environ["MCCE_AGENT_PDB"] = os.path.abspath(args.pdb)
    os.environ["MCCE_AGENT_PH"] = str((args.ph_min + args.ph_max) / 2)
    os.environ["MCCE_AGENT_CHARGE_METHOD"] = args.charge_method

    print(f"🌐 Launching MCCE4 Topology Agent GUI...")
    print(f"   PDB: {args.pdb}")
    print(f"   Open your browser to: http://localhost:{port}")
    print(f"   (If remote: ssh -L {port}:localhost:{port} user@server)")
    print()

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", gui_path,
            "--server.headless", "true",
            "--server.port", str(port),
            "--browser.gatherUsageStats", "false",
        ])
    except KeyboardInterrupt:
        print("\n  GUI stopped.")
    except FileNotFoundError:
        print("ERROR: Streamlit not installed — install: conda install -c conda-forge streamlit")
        sys.exit(1)


def _is_port_in_use(port: int) -> bool:
    """Check if a TCP port is in use."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def _find_pid_on_port(port: int):
    """Find PID of process using the given port. Returns int or None."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            # May return multiple PIDs; take the first
            return int(result.stdout.strip().split()[0])
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    # Fallback: try ss
    try:
        result = subprocess.run(
            ["ss", "-tlnp", f"sport = :{port}"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            import re
            m = re.search(r'pid=(\d+)', result.stdout)
            if m:
                return int(m.group(1))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


if __name__ == "__main__":
    main()
