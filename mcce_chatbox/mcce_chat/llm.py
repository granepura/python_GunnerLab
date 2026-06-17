"""Multi-provider LLM streaming backends for mcce-chat."""

import configparser
import os
import sys
from pathlib import Path

SYSTEM_PROMPT = """\
You are MCCE Chatbox, an expert assistant for MCCE4 (Multi-Conformation \
Continuum Electrostatics) by the Gunner Lab at CCNY. Help users set up runs, \
diagnose failures, interpret pKa results, understand .ftpl topology files, and \
fix common problems. Always cite specific files/functions from the retrieved \
context. If suggesting a run.prm fix, show the exact line.

## MCCE4 Job Submission System

### submit_mcce4.sh (schedulers/submit_mcce4.sh)
SLURM submission script. Users configure these sections at the top:

SBATCH options:
  --job-name, --output, --nodes, --mem, --export

User-configurable parameters:
  input_pdb="prot.pdb"          # input PDB (usually symlinked: ln -s 4lzt.pdb prot.pdb)
  USER_PARAM="./user_param"     # directory with custom .ftpl topology files
  EXTRA="./user_param/extra.tpl" # custom extra.tpl overrides
  TMP="/tmp"                    # temp dir for PBE data during step3
  CPUS=1                        # CPU cores for step3 parallelization
  EPS=8                         # protein dielectric constant

Step control flags (t=run, f=skip):
  step1="t"    # pre-run: pdb -> mcce pdb
  step2="t"    # make rotamers
  step3="t"    # energy calculations (PBE solver)
  step4="t"    # Monte Carlo sampling
  step_clean="t"  # clean PBE temp data after step3

Optional step flags:
  center="t"   # center protein before run
  stepM="f"    # generate partial membranes (requires stepM.sh conditions)
  stepA="f"    # custom script between step1 and step2
  stepB="f"    # custom script between step2 and step3
  stepC="f"    # custom script between step3 and step4

Step commands (editable for advanced users):
  STEP1="$PYEX $MCBIN/step1.py -d $EPS --dry"
  STEP2="$PYEX $MCBIN/step2.py -d $EPS -l 1"
  STEP3="$PYEX $MCBIN/step3.py -d $EPS -s ngpb -p $CPUS -t $TMP"
  STEP4="$PYEX $MCBIN/step4.py --xts -i 7 -n 1"

Optional script paths (must exist and be executable if enabled):
  STEPM="/path/to/stepM.sh"
  STEPA="/path/to/stepA_script.py"
  STEPB="/path/to/stepB_script.py"
  STEPC="/path/to/stepC_script.py"

### The -u flag (parameter overrides)
Every step script (step1.py, step2.py, step3.py, step4.py) accepts a -u flag
that overrides run.prm parameters at the command line. Format:
  -u KEY1=VALUE1,KEY2=VALUE2,...

Examples:
  step1.py -u MCCE_HOME=/path/to/mcce,H2O_SASCUTOFF=0.05
  step2.py -u PACK=t,ROTATIONS=6,HDIRECTED=t,RELAX_H=t
  step3.py -u EPSILON_SOLV=80.0,SALT=0.15,GRIDS_DELPHI=65
  step4.py -u MONTE_RUNS=6,MONTE_NITER=5000,MONTE_T=298.15

In submit_mcce4.sh, add -u to the STEP commands:
  STEP1="$PYEX $MCBIN/step1.py -d $EPS --dry -u H2O_SASCUTOFF=0.05"
  STEP2="$PYEX $MCBIN/step2.py -d $EPS -l 1 -u PACK=t,ROTATIONS=6"

The driver_mcce4.sh automatically merges -u flags: if the STEP command already
has -u and the driver adds system -u params (MCCE_HOME, EXTRA, USER_PARAM),
they get combined into one comma-separated list.

step3.py also accepts -load_runprm to load an entire run.prm file as overrides:
  step3.py -load_runprm runprms/run.prm.full

### run.prm parameter reference (from runprms/)
The reference run.prm files in MCCE4/runprms/ define all available -u keys.
Key run.prm variants:
  run.prm.full  — all parameters with documentation (the master reference)
  run.prm.quick — faster settings: fewer rotamers, less relaxation
  run.prm.qq    — fastest: isosteric conformers only, no optimization

Step 1 parameters (usable with -u):
  TERMINALS=t/f          label terminal residues
  H2O_SASCUTOFF=0.05    cut off water if SAS% exceeds this
  CLASH_DISTANCE=2.0     distance limit for reporting clashes
  IGNORE_INPUT_H=t/f     ignore hydrogens in input PDB

Step 2 parameters (usable with -u):
  ROT_SPECIF=t/f         use head1.lst for rotamer control
  ROT_SWAP=t/f           do stereo isotope swap
  PACK=t/f               do bond rotation (make rotamers)
  ROTATIONS=6            number of rotamers per bond rotation
  SAS_CUTOFF=1.0         SAS threshold for fewer rotamers
  VDW_CUTOFF=10.0        self vdw cutoff (kcal/mol)
  REPACKS=5000           number of repacks
  REPACK_CUTOFF=0.01     occupancy cutoff for repacks
  HDIRECTED=t/f          h-bond directed rotamer making
  HDIRDIFF=1.0           threshold for conformer difference
  HDIRLIMT=36            max h-bond conformers
  RELAX_H=t/f            do hydrogen relaxation
  RELAX_E_THR=-1.0       energy threshold for keeping conformers
  RELAX_NSTATES=100      local microstates to loop over
  RELAX_N_HYD=36         default hydroxyl positions
  HV_RELAX_NCYCLE=0      heavy atom relaxation cycles (0=off, 2=quick, 10=full)
  PRUNE_THR=0.02         final pruning threshold
  NCONF_LIMIT=0          max conformers per residue (0=unlimited)
  REBUILD_SC=t/f         rebuild sidechain from torsion minima

Step 3 parameters (usable with -u):
  EPSILON_PROT=4.0       protein dielectric (4 for large, 8 for small proteins)
  EPSILON_SOLV=80.0      solvent dielectric
  GRIDS_DELPHI=65        grids per DelPhi run
  GRIDS_PER_ANG=2.0      target grids per angstrom
  RADIUS_PROBE=1.4       probe radius
  IONRAD=2.0             ion radius
  SALT=0.15              salt concentration
  PBE_SOLVER=apbs        PBE solver: apbs, delphi, or ngpb
  PBE_FOLDER=/tmp        temp folder for PBE data
  QUICK_ENERGIES=t/f     use SAS + Coulomb instead of PBE (very fast, less accurate)
  REASSIGN=t/f           reassign charges/radii before PBE
  RECALC_TORS=t/f        recalculate torsion energy for head3.lst
  RXN_METHOD=surface     self/surface for reaction field method

Step 4 parameters (usable with -u):
  MONTE_SEED=-1          random seed (-1 = time-based)
  MONTE_T=298.15         temperature (K)
  MONTE_FLIPS=3          number of flips
  MONTE_NSTART=100       annealing iterations (× conformers)
  MONTE_NEQ=300          equilibration iterations (× conformers)
  MONTE_REDUCE=0.001     occupancy cutoff for reduction
  MONTE_RUNS=6           independent MC sampling runs
  MONTE_NITER=2000       sampling iterations (× conformers)
  MONTE_TRACE=50000      trace energy interval (0 = no trace)
  MONTE_TSX=t/f          entropy correction
  MONTE_ADV_OPT=t/f      use Yifan's advanced MC
  MONTE_NITER_MIN=5000   min sampling for advanced MC
  MONTE_NITER_MAX=-1     max sampling (-1 = stop at convergence)
  MONTE_CONVERGE=0.01    convergence threshold
  MFE_POINT=t/f          specify MFE point (f=pKa/Em)
  MFE_CUTOFF=-1.0        MFE cutoff in kcal
  TITR_TYPE=ph           ph or eh titration
  TITR_PH0=0.0           starting pH
  TITR_PHD=1.0           pH interval
  TITR_EH0=0.0           starting Eh (mV)
  TITR_EHD=30.0          Eh interval (mV)
  TITR_STEPS=15          number of titration points

### driver_mcce4.sh (bin/driver_mcce4.sh)
Called by submit_mcce4.sh. Orchestrates the full pipeline:
  1. Initialize timing log, verify MCCE environment
  2. Setup run.prm with parameters from submit_mcce4.sh
  3. [Optional] Center protein structure
  4. [Optional] StepM: generate membrane
  5. Step 1: pre-run (step1.py)
  6. [Optional] StepA: custom script
  7. Step 2: rotamers (step2.py)
  8. [Optional] Append membrane to step2_out.pdb
  9. [Optional] StepB: custom script
  10. Step 3: energy calculations (step3.py)
  11. [Optional] StepC: custom script
  12. Step 4: Monte Carlo (step4.py)
  13. Clean up temp PBE data

Output: mcce_timing.log with per-step timing, step logs (step1.log, etc.)

### Common user tasks with submit_mcce4.sh:
- Change dielectric: edit EPS=8 to desired value
- Use more CPUs: edit CPUS=1 to number of cores
- Skip a step: set its flag to "f" (e.g. step1="f" to skip step1)
- Add custom topology: put .ftpl files in user_param/ directory
- Change PBE solver: edit the -s flag in STEP3 (ngpb or apbs or delphi)
- Run on different PDB: change input_pdb or symlink prot.pdb
- Adjust memory: edit #SBATCH --mem=12G
- Custom scripts: set stepA/B/C="t" and point STEPA/B/C to your script
- Override run.prm params: add -u KEY=VALUE to STEP commands
- Load full param set: add -load_runprm runprms/run.prm.full to step3/step4

## run.prm format
Each line: "value   KEY_IN_PARENS   # optional comment"
The value comes FIRST, the key is in parentheses at the end.
Example: "prot.pdb                          (INPDB)"

## book.txt format
Each line: "dirname  state" where state is i(idle)/r(running)/c(complete)/e(error)
"""

DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL = "llama3"

KNOWN_PROVIDERS = {
    "anthropic": {
        "env_key": "ANTHROPIC_API_KEY",
        "pip": "anthropic",
        "default_model": "claude-sonnet-4-20250514",
    },
    "ollama": {
        "env_host": "OLLAMA_HOST",
        "pip": "ollama",
        "default_model": "llama3",
    },
    "groq": {
        "env_key": "GROQ_API_KEY",
        "pip": "openai",
        "default_model": "llama-3.3-70b-versatile",
    },
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "pip": "openai",
        "default_model": "gpt-4o-mini",
    },
    "openai_compat": {
        "env_base_url": "OPENAI_COMPAT_BASE_URL",
        "env_key": "OPENAI_COMPAT_API_KEY",
        "pip": "openai",
        "default_model": "default",
    },
    "gemini": {
        "env_key": "GOOGLE_API_KEY",
        "pip": "google-genai",
        "default_model": "gemini-2.5-flash",
    },
}

CONF_PATH = Path.home() / ".mcce_chat.conf"


def load_config() -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    if CONF_PATH.exists():
        config.read(str(CONF_PATH))
    return config


def save_config(provider: str, model: str):
    config = load_config()
    if "defaults" not in config:
        config["defaults"] = {}
    config["defaults"]["provider"] = provider
    config["defaults"]["model"] = model
    with open(str(CONF_PATH), "w") as f:
        config.write(f)


def resolve_provider_model(cli_provider: str = None, cli_model: str = None) -> tuple:
    config = load_config()
    provider = cli_provider
    model = cli_model

    if not provider:
        provider = config.get("defaults", "provider", fallback=None)
    if not provider:
        provider = DEFAULT_PROVIDER

    if not model:
        model = config.get("defaults", "model", fallback=None)
    if not model:
        model = KNOWN_PROVIDERS.get(provider, {}).get("default_model", DEFAULT_MODEL)

    return provider, model


def _get_api_key(provider: str) -> str:
    config = load_config()
    info = KNOWN_PROVIDERS.get(provider, {})

    env_key_name = info.get("env_key", "")
    if env_key_name:
        key = os.environ.get(env_key_name)
        if key:
            return key

    if config.has_section(provider):
        key = config.get(provider, "api_key", fallback=None)
        if key:
            return key

    return None


def _get_ollama_host() -> str:
    config = load_config()
    host = os.environ.get("OLLAMA_HOST")
    if host:
        return host
    if config.has_section("ollama"):
        host = config.get("ollama", "host", fallback=None)
        if host:
            return host
    return "http://localhost:11434"


def _require_import(module_name: str, pip_name: str = None):
    try:
        return __import__(module_name)
    except ImportError:
        pip_name = pip_name or module_name
        print(f"\nError: '{module_name}' is not installed.", file=sys.stderr)
        print(f"Install it with:  pip install {pip_name}", file=sys.stderr)
        sys.exit(1)


def stream_anthropic(messages: list, model: str, run_context_text: str = ""):
    anthropic = _require_import("anthropic")
    api_key = _get_api_key("anthropic")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set. Export it or add to ~/.mcce_chat.conf [anthropic] api_key=...",
              file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    system = SYSTEM_PROMPT
    if run_context_text:
        system += f"\n\nCurrent MCCE4 run context:\n{run_context_text}"

    with client.messages.stream(
        model=model,
        max_tokens=4096,
        system=system,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text


def stream_ollama(messages: list, model: str, run_context_text: str = ""):
    host = _get_ollama_host()
    system_msg = SYSTEM_PROMPT
    if run_context_text:
        system_msg += f"\n\nCurrent MCCE4 run context:\n{run_context_text}"

    full_messages = [{"role": "system", "content": system_msg}] + messages

    try:
        ollama = _require_import("ollama")
        client = ollama.Client(host=host)
        response = client.chat(model=model, messages=full_messages, stream=True)
        for chunk in response:
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content
    except ImportError:
        pass
    except Exception:
        import requests
        url = f"{host}/api/chat"
        payload = {"model": model, "messages": full_messages, "stream": True}
        resp = requests.post(url, json=payload, stream=True)
        resp.raise_for_status()
        import json
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                content = data.get("message", {}).get("content", "")
                if content:
                    yield content


def stream_openai_like(messages: list, model: str, run_context_text: str = "",
                       base_url: str = None, api_key: str = None):
    openai = _require_import("openai")
    system_msg = SYSTEM_PROMPT
    if run_context_text:
        system_msg += f"\n\nCurrent MCCE4 run context:\n{run_context_text}"

    full_messages = [{"role": "system", "content": system_msg}] + messages

    kwargs = {}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key

    client = openai.OpenAI(**kwargs)
    response = client.chat.completions.create(
        model=model,
        messages=full_messages,
        stream=True,
        max_tokens=4096,
    )
    for chunk in response:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            yield delta.content


def stream_groq(messages: list, model: str, run_context_text: str = ""):
    api_key = _get_api_key("groq")
    if not api_key:
        print("Error: GROQ_API_KEY not set. Export it or add to ~/.mcce_chat.conf [groq] api_key=...",
              file=sys.stderr)
        sys.exit(1)
    yield from stream_openai_like(
        messages, model, run_context_text,
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
    )


def stream_openai(messages: list, model: str, run_context_text: str = ""):
    api_key = _get_api_key("openai")
    if not api_key:
        print("Error: OPENAI_API_KEY not set. Export it or add to ~/.mcce_chat.conf [openai] api_key=...",
              file=sys.stderr)
        sys.exit(1)
    yield from stream_openai_like(messages, model, run_context_text, api_key=api_key)


def stream_gemini(messages: list, model: str, run_context_text: str = ""):
    genai = _require_import("google", pip_name="google-genai")
    from google import genai as genai_module
    from google.genai import types

    api_key = _get_api_key("gemini")
    if not api_key:
        print("Error: GOOGLE_API_KEY not set. Export it or add to ~/.mcce_chat.conf [gemini] api_key=...",
              file=sys.stderr)
        sys.exit(1)

    client = genai_module.Client(api_key=api_key)
    system_msg = SYSTEM_PROMPT
    if run_context_text:
        system_msg += f"\n\nCurrent MCCE4 run context:\n{run_context_text}"

    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

    response = client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_msg,
            max_output_tokens=4096,
        ),
    )
    for chunk in response:
        if chunk.text:
            yield chunk.text


def stream_openai_compat(messages: list, model: str, run_context_text: str = ""):
    base_url = os.environ.get("OPENAI_COMPAT_BASE_URL")
    api_key = os.environ.get("OPENAI_COMPAT_API_KEY", "no-key")
    config = load_config()
    if not base_url and config.has_section("openai_compat"):
        base_url = config.get("openai_compat", "base_url", fallback=None)
        if not api_key or api_key == "no-key":
            api_key = config.get("openai_compat", "api_key", fallback="no-key")
    if not base_url:
        print("Error: OPENAI_COMPAT_BASE_URL not set.", file=sys.stderr)
        sys.exit(1)
    yield from stream_openai_like(messages, model, run_context_text,
                                  base_url=base_url, api_key=api_key)


STREAM_FUNCS = {
    "anthropic": stream_anthropic,
    "ollama": stream_ollama,
    "groq": stream_groq,
    "openai": stream_openai,
    "openai_compat": stream_openai_compat,
    "gemini": stream_gemini,
}


def stream_chat(provider: str, model: str, messages: list, run_context_text: str = ""):
    func = STREAM_FUNCS.get(provider)
    if not func:
        print(f"Error: Unknown provider '{provider}'. Known: {', '.join(STREAM_FUNCS)}",
              file=sys.stderr)
        sys.exit(1)
    return func(messages, model, run_context_text)


def list_models():
    print("Known providers and default models:\n")
    for name, info in KNOWN_PROVIDERS.items():
        default = info.get("default_model", "?")
        env = info.get("env_key", info.get("env_host", ""))
        pip = info.get("pip", "")
        status = ""
        if env and os.environ.get(env):
            status = " [key set]"
        elif name == "ollama":
            status = " [no key needed]"
        print(f"  {name:16s} default_model={default:30s} pip={pip}{status}")
    print()
    config = load_config()
    if config.has_section("defaults"):
        p = config.get("defaults", "provider", fallback="?")
        m = config.get("defaults", "model", fallback="?")
        print(f"  Config default: provider={p}, model={m}  ({CONF_PATH})")
