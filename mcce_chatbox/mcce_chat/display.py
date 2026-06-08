"""Terminal display for mcce-chat — uses rich if available, ANSI fallback."""

import sys

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    _RICH = True
    _console = Console()
except ImportError:
    _RICH = False
    _console = None

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
RESET = "\033[0m"

MODE_COLORS = {
    "setup": ("yellow", YELLOW),
    "running": ("cyan", CYAN),
    "failed": ("red", RED),
    "done": ("green", GREEN),
    "empty": ("dim", DIM),
}


def print_banner(run_dir: str, mode: str, pdb_id: str, book_state: dict,
                 provider: str, model: str):
    mode_label = mode.upper()
    book_summary = ""
    if book_state:
        counts = {}
        for s in book_state.values():
            label = {"i": "idle", "r": "run", "c": "done", "e": "err"}.get(s, s)
            counts[label] = counts.get(label, 0) + 1
        book_summary = " | ".join(f"{v}{k}" for k, v in counts.items())

    if _RICH:
        rich_color, _ = MODE_COLORS.get(mode, ("white", ""))
        lines = [
            f"[bold]mcce-chat[/bold]  MCCE4 Assistant",
            f"Run dir:  {run_dir}",
            f"Mode:     [{rich_color}]{mode_label}[/{rich_color}]"
            + (f"  PDB: {pdb_id}" if pdb_id else ""),
        ]
        if book_summary:
            lines.append(f"Jobs:     {book_summary}")
        lines.append(f"LLM:      {provider}/{model}")
        _console.print(Panel("\n".join(lines), border_style="blue"))
    else:
        _, ansi_color = MODE_COLORS.get(mode, ("", ""))
        print(f"\n{BOLD}mcce-chat{RESET}  MCCE4 Assistant")
        print(f"  Run dir:  {run_dir}")
        print(f"  Mode:     {ansi_color}{mode_label}{RESET}"
              + (f"  PDB: {pdb_id}" if pdb_id else ""))
        if book_summary:
            print(f"  Jobs:     {book_summary}")
        print(f"  LLM:      {provider}/{model}")
        print()


def print_run_context(summary_text: str):
    if _RICH:
        _console.print(Panel(summary_text, title="Run Context", border_style="green"))
    else:
        print(f"\n{GREEN}--- Run Context ---{RESET}")
        print(summary_text)
        print(f"{GREEN}-------------------{RESET}\n")


def print_prompt():
    if _RICH:
        _console.print("[bold cyan]You>[/bold cyan] ", end="")
    else:
        sys.stdout.write(f"{BOLD}{CYAN}You> {RESET}")
        sys.stdout.flush()


def print_assistant_prefix():
    if _RICH:
        _console.print("[bold green]MCCE>[/bold green] ", end="")
    else:
        sys.stdout.write(f"{BOLD}{GREEN}MCCE> {RESET}")
        sys.stdout.flush()


def print_error(msg: str):
    if _RICH:
        _console.print(f"[bold red]Error:[/bold red] {msg}")
    else:
        print(f"{RED}Error:{RESET} {msg}", file=sys.stderr)


def print_info(msg: str):
    if _RICH:
        _console.print(f"[dim]{msg}[/dim]")
    else:
        print(f"{DIM}{msg}{RESET}")


def print_status_table(ctx):
    if _RICH:
        table = Table(title="Run Status")
        table.add_column("Item", style="cyan")
        table.add_column("Value")

        table.add_row("Directory", ctx.run_dir)
        table.add_row("Mode", ctx.mode())

        if ctx.params:
            table.add_row("Input PDB", ctx.params.inpdb)
            table.add_row("Titration", ctx.params.titr_type or "n/a")
            table.add_row("PBE Solver", ctx.params.pbe_solver or "n/a")
            table.add_row("Epsilon", str(ctx.params.epsilon_prot) if ctx.params.epsilon_prot else "n/a")

        if ctx.protein:
            table.add_row("PDB ID", ctx.protein.pdb_id or "n/a")
            table.add_row("Chains", ", ".join(ctx.protein.chains) if ctx.protein.chains else "n/a")
            table.add_row("Residues", str(ctx.protein.residue_count))
            table.add_row("Atoms", str(ctx.protein.atom_count))
            if ctx.protein.hetatm_ligands:
                table.add_row("Ligands", ", ".join(ctx.protein.hetatm_ligands))

        if ctx.conformers:
            table.add_row("Conformers", str(ctx.conformers.total_conformers))
            table.add_row("Titratable", str(len(ctx.conformers.titratable_residues)))

        table.add_row("Steps done", ", ".join(str(s) for s in ctx.steps_done) or "none")

        if ctx.log_errors:
            table.add_row("Errors", str(len(ctx.log_errors)))
        if ctx.log_warnings:
            table.add_row("Warnings", str(len(ctx.log_warnings)))

        _console.print(table)
    else:
        print(f"\n{BOLD}--- Run Status ---{RESET}")
        print(f"  Directory:  {ctx.run_dir}")
        print(f"  Mode:       {ctx.mode()}")
        if ctx.params:
            print(f"  Input PDB:  {ctx.params.inpdb}")
            print(f"  Titration:  {ctx.params.titr_type or 'n/a'}")
            print(f"  PBE Solver: {ctx.params.pbe_solver or 'n/a'}")
        if ctx.protein:
            print(f"  PDB ID:     {ctx.protein.pdb_id or 'n/a'}")
            print(f"  Chains:     {', '.join(ctx.protein.chains) if ctx.protein.chains else 'n/a'}")
            print(f"  Residues:   {ctx.protein.residue_count}")
        if ctx.conformers:
            print(f"  Conformers: {ctx.conformers.total_conformers}")
            print(f"  Titratable: {len(ctx.conformers.titratable_residues)}")
        print(f"  Steps done: {', '.join(str(s) for s in ctx.steps_done) or 'none'}")
        if ctx.log_errors:
            print(f"  Errors:     {len(ctx.log_errors)}")
        print()
