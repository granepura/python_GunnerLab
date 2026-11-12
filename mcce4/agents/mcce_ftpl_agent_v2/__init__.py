"""
mcce_ftpl_agent — MCCE4 Topology File AI Agent Package
=========================================================

An AI agent for creating MCCE4 topology files (.ftpl) from ligand PDB files.
Uses Dimorphite-DL for protonation state enumeration, RDKit for chemistry
validation, Google Gemini (free tier) for reasoning, and LangGraph for
agentic orchestration.

Install:
    conda install -c conda-forge google-genai langgraph dimorphite_dl rdkit streamlit

Usage:
    # CLI
    mcce_ftpl EMH.pdb --gui

    # Python
    from mcce_ftpl_agent.agent import run_agent
    run_agent("EMH.pdb")
"""

__version__ = "5.0.0"
__author__ = "Gehan / MCCE4 Team (GunnerLab)"
