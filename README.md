# Inductor-Driven Qubit Simulations

This repository contains Python simulations and analysis tools for driven transmon/qubit systems, including a serial-inductor drive, reflected-wave transient and frequency responses, and coefficient extraction from reflected spectra. The analysis scripts generate Matplotlib figures interactively; they do not save plots automatically.

## Files

- `Reflected_Wave_Transient.py` runs the reflected-wave transient analysis. It calculates drive and qubit-response waveforms, evaluates the reflected spectrum, extracts and fits qubit-state coefficients, prints results, and displays comparison plots. This is the main end-to-end analysis script.
- `Reflected_Wave_Solver.py` provides reusable functions and result dataclasses for extracting complex drive and qubit coefficients from reflected-spectrum measurements. Running it directly evaluates a built-in example.
- `Coefficients_Solver.py` contains an alternate version of the coefficient-extraction implementation, intended to be imported and used by other scripts.
- `Serial_Ind_Drive.py` models a qubit driven through a serial capacitor with QuTiP and plots the drive and qubit-state evolution.
- `Serial_Ind_Drive_TL.py` builds a finite-mode transmission-line model and runs a QuTiP master-equation solve for the driven system. It currently prints solver/model diagnostics; it does not produce plots.
- `Markov_Appox_Check.py` evaluates and plots an integral used to check a Markov approximation.
- `Graph_Lib.py` contains shared Matplotlib helpers for plot formatting, parameter annotations, axis ticks, and finding curve/level intersections.
- `Qubit_Lib.py` contains helpers for deriving relative state phases/frequencies and Bloch-sphere angles from simulated state data.
- `main.py` is the default PyCharm greeting example and is not connected to the simulations.

`Old_Versions/` contains archived scripts and is excluded from Git.

## Requirements

Use Python 3.10 or newer. The scripts use NumPy, SciPy, Matplotlib, QuTiP, fontTools, and Hypothesis. The latter two are imported by some scripts even though their imported names are not used in the calculations.

Create and activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib qutip fonttools hypothesis
```

If PowerShell blocks activation, allow scripts for the current terminal session and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## Run

Run commands from the repository root with the virtual environment active. For example:

```powershell
python Reflected_Wave_Transient.py
```

Other available scripts can be run independently:

```powershell
python Serial_Ind_Drive.py
python Serial_Ind_Drive_TL.py
python Markov_Appox_Check.py
python Reflected_Wave_Solver.py
```

The simulation scripts use parameters defined near the top of each file. Edit those values to change the modeled circuit, drive, initial state, or frequency range. Plotting scripts open interactive figure windows; close the windows to let the script exit. `python main.py` only prints the default greeting.