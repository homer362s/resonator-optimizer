# Resonator Optimizer

A Python/Tkinter GUI tool for designing and optimizing harmonic resonator networks (bandpass filters). It helps select inductor values and series capacitors that maximize harmonic rejection for a given stimulus waveform.

## Features

- **Frequency Analysis** — plots impedance / S-parameters vs. frequency for the current resonator network
- **Steady-State Analysis** — computes node voltages across harmonics under a specified stimulus
- **Cs Optimisation** — tunes series capacitors per resonator to maximize anti-resonance at target harmonics
- **Best Inductors Search** — Greedy + Genetic Algorithm search over a loaded inductor database to find the optimal set of inductors for maximum FOM (figure of merit)

## Requirements

```
Python 3.8+
numpy
pandas
matplotlib
tkinter (included in standard Python distributions)
```

Install dependencies:

```bash
pip install numpy pandas matplotlib
```

## Usage

```bash
python Resonator_Optimizer_1_22.py
```

1. **Load a stimulus file** (`Files → Open Stimulus`) — CSV with harmonic phase/amplitude data
2. **Load or enter resonators** (`Files → Open Resonators`) — `.res` file with L/C/R values
3. Optionally **load an inductor database** (`Files → Open Inductor Database`) — CSV with SPICE parameters
4. Run analysis from the **Analysis** menu or **Best Inductors** search from the **Optimize** menu

## File Formats

| Extension | Description |
|-----------|-------------|
| `.res` | Resonator definition file (L, Cs, Cp, Rs per resonator) |
| `.csv` (stimulus) | Harmonic stimulus — columns: harmonic index, amplitude, phase |
| `.csv` (inductor DB) | Inductor SPICE parameters — see `0805HP_SPICE_parameters.csv` for format |
| `.dat` | Saved analysis output |

## Versions

See the `CHANGELOG` in `Resonator_Optimizer_1_22.py` for full version history. Current version: **1.22**
