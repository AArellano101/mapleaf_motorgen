# Mapleaf Motorgen

This document outlines how `aaron_motorgen.py` (the refactored version) improves upon the original script by Zach. Written by *Aaron Arellano*.

---

## Overview

Both scripts accomplish the same core task: read static fire test data, extract the motor burn window, and write a MapleLeaf-compatible motor definition file. The refactored version re-implements this logic with improved structure, correctness, and extensibility.

---

## Key Improvements

### 1. Object-Oriented Architecture

The original script is a single flat file with global variables and close to no functions. The refactored version organizes everything into focused classes:

- **`MotorConfig`** - all tunable parameters in one place, with clear defaults and comments
- **`MotorDataProcessor`** - encapsulates the full processing pipeline
- **`MomentOfInertia`** - static utility methods for cylinder MOI calculations
- **`DynamicFlowRateCalculator`** - dedicated class for computing flow rates from load cell data

This separation makes it far easier to reuse, test, and extend each component independently.

---

### 2. Dynamic Flow Rate Calculation

The original script hardcodes the oxidizer flow rate. This refactored version introduces `DynamicFlowRateCalculator`, which:

- Dynamically computes flow rates from the load cell mass derivatives
- Applies a moving-average smoothing window to reduce noise

This allows the output file to reflect actual measured flow variation across the burn rather than assuming a constant average.

---

### 3. Improved Burn Extension Logic

The original extension algorithm collects "stable" thrust points into a list and inserts them semi-randomly. It has an off-by-one pop/insert sequence that can corrupt the time and thrust arrays at the insertion boundary, and it doesn't update MOI after extension.

The refactored version:

- Searches for a stable region using a sliding window with std deviation and mean thresholds
- Repeats the actual thrust pattern from that region
- **Allows for decimal burntimes (i.e. 3.4s, 2.1s)**

---

### 4. Configurable Parameters in One Place

The original script scatters magic numbers throughout the file (e.g., `OFRatio = 6.6`, `totalOxMass = 9.55`, hardcoded geometry). The refactored version centralizes every tunable value in `MotorConfig`, with descriptive names and inline comments explaining units and purpose. Changing a parameter no longer requires hunting through the script.

---

### 5. Unit Test Coverage

The original script has no tests. The refactored version includes a full `unittest` suite covering:

| Test Class | What It Covers |
|---|---|
| `TestMomentOfInertia` | MOI formulas for solid and hollow cylinders, zeroed mode |
| `TestDynamicFlowRateCalculator` | Constant-rate recovery, mass conservation error < 5%, noise reduction from smoothing, O/F ratio application |
| `TestMotorConfig` | Default values, runtime modification |
| `TestMotorDataProcessor` | Initialization, burn window detection, flow rate calculation, thrust scaling, MOI recalculation |
| `TestConfigureMotorFromMasses` | Utility function for back-calculating motor parameters from target masses |

---

### 6. `configure_motor_from_masses` Utility

A new standalone function that works backwards from desired propellant masses to compute burn time, flow rates, and total impulse. Useful for pre-test planning or cross-checking test results against design targets.

---

### 7. Enhanced Plotting

The original script opens separate blocking `plt.show()` windows at multiple points in the script, with no option to save. The refactored version:

- Places all plots into a single multi-panel figure
- Adds a statistics box (average thrust, max thrust, total impulse) directly on the plot
- Saves figures to disk via the `save_plots` config flag

---

### 8. Code Quality and Maintainability

| Aspect | Original | Refactored |
|---|---|---|
| Global mutable state | Yes | No |
| Magic numbers | Throughout | Centralized in `MotorConfig` |
| Error handling | None | Graceful fallbacks (e.g., stable region not found) |
| Docstrings | None | Every class and method documented |
| Unit tests | None | Full suite with 15+ test cases |
| Output file management | Manual `open/close` | Context manager (`with` block) |
| MOI after extension | Not updated | Fully recalculated |
