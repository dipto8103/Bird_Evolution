# Evo Birds Life-Cycle Simulator

An open research notebook that prototypes a large-scale, agent‑based reinforcement learning ecosystem. The project simulates multiple bird species moving through a 1024×1024 world divided into habitats, seasons, and elevation bands. Each bird is an RL agent with physiology (energy, hydration, satiety, health), evolutionary traits, and a life cycle (gather twigs → build nest → court → incubate → raise offspring). Pygame renders the world live, while instrumentation charts viability, reward, and per-species survival scores.

---

## Vision

> **Goal:** Build a reusable sandbox where reinforcement learning, ecology, and evolutionary computation converge. We want to see digital bird populations adapt their foraging, nesting, and mating strategies across seasons and generations, and to observe how trait mutations shift ecological niches over time.

Major research questions:
1. Can lightweight RL (linear Q/SARSA/MC) + meta-policy guardrails keep individual birds alive long enough to learn meaningful behaviors?
2. How do trait mutations affect resource specialization (nectar vs. insects vs. seeds vs. vertebrates) when the world imposes seasonal scarcity?
3. What telemetry and visualization tools make an ecological RL simulator debuggable and publishable directly from a notebook?

---

## What Has Been Built

### 1. **Simulation Core**
- **Terrain:** 1024×1024 grid, 16×16 regions, FBM elevation + sinuous river carving. Habitats (forest, shrubland, coastal, etc.) map to tile colors and resource mixes.
- **Seasons:** Summer, Monsoon, Winter cycles with per-season movement costs, dehydration rates, flooding chances, and resource multipliers.
- **Resource layers:** Four masks (Nectar, Insects, Seeds, Vertebrate) spawned according to habitat mix and seasonal multipliers. Desert/water constraints prevent unrealistic placements.

### 2. **Population Model**
- Array-based storage for every bird: position, stats, twigs, nests, mating flags, incubation timers, age, lifespan, species index.
- Traits derived from AVONET-style measurements feed efficiencies for each food type, cold/heat tolerance, and dispersal.
- `spawn_offspring` mutates traits when chicks hatch; `reinit_next_gen` resets stats while keeping learned weights.

### 3. **Policy Stack**
- **Meta-policy guardrails** ensure critical instincts: drink when dehydrated, forage when hungry, rest when weak, gather/build when nestless, court when ready.
- **RL backends**: Q-learning, SARSA, and an experimental MC branch share the same behavior policy but differ in TD update targets.
- **Linear function approximation** (φ·W) with 11 features and 10 discrete actions drives the RL layer.

### 4. **Renderer & Instrumentation**
- Pygame viewport with zoom/pan, species focus filters, HUD showing population counts, algorithm, minimap, and live charts for average reward, vitality, and per-species survival scores.
- Fast-forward toggle, manual season advance (`N`), forced generation reset (`G`), focus switching (`0–5`), and migration overlays.

### 5. **Documentation**
- Notebook is structured as numbered “steps” with markdown explaining imports, CLI, config, traits, world generation, population, policy, simulator, renderer, and conclusions (strengths vs. limitations).

---

## Current Work / Open Threads

| Area | Status | Notes |
|------|--------|-------|
| **Trait persistence across generations** | In progress | `reinit_next_gen` now resets stats but still calls `_init_traits()`, wiping accumulated mutations. Needs redesign to preserve evolved efficiencies for the initial cohort. |
| **Monte Carlo policy** | Experimental | `--algo mc` presently backs up immediate reward only. Needs true episodic return computation to justify the label. |
| **Torch / MP backends** | Stubs | CLI wires `--backend torch` and `--backend mp`, but both routes delegate to NumPy. Requires full tensor/multiprocessing implementations. |
| **Headless/batch runs** | Planned | Simulation currently tied to the renderer loop. A headless driver is needed for overnight training/experiments. |
| **Sensory modeling** | Missing | Birds act on local tile stats only. No explicit perception cones, occlusion, or signaling. |
| **Background intelligence services** | Missing | No swarm, formation, or cooperative behavior controllers; meta-policy only handles individual survival instincts. |
| **Data-driven calibration** | Missing | Habitat/resource parameters are hand-tuned. Integrating eBird/AVONET telemetry and automated sweeps would improve realism. |

---

## Roadmap / Future Enhancements

1. **True Monte Carlo Control**
   - Accumulate returns per episode, update `W` when a bird dies or season/year ends.
   - Support weighted importance sampling so we can evaluate off-policy variants.

2. **Headless Training Harness**
   - Provide a CLI flag (`--headless`) that skips Pygame, runs the backend at maximum speed, periodically checkpoints `pop.W` and metrics.
   - Enable integration with Weights & Biases / TensorBoard for remote monitoring.

3. **Persisted Policies**
   - Add helpers to save/load `pop.W`, mutation seeds, and trait arrays so long-running experiments can pause/resume.

4. **Trait Evolution Improvements**
   - Preserve mutated efficiencies across `reinit_next_gen`.
   - Introduce selection pressure (e.g., keep top‑k viability scores) instead of random resets.

5. **Richer Sensing and Social Cues**
   - Implement local perception (vision cone, neighbor list) feeding extra features to φ.
   - Add shared memory structures so birds can leave pheromone-like signals or warnings.

6. **Physics & Environmental Fidelity**
   - Prototype integration with a lightweight physics engine (e.g., Box2D) for flight arcs.
   - Model weather events (storms, temperature gradients) beyond the current scalar penalties.

7. **Testing & Modularization**
   - Extract world generation, population dynamics, and policy modules into Python packages with unit tests.
   - Maintain a script version for CI while keeping the notebook narrative for demos.

8. **Ecosystem Data Pipeline**
   - Use real telemetry (eBird, AVONET, GPS tags) to set habitat distributions, migration timings, and species-specific parameters.
   - Provide calibration notebooks that fit model parameters to real-world survival curves.

---

## Quick Start

```bash
pip install -r requirements.txt   # Pygame, NumPy, etc.
cd "Code files"
jupyter notebook test.ipynb       # or nbconvert to a script and run with python
```

Key controls in the Pygame window:

| Key | Action |
|-----|--------|
| `Space` | Toggle fast-forward (logic loop runs at 3–6× speed) |
| `N` | Force next season |
| `G` | Force new generation (reseed population, clear nests) |
| `M` | Toggle migration overlay |
| `C` | Toggle camera follow |
| `0–5` | Focus all species / individual species |
| `+` / `-` | Zoom in/out |
| Arrow keys / WASD | Pan camera when follow is off |

To switch RL algorithm in the notebook, set `sys.argv` before running the main cell:

```python
import sys
sys.argv = ["notebook", "--algo", "sarsa"]  # or "q", "mc"
```

---

## Project Layout

```
ReinforcementLearning_MegaProject/
├─ Code files/
│  ├─ test.ipynb         # Primary research notebook
│  ├─ test_review.py     # Script export for diff/review
│  └─ icons/, logs/…     # Assets and telemetry
├─ Datasets/
│  ├─ EDA/
│  └─ ELEData/…          # AVONET + trait datasets
├─ requirements.txt
└─ README.md             # This document
```

---

## Conclusion

The notebook now acts as both an experimental RL simulator and a narrative artifact. It demonstrates:
- A layered architecture where world → population → policy → renderer stay decoupled.
- Vectorized math capable of simulating large flocks deterministically.
- Instrumentation that keeps researchers in the loop via live charts and HUDs.

At the same time, it candidly exposes the missing pieces—no standardized “bird engine,” sparse sensory modeling, limited background intelligence services, and a reliance on handcrafted parameters. Closing those gaps will require dedicated physics/sensing modules, headless training pipelines, and data-driven calibration. Until then, Evo Birds remains a research sandbox: perfect for exploring ideas, not yet validated as a faithful ecological twin.

