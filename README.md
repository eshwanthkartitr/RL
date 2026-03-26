---
title: Desalination RL Protocol
emoji: 🌊
colorFrom: cyan
colorTo: blue
sdk: docker
pinned: false
---

# Advanced Municipal Desalination Plant (DesalEnv)

An incredibly unique, real-world RL environment that bridges continuous control, resource arbitrage, dynamic system physics, and environmental noise.

The agent operates an industrial reverse-osmosis water desalination plant providing drinking water to a municipality. It must balance massive trade-offs under high pressure. This goes **far** above basic control loops, presenting specific non-linear phenomena.

### Key Mechanics ⚙️
1. **Weather Shifts:** The environment continuously cycles through weather patterns (`Normal`, `Heatwave`, `Storm`) which violently alter both the Grid Energy Price and the sheer amount of water the city demands. 
2. **Maintenance Logistics:** Pushing water fouls the RO membranes, dragging up energy costs. You can trigger a `run_cleaning` action, however, crews are not instantly available! Doing so locks a `maintenance_cooldown`. Trying to clean while on cooldown results in idle time and fines.
3. **Biological Safety Limits:** Overworking a fouled membrane causes micro-tears resulting in salt leakage. The agent tracks `water_salinity`. Processing high water yields while fouled raises PPM levels. Tipping above 500PPM induces strict city health department fines. 

## 🧠 Environment Structure

### Observation Space

| Feature | Description | Type |
| :--- | :--- | :--- |
| `reservoir_level` | Fresh water stored (Megaliters). | `float` |
| `water_salinity` | PPM of salt in the water. >500 triggers penalties. | `float` |
| `energy_price` | Fluctuating grid energy price ($/MWh). | `float` |
| `membrane_fouling` | Hardware Degradation index (0.0=clean, 1.0=blocked). | `float` |
| `city_demand` | Fluctuating water consumption for the current step. | `float` |
| `weather_condition` | String literal tracking macro-events (`Heatwave`, etc.) | `string` |
| `maintenance_cooldown` | Steps until a cleaning crew is available again. | `int` |

### Action Space (Continuous & Discrete Hybrid)

| Feature | Description | Type |
| :--- | :--- | :--- |
| `production_rate` | Target water extraction flow rate (0.0 to 50.0). | `float` |
| `run_cleaning` | Set True to halt production and wash membranes (checks cooldown). | `bool` |

## Tasks

Provides 6 heavily distinct curriculums across 3 difficulty tiers to truly evaluate agent robustness:

**Tier 1: Standard Evaluation**
* `easy_spring`: Generous reservoir, standard normal weather variables.

**Tier 2: Volatile Environmental Shifts**
* `summer_crisis`: Back-to-back heatwaves and high energy prices. The agent has to aggressively juggle cleanings and salinity.
* `hurricane_season`: Erratic grids, lower demands, but requires extreme energy arbitrage. 

**Tier 3: Asymmetrical Shock Scenarios (Testing True Robustness)**
* `black_swan_drought`: Brutal. Demand stays critically high, reservoir is small. Tests the agent's ability to perfectly time maintenance cooldowns. If they miss one cleaning window, the city drys out.
* `grid_failure`: The ultimate energy arbitrage test. Standard demand, but grid energy pricing fluctuates by massive magnitudes (`price_volatility=250.0`). Pumping at the wrong time bankrupts the plant.
* `marathon_endurance`: A 500-step test where micro-degradations compound. Short-term greedy strategies (running fouled, taking salinity hits) will eventually snowball into total failure.
