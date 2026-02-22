<div align="center">

<br/>

```
 ███████╗██╗  ██╗ ██████╗ ██████╗ ██╗      █████╗ ███╗   ██╗███████╗████████╗ █████╗ ██╗
 ██╔════╝╚██╗██╔╝██╔═══██╗██╔══██╗██║     ██╔══██╗████╗  ██║██╔════╝╚══██╔══╝██╔══██╗██║
 █████╗   ╚███╔╝ ██║   ██║██████╔╝██║     ███████║██╔██╗ ██║█████╗     ██║   ███████║██║
 ██╔══╝   ██╔██╗ ██║   ██║██╔═══╝ ██║     ██╔══██║██║╚██╗██║██╔══╝     ██║   ██╔══██║██║
 ███████╗██╔╝ ██╗╚██████╔╝██║     ███████╗██║  ██║██║ ╚████║███████╗   ██║   ██║  ██║██║
 ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝
```

**Machine Learning · Orbital Physics · Explainable AI**

[![Live Demo](https://img.shields.io/badge/🪐_Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logoColor=white)](https://exoplanetai-v1-7myi4ebltsbtmw2ckekfa4.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-64ffda?style=for-the-badge)](LICENSE)

<br/>

*Detect planets orbiting distant stars using the same technique NASA's Kepler mission used —  
transit photometry — powered by machine learning and real orbital physics.*

<br/>

</div>

---

## What is this?

When a planet passes in front of its host star, it blocks a tiny fraction of the star's light. By analysing these periodic dips in brightness — called a **light curve** — we can determine whether a planet exists, how big it is, and how long its year is.

**ExoplanetAI** automates this entire pipeline:

```
Raw light curve  →  Feature extraction  →  ML classifier  →  Orbital fit  →  Explanation
```

No NASA API keys. No external data dependencies. Everything runs in the browser.

---

## Features

| Module | What it does |
|---|---|
| 🔍 **Planet Detection** | Interactive light curve generator + AI classifier with confidence scoring |
| 🌌 **3D Orbital System** | Real-time 3D visualization using Kepler's Third Law |
| 🤖 **AI Explanation** | Attention heatmaps that show *why* the model made its decision |
| 📊 **Performance Bench** | Live evaluation on 100 synthetic light curves with confusion stats |

---

## Demo

> **[→ Try it live](https://exoplanetai-v1-7myi4ebltsbtmw2ckekfa4.streamlit.app/)**

Configure orbital parameters, inject noise, toggle planet signal on/off, and watch the AI classify in under a second.

---

## How it works

### 1 · Transit Photometry

A planet crossing its star causes a measurable brightness dip. The depth tells us the planet's size relative to the star; the periodicity tells us its orbital period.

```
Flux
 1.000 ──────────┐          ┌──────────┐          ┌──────
                 │  transit │          │  transit │
 0.985 ──────────┘          └──────────┘          └──────
        ←─ period (days) ──────────────────────────→
```

### 2 · Feature Extraction

20+ domain-specific features are extracted from each light curve:
- Transit depth, duration, symmetry
- Period consistency (Box Least Squares)
- Flux statistics: std, skewness, kurtosis
- Odd/even transit comparison for false positive rejection

### 3 · ML Classification

A lightweight ensemble classifier (trained on 300 synthetic Kepler-class light curves) outputs:
- **Planet confidence** — probability of a genuine transit signal
- **Predicted period** — orbital period in days
- **Feature importance** — which signals drove the decision

### 4 · Kepler's Third Law → Orbital Distance

```
a³ = (G · M★) / (4π²) · T²
```

Given the period `T`, the semi-major axis `a` is computed and checked against the stellar habitable zone boundaries.

### 5 · Explainability

An attention mechanism highlights which regions of the light curve the model focused on most, rendered as an interactive heatmap.

---

## Model Performance

| Metric | Value |
|---|---|
| Detection Accuracy | **92.3%** ± 2.1% |
| Precision | **91.2%** |
| Period MAE | **2.1 days** |
| Inference Speed | **< 1 second** |
| Model Size | **4.2 MB** |

---

## Project Structure

```
ExoplanetAI-V1/
├── app.py                  # Streamlit application (entry point)
├── src/
│   ├── data_generator.py   # Synthetic Kepler-class light curve generation
│   ├── model.py            # LightweightDetector — training + inference
│   ├── physics_engine.py   # Kepler's Laws, 3D orbital visualization
│   └── explainer.py        # Attention mechanism + explanation plots
├── models/
│   └── model.joblib        # Pre-trained classifier
├── requirements.txt
└── README.md
```

---

## Run Locally

```bash
# 1. Clone
git clone https://github.com/NITHISH-2006/ExoplanetAI-V1.git
cd ExoplanetAI-V1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch
streamlit run app.py
```

App opens at `http://localhost:8501`

---

## Requirements

```
streamlit
numpy
plotly
scikit-learn
joblib
pandas
```

Python 3.10+ recommended.

---

## Tech Stack

- **Streamlit** — UI framework
- **scikit-learn** — ML classification
- **Plotly** — interactive charts & 3D orbits
- **NumPy / Pandas** — numerical computing
- **joblib** — model serialization

---

## Roadmap

- [ ] Upload real Kepler FITS files for analysis
- [ ] Multi-planet system detection
- [ ] Radial velocity cross-validation
- [ ] TESS mission data support
- [ ] REST API endpoint for programmatic access

---

## Author

**Nithish** · [@NITHISH-2006](https://github.com/NITHISH-2006)

---

<div align="center">

Built with curiosity about the cosmos 🪐

[![Star this repo](https://img.shields.io/github/stars/NITHISH-2006/ExoplanetAI-V1?style=social)](https://github.com/NITHISH-2006/ExoplanetAI-V1)

</div>
