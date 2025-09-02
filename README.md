# Fake or Real: The Impostor Hunt in Texts

## Competition Overview
This project is part of the **Secure Your AI** series of competitions by the European Space Agency (ESA). The main task is to detect fake texts in a dataset where each sample contains two texts - one real and one fake.

## Problem Description
- **Real texts**: Optimal for the recipient, as close as possible to the hidden original text
- **Fake texts**: More or much more distant from the hidden original text
- Both texts have been significantly modified using LLMs
- Documents focus on space-related projects (research, devices, workshops, astronauts)
- Language: English

## Evaluation Metric
- **Pairwise Accuracy**: Evaluates how well models align with human preferences
- Goal: Minimize the number of wrongly chosen texts from each pair
- Submission format: CSV with columns `id` and `real_text_id` (1 or 2)

## Project Structure
```
├── data/                   # Data files (not included in repo)
├── notebooks/             # Jupyter notebooks
├── src/                   # Source code
│   ├── models/           # Model implementations
│   ├── data/             # Data processing utilities
│   ├── evaluation/       # Evaluation metrics
│   └── utils/            # Utility functions
├── requirements.txt       # Python dependencies
├── config.yaml           # Configuration file
└── README.md             # This file
```

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Download competition data to `data/` folder
3. Run baseline notebook: `notebooks/01_baseline_solution.ipynb`
4. Experiment with different approaches in `notebooks/`

## Key Challenges
- Texts are significantly modified using LLMs
- Not all modification types may be in public test sets
- Manual/rule-based solutions may not be optimal
- Focus on distinguishing between good and corrupted outputs

## Timeline
- **Start Date**: 23 June 2025
- **Final Submission**: 23 September 2025
- **Private Leaderboard**: 30 September 2025

## References
- [DataX Strategy Paper](https://iafastro.directory/iac/paper/id/89097/summary/)
- [AI Security for Space Operations](https://star.spaceops.org/2025/user_manudownload.php?doc=150__traivhwa.pdf)
- [Resistance Against Manipulative AI](https://arxiv.org/abs/2404.14230)
- [Dark Patterns in LLMs](https://arxiv.org/abs/2411.06008)
