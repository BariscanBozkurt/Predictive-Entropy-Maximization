# Predictive Entropy Maximization

This repository contains the code used for the anonymous paper submission.

## Repository structure

- `src/` contains the implementations of predictive entropy maximization (PEM; code class `PredictiveDecorrBSS`), unnormalized predictive entropy maximization (u-PEM; code class `PredictiveDecorrBSSSimple`), and the benchmark methods included in the submission.
- `Simulations/PredictiveBSS/` and `Simulations/u-PEM/` contain the main simulation scripts.
- `Simulations/AblationStudies/` contains the ablation-study scripts.
- `Simulations/AnalyzeSimulationResults/` contains the Jupyter notebooks used to load result files from `Simulations/Results/` and generate the corresponding tables and figures.

## Reproducing representative results

Run each script from the directory that contains it so that the relative paths resolve correctly.

Example 1: main simplex experiment for PEM

```bash
cd Simulations/PredictiveBSS
python PredictiveBSS_Noisy_Simplex_10by5.py
```

This produces `Simulations/Results/predictive_bss_noisy_simplex_10by5_results.pkl`.

Example 2: main simplex experiment for u-PEM

```bash
cd Simulations/u-PEM
python PredictiveBSS_Noisy_Simplex_10by5.py
```

This produces `Simulations/Results/upem_noisy_simplex_10by5_results.pkl`.

To visualize these results, open the matching notebook:

```bash
cd Simulations/AnalyzeSimulationResults
jupyter notebook Simplex_Simulations.ipynb
```

The notebook reads the pickle files from `../Results/` and saves the generated figures in `Simulations/AnalyzeSimulationResults/Figures/`.

## Other experiments

Other settings follow the same pattern: run the corresponding script in `Simulations/PredictiveBSS/`, `Simulations/u-PEM/`, or `Simulations/AblationStudies/`, then open the matching notebook in `Simulations/AnalyzeSimulationResults/`.

For example, ablation studies can be reproduced by running a script such as:

```bash
cd Simulations/AblationStudies
python PredictiveBSS_Ablation_NumberOfMixtures_Sparse.py
```

This produces a result file in `Simulations/Results/`. The corresponding visualizations can then be generated from:

```bash
cd Simulations/AnalyzeSimulationResults
jupyter notebook Ablation_Studies.ipynb
```

Other analysis notebooks follow similarly, including `Sparse_Simulations.ipynb`, `NNSparse_Simulations.ipynb`, and `NNAntisparse_Simulations.ipynb`.
