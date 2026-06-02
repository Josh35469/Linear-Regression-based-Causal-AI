# Datathon--Team-Datahunters
Causal Inference Framework for Scrap Reduction in Injection Molding Overview

This project implements a causal inference pipeline to identify and optimize process interventions for reducing scrap rate in an injection molding system.

Traditional data analysis in manufacturing often relies on correlations, which can be misleading due to confounding variables. This project addresses that limitation by applying causal reasoning to determine:

Which variables should be changed In what direction By how much To achieve a measurable reduction in scrap rate

The implementation follows a structured approach combining domain knowledge (DAG), statistical modeling, and validation techniques.

Key Features Causal modeling using a predefined Directed Acyclic Graph (DAG) Backdoor-adjusted estimation of causal effects (ATE) Identification of: Controllable variables Confounders Mediators Bootstrap confidence intervals for uncertainty quantification Refutation testing: Placebo test Subset stability test Feasibility-constrained intervention recommendations Counterfactual simulation engine Visualizations for: Correlation analysis Cooling paradox Intervention impact Pathway attribution Dataset

The model uses a synthetic but realistic injection molding dataset:

~5000 production intervals 33 variables Target KPI: scrap_rate_pct

Each row represents a 30-minute production interval for a specific machine, mold, product variant, and material batch.

Methodology

The pipeline follows a five-step causal framework:

Variable Classification
Using the DAG, variables are categorized into:

Controllable levers Confounders Mediators Non-controllable context 2. Causal Identification

Backdoor adjustment is applied to remove bias from confounders and estimate interventional effects.

Effect Estimation
Linear regression is used to compute Average Treatment Effects (ATE) for each controllable variable.

Validation
Results are validated using:

Bootstrap confidence intervals (95%) Placebo tests Subset stability checks 5. Decision Layer

Feasible interventions are applied (typically ±10% changes), and expected impact on scrap rate is calculated.

Key Insight

A major finding is the Cooling Time Paradox:

Raw data shows: Cooling time ↑ → Scrap ↑ After causal adjustment: Cooling adequacy ↑ → Scrap ↓

This demonstrates how confounding (e.g., part weight) can mislead correlation-based analysis.

Results Baseline scrap rate: ~4.44% Predicted scrap rate after intervention: ~2.99% Improvement: ~32%

Top actionable levers include:

Cooling adequacy Mold temperature Injection pressure Dryer dew point Maintenance scheduling Visual Outputs

The code generates the following plots:

scrap_by_defect.png correlation_heatmap.png impact_chart.png cooling_paradox.png pathway_pie.png refutation_plot.png

These help interpret causal relationships and validate findings.

Counterfactual Simulator

An interactive module allows users to input proposed process settings and estimate the resulting scrap rate.

Example:

predict_scrap_change({ 'cooling_adequacy': new_value, 'mold_temperature_c': new_value })

This enables decision-makers to evaluate “what-if” scenarios before implementing changes.

Technologies Used Python pandas, numpy scikit-learn (Linear Regression) DoWhy (causal modeling) matplotlib, seaborn (visualization) Limitations Linear model assumption for causal effects Adjustment sets defined manually using DAG knowledge No full structural causal simulation Results depend on correctness of the causal graph Conclusion

This project demonstrates how causal inference can transform manufacturing analytics from descriptive insights into actionable decision-making.

Instead of asking “what is correlated with scrap?”, the framework answers:

“what should we change to reduce scrap?”

AI Usage Disclosure

AI tools were used for assistance in structuring code, refining explanations, and improving clarity of documentation.

All core components, including:

causal modeling variable classification adjustment set design implementation interpretation of results

were developed, verified, and fully understood by the author.