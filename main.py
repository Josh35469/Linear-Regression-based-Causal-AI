import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("data/synthetic_injection_molding_demo.csv")
print(df.shape)
print(df.columns.tolist())
print(df.head())
print(df.isnull().sum())
print(df['scrap_rate_pct'].describe())
print(df['defect_type'].value_counts())#check

shift_map = {
    'A_Day': 0,
    'B_Evening': 1,
    'C_Night': 2
}
#operator_shift is text (A_Day, B_Evening, C_Night).
#using a dictionary to store it as a new column called operator_shift_enc
df['operator_shift_enc'] = df['operator_shift'].map(shift_map)

#Cooling adequacy appears scaled higher than typical due to dataset units, 
#relative changes are used for intervention analysis
df['cooling_adequacy'] = df['cooling_time_s'] / (df['mold_temperature_c'] / 70.0)
print("Mean cooling_adequacy:", df['cooling_adequacy'].mean())

#This captures the ratio of cooling time to temperature demand
corr = df['cooling_adequacy'].corr(df['scrap_rate_pct'])
print("Correlation with scrap rate:", corr)

#Correlation matrix of numeric variables with KPI on numeric columns
numeric_df = df.select_dtypes(include=['number'])
corr_series = numeric_df.corrwith(df['scrap_rate_pct'])
corr_sorted = corr_series.abs().sort_values(ascending=False)
top_15 = corr_series.loc[corr_sorted.index].head(15)

print("Top 15 correlations with scrap_rate_pct (correlation, not causation)")
print(top_15.round(3))#cleaner output with rounding

# Raw correlation:cooling time vs scrap
corr_cooling = df['cooling_time_s'].corr(df['scrap_rate_pct'])

# Engineered feature correlation
corr_adequacy = df['cooling_adequacy'].corr(df['scrap_rate_pct'])

# Correlation with part weight(confounder)
corr_weight = df['cooling_time_s'].corr(df['part_weight_g'])

print("Cooling time vs scrap_rate_pct:", round(corr_cooling, 3))
print("Cooling adequacy vs scrap_rate_pct:", round(corr_adequacy, 3))
print("Cooling time vs part_weight_g:", round(corr_weight, 3))

# important
# Cooling_time_s appears correlated with scrap, but this is misleading.
# It is strongly correlated with part_weight_g (process requirement),
# meaning heavier parts naturally require longer cooling.
# Once we normalize cooling by temperature demand (cooling_adequacy),
# the relationship with scrap largely disappears.
# This demonstrates confounding, correlation does not imply causation.

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

sns.boxplot(x='defect_type', y='scrap_rate_pct', data=df)

plt.xticks(rotation=45)
plt.title("Scrap Rate by Defect Type")

plt.tight_layout()
plt.savefig("scrap_by_defect.png")
plt.show()

selected_cols=[
    'dryer_dewpoint_c',          # dryer setpoint — controls resin dryness
    'barrel_temperature_c',      # barrel heater setpoint
    'mold_temperature_c',        # mold temperature setpoint
    'injection_pressure_bar',    # injection pressure setpoint
    'hold_pressure_bar',         # hold pressure setpoint
    'screw_speed_rpm',           # screw speed setpoint
    'cooling_time_s',            # cooling duration
    'shot_size_g',               # shot size setpoint
    'maintenance_days_since_last', # scheduling lever
    'calibration_drift_index',   # controllable through maintenance
    'energy_kwh_interval',       # monitor for trade-off

]
#Plot correlation heatmap for key process variables
#why you need causal methods — the correlations are confounded.
plt.figure(figsize=(12, 8))

sns.heatmap(df[selected_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm')

plt.title("Correlation Heatmap of Process Variables")

plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

# Variables Marcus can directly tune
controllable_levers = [
    'dryer_dewpoint_c',          # dryer setpoint — controls resin dryness
    'barrel_temperature_c',      # barrel heater setpoint
    'mold_temperature_c',        # mold temperature setpoint
    'injection_pressure_bar',    # injection pressure setpoint
    'hold_pressure_bar',         # hold pressure setpoint
    'screw_speed_rpm',           # screw speed setpoint
    'cooling_time_s',            # cooling duration
    'shot_size_g',               # shot size setpoint
    'maintenance_days_since_last', # scheduling lever
    'calibration_drift_index',   # controllable through maintenance
    'energy_kwh_interval'
]

# Variables that influence both process settings and scrap (confounders)
confounders = [
    'part_weight_g',
    'product_variant',
    'mold_id',
    'cavity_count',
    'ambient_temperature_c',
    'ambient_humidity_pct',
    'resin_moisture_pct',
    'resin_batch_quality_index',
    'tool_wear_index',
    'maintenance_days_since_last',
    'operator_experience_level',
    'calibration_drift_index'
]
# Variables on the causal pathway (do not control for these)
mediators = [
    'cycle_time_s',     # often downstream of process settings
    'part_weight_g',    # can be influenced by settings in some processes
    'energy_kwh_interval'
]

# Cannot be controlled but must be accounted for
non_controllable = [
    'ambient_humidity_pct',        # weather — cannot control
    'ambient_temperature_c',       # weather — cannot control
    'operator_shift_enc',          # scheduling context
    'operator_experience_level',   # cannot change person's experience
    'resin_batch_quality_index',   # upstream supplier quality
    'tool_wear_index',             # degrades over time, not a setpoint
    'resin_moisture_pct'
]

# Print them clearly
print("Controllable Levers:", controllable_levers)
print("\nConfounders:", confounders)
print("\nMediators (dont control):", mediators)
print("\nNon-controllable:", non_controllable)

#part_weight_g appears in confounders AND mediators
#That’s intentional ambiguity.
#In injection molding:
#It’s usually a design constraint (confounder)
#But sometimes partially affected by process- mediator-like behavior


#For each controllable variable, write a list of confounders to control for.
#what are the common causes of both this variable AND scrap rate? Those go in the adjustment set
adjustment_sets = {

    'mold_temperature_c': [
        'part_weight_g',
        'product_variant',
        'mold_id',
        'ambient_humidity_pct',
        'resin_moisture_pct',
        'tool_wear_index'
    ],

    'barrel_temperature_c': [
        'resin_moisture_pct',
        'resin_batch_quality_index',
        'product_variant',
        'ambient_temperature_c'
    ],

    'injection_pressure_bar': [
        'part_weight_g',
        'product_variant',
        'mold_id',
        'tool_wear_index',
        'calibration_drift_index'
    ],

    'hold_pressure_bar': [
        'part_weight_g',
        'product_variant',
        'mold_id',
        'tool_wear_index'
    ],

    'cooling_adequacy': [
        'part_weight_g',
        'product_variant',
        'ambient_temperature_c',
        'ambient_humidity_pct'
    ],

  'dryer_dewpoint_c': [
        'ambient_humidity_pct',
        'resin_batch_quality_index',
        'operator_experience_level'
    ],
 
    'maintenance_days_since_last': [
        'operator_shift_enc',
        'operator_experience_level'
    ]
}

print("\nAdjustment Sets:")
for k, v in adjustment_sets.items():
    print(f"{k}: {v}")


#dag as graphviz string
causal_graph = """
digraph {

ambient_humidity_pct -> dryer_dewpoint_c;
dryer_dewpoint_c -> resin_moisture_pct;
resin_batch_quality_index -> resin_moisture_pct;

maintenance_days_since_last -> calibration_drift_index;
maintenance_days_since_last -> tool_wear_index;

calibration_drift_index -> injection_pressure_bar;
calibration_drift_index -> hold_pressure_bar;
calibration_drift_index -> barrel_temperature_c;
calibration_drift_index -> cooling_time_s;

tool_wear_index -> injection_pressure_bar;
tool_wear_index -> defect_type;

ambient_temperature_c -> mold_temperature_c;

mold_temperature_c -> cooling_time_s;
operator_experience_level -> cooling_time_s;

screw_speed_rpm -> injection_pressure_bar;
injection_pressure_bar -> hold_pressure_bar;

resin_moisture_pct -> defect_type;

mold_temperature_c -> defect_type;
cooling_time_s -> defect_type;

injection_pressure_bar -> defect_type;
hold_pressure_bar -> defect_type;

defect_type -> scrap_rate_pct;

resin_moisture_pct -> scrap_rate_pct;
mold_temperature_c -> scrap_rate_pct;
cooling_time_s -> scrap_rate_pct;
injection_pressure_bar -> scrap_rate_pct;

}
"""
from dowhy import CausalModel

model = CausalModel(
    data=df,
    treatment='cooling_adequacy',
    outcome='scrap_rate_pct',
    graph=causal_graph
)
model.view_model()

#function to estimate ATE
from sklearn.linear_model import LinearRegression
import pandas as pd

def estimate_ate(df, treatment, adjustment_set):
    cols = [treatment] + adjustment_set + ['scrap_rate_pct']
    data = df[cols].dropna()

    # Split X and y
    X = data[[treatment] + adjustment_set]
    y = data['scrap_rate_pct']

    # One-hot encode categorical variables
    X = pd.get_dummies(X, drop_first=True)

    model = LinearRegression()
    model.fit(X, y)

    # Treatment column might not be index 0 anymore after encoding
    # So find it explicitly:
    treatment_cols = [col for col in X.columns if col.startswith(treatment)]

    # For numeric treatment → single column
    # For categorical → multiple columns (rare here)
    coef = model.coef_[X.columns.get_loc(treatment_cols[0])]

    return coef
#loop + bootstrap Cl
ate_results = {}
n_bootstrap = 500

for treatment, adjustment_set in adjustment_sets.items():

    ate = estimate_ate(df, treatment, adjustment_set)
    boot_ates = []

    cols = [treatment] + adjustment_set + ['scrap_rate_pct']
    data = df[cols].dropna()

    n = len(data)

    for _ in range(n_bootstrap):
        # sample with replacement
        sample = data.sample(n=n, replace=True)

        # Split
        X_sample = sample[[treatment] + adjustment_set]
        y_sample = sample['scrap_rate_pct']

        # One-hot encode
        X_sample = pd.get_dummies(X_sample, drop_first=True)

        # Fit model
        model_boot = LinearRegression()
        model_boot.fit(X_sample, y_sample)

        #Get treatment coefficient safely
        if treatment in X_sample.columns:
            coef = model_boot.coef_[X_sample.columns.get_loc(treatment)]
        else:
            # fallback (in case of weird encoding edge cases)
            treatment_cols = [col for col in X_sample.columns if col.startswith(treatment)]
            coef = model_boot.coef_[X_sample.columns.get_loc(treatment_cols[0])]

        boot_ates.append(coef)

    #Confidence Interval 
    ci_low, ci_high = np.percentile(boot_ates, [2.5, 97.5])

    # store results
    ate_results[treatment] = {
        'ate': ate,
        'ci_low': ci_low,
        'ci_high': ci_high
    }

print("\nCausal Effect Estimates (ATE with 95% CI)\n")
print(f"{'Variable':<30} {'ATE':<12} {'CI Low':<12} {'CI High':<12} {'Direction'}")
print("-" * 80)

for var, res in ate_results.items():
    ate = res['ate']
    ci_low = res['ci_low']
    ci_high = res['ci_high']

    direction = "↑ increases scrap" if ate > 0 else "↓ decreases scrap"
    crosses_zero = "YES — unreliable" if (ci_low < 0 < ci_high) else "no"
    print(f"{var:<30} {ate:<12.4f} {ci_low:<12.4f} {ci_high:<12.4f} {crosses_zero:<18}  {direction}")

#lets do the prove its real step lessgooo
#First, pick top 3 by absolute ATE
# Get top 3 variables by absolute effect size

top_3 = sorted(ate_results.items(), key=lambda x: abs(x[1]['ate']), reverse=True)[:3]
top_3_vars = [var for var, _ in top_3]

def placebo_test(df, treatment, adjustment_set, n_iter=50):
    placebo_ates = []

    cols = [treatment] + adjustment_set + ['scrap_rate_pct']
    data = df[cols].dropna()

    for _ in range(n_iter):
        sample = data.copy()

        # Replace treatment with random permutation
        sample[treatment] = np.random.permutation(sample[treatment].values)

        X = sample[[treatment] + adjustment_set]
        y = sample['scrap_rate_pct']

        X = pd.get_dummies(X, drop_first=True)

        model = LinearRegression()
        model.fit(X, y)

        # extract coef safely
        if treatment in X.columns:
            coef = model.coef_[X.columns.get_loc(treatment)]
        else:
            treatment_cols = [col for col in X.columns if col.startswith(treatment)]
            coef = model.coef_[X.columns.get_loc(treatment_cols[0])]

        placebo_ates.append(coef)

    return np.mean(placebo_ates)

#subset stability test
def subset_stability_test(df, treatment, adjustment_set, n_iter=30):
    subset_ates = []

    cols = [treatment] + adjustment_set + ['scrap_rate_pct']
    data = df[cols].dropna()

    n = len(data)
    subset_size = int(0.8 * n)

    for _ in range(n_iter):
        sample = data.sample(n=subset_size, replace=False)

        X = sample[[treatment] + adjustment_set]
        y = sample['scrap_rate_pct']

        X = pd.get_dummies(X, drop_first=True)

        model = LinearRegression()
        model.fit(X, y)

        if treatment in X.columns:
            coef = model.coef_[X.columns.get_loc(treatment)]
        else:
            treatment_cols = [col for col in X.columns if col.startswith(treatment)]
            coef = model.coef_[X.columns.get_loc(treatment_cols[0])]

        subset_ates.append(coef)

    return np.std(subset_ates)
#print summary
print("\nRefutation Results\n")
print(f"{'Variable':<30} {'ATE':<10} {'Placebo Mean':<15} {'Stability Ratio':<18} {'Placebo Test':<15} {'Stability Test'}")
print("-" * 100)

for var in top_3_vars:
    ate = ate_results[var]['ate']
    adjustment_set = adjustment_sets[var]

    # Placebo
    placebo_mean = placebo_test(df, var, adjustment_set)
    placebo_pass = abs(placebo_mean) < abs(ate) * 0.2

    #Stability
    std_dev = subset_stability_test(df, var, adjustment_set)
    stability_ratio = std_dev / abs(ate) if ate != 0 else np.inf
    stability_pass = stability_ratio < 0.15

    print(f"{var:<30} {ate:<10.4f} {placebo_mean:<15.4f} {stability_ratio:<18.4f} "
          f"{'pass' if placebo_pass else 'fail':<15} {'pass' if stability_pass else 'fail'}")
    
    #compute recommendations
    #less do decision layer

    #defining current+ ranges+ recommendations

recommendations = {
    'cooling_adequacy': {
        'current_avg': df['cooling_adequacy'].mean(),
        'feasible_min': 10,
        'feasible_max': 30,
        'recommended_value': df['cooling_adequacy'].mean() * 1.10  # increase by 10%
    },
    'mold_temperature_c': {
        'current_avg': df['mold_temperature_c'].mean(),
        'feasible_min': 35,
        'feasible_max': 95,
        'recommended_value': df['mold_temperature_c'].mean() * 0.90  # decrease by 10%
    },
    'injection_pressure_bar': {
        'current_avg': df['injection_pressure_bar'].mean(),
        'feasible_min': 450,
        'feasible_max': 1700,
        'recommended_value': df['injection_pressure_bar'].mean() * 0.90  # decrease by 10%
    },
    'hold_pressure_bar': {
        'current_avg': df['hold_pressure_bar'].mean(),
        'feasible_min': 300,
        'feasible_max': 1200,
        'recommended_value': df['hold_pressure_bar'].mean() * 0.90  # decrease by 10%
    },
    'barrel_temperature_c': {
        'current_avg': df['barrel_temperature_c'].mean(),
        'feasible_min': 195,
        'feasible_max': 290,
        'recommended_value': df['barrel_temperature_c'].mean() * 1.10  # increase by 10% (ATE is negative)
    },
    'dryer_dewpoint_c': {
        'current_avg': df['dryer_dewpoint_c'].mean(),
        'feasible_min': -48,
        'feasible_max': -18,
        'recommended_value': df['dryer_dewpoint_c'].mean() * 1.06  # move 6% more negative (lower dewpoint)
    },
    'maintenance_days_since_last': {
        'current_avg': df['maintenance_days_since_last'].mean(),
        'feasible_min': 0,
        'feasible_max': 60,
        'recommended_value': df['maintenance_days_since_last'].mean() * 0.90  # decrease by 10%
    }
}

# Impact + % change 
results = []

for var, config in recommendations.items():
    if var not in ate_results:
        continue

    ate = ate_results[var]['ate']
    ci_low = ate_results[var]['ci_low']
    ci_high = ate_results[var]['ci_high']

    if ci_low < 0 < ci_high:
        print(f"Skipping {var} — CI crosses zero, effect not reliable")
        continue

    current = config['current_avg']
    recommended = config['recommended_value']

    delta_units = recommended - current
 
    # expected scrap change = ate * change in variable (in original units)
    expected_impact = ate * delta_units
 
    # % change for display
    delta_pct = ((recommended - current) / abs(current)) * 100
 
    results.append({
        'variable': var,
        'current': current,
        'recommended': recommended,
        'delta_pct': delta_pct,
        'impact': expected_impact,
        'ci_low': ci_low,
        'ci_high': ci_high
    })


#sort by absolute impact
results_sorted = sorted(results, key=lambda x: abs(x['impact']), reverse=True)

# recommendation table
print("\nMarcus's intervention results\n")
print(f"{'Rank':<5} {'Variable':<30} {'Current':<12} {'Recommended':<15} {'Change %':<12} {'Δ Scrap':<12} {'95% CI(ATE)'}")
print("-" * 105)

for i, r in enumerate(results_sorted, 1):
    print(f"{i:<5} {r['variable']:<30} "
          f"{r['current']:<12.2f} {r['recommended']:<15.2f} "
          f"{r['delta_pct']:<12.2f} {r['impact']:<12.4f} "
          f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]")


# summary
baseline_scrap = df['scrap_rate_pct'].mean()

total_impact = sum(r['impact'] for r in results_sorted)

new_scrap = baseline_scrap + total_impact

improvement_pct = (total_impact / baseline_scrap) * 100 if baseline_scrap != 0 else 0

print("\nSUMMARY\n")
print(f"Baseline Scrap Rate: {baseline_scrap:.4f}")
print(f"Net Impact on Scrap: {total_impact:.4f}")  
print(f"New Scrap Rate: {new_scrap:.4f}")
print(f"Relative Change: {improvement_pct:.2f}%")

#split data by dominant defect type
warpage_df = df[df['defect_type'] == 'warpage']
flash_df = df[df['defect_type'] == 'flash']
splay_df = df[df['defect_type'] == 'splay_moisture']

print("Mean scrap rate by defect type:")
print("Warpage:", warpage_df['scrap_rate_pct'].mean())
print("Flash:", flash_df['scrap_rate_pct'].mean())
print("Splay:", splay_df['scrap_rate_pct'].mean())

#estimate causal effects seperately per pathway
# Warpage pathway
warpage_temp_ate = estimate_ate(warpage_df, 'mold_temperature_c', adjustment_sets['mold_temperature_c'])
warpage_cooling_ate = estimate_ate(warpage_df, 'cooling_adequacy', adjustment_sets['cooling_adequacy'])

# Flash pathway
flash_pressure_ate = estimate_ate(flash_df, 'injection_pressure_bar', adjustment_sets['injection_pressure_bar'])

# Splay pathway
splay_dryer_ate = estimate_ate(splay_df, 'dryer_dewpoint_c', adjustment_sets['dryer_dewpoint_c'])

print("\nPathway-specific ATEs:")
print("Warpage - Mold Temp:", warpage_temp_ate)
print("Warpage - Cooling Adequacy:", warpage_cooling_ate)
print("Flash - Injection Pressure:", flash_pressure_ate)
print("Splay - Cooling Time:", splay_dryer_ate)

# pathway attribution
total_count = len(df)

def pathway_contribution(group):
    return (group['scrap_rate_pct'].mean() * len(group)) / (df['scrap_rate_pct'].mean() * total_count)

print("\nPathway Contributions (% of total scrap):")
print("Warpage:", pathway_contribution(warpage_df) * 100)
print("Flash:", pathway_contribution(flash_df) * 100)
print("Splay:", pathway_contribution(splay_df) * 100)

#humidity split
low_hum = df[df['ambient_humidity_pct'] < df['ambient_humidity_pct'].median()]
high_hum = df[df['ambient_humidity_pct'] >= df['ambient_humidity_pct'].median()]

low_ate = estimate_ate(low_hum, 'cooling_adequacy', adjustment_sets['cooling_adequacy'])
high_ate = estimate_ate(high_hum, 'cooling_adequacy', adjustment_sets['cooling_adequacy'])

print("\nCATE - Cooling Time Effect by Humidity:")
print("Low humidity:", low_ate)
print("High humidity:", high_ate)

#machine split
machines = df['machine_id'].unique()[:2]

for m in machines:
    m_df = df[df['machine_id'] == m]
    ate = estimate_ate(m_df, 'mold_temperature_c', adjustment_sets['mold_temperature_c'])
    print(f"Effect of mold_temperature_c on {m}: {ate}")

    #cate summary
print("\nCconditonal effects—when do interventions matter most?\n")

print("Cooling Adequacy:")
print(" - Low humidity effect:", round(low_ate,4))
print(" - High humidity effect:", round(high_ate,4))
print(" - Interpretation: cooling adequacy has a slightly stronger effect under high humidity")
print("   (high humidity increases resin moisture risk, making cooling control more critical)")
for m in machines:
    m_df = df[df['machine_id'] == m]
    ate = estimate_ate(m_df, 'mold_temperature_c', adjustment_sets['mold_temperature_c'])
    print(f"Mold temperature effect on {m}: {round(ate,4)}")

#impact bar chart
import matplotlib.pyplot as plt

vars_ = [r['variable'] for r in results_sorted]
impacts = [r['impact'] for r in results_sorted]

plt.figure(figsize=(10,6))
plt.barh(vars_, impacts,color='steelblue')
plt.axvline(0,color='black',linewidth=0.8)
plt.xlabel("Expected change in scrap_rate_pct (%)")
plt.title("Causal Effect of Interventions on Scrap Rate")

plt.tight_layout()
plt.savefig("impact_chart.png")
plt.show()

#cooling paradox plot
import numpy as np

fig, ax = plt.subplots(1, 2, figsize=(12,5))

# Left: raw cooling time
x = df['cooling_time_s']
y = df['scrap_rate_pct']
ax[0].scatter(x, y,alpha=0.3,s=8)
ax[0].set_title("Cooling Time vs Scrap\n(raw — misleading positive correlation)")
ax[0].set_xlabel("cooling_time_s")
ax[0].set_ylabel("scrap_rate_pct")

m, b = np.polyfit(x.dropna(), y.loc[x.dropna().index], 1)
x_line = np.linspace(x.min(), x.max(), 100)
ax[0].plot(x_line, m * x_line + b, color='red', linewidth=2)
# Right: cooling adequacy— shows true relationship after removing confounder

x2 = df['cooling_adequacy']
ax[1].scatter(x2, y,alpha=0.3, s=8)
ax[1].set_title("Cooling Adequacy vs Scrap\n(after removing part_weight confounder)")
ax[1].set_xlabel("cooling_adequacy (cooling_time / mold_temp demand)")
ax[1].set_ylabel("scrap_rate_pct")
 
m2, b2 = np.polyfit(x2.dropna(), y.loc[x2.dropna().index], 1)
x2_line = np.linspace(x2.min(), x2.max(), 100)
ax[1].plot(x2_line, m2 * x2_line + b2, color='red', linewidth=2)

plt.tight_layout()
plt.savefig("cooling_paradox.png")
plt.show()

#pie chart
counts = df['defect_type'].value_counts()

plt.figure(figsize=(6,6))
plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
plt.title("Pathway Attribution by Defect Type")

plt.savefig("pathway_pie.png")
plt.show()

#refutation plot
placebo_means_for_plot = []
for var in top_3_vars:
    pm = placebo_test(df, var, adjustment_sets[var], n_iter=30)
    placebo_means_for_plot.append(pm)
 
orig = [ate_results[v]['ate'] for v in top_3_vars]
 
x = np.arange(len(top_3_vars))
 
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - 0.2, orig, 0.4, label='Original ATE', color='steelblue')
ax.bar(x + 0.2, placebo_means_for_plot, 0.4, label='Placebo ATE (should be ~0)', color='orange')
 
ax.set_xticks(x)
ax.set_xticklabels(top_3_vars, rotation=10)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel("ATE value")
ax.set_title("Refutation Test: Original vs Placebo ATE")
 
# annotate placebo values so they're visible even when small
for i, pm in enumerate(placebo_means_for_plot):
    ax.annotate(f'{pm:.4f}', xy=(x[i] + 0.2, pm), ha='center',
                va='bottom' if pm >= 0 else 'top', fontsize=9, color='darkorange')
 
ax.legend()
plt.tight_layout()
plt.savefig("refutation_plot.png")
plt.show()
 
# counterfactual simulator
# Marcus types in proposed setpoints, gets predicted scrap rate
 
def predict_scrap_change(proposed_settings):
    # proposed_settings: dict of {variable: new_value}
    # returns predicted new scrap rate
 
    baseline = df['scrap_rate_pct'].mean()
    total_effect = 0
 
    for var, new_val in proposed_settings.items():
        if var not in ate_results:
            print(f"  Warning: {var} not in ATE results, skipping")
            continue
        if var not in recommendations:
            continue
 
        ate = ate_results[var]['ate']
        current = recommendations[var]['current_avg']
        delta_units = new_val - current
        effect = ate * delta_units
        total_effect += effect
 
    return baseline + total_effect
 
# example scenarios
print("\n Counterfactual Scenarios\n")
 
scenario_a = {'cooling_adequacy': df['cooling_adequacy'].mean() * 1.10}
scenario_b = {
    'cooling_adequacy': df['cooling_adequacy'].mean() * 1.10,
    'mold_temperature_c': df['mold_temperature_c'].mean() * 0.90
}
scenario_c = {var: config['recommended_value'] for var, config in recommendations.items()}
 
print(f"Scenario A (cooling only):         {predict_scrap_change(scenario_a):.4f}%")
print(f"Scenario B (cooling + mold temp):  {predict_scrap_change(scenario_b):.4f}%")
print(f"Scenario C (all interventions):    {predict_scrap_change(scenario_c):.4f}%")
print(f"Baseline:                          {df['scrap_rate_pct'].mean():.4f}%")
 
# interactive mode
if __name__ == "__main__":
    print("\n--- Interactive Simulator ---")
    print("Enter proposed values for each variable (press Enter to keep current avg):\n")
 
    proposed = {}
    for var, config in recommendations.items():
        current = config['current_avg']
        user_input = input(f"  {var} (current avg = {current:.2f}): ").strip()
        if user_input != "":
            try:
                proposed[var] = float(user_input)
            except ValueError:
                print(f"  Invalid input for {var}, keeping current")
                proposed[var] = current
        else:
            proposed[var] = current
 
    predicted = predict_scrap_change(proposed)
    print(f"\nPredicted scrap rate: {predicted:.4f}%")
    print(f"Baseline scrap rate:  {df['scrap_rate_pct'].mean():.4f}%")
    print(f"Expected change:      {predicted - df['scrap_rate_pct'].mean():+.4f}%")