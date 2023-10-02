import pandas as pd
import plotly.express as px
from policyengine_core.charts import format_fig, BLUE, GRAY, DARK_GRAY
import streamlit as st

st.title("PolicyEngine US calibration")

training_log_puf_extended_cps = pd.read_csv(
    "training_log.csv.gz", compression="gzip"
)
training_log_cps = pd.read_csv("training_log_cps.csv.gz", compression="gzip")
calibration_final_results = pd.read_csv(
    "calibration_final_results.csv.gz", compression="gzip"
)

training_log_puf_extended_cps["Source dataset"] = "PUF-extended CPS"
training_log_cps["Source dataset"] = "CPS"

training_log = pd.concat([training_log_puf_extended_cps, training_log_cps])
training_log_targets = training_log.copy()
training_log_targets["value"] = training_log_targets["target"]
training_log_targets["Source dataset"] = "Official"
training_log = pd.concat([training_log, training_log_targets])

names = st.multiselect(
    "Metric",
    [x for x in list(training_log.name.unique()) if "population" not in x],
    default=["employment income aggregate", "federal income tax", "SSI"],
)

if len(names) == 0:
    names = ["employment income aggregate", "federal income tax", "SSI"]

performance_df = pd.DataFrame() # columns: [name, source_dataset, deviation]

for name in names:
    official_value = calibration_final_results[
        (calibration_final_results.Variable == name)
        & (calibration_final_results["Source dataset"] == "Official")
    ].Value.iloc[0]
    calibrated_value = calibration_final_results[
        (calibration_final_results.Variable == name)
        & (calibration_final_results["Source dataset"] == "Calibrated CPS")
    ].Value.iloc[0]
    enhanced_value = calibration_final_results[
        (calibration_final_results.Variable == name)
        & (calibration_final_results["Source dataset"] == "Enhanced CPS")
    ].Value.iloc[0]
    cps_value = calibration_final_results[
        (calibration_final_results.Variable == name)
        & (calibration_final_results["Source dataset"] == "CPS")
    ].Value.iloc[0]
    performance_df = pd.concat([
        performance_df,
        pd.DataFrame({
            "Metric": [name] * 3,
            "Source dataset": ["CPS", "Calibrated CPS", "Enhanced CPS"],
            "Deviation": [
                abs((cps_value - official_value) / official_value),
                abs((calibrated_value - official_value) / official_value),
                abs((enhanced_value - official_value) / official_value),
            ]
        })
    ])

performance_df = performance_df.sort_values("Source dataset", ascending=False)
performance_df["Deviation text"] = performance_df.Deviation.apply(lambda x: f"{x:.1%}")
performance_df.Deviation = performance_df.Deviation.apply(lambda x: max(x, 0.0049))

def capitalise(string):
    return string[0].upper() + string[1:]

performance_df["Metric"] = performance_df["Metric"].apply(capitalise)

fig = px.bar(
    performance_df,
    y="Metric",
    x="Deviation",
    color="Source dataset",
    barmode="group",
    color_discrete_map={
        "Enhanced CPS": BLUE,
        "Calibrated CPS": GRAY,
        "CPS": DARK_GRAY,
    },
    text=performance_df["Deviation text"]
)
fig = format_fig(fig.update_layout(
    title="Deviation from official values, by dataset",
    xaxis_tickformat=".0%",
    xaxis_range=[0, 1],
    # min font size for text
    uniformtext_minsize=12,
    legend_traceorder="reversed",
))
fig

fig = px.scatter(
    performance_df,
    y="Metric",
    x="Deviation",
    color="Source dataset",
    #barmode="group",
    #color_discrete_map={
    #    "Enhanced CPS": BLUE,
    #    "Calibrated CPS": GRAY,
    #    "CPS": DARK_GRAY,
    #},
)
fig

total_loss_log = training_log[training_log.name == name]

fig = px.line(
    total_loss_log,
    x="epoch",
    y="value",
    color="Source dataset",
    color_discrete_map={
        "PUF-extended CPS": BLUE,
        "CPS": GRAY,
        "Official": DARK_GRAY,
    },
)

fig = format_fig(fig)
fig.update_layout(
    title=f"{name} during reweighting, by source dataset",
    yaxis_title="Relative loss change",
    xaxis_title="Epoch",
    legend_title="",
)

# Add annotations for final values
for source_dataset in ["PUF-extended CPS", "CPS", "Official"]:
    final_value = total_loss_log[
        total_loss_log["Source dataset"] == source_dataset
    ].iloc[-1]["value"]
    fig.add_annotation(
        x=total_loss_log["epoch"].max(),
        y=final_value,
        text=f"{final_value:,.0f}",
        arrowhead=1,
        yshift=10,
        ax=0,
        ay=-20,
        bgcolor="white",
    )

fig


def plot_metric(name):
    fig = px.bar(
        calibration_final_results[
            calibration_final_results.Variable == name
        ].sort_values("Value"),
        x="Source dataset",
        y="Value",
        color_discrete_map={
            "Enhanced CPS": BLUE,
            "Calibrated CPS": GRAY,
            "CPS": GRAY,
            "Official": DARK_GRAY,
        },
        color="Source dataset",
    )
    fig = format_fig(fig).update_layout(
        title=f"{name} after reweighting, by source dataset",
    )
    return fig


st.plotly_chart(plot_metric(name))
