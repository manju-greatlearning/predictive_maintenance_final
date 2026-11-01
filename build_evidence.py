"""Create lightweight evidence images for final report placeholders."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def save_text_image(filename, title, lines, figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.text(0.03, 0.94, title, fontsize=18, fontweight="bold", color="#163752", va="top")
    y = 0.84
    for line in lines:
        ax.text(0.05, y, line, fontsize=12, family="monospace", color="#243b4a", va="top")
        y -= 0.055
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)

save_text_image(
    "fig_repo_structure.png",
    "GitHub Repository Folder Structure",
    [
        "predictive_maintenance_final/",
        "├── .github/workflows/pipeline.yml",
        "├── app.py                         # Streamlit app",
        "├── data/ai4i2020.csv              # registered dataset",
        "├── data/train.csv, data/test.csv  # workflow artifacts",
        "├── models/best_model.joblib       # committed model",
        "├── models/metrics.json            # experiment tracking log",
        "├── notebooks/Predictive_Maintenance_Final_Notebook.ipynb",
        "├── reports/figures/",
        "├── src/validate_dataset.py",
        "├── src/prepare_data.py",
        "└── src/train.py",
    ],
)

fig, ax = plt.subplots(figsize=(11, 3.2))
ax.axis("off")
steps = ["Validate\ndataset", "Prepare\ndata", "Upload\nartifact", "Train + tune\nmodels", "Smoke test\nStreamlit", "Commit\noutputs"]
for i, step in enumerate(steps):
    x = 0.03 + i * 0.16
    box = FancyBboxPatch((x, 0.35), 0.12, 0.32, boxstyle="round,pad=0.02", fc="#e9f2f0", ec="#167478", lw=1.5)
    ax.add_patch(box)
    ax.text(x + 0.06, 0.51, step, ha="center", va="center", fontsize=10, color="#163752")
    if i < len(steps) - 1:
        ax.annotate("", xy=(x + 0.155, 0.51), xytext=(x + 0.125, 0.51), arrowprops=dict(arrowstyle="->", color="#167478", lw=1.5))
ax.text(0.03, 0.88, "Automated GitHub Actions Workflow", fontsize=17, fontweight="bold", color="#163752")
fig.savefig(OUT / "fig_workflow_diagram.png", dpi=160, bbox_inches="tight")
plt.close(fig)

save_text_image(
    "fig_workflow_status.png",
    "GitHub Actions Evidence",
    [
        "Status after push: to be verified after pushing project-18 repository.",
        "Expected jobs:",
        "  ✓ validate-prepare",
        "  ✓ train-evaluate-deploy-check",
        "Expected artifacts:",
        "  • train-test-splits",
        "  • trained-model-and-metrics",
        "Expected commit behavior:",
        "  • Generated train/test/model/metrics are committed back to main if changed.",
    ],
)

save_text_image(
    "fig_streamlit_app.png",
    "Streamlit Community Cloud Evidence",
    [
        "Deployment app file: app.py",
        "Dependency file: requirements.txt",
        "Main model loaded: models/best_model.joblib",
        "Inputs captured into a dataframe:",
        "  Type, Air_temperature, Process_temperature, Rotational_speed, Torque, Tool_wear",
        "Output displayed:",
        "  Failure probability and low/high risk recommendation",
        "Deployment URL: add after Streamlit Community Cloud deployment.",
    ],
)

print(f"Evidence figures written to {OUT}")
