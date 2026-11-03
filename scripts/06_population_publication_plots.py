"""
06_population_publication_plots.py
Publication-style plotting for ISIMIP3a histsoc total population.

Creates:
1) Absolute population with SAME y-axis across Jolo, Masinloc, Palapag.
2) Relative population growth indexed to 1900 = 100.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

ROOT = Path(r"D:\temporal_environmental_data")
INPUT = ROOT / "03_extracted" / "population_ISIMIP3a_5arcmin_annual_1900_2018.csv"
FIG_DIR = ROOT / "05_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SITES = ["Jolo", "Masinloc", "Palapag"]
BUFFERS = [25, 50, 70]

df = pd.read_csv(INPUT)

required = {"site","year","population_25km","population_50km","population_70km"}
missing = required - set(df.columns)
if missing:
    raise RuntimeError(f"Missing columns: {sorted(missing)}")

def pop_fmt(x, pos):
    if abs(x) < 1e-9:
        return "0"
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:.1f} M"
    if abs(x) >= 1_000:
        return f"{x/1_000:.0f}k"
    return f"{x:.0f}"

def nice_ymax(x):
    step = 500_000
    return math.ceil(x / step) * step

# =========================
# Figure 1: absolute values
# =========================
pop_cols = [f"population_{b}km" for b in BUFFERS]
global_max = float(df[pop_cols].max().max())
shared_ymax = nice_ymax(global_max * 1.04)

fig, axes = plt.subplots(3, 1, figsize=(9, 9.2), sharex=True, sharey=True)

for ax, site, panel in zip(axes, SITES, ["A","B","C"]):
    sdf = df[df["site"] == site].sort_values("year")

    for b in BUFFERS:
        ax.plot(
            sdf["year"],
            sdf[f"population_{b}km"],
            linewidth=1.9,
            label=f"{b} km",
        )

    ax.set_ylim(0, shared_ymax)
    ax.yaxis.set_major_formatter(FuncFormatter(pop_fmt))
    if shared_ymax >= 2_000_000:
        ax.yaxis.set_major_locator(MultipleLocator(500_000))

    ax.grid(alpha=0.20, linewidth=0.7)
    ax.set_title(site, fontsize=12, pad=5)
    ax.set_ylabel("Population (people)", fontsize=10)
    ax.text(
        0.01, 0.94, panel,
        transform=ax.transAxes,
        va="top",
        fontweight="bold",
        fontsize=12,
    )

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.945),
    ncol=3,
    frameon=False,
)

axes[-1].set_xlabel("Year", fontsize=10)
axes[-1].set_xlim(df["year"].min(), df["year"].max())
axes[-1].set_xticks([1900,1920,1940,1960,1980,2000,2018])

fig.suptitle(
    "Historical population around Philippine marine sites",
    fontsize=14,
    y=0.995,
)
fig.text(
    0.5, 0.965,
    "ISIMIP3a histsoc total population, 5 arcmin, 1900–2018",
    ha="center",
    fontsize=10,
)

fig.subplots_adjust(top=0.90, bottom=0.08, left=0.12, right=0.98, hspace=0.23)

main_png = FIG_DIR / "Fig_population_ISIMIP3a_COMMON_Y_1900_2018.png"
main_pdf = FIG_DIR / "Fig_population_ISIMIP3a_COMMON_Y_1900_2018.pdf"

fig.savefig(main_png, dpi=400, bbox_inches="tight")
fig.savefig(main_pdf, bbox_inches="tight")
plt.close(fig)

# ==================================
# Figure 2: relative change, 1900=100
# ==================================
indexed = df.copy()

for b in BUFFERS:
    col = f"population_{b}km"
    out = f"index_{b}km"
    indexed[out] = np.nan

    for site in SITES:
        m = indexed["site"] == site
        base_rows = indexed.loc[m & (indexed["year"] == 1900), col]
        if base_rows.empty:
            continue
        base = float(base_rows.iloc[0])
        if np.isfinite(base) and base > 0:
            indexed.loc[m, out] = indexed.loc[m, col] / base * 100.0

index_cols = [f"index_{b}km" for b in BUFFERS]
idx_max = float(indexed[index_cols].max().max())
idx_ymax = math.ceil(idx_max / 500) * 500

fig, axes = plt.subplots(3, 1, figsize=(9, 9.2), sharex=True, sharey=True)

for ax, site, panel in zip(axes, SITES, ["A","B","C"]):
    sdf = indexed[indexed["site"] == site].sort_values("year")

    for b in BUFFERS:
        y = sdf[f"index_{b}km"]
        if y.notna().any():
            ax.plot(sdf["year"], y, linewidth=1.9, label=f"{b} km")

    ax.axhline(100, linestyle="--", linewidth=0.9, alpha=0.6)
    ax.set_ylim(0, idx_ymax)
    ax.grid(alpha=0.20, linewidth=0.7)
    ax.set_title(site, fontsize=12, pad=5)
    ax.set_ylabel("Population index\n(1900 = 100)", fontsize=10)
    ax.text(
        0.01, 0.94, panel,
        transform=ax.transAxes,
        va="top",
        fontweight="bold",
        fontsize=12,
    )

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.945),
    ncol=3,
    frameon=False,
)

axes[-1].set_xlabel("Year", fontsize=10)
axes[-1].set_xlim(df["year"].min(), df["year"].max())
axes[-1].set_xticks([1900,1920,1940,1960,1980,2000,2018])

fig.suptitle(
    "Relative population growth around Philippine marine sites",
    fontsize=14,
    y=0.995,
)
fig.text(
    0.5, 0.965,
    "ISIMIP3a histsoc total population; each site–buffer series indexed to 1900 = 100",
    ha="center",
    fontsize=10,
)

fig.subplots_adjust(top=0.90, bottom=0.08, left=0.13, right=0.98, hspace=0.23)

idx_png = FIG_DIR / "Fig_population_ISIMIP3a_INDEX_1900eq100.png"
idx_pdf = FIG_DIR / "Fig_population_ISIMIP3a_INDEX_1900eq100.pdf"

fig.savefig(idx_png, dpi=400, bbox_inches="tight")
fig.savefig(idx_pdf, bbox_inches="tight")
plt.close(fig)

print("="*78)
print("PUBLICATION PLOTS COMPLETE")
print("="*78)
print(f"Maximum population: {global_max:,.0f}")
print(f"Common y-axis maximum: {shared_ymax:,.0f}")
print(f"Main PNG: {main_png}")
print(f"Main PDF: {main_pdf}")
print(f"Indexed PNG: {idx_png}")
print(f"Indexed PDF: {idx_pdf}")
