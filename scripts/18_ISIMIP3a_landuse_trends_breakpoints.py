"""
SCRIPT 18
FORMAL TREND + CHANGE-POINT ANALYSIS
ISIMIP3a HISTORICAL LAND USE, 1900–2018

Input
-----
D:\\temporal_environmental_data\\03_extracted\\
landuse_ISIMIP3a_buffers_1900_2018.csv

Analyses
--------
1. Ordinary least-squares trend
2. Sen's slope
3. Mann-Kendall monotonic trend test
4. 1900 versus 2018 change
5. 1900–1919 versus 1999–2018 mean change
6. Approximate structural breakpoint using segmented linear regression
7. Spatial sensitivity across 25, 50 and 70 km

IMPORTANT
---------
ISIMIP3a histsoc land use is a historical reconstruction,
not direct annual observation.

The breakpoint analysis is exploratory. It identifies the year at which
a two-segment linear model gives the strongest improvement over a single
linear trend. It should NOT be interpreted as proof of a historical
causal event.
"""

from pathlib import Path
import math
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats


warnings.filterwarnings("ignore", category=RuntimeWarning)


# ======================================================================
# PATHS
# ======================================================================

ROOT = Path(r"D:\temporal_environmental_data")

INPUT = (
    ROOT
    / "03_extracted"
    / "landuse_ISIMIP3a_buffers_1900_2018.csv"
)

TABLES = ROOT / "04_tables"
FIGURES = ROOT / "05_figures"
LOGS = ROOT / "07_logs"

for directory in [
    TABLES,
    FIGURES,
    LOGS
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ======================================================================
# OUTPUTS
# ======================================================================

OUT_TRENDS = (
    TABLES
    / "landuse_ISIMIP3a_trend_statistics.csv"
)

OUT_CHANGE = (
    TABLES
    / "landuse_ISIMIP3a_period_change_statistics.csv"
)

OUT_BREAKPOINTS = (
    TABLES
    / "landuse_ISIMIP3a_breakpoint_analysis.csv"
)

OUT_SENSITIVITY = (
    TABLES
    / "landuse_ISIMIP3a_trend_buffer_sensitivity.csv"
)

OUT_PRIMARY = (
    TABLES
    / "landuse_ISIMIP3a_primary_70km_statistics.csv"
)

FIG_TRENDS_PNG = (
    FIGURES
    / "Fig_ISIMIP3a_landuse_trends_70km.png"
)

FIG_TRENDS_PDF = (
    FIGURES
    / "Fig_ISIMIP3a_landuse_trends_70km.pdf"
)

FIG_SLOPES_PNG = (
    FIGURES
    / "Fig_ISIMIP3a_landuse_Sen_slopes_70km.png"
)

FIG_SLOPES_PDF = (
    FIGURES
    / "Fig_ISIMIP3a_landuse_Sen_slopes_70km.pdf"
)

FIG_BREAKPOINTS_PNG = (
    FIGURES
    / "Fig_ISIMIP3a_landuse_breakpoints_70km.png"
)

FIG_BREAKPOINTS_PDF = (
    FIGURES
    / "Fig_ISIMIP3a_landuse_breakpoints_70km.pdf"
)

OUT_LOG = (
    LOGS
    / "18_ISIMIP3a_landuse_trends_breakpoints.log"
)


# ======================================================================
# LOGGING
# ======================================================================

log_lines = []


def log(message=""):

    print(message)

    log_lines.append(
        str(message)
    )


# ======================================================================
# PLOT SETTINGS
# ======================================================================

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# ======================================================================
# VARIABLES
# ======================================================================

VARIABLES = {

    "cropland_total_percent_of_buffer":
        "Cropland",

    "pastures_percent_of_buffer":
        "Pasture",

    "forests_natural_percent_of_buffer":
        "Forest + natural vegetation",

    "urban_percent_of_buffer":
        "Urban area"
}


SITES = [
    "Jolo",
    "Masinloc",
    "Palapag"
]

BUFFERS = [
    25,
    50,
    70
]


# ======================================================================
# LOAD
# ======================================================================

log("=" * 78)

log(
    "SCRIPT 18 — ISIMIP3a LAND-USE "
    "TRENDS + BREAKPOINTS"
)

log("=" * 78)


if not INPUT.exists():

    sys.exit(
        "\nInput file not found:\n"
        f"{INPUT}\n\n"
        "Run Script 16 first."
    )


df = pd.read_csv(
    INPUT
)


log("")

log(
    f"Loaded: {INPUT.name}"
)

log(
    f"Rows: {len(df):,}"
)


# ======================================================================
# INPUT QC
# ======================================================================

required = [
    "site",
    "year",
    "buffer_km",
    *VARIABLES.keys()
]


missing = [
    x
    for x in required
    if x not in df.columns
]


if missing:

    sys.exit(
        "\nMissing columns:\n"
        + "\n".join(missing)
    )


df["year"] = pd.to_numeric(
    df["year"],
    errors="raise"
).astype(int)


df["buffer_km"] = pd.to_numeric(
    df["buffer_km"],
    errors="raise"
).astype(int)


expected_years = np.arange(
    1900,
    2019
)


for site in SITES:

    for radius in BUFFERS:

        sub = df[
            (df["site"] == site)
            &
            (df["buffer_km"] == radius)
        ]

        years = np.sort(
            sub["year"].unique()
        )

        if not np.array_equal(
            years,
            expected_years
        ):

            raise ValueError(
                f"Year QC failed: "
                f"{site}, {radius} km"
            )


log("")

log("INPUT QC PASSED")

log("-" * 78)

log(
    "3 sites × 3 buffers × 119 years"
)

log(
    "Expected rows = 1,071"
)

log(
    f"Actual rows   = {len(df):,}"
)


# ======================================================================
# MANN-KENDALL
# ======================================================================

def mann_kendall(values):

    """
    Mann-Kendall monotonic trend test.

    Includes tie correction in variance.

    Returns:
        S
        variance
        Z
        two-sided p-value
        Kendall tau
        direction
    """

    x = np.asarray(
        values,
        dtype=float
    )

    x = x[
        np.isfinite(x)
    ]

    n = len(x)

    if n < 3:

        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            "insufficient"
        )


    S = 0


    for k in range(
        n - 1
    ):

        S += np.sum(
            np.sign(
                x[
                    k + 1:
                ]
                -
                x[k]
            )
        )


    unique_values, counts = np.unique(
        x,
        return_counts=True
    )


    tie_term = np.sum(
        counts
        *
        (
            counts - 1
        )
        *
        (
            2 * counts + 5
        )
    )


    variance = (
        n
        *
        (
            n - 1
        )
        *
        (
            2 * n + 5
        )
        -
        tie_term
    ) / 18.0


    if variance <= 0:

        z = 0.0

    elif S > 0:

        z = (
            S - 1
        ) / np.sqrt(
            variance
        )

    elif S < 0:

        z = (
            S + 1
        ) / np.sqrt(
            variance
        )

    else:

        z = 0.0


    p = 2 * (
        1
        -
        stats.norm.cdf(
            abs(z)
        )
    )


    tau_denominator = (
        0.5
        *
        n
        *
        (
            n - 1
        )
    )


    tau = (
        S
        /
        tau_denominator
    )


    if p < 0.05:

        if z > 0:

            direction = (
                "significant increase"
            )

        elif z < 0:

            direction = (
                "significant decrease"
            )

        else:

            direction = (
                "no monotonic trend"
            )

    else:

        direction = (
            "no significant monotonic trend"
        )


    return (
        float(S),
        float(variance),
        float(z),
        float(p),
        float(tau),
        direction
    )


# ======================================================================
# SEN'S SLOPE
# ======================================================================

def sens_slope(
    years,
    values
):

    x = np.asarray(
        years,
        dtype=float
    )

    y = np.asarray(
        values,
        dtype=float
    )


    mask = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[mask]
    y = y[mask]


    result = stats.theilslopes(
        y,
        x,
        alpha=0.95
    )


    return {
        "sen_slope_per_year":
            float(result.slope),

        "sen_slope_per_decade":
            float(result.slope * 10),

        "sen_intercept":
            float(result.intercept),

        "sen_ci95_low_per_year":
            float(result.low_slope),

        "sen_ci95_high_per_year":
            float(result.high_slope),

        "sen_ci95_low_per_decade":
            float(result.low_slope * 10),

        "sen_ci95_high_per_decade":
            float(result.high_slope * 10)
    }


# ======================================================================
# OLS
# ======================================================================

def ols_statistics(
    years,
    values
):

    x = np.asarray(
        years,
        dtype=float
    )

    y = np.asarray(
        values,
        dtype=float
    )


    mask = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[mask]
    y = y[mask]


    result = stats.linregress(
        x,
        y
    )


    n = len(x)

    df_resid = (
        n - 2
    )


    tcrit = stats.t.ppf(
        0.975,
        df_resid
    )


    ci_low = (
        result.slope
        -
        tcrit
        *
        result.stderr
    )


    ci_high = (
        result.slope
        +
        tcrit
        *
        result.stderr
    )


    return {

        "ols_slope_per_year":
            float(result.slope),

        "ols_slope_per_decade":
            float(result.slope * 10),

        "ols_ci95_low_per_year":
            float(ci_low),

        "ols_ci95_high_per_year":
            float(ci_high),

        "ols_ci95_low_per_decade":
            float(ci_low * 10),

        "ols_ci95_high_per_decade":
            float(ci_high * 10),

        "ols_intercept":
            float(result.intercept),

        "ols_r2":
            float(result.rvalue ** 2),

        "ols_p":
            float(result.pvalue),

        "ols_stderr":
            float(result.stderr)
    }


# ======================================================================
# PERIOD CHANGE
# ======================================================================

def period_change(
    years,
    values
):

    x = np.asarray(
        years,
        dtype=int
    )

    y = np.asarray(
        values,
        dtype=float
    )


    v1900 = float(
        y[
            x == 1900
        ][0]
    )

    v2018 = float(
        y[
            x == 2018
        ][0]
    )


    early = y[
        (
            x >= 1900
        )
        &
        (
            x <= 1919
        )
    ]


    late = y[
        (
            x >= 1999
        )
        &
        (
            x <= 2018
        )
    ]


    early_mean = float(
        np.mean(
            early
        )
    )

    late_mean = float(
        np.mean(
            late
        )
    )


    early_sd = float(
        np.std(
            early,
            ddof=1
        )
    )


    endpoint_change = (
        v2018
        -
        v1900
    )


    period_change_value = (
        late_mean
        -
        early_mean
    )


    if (
        np.isfinite(
            early_sd
        )
        and
        early_sd > 0
    ):

        standardized_change = (
            period_change_value
            /
            early_sd
        )

    else:

        standardized_change = (
            np.nan
        )


    return {

        "value_1900":
            v1900,

        "value_2018":
            v2018,

        "change_1900_2018_pp":
            float(
                endpoint_change
            ),

        "mean_1900_1919":
            early_mean,

        "sd_1900_1919":
            early_sd,

        "mean_1999_2018":
            late_mean,

        "change_20yr_means_pp":
            float(
                period_change_value
            ),

        "standardized_change_vs_early_sd":
            float(
                standardized_change
            )
            if np.isfinite(
                standardized_change
            )
            else np.nan
    }


# ======================================================================
# SEGMENTED BREAKPOINT
# ======================================================================

def fit_line(
    x,
    y
):

    result = stats.linregress(
        x,
        y
    )


    fitted = (
        result.intercept
        +
        result.slope
        *
        x
    )


    residuals = (
        y
        -
        fitted
    )


    sse = float(
        np.sum(
            residuals ** 2
        )
    )


    return (
        result,
        fitted,
        sse
    )


def breakpoint_search(
    years,
    values,
    min_segment_years=15
):

    """
    Exploratory two-segment linear breakpoint search.

    Candidate breakpoint must leave at least
    min_segment_years observations on each side.

    The breakpoint is chosen by minimum combined SSE.

    Also calculates BIC for:
        single-line model
        two-line model

    This is an exploratory structural-change descriptor,
    not a causal test.
    """

    x = np.asarray(
        years,
        dtype=float
    )

    y = np.asarray(
        values,
        dtype=float
    )


    mask = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[mask]
    y = y[mask]


    n = len(x)


    single_result, (
        single_fit
    ), single_sse = fit_line(
        x,
        y
    )


    # Single line:
    # intercept + slope = 2 parameters
    k_single = 2


    single_sse_safe = max(
        single_sse,
        1e-15
    )


    bic_single = (
        n
        *
        np.log(
            single_sse_safe
            /
            n
        )
        +
        k_single
        *
        np.log(
            n
        )
    )


    candidates = []


    start_index = (
        min_segment_years - 1
    )

    end_index = (
        n
        -
        min_segment_years
    )


    for i in range(
        start_index,
        end_index + 1
    ):

        breakpoint_year = int(
            x[i]
        )


        left_mask = (
            x <= breakpoint_year
        )

        right_mask = (
            x > breakpoint_year
        )


        if (
            left_mask.sum()
            <
            min_segment_years
        ):

            continue


        if (
            right_mask.sum()
            <
            min_segment_years
        ):

            continue


        left_result, (
            left_fit
        ), left_sse = fit_line(

            x[
                left_mask
            ],

            y[
                left_mask
            ]
        )


        right_result, (
            right_fit
        ), right_sse = fit_line(

            x[
                right_mask
            ],

            y[
                right_mask
            ]
        )


        total_sse = (
            left_sse
            +
            right_sse
        )


        candidates.append(
            {
                "breakpoint_year":
                    breakpoint_year,

                "total_sse":
                    total_sse,

                "left_slope_per_year":
                    float(
                        left_result.slope
                    ),

                "right_slope_per_year":
                    float(
                        right_result.slope
                    ),

                "left_intercept":
                    float(
                        left_result.intercept
                    ),

                "right_intercept":
                    float(
                        right_result.intercept
                    )
            }
        )


    if not candidates:

        return None


    best = min(
        candidates,
        key=lambda d: d[
            "total_sse"
        ]
    )


    segmented_sse = max(
        best[
            "total_sse"
        ],
        1e-15
    )


    # Two independent lines:
    # 2 intercepts + 2 slopes + breakpoint = 5 parameters
    k_segmented = 5


    bic_segmented = (
        n
        *
        np.log(
            segmented_sse
            /
            n
        )
        +
        k_segmented
        *
        np.log(
            n
        )
    )


    delta_bic = (
        bic_single
        -
        bic_segmented
    )


    sse_improvement_pct = (
        100
        *
        (
            single_sse
            -
            best[
                "total_sse"
            ]
        )
        /
        single_sse
        if single_sse > 0
        else np.nan
    )


    best[
        "single_line_sse"
    ] = single_sse


    best[
        "segmented_sse"
    ] = best[
        "total_sse"
    ]


    best[
        "sse_improvement_percent"
    ] = float(
        sse_improvement_pct
    )


    best[
        "bic_single"
    ] = float(
        bic_single
    )


    best[
        "bic_segmented"
    ] = float(
        bic_segmented
    )


    best[
        "delta_bic_single_minus_segmented"
    ] = float(
        delta_bic
    )


    best[
        "left_slope_per_decade"
    ] = (
        best[
            "left_slope_per_year"
        ]
        *
        10
    )


    best[
        "right_slope_per_decade"
    ] = (
        best[
            "right_slope_per_year"
        ]
        *
        10
    )


    if delta_bic > 10:

        evidence = (
            "strong segmented support"
        )

    elif delta_bic > 6:

        evidence = (
            "moderate segmented support"
        )

    elif delta_bic > 2:

        evidence = (
            "weak segmented support"
        )

    else:

        evidence = (
            "little/no segmented support"
        )


    best[
        "breakpoint_evidence"
    ] = evidence


    return best


# ======================================================================
# RUN STATISTICS
# ======================================================================

trend_rows = []
change_rows = []
breakpoint_rows = []


log("")

log(
    "CALCULATING STATISTICS"
)

log("-" * 78)


for site in SITES:

    log(
        f"Processing: {site}"
    )


    for radius in BUFFERS:

        subset = df[
            (df["site"] == site)
            &
            (
                df["buffer_km"]
                ==
                radius
            )
        ].sort_values(
            "year"
        )


        years = subset[
            "year"
        ].to_numpy(
            dtype=float
        )


        for (
            variable,
            label
        ) in VARIABLES.items():

            values = subset[
                variable
            ].to_numpy(
                dtype=float
            )


            # ----------------------------------------------------------
            # OLS
            # ----------------------------------------------------------

            ols = ols_statistics(
                years,
                values
            )


            # ----------------------------------------------------------
            # SEN
            # ----------------------------------------------------------

            sen = sens_slope(
                years,
                values
            )


            # ----------------------------------------------------------
            # MANN-KENDALL
            # ----------------------------------------------------------

            (
                mk_s,
                mk_variance,
                mk_z,
                mk_p,
                mk_tau,
                mk_direction
            ) = mann_kendall(
                values
            )


            trend_row = {

                "site":
                    site,

                "buffer_km":
                    radius,

                "variable":
                    variable,

                "variable_label":
                    label,

                "n_years":
                    len(values),

                **ols,

                **sen,

                "mk_S":
                    mk_s,

                "mk_variance":
                    mk_variance,

                "mk_z":
                    mk_z,

                "mk_p":
                    mk_p,

                "mk_tau":
                    mk_tau,

                "mk_direction":
                    mk_direction,

                "data_class":
                    "historical_landuse_reconstruction"
            }


            trend_rows.append(
                trend_row
            )


            # ----------------------------------------------------------
            # PERIOD CHANGE
            # ----------------------------------------------------------

            change_stats = period_change(
                years,
                values
            )


            change_rows.append(
                {

                    "site":
                        site,

                    "buffer_km":
                        radius,

                    "variable":
                        variable,

                    "variable_label":
                        label,

                    **change_stats
                }
            )


            # ----------------------------------------------------------
            # BREAKPOINT
            # ----------------------------------------------------------

            bp = breakpoint_search(
                years,
                values,
                min_segment_years=15
            )


            if bp is not None:

                breakpoint_rows.append(
                    {

                        "site":
                            site,

                        "buffer_km":
                            radius,

                        "variable":
                            variable,

                        "variable_label":
                            label,

                        **bp,

                        "analysis_status":
                            "exploratory"
                    }
                )


# ======================================================================
# TABLES
# ======================================================================

trends = pd.DataFrame(
    trend_rows
)

changes = pd.DataFrame(
    change_rows
)

breakpoints = pd.DataFrame(
    breakpoint_rows
)


trends.to_csv(
    OUT_TRENDS,
    index=False
)

changes.to_csv(
    OUT_CHANGE,
    index=False
)

breakpoints.to_csv(
    OUT_BREAKPOINTS,
    index=False
)


# ======================================================================
# BUFFER SENSITIVITY TABLE
# ======================================================================

sensitivity = trends[
    [
        "site",
        "buffer_km",
        "variable",
        "variable_label",
        "sen_slope_per_decade",
        "sen_ci95_low_per_decade",
        "sen_ci95_high_per_decade",
        "mk_tau",
        "mk_p",
        "mk_direction",
        "ols_slope_per_decade",
        "ols_r2",
        "ols_p"
    ]
].copy()


sensitivity.to_csv(
    OUT_SENSITIVITY,
    index=False
)


# ======================================================================
# PRIMARY 70-KM TABLE
# ======================================================================

primary_trends = trends[
    trends["buffer_km"] == 70
].copy()


primary_changes = changes[
    changes["buffer_km"] == 70
].copy()


primary_breakpoints = breakpoints[
    breakpoints["buffer_km"] == 70
].copy()


primary = primary_trends.merge(
    primary_changes,
    on=[
        "site",
        "buffer_km",
        "variable",
        "variable_label"
    ],
    how="left"
)


primary = primary.merge(
    primary_breakpoints,
    on=[
        "site",
        "buffer_km",
        "variable",
        "variable_label"
    ],
    how="left",
    suffixes=(
        "",
        "_breakpoint"
    )
)


primary.to_csv(
    OUT_PRIMARY,
    index=False
)


# ======================================================================
# TERMINAL — PRIMARY 70 KM
# ======================================================================

log("")

log("=" * 78)

log(
    "PRIMARY 70-KM TREND RESULTS"
)

log(
    "Sen slope = percentage points per decade"
)

log("=" * 78)


display_columns = [
    "site",
    "variable_label",
    "sen_slope_per_decade",
    "mk_tau",
    "mk_p",
    "ols_r2"
]


log(
    primary_trends[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)


# ======================================================================
# TERMINAL — BREAKPOINTS
# ======================================================================

log("")

log("=" * 78)

log(
    "PRIMARY 70-KM EXPLORATORY BREAKPOINTS"
)

log("=" * 78)


bp_display = [
    "site",
    "variable_label",
    "breakpoint_year",
    "left_slope_per_decade",
    "right_slope_per_decade",
    "sse_improvement_percent",
    "delta_bic_single_minus_segmented",
    "breakpoint_evidence"
]


log(
    primary_breakpoints[
        bp_display
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}"
    )
)


# ======================================================================
# FIGURE 1
# HISTORICAL TRAJECTORIES + SEN TREND
# ======================================================================

log("")

log(
    "Creating primary 70-km trend figure..."
)


df70 = df[
    df["buffer_km"] == 70
].copy()


fig, axes = plt.subplots(
    nrows=4,
    ncols=1,
    figsize=(8.5, 10.5),
    sharex=True
)


for ax, (
    variable,
    label
) in zip(
    axes,
    VARIABLES.items()
):


    for site in SITES:

        sub = df70[
            df70["site"] == site
        ].sort_values(
            "year"
        )


        years = sub[
            "year"
        ].to_numpy(
            dtype=float
        )


        values = sub[
            variable
        ].to_numpy(
            dtype=float
        )


        line = ax.plot(
            years,
            values,
            linewidth=1.8,
            alpha=0.70,
            label=site
        )[0]


        sen_result = primary_trends[
            (
                primary_trends[
                    "site"
                ]
                ==
                site
            )
            &
            (
                primary_trends[
                    "variable"
                ]
                ==
                variable
            )
        ].iloc[0]


        slope = sen_result[
            "sen_slope_per_year"
        ]


        # Anchor fitted Sen trend at median year/value.
        x0 = np.median(
            years
        )

        y0 = np.median(
            values
        )


        fitted = (
            y0
            +
            slope
            *
            (
                years - x0
            )
        )


        ax.plot(
            years,
            fitted,
            linestyle="--",
            linewidth=1.2,
            color=line.get_color()
        )


    ax.set_title(
        label,
        loc="left",
        fontweight="bold"
    )


    ax.set_ylabel(
        "% of 70-km buffer"
    )


    ax.grid(
        axis="y",
        alpha=0.25,
        linewidth=0.6
    )


    ax.set_xlim(
        1900,
        2018
    )


axes[-1].set_xlabel(
    "Year"
)


handles, labels = axes[
    0
].get_legend_handles_labels()


fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(
        0.5,
        0.995
    ),
    ncol=3,
    frameon=False
)


fig.suptitle(
    "Historical land-use trajectories and Sen trends within 70 km",
    fontsize=13,
    fontweight="bold",
    y=1.02
)


fig.text(
    0.5,
    0.004,
    (
        "Solid lines = ISIMIP3a reconstructed annual values; "
        "dashed lines = Sen monotonic trends. "
        "Percentages refer to the full circular buffer."
    ),
    ha="center",
    fontsize=8
)


fig.tight_layout(
    rect=[
        0,
        0.025,
        1,
        0.965
    ]
)


fig.savefig(
    FIG_TRENDS_PNG,
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    FIG_TRENDS_PDF,
    bbox_inches="tight"
)

plt.close(
    fig
)


# ======================================================================
# FIGURE 2
# SEN SLOPES
# ======================================================================

log(
    "Creating Sen-slope comparison figure..."
)


variables_order = list(
    VARIABLES.keys()
)


labels_order = [
    VARIABLES[x]
    for x in variables_order
]


fig, ax = plt.subplots(
    figsize=(
        9,
        5.5
    )
)


x = np.arange(
    len(
        variables_order
    )
)


width = 0.24


for i, site in enumerate(
    SITES
):

    sub = primary_trends[
        primary_trends[
            "site"
        ]
        ==
        site
    ].set_index(
        "variable"
    )


    values = np.array(
        [
            sub.loc[
                variable,
                "sen_slope_per_decade"
            ]
            for variable
            in variables_order
        ]
    )


    lower = np.array(
        [
            sub.loc[
                variable,
                "sen_ci95_low_per_decade"
            ]
            for variable
            in variables_order
        ]
    )


    upper = np.array(
        [
            sub.loc[
                variable,
                "sen_ci95_high_per_decade"
            ]
            for variable
            in variables_order
        ]
    )


    errors = np.vstack(
        [
            values - lower,
            upper - values
        ]
    )


    offset = (
        i - 1
    ) * width


    ax.bar(
        x + offset,
        values,
        width=width,
        label=site
    )


    ax.errorbar(
        x + offset,
        values,
        yerr=errors,
        fmt="none",
        capsize=3,
        linewidth=1
    )


ax.axhline(
    0,
    linewidth=0.8
)


ax.set_xticks(
    x
)


ax.set_xticklabels(
    [
        "Cropland",
        "Pasture",
        "Forest + natural\nvegetation",
        "Urban area"
    ]
)


ax.set_ylabel(
    "Sen slope\n"
    "(percentage points per decade)"
)


ax.set_title(
    "Historical land-use trends within 70 km",
    fontweight="bold"
)


ax.legend(
    frameon=False,
    ncol=3
)


ax.grid(
    axis="y",
    alpha=0.25,
    linewidth=0.6
)


fig.tight_layout()


fig.savefig(
    FIG_SLOPES_PNG,
    dpi=600,
    bbox_inches="tight"
)


fig.savefig(
    FIG_SLOPES_PDF,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ======================================================================
# FIGURE 3
# EXPLORATORY BREAKPOINTS — CROPLAND + FOREST ONLY
# ======================================================================

log(
    "Creating exploratory breakpoint figure..."
)


bp_variables = [
    (
        "cropland_total_percent_of_buffer",
        "Cropland"
    ),
    (
        "forests_natural_percent_of_buffer",
        "Forest + natural vegetation"
    )
]


fig, axes = plt.subplots(
    nrows=3,
    ncols=2,
    figsize=(
        11,
        9
    ),
    sharex=True
)


for row_i, site in enumerate(
    SITES
):

    for col_i, (
        variable,
        label
    ) in enumerate(
        bp_variables
    ):

        ax = axes[
            row_i,
            col_i
        ]


        sub = df70[
            df70["site"] == site
        ].sort_values(
            "year"
        )


        years = sub[
            "year"
        ].to_numpy(
            dtype=float
        )


        values = sub[
            variable
        ].to_numpy(
            dtype=float
        )


        ax.plot(
            years,
            values,
            linewidth=2
        )


        bp_row = primary_breakpoints[
            (
                primary_breakpoints[
                    "site"
                ]
                ==
                site
            )
            &
            (
                primary_breakpoints[
                    "variable"
                ]
                ==
                variable
            )
        ]


        if not bp_row.empty:

            bp_row = bp_row.iloc[0]

            bp_year = int(
                bp_row[
                    "breakpoint_year"
                ]
            )


            ax.axvline(
                bp_year,
                linestyle="--",
                linewidth=1.2
            )


            ymin, ymax = ax.get_ylim()


            ax.text(
                bp_year + 1,
                ymax
                -
                0.06
                *
                (
                    ymax - ymin
                ),
                f"{bp_year}",
                fontsize=8,
                va="top"
            )


        if row_i == 0:

            ax.set_title(
                label,
                fontweight="bold"
            )


        if col_i == 0:

            ax.set_ylabel(
                f"{site}\n% of 70-km buffer"
            )

        else:

            ax.set_ylabel(
                "% of 70-km buffer"
            )


        if row_i == 2:

            ax.set_xlabel(
                "Year"
            )


        ax.grid(
            axis="y",
            alpha=0.25,
            linewidth=0.6
        )


fig.suptitle(
    "Exploratory structural breakpoints in reconstructed land use",
    fontsize=13,
    fontweight="bold",
    y=1.01
)


fig.text(
    0.5,
    0.003,
    (
        "Dashed vertical lines show the minimum-SSE two-segment "
        "breakpoint. These are exploratory structural descriptors, "
        "not evidence of causal historical events."
    ),
    ha="center",
    fontsize=8
)


fig.tight_layout(
    rect=[
        0,
        0.025,
        1,
        0.97
    ]
)


fig.savefig(
    FIG_BREAKPOINTS_PNG,
    dpi=600,
    bbox_inches="tight"
)


fig.savefig(
    FIG_BREAKPOINTS_PDF,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ======================================================================
# BUFFER ROBUSTNESS SUMMARY
# ======================================================================

log("")

log("=" * 78)

log(
    "BUFFER SENSITIVITY — SEN SLOPE DIRECTION"
)

log("=" * 78)


for site in SITES:

    log("")

    log(site)

    log("-" * 50)


    for (
        variable,
        label
    ) in VARIABLES.items():

        sub = sensitivity[
            (
                sensitivity[
                    "site"
                ]
                ==
                site
            )
            &
            (
                sensitivity[
                    "variable"
                ]
                ==
                variable
            )
        ].sort_values(
            "buffer_km"
        )


        slopes = [
            f"{int(row.buffer_km)} km="
            f"{row.sen_slope_per_decade:+.3f}"
            for row
            in sub.itertuples()
        ]


        signs = np.sign(
            sub[
                "sen_slope_per_decade"
            ].to_numpy()
        )


        if np.all(
            signs > 0
        ):

            robustness = (
                "consistent increase"
            )

        elif np.all(
            signs < 0
        ):

            robustness = (
                "consistent decrease"
            )

        else:

            robustness = (
                "direction varies by buffer"
            )


        log(
            f"{label}: "
            +
            ", ".join(
                slopes
            )
            +
            f" | {robustness}"
        )


# ======================================================================
# SCIENTIFIC CAUTIONS
# ======================================================================

log("")

log("=" * 78)

log(
    "INTERPRETATION RULES"
)

log("=" * 78)


log(
    "1. ISIMIP3a histsoc land use is a "
    "historical reconstruction, not direct annual observation."
)


log(
    "2. Sen slope and Mann-Kendall quantify monotonic "
    "historical tendencies; they do not establish causality."
)


log(
    "3. OLS is reported for comparability, but reconstructed "
    "annual values are temporally autocorrelated, so OLS p-values "
    "should not be treated as fully independent-year evidence."
)


log(
    "4. Breakpoints are exploratory structural descriptors. "
    "Do not label them as known historical events without "
    "independent documentary evidence."
)


log(
    "5. 70 km is the primary scale; 25 and 50 km are "
    "spatial sensitivity analyses."
)


log(
    "6. Percentages refer to the entire circular coastal buffer, "
    "including marine area, not terrestrial land alone."
)


# ======================================================================
# OUTPUT REPORT
# ======================================================================

log("")

log("=" * 78)

log(
    "SCRIPT 18 COMPLETE"
)

log("=" * 78)


log("")

log("TABLES")

log("-" * 78)

log(
    str(
        OUT_TRENDS
    )
)

log(
    str(
        OUT_CHANGE
    )
)

log(
    str(
        OUT_BREAKPOINTS
    )
)

log(
    str(
        OUT_SENSITIVITY
    )
)

log(
    str(
        OUT_PRIMARY
    )
)


log("")

log("FIGURES")

log("-" * 78)

log(
    str(
        FIG_TRENDS_PNG
    )
)

log(
    str(
        FIG_TRENDS_PDF
    )
)

log(
    str(
        FIG_SLOPES_PNG
    )
)

log(
    str(
        FIG_SLOPES_PDF
    )
)

log(
    str(
        FIG_BREAKPOINTS_PNG
    )
)

log(
    str(
        FIG_BREAKPOINTS_PDF
    )
)


# ======================================================================
# WRITE LOG
# ======================================================================

OUT_LOG.write_text(
    "\n".join(
        log_lines
    ),
    encoding="utf-8"
)


print()

print(
    f"Log saved: {OUT_LOG}"
)