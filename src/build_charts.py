"""
build_charts.py

Interactive HTML charts showing TX House/Senate seat-distribution forecasts
across environment scenarios, with confidence intervals from Monte Carlo.

Two chambers × two panels each:
  Top:    Seat distribution — expected D seats + 50% CI (p25-p75) + 80% CI (p10-p90)
          Dashed majority threshold line.  Current scenario column highlighted.
  Bottom: P(D controls chamber) — diverging bar chart.

Output:  output/model_2026_control_chart.html

USAGE:
  python src/build_charts.py
  python src/model.py && python src/build_charts.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT   = Path(__file__).parent.parent
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

HOUSE_MAJORITY  = 76
SENATE_MAJORITY = 16

# Color constants
C_BAND_OUTER = "rgba(59, 130, 190, 0.18)"
C_BAND_INNER = "rgba(59, 130, 190, 0.42)"
C_EXPECTED   = "#1A5BA8"
C_EXPECTED_R = "#8B0000"
C_MAJORITY   = "#CC2200"
C_CURRENT_BG = "rgba(255, 220, 80, 0.15)"
C_CTRL_D     = "#2166AC"
C_CTRL_R     = "#B2182B"
C_CTRL_MID   = "#888888"


def load_summary() -> pd.DataFrame:
    path = OUTPUT / "model_2026_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path}\nRun: python src/model.py")
    return pd.read_csv(path).sort_values("env_dial").reset_index(drop=True)


def get_current_env() -> float:
    sys.path.insert(0, str(Path(__file__).parent))
    from model_config import GENERIC_BALLOT_TOPLINE_D_2P
    return round((GENERIC_BALLOT_TOPLINE_D_2P - 0.5) * 200, 1)


def scen_label(env_dial: float, current_env: float) -> str:
    base = f"D+{int(env_dial)}" if env_dial >= 0 else f"R+{int(-env_dial)}"
    return f"{base} ★" if abs(env_dial - current_env) < 0.5 else base


def ctrl_color(prob: float) -> str:
    if prob >= 0.55: return C_CTRL_D
    if prob <= 0.45: return C_CTRL_R
    return C_CTRL_MID


def build_figure(df: pd.DataFrame, current_env: float) -> go.Figure:
    labels = [scen_label(r["env_dial"], current_env) for _, r in df.iterrows()]

    current_env_str = f"D+{current_env:.1f}" if current_env >= 0 else f"R+{-current_env:.1f}"

    fig = make_subplots(
        rows=4, cols=1,
        row_heights=[0.37, 0.13, 0.37, 0.13],
        vertical_spacing=0.05,
        subplot_titles=[
            "TX House — Expected D Seats won  (150 total; need 76 for majority)",
            "P(Democrats control TX House)",
            "TX Senate seats on 2026 ballot — Expected D wins  (16 seats; need 16 total for D majority)",
            "P(Democrats control TX Senate)",
        ],
    )

    def add_chamber(chamber: str, seat_row: int, ctrl_row: int, majority: int):
        col_exp  = f"expected_{chamber}_seats"
        col_p10  = f"{chamber}_seats_p10"
        col_p25  = f"{chamber}_seats_p25"
        col_p75  = f"{chamber}_seats_p75"
        col_p90  = f"{chamber}_seats_p90"
        col_ctrl = f"{chamber}_control_prob"

        show_leg = (seat_row == 1)   # legend entries only from first chamber

        # 80% CI bar (p10 → p90)
        fig.add_trace(go.Bar(
            x=labels,
            y=(df[col_p90] - df[col_p10]).tolist(),
            base=df[col_p10].tolist(),
            width=0.55,
            marker=dict(color=C_BAND_OUTER, line=dict(width=0)),
            name="80% CI  (p10–p90)",
            legendgroup="ci_outer",
            showlegend=show_leg,
            customdata=df[[col_p10, col_p90]].values.tolist(),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "80% range: %{customdata[0]:.0f}–%{customdata[1]:.0f} seats"
                "<extra></extra>"
            ),
        ), row=seat_row, col=1)

        # 50% CI bar (p25 → p75)
        fig.add_trace(go.Bar(
            x=labels,
            y=(df[col_p75] - df[col_p25]).tolist(),
            base=df[col_p25].tolist(),
            width=0.30,
            marker=dict(color=C_BAND_INNER, line=dict(width=0)),
            name="50% CI  (p25–p75)",
            legendgroup="ci_inner",
            showlegend=show_leg,
            customdata=df[[col_p25, col_p75]].values.tolist(),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "50% range: %{customdata[0]:.0f}–%{customdata[1]:.0f} seats"
                "<extra></extra>"
            ),
        ), row=seat_row, col=1)

        # Expected value diamonds — blue if at/above majority, dark red if below
        diamond_colors = [
            C_EXPECTED if v >= majority else C_EXPECTED_R
            for v in df[col_exp]
        ]
        ctrl_pcts = (df[col_ctrl] * 100).tolist()
        fig.add_trace(go.Scatter(
            x=labels,
            y=df[col_exp].tolist(),
            mode="markers+text",
            marker=dict(
                symbol="diamond",
                size=14,
                color=diamond_colors,
                line=dict(color="white", width=1.5),
            ),
            text=[f"  {v:.1f}" for v in df[col_exp]],
            textposition="middle right",
            textfont=dict(size=11, color="#222222"),
            name="Expected D seats",
            legendgroup="expected",
            showlegend=show_leg,
            customdata=[[v, c] for v, c in zip(df[col_exp], ctrl_pcts)],
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Expected D seats: <b>%{customdata[0]:.1f}</b><br>"
                "P(D majority): <b>%{customdata[1]:.1f}%</b>"
                "<extra></extra>"
            ),
        ), row=seat_row, col=1)

        # Majority threshold line
        fig.add_hline(
            y=majority,
            row=seat_row, col=1,
            line=dict(color=C_MAJORITY, width=1.8, dash="dash"),
            annotation_text=f"  Majority ({majority})",
            annotation_position="right",
            annotation_font=dict(color=C_MAJORITY, size=10),
        )

        # Control probability bars
        fig.add_trace(go.Bar(
            x=labels,
            y=ctrl_pcts,
            marker=dict(
                color=[ctrl_color(p / 100) for p in ctrl_pcts],
                line=dict(color="white", width=0.8),
            ),
            text=[f"{p:.1f}%" for p in ctrl_pcts],
            textposition="outside",
            textfont=dict(size=12, color="#222"),
            name="P(D controls)",
            showlegend=False,
            customdata=[100 - p for p in ctrl_pcts],
            hovertemplate=(
                "<b>%{x}</b><br>"
                "P(D controls): <b>%{y:.1f}%</b><br>"
                "P(R controls): <b>%{customdata:.1f}%</b>"
                "<extra></extra>"
            ),
        ), row=ctrl_row, col=1)

        # 50% reference line on control chart
        fig.add_hline(
            y=50, row=ctrl_row, col=1,
            line=dict(color="#888888", width=1, dash="dot"),
        )

        # Y-axis ranges
        y_lo = max(0, int(df[col_p10].min()) - 4)
        y_hi = min(150 if chamber == "house" else 16,
                   int(df[col_p90].max()) + 8)
        fig.update_yaxes(range=[y_lo, y_hi], title_text="D seats",
                         row=seat_row, col=1, gridcolor="#EBEBEB")

        ctrl_max = df[col_ctrl].max() * 100
        fig.update_yaxes(
            range=[0, max(12, ctrl_max + 10)],
            title_text="Prob. (%)",
            row=ctrl_row, col=1,
            gridcolor="#EBEBEB",
        )

    add_chamber("house",   seat_row=1, ctrl_row=2, majority=HOUSE_MAJORITY)
    add_chamber("senate",  seat_row=3, ctrl_row=4, majority=SENATE_MAJORITY)

    # Shade current scenario column in all panels
    current_lbl = next(
        (scen_label(r["env_dial"], current_env)
         for _, r in df.iterrows()
         if abs(r["env_dial"] - current_env) < 0.5),
        None,
    )
    if current_lbl:
        for row in (1, 2, 3, 4):
            fig.add_vrect(
                x0=current_lbl, x1=current_lbl,
                row=row, col=1,
                fillcolor=C_CURRENT_BG,
                layer="below",
                line_width=0,
            )

    # Global layout
    fig.update_layout(
        title=dict(
            text=(
                f"2026 TX Legislature — Seat Forecast & Majority Probability by Scenario<br>"
                f"<sup>Current environment: {current_env_str}  (marked ★)  ·  "
                f"Bands = 50% and 80% confidence intervals  ·  "
                f"Monte Carlo n=10,000 simulations</sup>"
            ),
            x=0.5, xanchor="center",
            font=dict(size=15),
        ),
        barmode="overlay",
        legend=dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=1.015, yanchor="bottom",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#DDDDDD",
            borderwidth=1,
            font=dict(size=11),
            traceorder="normal",
        ),
        height=900,
        margin=dict(t=110, b=40, l=70, r=110),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=12),
    )

    # Subtitle font sizes
    for ann in fig.layout.annotations:
        ann.font.size = 12

    return fig


def main():
    current_env = get_current_env()
    env_str = f"D+{current_env:.1f}" if current_env >= 0 else f"R+{-current_env:.1f}"
    print(f"Building control probability charts  (current env: {env_str})")

    df = load_summary()
    print(f"  Scenarios loaded: {df['scenario'].tolist()}")

    fig = build_figure(df, current_env)

    out_path = OUTPUT / "model_2026_control_chart.html"
    fig.write_html(
        out_path,
        include_plotlyjs="cdn",
        full_html=True,
        config={
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
            "toImageButtonOptions": {
                "format": "png",
                "filename": "tx_legislature_2026_control_chart",
                "height": 950, "width": 1050, "scale": 2,
            },
        },
    )
    size_kb = out_path.stat().st_size / 1024
    print(f"  Written → {out_path.name}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
