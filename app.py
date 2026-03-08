from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import sys
import warnings

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing


REQUIRED_COLUMNS = ["Month", "GMV"]
FORECAST_HORIZON = 12
MIN_HISTORY_FOR_SEASONAL = 24
MIN_HISTORY_FOR_SEASONAL_FALLBACK = 12
BRAND_ACCENT = "#0F766E"
BRAND_HIGHLIGHT = "#F59E0B"
SURFACE = "#F7F4EC"
INK = "#1F2937"


@dataclass
class ForecastResult:
    metric: str
    historical: pd.Series
    forecast: pd.Series
    components: pd.DataFrame


@dataclass
class HoltWintersConfig:
    trend: str | None
    seasonal: str | None
    seasonal_periods: int
    damped_trend: bool
    smoothing_level: float | None
    smoothing_trend: float | None
    smoothing_seasonal: float | None
    damping_trend: float | None


def resource_path(*parts: str) -> Path:
    base_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base_path.joinpath(*parts)


def main() -> None:
    st.set_page_config(page_title="OTA GMV Forecasting Tool", layout="wide")
    inject_styles()

    st.markdown(
        """
        <div class="hero-card">
            <div class="eyebrow">OTA GMV Forecasting</div>
            <h1>Upload monthly GMV history and get a simple 12-month forecast</h1>
            <p>This version keeps the input lightweight while still allowing optional tags like campaigns, holidays, or promos to enrich the explanations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        render_input_panel()
    with right:
        render_info_panel()


def render_input_panel() -> None:
    st.markdown("### Upload & Run")

    with st.container(border=True):
        st.markdown(
            "<div class='mini-note'>Required: <code>Month</code>, <code>GMV</code><br/>Optional: tag columns like <code>Campaign</code>, <code>Holiday</code>, <code>Promo</code></div>",
            unsafe_allow_html=True,
        )

        try:
            hw_config = render_holt_winters_controls()
        except ValueError as exc:
            st.error(str(exc))
            return

        uploaded_file = st.file_uploader(
            "Monthly CSV or Excel file",
            type=["csv", "xlsx"],
            accept_multiple_files=False,
        )

        sample_path = resource_path("data", "sample.csv")
        with sample_path.open("rb") as handle:
            st.download_button(
                label="Download sample input",
                data=handle.read(),
                file_name="sample.csv",
                mime="text/csv",
                use_container_width=True,
            )

    if uploaded_file is None:
        st.info("Upload a monthly GMV dataset to unlock the forecast workspace.")
        return

    try:
        raw_df = load_input(uploaded_file)
        cleaned_df, tag_columns, warning_list = validate_and_prepare(raw_df)
    except ValueError as exc:
        st.error(str(exc))
        return

    for warning_text in warning_list:
        st.warning(warning_text)

    st.markdown("### Data Check")
    data_col, stats_col = st.columns([1.5, 1], gap="large")
    with data_col:
        st.dataframe(cleaned_df, use_container_width=True, height=280)
    with stats_col:
        render_dataset_snapshot(cleaned_df, tag_columns)

    if st.button("Generate Forecast", type="primary", use_container_width=True):
        try:
            result = forecast_metric(cleaned_df, "GMV", FORECAST_HORIZON, hw_config)
        except ValueError as exc:
            st.error(str(exc))
            return
        explanation_df = build_explanations(cleaned_df, result, tag_columns)
        forecast_df = build_forecast_table(result, explanation_df)
        summary_df = build_summary_table(result)
        executive_export = build_executive_export_text(cleaned_df, result, explanation_df)

        render_results(
            cleaned_df=cleaned_df,
            result=result,
            forecast_df=forecast_df,
            explanation_df=explanation_df,
            summary_df=summary_df,
            executive_export=executive_export,
        )


def render_info_panel() -> None:
    st.markdown("### What This Version Does")
    st.markdown(
        """
        <div class="side-card">
            <div class="pill-row">
                <span class="pill">GMV only</span>
                <span class="pill">Seasonality-aware</span>
                <span class="pill">Tag-aware explanations</span>
            </div>
            <p>This version keeps the forecast simple:</p>
            <ul>
                <li>required columns are just <code>Month</code> and <code>GMV</code>,</li>
                <li>optional tag columns improve the narrative,</li>
                <li>the output stays focused on one GMV forecast.</li>
            </ul>
            <p>Explanations are based on recent trend, recurring monthly seasonality, and recurring tags when they are present.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Best Input Setup")
    st.markdown(
        """
        - Use at least 12 months if you want the forecast to preserve month-of-year seasonality.
        - Use 24 months or more for a more stable seasonal fit.
        - Required columns: `Month`, `GMV`.
        - Optional text tags: `Campaign`, `Holiday`, `Promo`, `Market Event`.
        - Format the month values like `2025-01-01` or `2025-01`.
        """
    )


def render_holt_winters_controls() -> HoltWintersConfig:
    with st.expander("Holt-Winters Parameters (Optional Tuning)", expanded=False):
        st.markdown("<div class='tuning-scope'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='tuning-card'>
                <div class='tuning-eyebrow'>Optional Tuning</div>
                <div class='tuning-title'>Adjust only if needed</div>
                <p>Use the default values unless you want finer control over trend, seasonality, and smoothing behavior.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        trend = st.selectbox(
            "trend",
            options=["add", "mul", "None"],
            index=0,
            help="Controls whether the long-term movement is additive or multiplicative.",
        )
        st.caption(
            "Options: None, add, mul. Use add for fixed-amount growth and mul for percentage-style growth."
        )

        seasonal = st.selectbox(
            "seasonal",
            options=["add", "mul", "None"],
            index=0,
            help="Controls whether seasonal movement is additive or multiplicative.",
        )
        st.caption(
            "Options: None, add, mul. Use add for fixed seasonal lift and mul when peaks scale with GMV level."
        )

        seasonal_periods = int(
            st.number_input(
                "seasonal_periods",
                min_value=1,
                max_value=60,
                value=12,
                step=1,
                help="How many observations make up one full seasonal cycle.",
            )
        )
        st.caption("For monthly data with yearly seasonality, this is usually 12.")

        damped_trend = st.checkbox(
            "damped_trend",
            value=True,
            help="Gradually flattens the projected trend in later forecast periods.",
        )
        st.caption("If enabled, the trend gradually flattens in the future instead of extending in a straight line.")

        st.caption("These are optional tuning inputs. We prefill them with practical starting values so you can adjust from there.")

        smoothing_level = st.text_input("smoothing_level (alpha)", value="0.40")
        st.caption("How fast the model reacts to new level changes.")

        smoothing_trend = st.text_input("smoothing_trend (beta)", value="0.10")
        st.caption("How fast the trend adapts.")

        smoothing_seasonal = st.text_input("smoothing_seasonal (gamma)", value="0.10")
        st.caption("How fast seasonal factors adapt.")

        damping_trend = st.text_input("damping_trend (phi)", value="0.98")
        st.caption("Only relevant if damped_trend=True; controls how strongly the trend is damped.")

    return HoltWintersConfig(
        trend=normalize_hw_component(trend),
        seasonal=normalize_hw_component(seasonal),
        seasonal_periods=seasonal_periods,
        damped_trend=damped_trend,
        smoothing_level=parse_optional_float(smoothing_level, "smoothing_level"),
        smoothing_trend=parse_optional_float(smoothing_trend, "smoothing_trend"),
        smoothing_seasonal=parse_optional_float(smoothing_seasonal, "smoothing_seasonal"),
        damping_trend=parse_optional_float(damping_trend, "damping_trend"),
    )


def render_dataset_snapshot(cleaned_df: pd.DataFrame, tag_columns: list[str]) -> None:
    start_month = cleaned_df["Month"].min().strftime("%b %Y")
    end_month = cleaned_df["Month"].max().strftime("%b %Y")
    latest_gmv = format_number(cleaned_df["GMV"].iloc[-1])
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">History Range</div>
            <div class="metric-value">{start_month} to {end_month}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Months Loaded</div>
            <div class="metric-value">{len(cleaned_df)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Latest GMV</div>
            <div class="metric-value">{latest_gmv}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Tag Columns</div>
            <div class="metric-value">{len(tag_columns)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results(
    cleaned_df: pd.DataFrame,
    result: ForecastResult,
    forecast_df: pd.DataFrame,
    explanation_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    executive_export: str,
) -> None:
    st.markdown("### Forecast Workspace")

    cards = [
        {
            "label": "12M Avg Forecast",
            "value": summary_df.iloc[0]["12M Average"],
            "subtext": f"vs recent 12M: {summary_df.iloc[0]['Vs Recent 12M']}",
        },
        {
            "label": "Peak Month",
            "value": summary_df.iloc[0]["Peak Month"],
            "subtext": summary_df.iloc[0]["Peak GMV"],
        },
        {
            "label": "Low Month",
            "value": summary_df.iloc[0]["Low Month"],
            "subtext": summary_df.iloc[0]["Low GMV"],
        },
    ]
    cols = st.columns(len(cards), gap="medium")
    for idx, card in enumerate(cards):
        cols[idx].markdown(
            f"""
            <div class="metric-card strong">
                <div class="metric-label">{card['label']}</div>
                <div class="metric-value">{card['value']}</div>
                <div class="metric-sub">{card['subtext']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Historical vs Forecast")
    st.altair_chart(build_history_forecast_chart(result), use_container_width=True)
    st.caption(
        "Historical GMV is shown in teal. Forecast GMV is shown in amber with a dashed line, and the split marks where the forecast starts."
    )

    with st.expander("Model Notes", expanded=True):
        render_model_notes(cleaned_df)

    tab_overview, tab_drivers, tab_explain, tab_export = st.tabs(
        ["Executive View", "Driver View", "Explainability", "Exports"]
    )

    with tab_overview:
        left, right = st.columns([1.2, 1], gap="large")
        with left:
            st.markdown("#### Summary")
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            st.markdown("#### Forecast Table")
            st.dataframe(forecast_df, use_container_width=True, height=360)
        with right:
            st.markdown("#### Narrative Brief")
            st.markdown(format_brief_html(executive_export), unsafe_allow_html=True)

    with tab_drivers:
        comp_df = result.components.copy()
        comp_df.index = comp_df.index.strftime("%Y-%m")
        st.markdown("#### Forecast Driver Breakdown")
        st.dataframe(
            comp_df[["trend_effect", "seasonal_effect", "forecast"]].round(2),
            use_container_width=True,
        )
        st.markdown(
            "The `trend_effect` captures recent momentum, while `seasonal_effect` captures recurring month-of-year lift or softness."
        )

    with tab_explain:
        st.markdown("#### Month-by-Month Explanation")
        st.dataframe(explanation_df, use_container_width=True, hide_index=True)

    with tab_export:
        excel_bytes = build_excel_export(cleaned_df, forecast_df, summary_df, explanation_df)
        st.download_button(
            label="Download Excel Export",
            data=excel_bytes,
            file_name="ota_gmv_forecast.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        st.download_button(
            label="Download Executive Summary",
            data=executive_export,
            file_name="ota_gmv_forecast_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )


def load_input(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Please upload a CSV or Excel file.")


def validate_and_prepare(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    if raw_df.empty:
        raise ValueError("The uploaded file is empty.")

    df = raw_df.copy()
    df.columns = [str(column).strip() for column in df.columns]

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    df["GMV"] = pd.to_numeric(df["GMV"], errors="coerce")
    df = df.dropna(subset=["Month", "GMV"]).copy()
    if df.empty:
        raise ValueError("No valid rows were found after parsing Month and GMV.")

    df["Month"] = df["Month"].dt.to_period("M").dt.to_timestamp()
    tag_columns = [column for column in df.columns if column not in REQUIRED_COLUMNS]

    warning_list: list[str] = []
    duplicate_months = int(df["Month"].duplicated().sum())
    if duplicate_months:
        warning_list.append(
            f"Found {duplicate_months} duplicate month rows. They were rolled up into a single monthly record."
        )

    aggregations: dict[str, str] = {"GMV": "sum"}
    for column in tag_columns:
        df[column] = df[column].fillna("").astype(str).str.strip()
        aggregations[column] = " | ".join

    monthly_df = df.groupby("Month", as_index=False).agg(aggregations)
    for column in tag_columns:
        monthly_df[column] = monthly_df[column].apply(lambda value: value if value else np.nan).fillna("")
        monthly_df[column] = monthly_df[column].apply(normalize_tag_text)

    monthly_df = monthly_df.sort_values("Month").reset_index(drop=True)
    if len(monthly_df) < 6:
        raise ValueError("Please provide at least 6 months of history for a usable forecast.")

    full_index = pd.date_range(monthly_df["Month"].min(), monthly_df["Month"].max(), freq="MS")
    if len(full_index) != len(monthly_df):
        warning_list.append(
            "Some months were missing in the history. The app inserted those months and linearly interpolated GMV for model continuity."
        )
        monthly_df = monthly_df.set_index("Month").reindex(full_index)
        monthly_df.index.name = "Month"
        monthly_df["GMV"] = monthly_df["GMV"].interpolate(method="linear").bfill().ffill()
        for column in tag_columns:
            monthly_df[column] = monthly_df[column].fillna("")
        monthly_df = monthly_df.reset_index().rename(columns={"index": "Month"})

    return monthly_df, tag_columns, warning_list


def normalize_tag_text(value: str) -> str:
    parts = [part.strip() for part in value.split("|") if part.strip()]
    seen: list[str] = []
    for part in parts:
        if part not in seen:
            seen.append(part)
    return " | ".join(seen)


def normalize_hw_component(value: str) -> str | None:
    return None if value == "None" else value


def parse_optional_float(raw_value: str, field_name: str) -> float | None:
    value = raw_value.strip()
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a number between 0 and 1.") from exc
    if not 0 <= parsed <= 1:
        raise ValueError(f"{field_name} must be between 0 and 1.")
    return parsed


def forecast_metric(
    cleaned_df: pd.DataFrame,
    metric: str,
    forecast_horizon: int,
    hw_config: HoltWintersConfig,
) -> ForecastResult:
    historical = cleaned_df.set_index("Month")[metric].astype(float)

    if (hw_config.trend == "mul" or hw_config.seasonal == "mul") and (historical <= 0).any():
        raise ValueError(
            "Multiplicative trend or seasonality requires all GMV values to be greater than zero."
        )

    if hw_config.seasonal is not None and len(historical) < hw_config.seasonal_periods * 2:
        raise ValueError(
            f"Seasonal Holt-Winters needs at least {hw_config.seasonal_periods * 2} months of history "
            f"for seasonal_periods={hw_config.seasonal_periods}."
        )

    if not hw_config.damped_trend and hw_config.damping_trend is not None:
        raise ValueError("damping_trend (phi) can only be set when damped_trend is enabled.")

    fit_kwargs = build_fit_kwargs(hw_config)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            historical,
            trend=hw_config.trend,
            seasonal=hw_config.seasonal,
            seasonal_periods=hw_config.seasonal_periods if hw_config.seasonal is not None else None,
            damped_trend=hw_config.damped_trend,
            initialization_method="estimated",
        )
        fit = model.fit(use_brute=True, **fit_kwargs)

    forecast_index = pd.date_range(
        historical.index[-1] + pd.offsets.MonthBegin(1),
        periods=forecast_horizon,
        freq="MS",
    )
    forecast_values = pd.Series(fit.forecast(forecast_horizon).values, index=forecast_index, name=metric)
    forecast_values = forecast_values.clip(lower=0)

    base_level = float(historical.iloc[-1])
    recent_slope = compute_recent_slope(historical)
    trend_effect = pd.Series(
        [max(base_level + recent_slope * (step + 1), 0.0) for step in range(forecast_horizon)],
        index=forecast_index,
        name="trend_effect",
    )
    seasonal_effect = forecast_values - trend_effect

    components = pd.DataFrame(
        {
            "trend_effect": trend_effect,
            "seasonal_effect": seasonal_effect,
            "forecast": forecast_values,
        }
    )

    return ForecastResult(
        metric=metric,
        historical=historical,
        forecast=forecast_values,
        components=components,
    )


def build_fit_kwargs(hw_config: HoltWintersConfig) -> dict[str, float | bool]:
    fit_kwargs: dict[str, float | bool] = {"optimized": True}
    manual_params = {
        "smoothing_level": hw_config.smoothing_level,
        "smoothing_trend": hw_config.smoothing_trend,
        "smoothing_seasonal": hw_config.smoothing_seasonal,
        "damping_trend": hw_config.damping_trend,
    }
    if any(value is not None for value in manual_params.values()):
        fit_kwargs["optimized"] = False
        for key, value in manual_params.items():
            if value is not None:
                fit_kwargs[key] = value
    return fit_kwargs


def compute_recent_slope(history: pd.Series) -> float:
    if len(history) < 2:
        return 0.0
    recent_points = min(len(history), 6)
    recent = history.tail(recent_points).reset_index(drop=True)
    x_values = np.arange(len(recent), dtype=float)
    slope, _ = np.polyfit(x_values, recent.values.astype(float), 1)
    return float(slope)


def build_explanations(
    cleaned_df: pd.DataFrame,
    result: ForecastResult,
    tag_columns: list[str],
) -> pd.DataFrame:
    recurring_tags = build_recurring_tags(cleaned_df, tag_columns)
    explanations: list[dict[str, str | float]] = []
    previous_value = float(result.historical.iloc[-1])

    for month, forecast_value in result.forecast.items():
        seasonal_effect = float(result.components.loc[month, "seasonal_effect"])
        explanation = explain_month(
            month=month,
            forecast_value=float(forecast_value),
            previous_value=previous_value,
            recent_slope=compute_recent_slope(result.historical),
            seasonal_effect=seasonal_effect,
            recurring_tags=recurring_tags,
        )
        explanations.append(
            {
                "Month": month.strftime("%Y-%m"),
                "Forecast GMV": round(float(forecast_value), 2),
                "Explanation": explanation,
            }
        )
        previous_value = float(forecast_value)

    return pd.DataFrame(explanations)


def build_recurring_tags(cleaned_df: pd.DataFrame, tag_columns: list[str]) -> dict[int, list[str]]:
    recurring_tags: dict[int, list[str]] = {}
    if not tag_columns:
        return recurring_tags

    for month_number in range(1, 13):
        month_slice = cleaned_df.loc[cleaned_df["Month"].dt.month == month_number, tag_columns]
        found_tags: list[str] = []
        for column in tag_columns:
            for value in month_slice[column].tolist():
                if not value:
                    continue
                for part in [item.strip() for item in str(value).split("|") if item.strip()]:
                    if part not in found_tags:
                        found_tags.append(part)
        if found_tags:
            recurring_tags[month_number] = found_tags

    return recurring_tags


def explain_month(
    month: pd.Timestamp,
    forecast_value: float,
    previous_value: float,
    recent_slope: float,
    seasonal_effect: float,
    recurring_tags: dict[int, list[str]],
) -> str:
    change_pct = ((forecast_value - previous_value) / previous_value * 100.0) if previous_value else 0.0
    if change_pct > 2:
        movement = "higher"
    elif change_pct < -2:
        movement = "lower"
    else:
        movement = "steady"

    driver_bits: list[str] = []
    if seasonal_effect > max(forecast_value * 0.05, 1.0):
        driver_bits.append("positive seasonal lift")
    elif seasonal_effect < -max(forecast_value * 0.05, 1.0):
        driver_bits.append("seasonal softness")

    if recent_slope > 0:
        driver_bits.append("positive recent trend")
    elif recent_slope < 0:
        driver_bits.append("recent softening trend")
    else:
        driver_bits.append("stable recent pattern")

    month_tags = recurring_tags.get(month.month, [])
    if month_tags:
        driver_bits.append(f"recurring tags: {', '.join(month_tags[:2])}")

    if movement == "higher":
        opening = f"GMV is expected to rise in {month.strftime('%b %Y')}"
    elif movement == "lower":
        opening = f"GMV is expected to dip in {month.strftime('%b %Y')}"
    else:
        opening = f"GMV is expected to stay fairly steady in {month.strftime('%b %Y')}"

    return f"{opening}, mainly due to {' and '.join(driver_bits)}."


def build_forecast_table(result: ForecastResult, explanation_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Month": result.forecast.index.strftime("%Y-%m"),
            "GMV": result.forecast.values.round(2),
            "Key Explanation": explanation_df["Explanation"],
        }
    )


def build_summary_table(result: ForecastResult) -> pd.DataFrame:
    recent_mean = result.historical.tail(12).mean()
    forecast_mean = result.forecast.mean()
    growth_pct = ((forecast_mean - recent_mean) / recent_mean * 100) if recent_mean else 0.0
    peak_month = result.forecast.idxmax()
    trough_month = result.forecast.idxmin()
    return pd.DataFrame(
        [
            {
                "Metric": "GMV",
                "Latest Actual": format_number(result.historical.iloc[-1]),
                "12M Average": format_number(forecast_mean),
                "Vs Recent 12M": format_pct(growth_pct),
                "Peak Month": peak_month.strftime("%b %Y"),
                "Peak GMV": format_number(result.forecast.max()),
                "Low Month": trough_month.strftime("%b %Y"),
                "Low GMV": format_number(result.forecast.min()),
            }
        ]
    )


def build_executive_export_text(
    cleaned_df: pd.DataFrame,
    result: ForecastResult,
    explanation_df: pd.DataFrame,
) -> str:
    start_month = cleaned_df["Month"].min().strftime("%b %Y")
    end_month = cleaned_df["Month"].max().strftime("%b %Y")
    recent_mean = result.historical.tail(12).mean()
    forecast_mean = result.forecast.mean()
    growth_pct = ((forecast_mean - recent_mean) / recent_mean * 100) if recent_mean else 0.0
    peak_month = result.forecast.idxmax()
    trough_month = result.forecast.idxmin()
    peak_reason = explanation_df.loc[
        explanation_df["Month"] == peak_month.strftime("%Y-%m"),
        "Explanation",
    ].iloc[0]
    trough_reason = explanation_df.loc[
        explanation_df["Month"] == trough_month.strftime("%Y-%m"),
        "Explanation",
    ].iloc[0]

    lines = [
        "OTA GMV Forecast Summary",
        f"Historical range: {start_month} to {end_month}",
        f"Forecast horizon: {len(result.forecast)} months",
        "Overview:",
        f"- Average GMV over the next 12 months is {format_number(forecast_mean)} ({format_pct(growth_pct)} vs recent 12-month average).",
        f"- Peak month is {peak_month.strftime('%b %Y')} at {format_number(result.forecast.max())}.",
        f"- Low month is {trough_month.strftime('%b %Y')} at {format_number(result.forecast.min())}.",
        "Drivers:",
        f"- Peak driver: {compress_driver(peak_reason)}.",
        f"- Low driver: {compress_driver(trough_reason)}.",
    ]
    return "\n".join(lines)


def build_excel_export(
    cleaned_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    explanation_df: pd.DataFrame,
) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        cleaned_df.to_excel(writer, sheet_name="History", index=False)
        forecast_df.to_excel(writer, sheet_name="Forecast", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        explanation_df.to_excel(writer, sheet_name="Explainability", index=False)
    buffer.seek(0)
    return buffer.getvalue()


def build_history_forecast_chart(result: ForecastResult) -> alt.Chart:
    historical_df = pd.DataFrame(
        {
            "Month": result.historical.index,
            "GMV": result.historical.values,
            "Series": "Historical",
        }
    )
    forecast_df = pd.DataFrame(
        {
            "Month": result.forecast.index,
            "GMV": result.forecast.values,
            "Series": "Forecast",
        }
    )
    combined = pd.concat([historical_df, forecast_df], ignore_index=True)

    line = (
        alt.Chart(combined)
        .mark_line(strokeWidth=3)
        .encode(
            x=alt.X("Month:T", title="Month"),
            y=alt.Y("GMV:Q", title="GMV"),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(domain=["Historical", "Forecast"], range=[BRAND_ACCENT, BRAND_HIGHLIGHT]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            strokeDash=alt.condition(
                alt.datum.Series == "Forecast",
                alt.value([7, 5]),
                alt.value([1, 0]),
            ),
            tooltip=[alt.Tooltip("Month:T"), alt.Tooltip("GMV:Q", format=",.2f"), "Series:N"],
        )
    )

    split_rule = (
        alt.Chart(pd.DataFrame({"Month": [result.forecast.index.min()]}))
        .mark_rule(color="#9CA3AF", strokeDash=[4, 4])
        .encode(x="Month:T")
    )

    return (line + split_rule).properties(height=360)


def render_model_notes(cleaned_df: pd.DataFrame) -> None:
    has_seasonality = len(cleaned_df) >= MIN_HISTORY_FOR_SEASONAL_FALLBACK
    st.markdown(
        "\n".join(
            [
                "- Model family: Holt-Winters exponential smoothing.",
                f"- Seasonality enabled: {'yes' if has_seasonality else 'no, due to limited history'}.",
                "- Forecasts are best used as a directional planning baseline, not as a substitute for commercial judgment.",
                "- Optional text tags do not directly fit the model; they are used to improve the narrative explanations.",
            ]
        )
    )


def format_brief_html(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "<div class='brief-card'></div>"

    title = lines[0]
    body_lines = lines[1:]
    html_parts = [f"<div class='brief-card'><h4>{title}</h4>"]
    for line in body_lines:
        if line.endswith(":"):
            html_parts.append(f"<p class='brief-section'>{line}</p>")
        elif line.startswith("- "):
            html_parts.append(f"<p class='brief-bullet'>{line[2:]}</p>")
        else:
            html_parts.append(f"<p>{line}</p>")
    html_parts.append("</div>")
    return "".join(html_parts)


def compress_driver(text: str) -> str:
    trimmed = text.split(", mainly due to ")[-1]
    return trimmed[0].upper() + trimmed[1:].rstrip(".") if trimmed else "N/A"


def format_number(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.2f}"


def format_pct(value: float) -> str:
    return f"{value:+.1f}%"


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
            .stApp {{
                background: linear-gradient(180deg, #fcfbf7 0%, {SURFACE} 55%, #eef6f4 100%);
                color: {INK};
            }}
            .hero-card {{
                padding: 1.4rem 1.6rem;
                border-radius: 22px;
                background: linear-gradient(135deg, rgba(15,118,110,0.98) 0%, rgba(13,148,136,0.88) 55%, rgba(245,158,11,0.72) 100%);
                color: white;
                margin-bottom: 1rem;
                box-shadow: 0 20px 60px rgba(15, 118, 110, 0.18);
            }}
            .hero-card h1 {{
                font-size: 2.2rem;
                margin: 0.2rem 0 0.5rem 0;
                line-height: 1.1;
            }}
            .hero-card p {{
                margin: 0;
                font-size: 1rem;
                max-width: 52rem;
            }}
            .eyebrow {{
                text-transform: uppercase;
                letter-spacing: 0.14em;
                font-size: 0.72rem;
                opacity: 0.85;
                font-weight: 700;
            }}
            .side-card, .metric-card, .brief-card {{
                background: rgba(255,255,255,0.8);
                border: 1px solid rgba(15,118,110,0.12);
                border-radius: 18px;
                padding: 1rem 1.1rem;
                box-shadow: 0 12px 30px rgba(15,118,110,0.06);
            }}
            .metric-card {{
                margin-bottom: 0.75rem;
            }}
            .metric-card.strong {{
                background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(245,158,11,0.08));
            }}
            .metric-label {{
                color: #4b5563;
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-weight: 700;
            }}
            .metric-value {{
                font-size: 1.55rem;
                font-weight: 700;
                color: {INK};
                margin-top: 0.2rem;
            }}
            .metric-sub {{
                margin-top: 0.15rem;
                color: #6b7280;
                font-size: 0.92rem;
            }}
            .pill-row {{
                display: flex;
                gap: 0.45rem;
                flex-wrap: wrap;
                margin-bottom: 0.8rem;
            }}
            .pill {{
                background: rgba(15,118,110,0.1);
                color: {BRAND_ACCENT};
                border: 1px solid rgba(15,118,110,0.18);
                border-radius: 999px;
                padding: 0.25rem 0.7rem;
                font-size: 0.8rem;
                font-weight: 700;
            }}
            .mini-note {{
                background: rgba(245,158,11,0.1);
                color: {INK};
                border-left: 4px solid {BRAND_HIGHLIGHT};
                padding: 0.7rem 0.8rem;
                border-radius: 12px;
                font-size: 0.9rem;
                margin-bottom: 1rem;
            }}
            .tuning-scope {{
                display: none;
            }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.tuning-scope) {{
                background: linear-gradient(180deg, rgba(237,245,242,0.92) 0%, rgba(226,238,234,0.96) 100%);
                border: 1px solid rgba(15,118,110,0.22);
                border-radius: 22px;
                box-shadow: 0 18px 44px rgba(15,118,110,0.08);
            }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.tuning-scope) > div {{
                padding: 0.45rem 0.55rem 0.8rem 0.55rem;
            }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.tuning-scope) label,
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.tuning-scope) .stMarkdown,
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.tuning-scope) p,
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.tuning-scope) span,
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.tuning-scope) small {{
                color: #102a43;
            }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.tuning-scope) [data-testid="stWidgetLabel"] {{
                color: #0f172a;
                font-weight: 700;
            }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.tuning-scope) [data-testid="stExpander"] summary,
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.tuning-scope) [data-testid="stExpander"] summary p {{
                color: #12344d;
                font-weight: 600;
            }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.tuning-scope) .stCaptionContainer,
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.tuning-scope) .stCaptionContainer p {{
                color: #365266;
            }}
            .tuning-card {{
                background: linear-gradient(135deg, rgba(15,118,110,0.18), rgba(13,148,136,0.10) 60%, rgba(245,158,11,0.16));
                border: 1px solid rgba(15,118,110,0.22);
                border-radius: 18px;
                padding: 1rem 1rem 0.9rem 1rem;
                margin: 0.2rem 0 1rem 0;
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.45);
            }}
            .tuning-eyebrow {{
                text-transform: uppercase;
                letter-spacing: 0.12em;
                font-size: 0.72rem;
                font-weight: 800;
                color: {BRAND_ACCENT};
                margin-bottom: 0.2rem;
            }}
            .tuning-title {{
                font-size: 1.05rem;
                font-weight: 800;
                color: #0f172a;
                margin-bottom: 0.35rem;
            }}
            .tuning-card p {{
                margin: 0;
                color: #334155;
                font-size: 0.93rem;
                line-height: 1.45;
            }}
            div[data-testid="stFileUploaderDropzone"] > div,
            div[data-testid="stFileUploaderDropzone"] small,
            div[data-testid="stFileUploaderDropzoneInstructions"] span,
            div[data-testid="stFileUploaderDropzoneInstructions"] small,
            div[data-testid="stFileUploader"] label,
            div[data-testid="stFileUploader"] section small {{
                color: #ffffff !important;
            }}
            div[data-testid="stFileUploaderDropzone"] {{
                background: rgba(255,255,255,0.72);
                border: 1px dashed rgba(15,118,110,0.28);
            }}
            div[data-testid="stDownloadButton"] button,
            div[data-testid="stDownloadButton"] button p,
            div[data-testid="stDownloadButton"] button span {{
                color: #ffffff !important;
                font-weight: 700;
            }}
            .brief-card h4 {{
                margin: 0 0 0.7rem 0;
                font-size: 1rem;
                color: #0f172a;
            }}
            .brief-card p {{
                margin: 0 0 0.65rem 0;
                line-height: 1.55;
                font-size: 0.95rem;
                color: #334155;
            }}
            .brief-card .brief-section {{
                margin-top: 0.9rem;
                font-weight: 700;
                color: #0f172a;
            }}
            .brief-card .brief-bullet {{
                padding-left: 0.9rem;
                position: relative;
            }}
            .brief-card .brief-bullet:before {{
                content: '*';
                position: absolute;
                left: 0;
                color: #0F766E;
                font-weight: 700;
            }}
            div[data-testid="stDataFrame"] {{
                border-radius: 14px;
                overflow: hidden;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

