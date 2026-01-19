import pandas as pd
import requests
from pathlib import Path

OUT_PATH = Path("data/live_canada_macro.csv")

# --- BoC Valet endpoints (CSV) ---
# Total CPI (monthly index). Series id shown in BoC Valet group pages.
CPI_URL = "https://www.bankofcanada.ca/valet/observations/V41690973/csv"
# Policy rate / Target for overnight rate (BoC Policy Instrument group exposes this series)
POLICY_URL = "https://www.bankofcanada.ca/valet/observations/STATIC_ATABLE_V39079/csv"


def fetch_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    text = r.text
    lines = text.splitlines()

    # Try to find the first line that contains BOTH 'date' and a comma (CSV header-ish)
    header_idx = None
    for i, line in enumerate(lines[:200]):  # scan early part
        low = line.strip().lower()
        if ("date" in low) and ("," in low):
            header_idx = i
            break

    # If not found, show a helpful preview so we can see what's coming back
    if header_idx is None:
        preview = "\n".join(lines[:40])
        raise ValueError(
            "Could not detect a CSV header line containing 'date'.\n"
            "First 40 lines received:\n"
            f"{preview}"
        )

    from io import StringIO
    csv_text = "\n".join(lines[header_idx:])
    return pd.read_csv(StringIO(csv_text))




def clean_boc_obs(df_raw: pd.DataFrame, value_col_guess: str = None) -> pd.DataFrame:
    # BoC CSV observations typically have columns like: date, <SERIES_ID>
    # Find the date column and the single value column.
    cols = list(df_raw.columns)
    date_col = "date" if "date" in cols else cols[0]

    value_cols = [c for c in cols if c != date_col]
    if value_col_guess and value_col_guess in value_cols:
        value_col = value_col_guess
    else:
        # pick the first non-date column
        value_col = value_cols[0]

    out = df_raw[[date_col, value_col]].copy()
    out.columns = ["date", "value"]
    out["date"] = pd.to_datetime(out["date"])
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna().sort_values("date")
    return out


def main():
    # CPI level (index)
    cpi_raw = fetch_csv(CPI_URL)
    cpi = clean_boc_obs(cpi_raw, value_col_guess="V41690973")
    # YoY % change
    cpi["cpi_yoy"] = cpi["value"].pct_change(12) * 100.0

    # Policy rate (target for overnight rate)
    pol_raw = fetch_csv(POLICY_URL)
    pol = clean_boc_obs(pol_raw, value_col_guess="STATIC_ATABLE_V39079")
    pol = pol.rename(columns={"value": "policy_rate"})

    # Merge on date (monthly)
    df = pd.merge(cpi[["date", "cpi_yoy"]], pol[["date", "policy_rate"]], on="date", how="outer")
    df = df.sort_values("date")

    # Keep only rows where we have CPI YoY (needs 12 months history)
    df = df.dropna(subset=["cpi_yoy"])

    # Optional placeholders (until you add StatCan later)
    df["unemployment"] = pd.NA
    df["wage_growth"] = pd.NA

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH.resolve()}")
    print(df.tail(5))


if __name__ == "__main__":
    main()
