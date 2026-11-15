import os
import numpy as np
import pandas as pd

RAW_DIR = os.path.join("data", "my_reports")
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)


def build_isone_csv(raw_dir: str = RAW_DIR, out_dir: str = OUT_DIR) -> str:
    years = [2019, 2020, 2021, 2022, 2023]
    sheet_name = "ISO NE CA"
    dfs = []

    for yr in years:
        fpath = os.path.join(raw_dir, f"{yr}_smd_hourly.xlsx")
        if not os.path.exists(fpath):
            fpath = f"{yr}_smd_hourly.xlsx"
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Could not find {yr}_smd_hourly.xlsx in '{raw_dir}' or current directory.")
        print(f"Reading {fpath} (sheet='{sheet_name}')...")
        dfs.append(pd.read_excel(fpath, sheet_name=sheet_name))

    raw = pd.concat(dfs, ignore_index=True)
    timestamps = pd.to_datetime(raw["Date"]) + pd.to_timedelta(raw["Hr_End"] - 1, unit="h")

    temp_c = (raw["Dry_Bulb"] - 32.0) * 5.0 / 9.0
    dew_c = (raw["Dew_Point"] - 32.0) * 5.0 / 9.0
    rh = np.clip(
        100.0 * (np.exp((17.625 * dew_c) / (243.04 + dew_c)) / np.exp((17.625 * temp_c) / (243.04 + temp_c))),
        0.0,
        100.0,
    )

    df_out = pd.DataFrame({
        "timestamp": timestamps,
        "load": raw["RT_Demand"].round(3),
        "da_demand": raw["DA_Demand"].round(3),
        "da_lmp": raw["DA_LMP"].round(3),
        "temperature": temp_c.round(2),
        "humidity": rh.round(2),
    })

    out_csv = os.path.join(out_dir, "isone_hourly.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"Saved '{out_csv}' with {len(df_out):,} hourly rows.")
    return out_csv


if __name__ == "__main__":
    build_isone_csv()