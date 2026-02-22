import sys
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def to_ms(row):
    t = float(row["real_time"])
    unit = str(row["time_unit"]).strip()
    if unit == "ns": return t / 1e6
    if unit == "us": return t / 1e3
    if unit == "ms": return t
    if unit == "s":  return t * 1e3
    return t

def parse_hw(name: str):
    parts = name.split("/")
    if len(parts) < 3:
        return None, None, parts[0]
    bench = parts[0]
    try:
        h = int(parts[-2]); w = int(parts[-1])
        return h, w, bench
    except:
        return None, None, bench

def main(csv_path, out_pdf="Ex6_plots.pdf"):
    df = pd.read_csv(csv_path)
    df = df[df["name"].astype(str).str.startswith("get_max_")].copy()

    rows = []
    for _, r in df.iterrows():
        h, w, bench = parse_hw(str(r["name"]))
        if h is None: 
            continue
        n = h * w
        rows.append({
            "bench": bench,
            "h": h, "w": w, "pixels": n,
            "time_ms": to_ms(r)
        })

    data = pd.DataFrame(rows)
    if data.empty:
        raise RuntimeError("No benchmark rows found in CSV.")

    agg = data.groupby(["bench", "pixels"], as_index=False)["time_ms"].mean()

    with PdfPages(out_pdf) as pdf:
        plt.figure()
        for bench_name, g in agg.groupby("bench"):
            g2 = g.sort_values("pixels")
            plt.plot(g2["pixels"], g2["time_ms"], marker="o", label=bench_name)
        plt.xlabel("Total pixels")
        plt.ylabel("Runtime (ms)")
        plt.title("get_max_value: Runtime vs input size")
        plt.legend()
        plt.grid(True)
        pdf.savefig()
        plt.close()

        serial = agg[agg["bench"] == "get_max_serial"][["pixels", "time_ms"]].rename(columns={"time_ms":"t_serial"})
        plt.figure()

        for bench_name, g in agg.groupby("bench"):
            if bench_name == "get_max_serial":
                continue
            merged = pd.merge(serial, g, on="pixels", how="inner")
            if merged.empty:
                continue
            merged["speedup"] = merged["t_serial"] / merged["time_ms"]
            merged = merged.sort_values("pixels")
            plt.plot(merged["pixels"], merged["speedup"], marker="o", label=bench_name)

        plt.xlabel("Total pixels")
        plt.ylabel("Speedup vs serial")
        plt.title("get_max_value: Speedup (parallel vs serial)")
        plt.legend()
        plt.grid(True)
        pdf.savefig()
        plt.close()

    print(f"Wrote {out_pdf}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 plot_ex6.py <benchmark_csv> [out_pdf]")
        sys.exit(1)
    csv_path = sys.argv[1]
    out_pdf = sys.argv[2] if len(sys.argv) >= 3 else "Ex6_plots.pdf"
    main(csv_path, out_pdf)
