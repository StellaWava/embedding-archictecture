"""
STEP 7: Local Experiment
Summarize the fresh local test results

"""
import pandas as pd

INPUT_FILE = "results/local_results.csv"

df = pd.read_csv(INPUT_FILE)

print("\n=== Full results ===")
print(df)

print("\n=== Best recall@10 by architecture/freshness ===")
best = (
    df.sort_values(["arch", "freshness", "recall_at_10", "latency_p95_ms"], ascending=[True, True, False, True])
      .groupby(["arch", "freshness"], as_index=False)
      .first()
)
print(best)

print("\n=== Average latency trend ===")
lat = df.groupby(["arch", "freshness", "knob_name", "knob_value"], as_index=False)[
    ["latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "recall_at_10"]
].mean()
print(lat)