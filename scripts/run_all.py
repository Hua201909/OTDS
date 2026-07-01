import subprocess
import os

scripts = [
    "run_lstm_baseline.py", "run_lstm_fgsm.py", "run_lstm_sam.py", "run_lstm_sam_fgsm.py",
    "run_bilstm_baseline.py", "run_bilstm_fgsm.py", "run_bilstm_sam.py", "run_bilstm_sam_fgsm.py",
    "run_cnn_baseline.py", "run_cnn_fgsm.py", "run_cnn_sam.py", "run_cnn_sam_fgsm.py"
]

for script in scripts:
    print(f"Running {script}...")
    subprocess.run(["python", f"scripts/{script}"], check=True)
    print(f"Finished {script}\n")

# Summarize all the results and generate a comparison table
import json, glob, pandas as pd
results = []
for json_file in glob.glob("results/*.json"):
    name = os.path.basename(json_file).replace(".json", "")
    with open(json_file) as f:
        data = json.load(f)
        data["experiment"] = name
        results.append(data)
df = pd.DataFrame(results)
df.to_csv("results/metrics_summary.csv", index=False)
print("The summary results have been saved to results/metrics_summary.csv")