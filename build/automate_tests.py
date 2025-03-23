#!/usr/bin/env python3
"""
automate_tests.py

Automates running the mmult benchmark for the five datasets:
testing, small, medium, large, and native. Each dataset is run
twice: once with random data (no --load) and once with --load.
For the --load run, it first removes any existing matrix files,
then generates matrices using generate_matrices.py (based on the
dataset's dimensions) and waits until the files are ready before
starting the benchmark.
The console output of each mmult command is saved into a text file
named by the dataset and run type, with the command used printed at the top.
"""

import subprocess
import re
import argparse
import os
import time

# (dataset_name) -> (M, N, P)
DATASETS = {
    "testing": (16, 12, 8),
    "small": (121, 180, 115),
    "medium": (550, 620, 480),
    "large": (962, 1012, 1221),
    "native": (2500, 3000, 2100),
}

def save_console_output(dataset, suffix, header, output):
    """
    Saves the provided header and output to a file named "<dataset>_<suffix>.txt".
    """
    filename = f"{dataset}_{suffix}.txt"
    try:
        with open(filename, "w") as f:
            f.write(header + "\n" + ("-" * 80) + "\n")
            f.write(output)
        print(f"[INFO] Saved output to {filename}")
    except Exception as e:
        print(f"[ERROR] Could not save output to {filename}: {e}")

def wait_for_file(filename, timeout=10, interval=1):
    """
    Waits until the file 'filename' exists and is non-empty.
    """
    print(f"[INFO] Waiting for {filename} to be ready...")
    total_wait = 0
    while total_wait < timeout:
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            print(f"[INFO] {filename} is ready.")
            return True
        time.sleep(interval)
        total_wait += interval
    print(f"[WARN] Timeout reached: {filename} may not be fully written.")
    return False

def remove_old_files():
    """
    Removes A.bin, B.bin, and python_ref.bin if they exist.
    """
    for filename in ["A.bin", "B.bin", "python_ref.bin"]:
        if os.path.exists(filename):
            try:
                os.remove(filename)
                print(f"[INFO] Removed old file: {filename}")
            except Exception as e:
                print(f"[WARN] Could not remove {filename}: {e}")

def generate_matrices(dataset, M, N, P):
    """
    Removes existing matrix files then calls generate_matrices.py
    to create A.bin, B.bin, and python_ref.bin for the given dimensions.
    """
    print(f"[INFO] Preparing to generate matrices for dataset '{dataset}' "
          f"with dimensions {M}x{N} and {N}x{P} ...")
    remove_old_files()
    cmd = ["python", "generate_matrices.py", str(M), str(N), str(P)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] generate_matrices.py failed for dataset '{dataset}':\n{result.stderr}")
        raise RuntimeError("Matrix generation failed")
    else:
        print(result.stdout.strip())
    # Wait for python_ref.bin to be fully written.
    wait_for_file("python_ref.bin", timeout=10, interval=1)

def run_mmult(dataset, M, N, P, load, nruns):
    """
    Runs the mmult benchmark for the given dataset/dimensions.
    If load=True, it uses the previously generated matrices (A.bin, B.bin, python_ref.bin).
    Otherwise, it runs with random data.
    Saves the console output (with the command used as header) into a file.
    Parses the output to extract runtime statistics and the reference comparison.
    """
    cmd = [
        "./mmult",
        "-i", "naive",  # run the naive implementation
        "--M", str(M),
        "--N", str(N),
        "--P", str(P),
        "--nruns", str(nruns),
    ]
    if load:
        cmd.append("--load")

    cmd_str = "Command used: " + " ".join(cmd)
    print(f"[INFO] Running: {cmd_str}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + "\n" + result.stderr

    suffix = "with_load" if load else "no_load"
    save_console_output(dataset, suffix, cmd_str, output)

    # Parse output for runtime stats and reference comparison.
    stdev = None
    average = None
    match_result = "Unknown"

    std_re = re.compile(r"Standard deviation = (\d+)")
    avg_re = re.compile(r"Average = (\d+)")
    runtime_re = re.compile(r"Runtimes \((Success|Fail|MATCHING)\):\s+(\d+)\s+ns")

    for line in result.stdout.splitlines():
        line = line.strip()
        m_std = std_re.search(line)
        if m_std:
            stdev = int(m_std.group(1))
        m_avg = avg_re.search(line)
        if m_avg:
            average = int(m_avg.group(1))
        m_run = runtime_re.search(line)
        if m_run:
            match_result = "Success" if m_run.group(1) in ("Success", "MATCHING") else "Fail"
            average = int(m_run.group(2))
    
    # Check for python_ref comparison.
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("Comparison with python_ref.bin:"):
            if "Success" in line:
                match_result = "Success"
            else:
                match_result = "Fail(ref)"
    
    if result.returncode != 0:
        print(f"[ERROR] mmult for dataset '{dataset}' returned non-zero exit code:\n{result.stderr}")
    else:
        print(f"[INFO] mmult for dataset '{dataset}' completed with comparison: {match_result}.")

    return {
        "dataset": dataset,
        "M": M,
        "N": N,
        "P": P,
        "load": load,
        "avg_ns": average,
        "std_ns": stdev,
        "match": match_result,
        "console_output": output,
    }

def main():
    parser = argparse.ArgumentParser(description="Automate mmult testing.")
    parser.add_argument("--nruns", type=int, default=10,
                        help="Number of runs for each dataset (default=10)")
    args = parser.parse_args()

    nruns = args.nruns
    print(f"[INFO] Starting automation with nruns={nruns}...")

    for dataset_name, (M, N, P) in DATASETS.items():
        print(f"\n=== Testing dataset: {dataset_name} (M={M}, N={N}, P={P}) ===")

        # Run without --load (using random data).
        print("[STEP] Running without --load ...")
        run_mmult(dataset_name, M, N, P, load=False, nruns=nruns)

        # For --load run, generate matrices first.
        print("[STEP] Generating matrices for --load ...")
        generate_matrices(dataset_name, M, N, P)
        print("[STEP] Running with --load ...")
        run_mmult(dataset_name, M, N, P, load=True, nruns=nruns)

    print("\n[INFO] All tests completed.\n")

if __name__ == "__main__":
    main()

