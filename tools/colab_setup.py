#!/usr/bin/env python3
"""
colab_setup.py - One-Command Remote GPU Worker Launcher for Google Colab

Run this script directly in a Google Colab GPU cell:
  !python3 tools/colab_setup.py

It verifies GPU availability, builds the CUDA solver, launches the API worker,
and opens a free Cloudflare tunnel providing an HTTPS endpoint.
"""

import os
import sys
import time
import subprocess

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT_DIR)

print("\n" + "=" * 70)
print("  INITIALIZING GOOGLE COLAB CUDA POWER FLOW COMPUTE WORKER")
print("=" * 70)

# 1. Check GPU
try:
    smi = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]).decode().strip()
    print(f"✓ Detected NVIDIA GPU: {smi}")
except Exception as e:
    print("⚠ WARNING: No NVIDIA GPU detected. In Colab, go to: Runtime -> Change runtime type -> T4 GPU")

# 2. Compile CUDA binary for Tesla T4 (sm_75)
print("\n[1/3] Building CUDA Newton-Raphson Solver (make ARCH=sm_75)...")
try:
    subprocess.check_call(["make", "ARCH=sm_75"], cwd=ROOT_DIR)
    print("✓ pf_cuda binary built successfully.")
except Exception as e:
    print(f"⚠ Build note: {e}")

# 3. Download cloudflared if not present
cf_bin = os.path.join(ROOT_DIR, "cloudflared")
if not os.path.exists(cf_bin):
    print("\n[2/3] Downloading Cloudflare Tunnel (free zero-config HTTPS tunnel)...")
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    subprocess.check_call(["wget", "-q", "-nc", "-O", cf_bin, url])
    subprocess.check_call(["chmod", "+x", cf_bin])
    print("✓ cloudflared ready.")

# 4. Start cloud_gpu_worker.py in background
print("\n[3/3] Starting Cloud GPU Worker API on port 8000...")
worker_proc = subprocess.Popen([sys.executable, "tools/cloud_gpu_worker.py", "--port", "8000"], cwd=ROOT_DIR)
time.sleep(1.5)

# 5. Start Cloudflare tunnel
print("Starting Cloudflare tunnel to expose remote worker...")
tunnel_proc = subprocess.Popen(
    [cf_bin, "tunnel", "--url", "http://localhost:8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    cwd=ROOT_DIR
)

public_url = None
print("Waiting for tunnel URL to generate...")
for line in tunnel_proc.stdout:
    if "trycloudflare.com" in line:
        for token in line.split():
            if "trycloudflare.com" in token and token.startswith("http"):
                public_url = token.strip()
                break
        if public_url:
            break

if public_url:
    print("\n" + "=" * 70)
    print("⚡ REMOTE COLAB GPU COMPUTE WORKER IS LIVE!")
    print(f"  ENDPOINT URL:  {public_url}")
    print("=" * 70)
    print("👉 HOW TO CONNECT:")
    print("  1. In the Web Simulation Lab, click the 'Cloud GPU Config' button.")
    print("  2. Select: 'NVIDIA Tesla T4 (Cloud Colab / AWS G4dn - Turing sm_75)'.")
    print(f"  3. In Remote Endpoint URL, paste: {public_url}")
    print("  4. Click 'Save & Connect'.")
    print("======================================================================\n")
else:
    print("\n⚠ Could not parse tunnel URL. Check cloudflared output.")

try:
    tunnel_proc.wait()
except KeyboardInterrupt:
    print("\nShutting down worker...")
    worker_proc.terminate()
    tunnel_proc.terminate()
