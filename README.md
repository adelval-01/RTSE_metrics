# 🐍🔉📞 RTSE Metrics

## 📘 Introduction

This repository implements a framework for **Real-Time Speech Enhancement (RTSE)** performance profiling, specifically focused on **latency measurement**. It allows evaluating how different system parameters affect real-time performance by processing a list of audio files and logging latency metrics for each configuration.

The core script, `rtse_latency.py`, loads configuration files (`configs/*.json`) that define parameters affecting latency, such as:

- **Diezmation Factor**: Controls sub-sampling or frame skipping strategies to reduce computational load.
- **Progressive Startup**: Gradually adding more context to the RTSE system.
- **Warm-Up Factor**: Repeats audio processing to simulate the caching and optimization behavior of the GPU or CPU over time.

Each configuration is processed independently, and for every run, the latency results are saved as a `.mat` file. This format is compatible with MATLAB and Python (via `scipy.io`) and facilitates post-processing, plotting, and statistical analysis.

---

## 📥 Input

- A list of audio file paths is provided via a `.list` text file (e.g., `minitest_8k.wav.list`).
- The audio is processed using the parameters defined in a JSON configuration file.

---

## 📤 Output

- A `.mat` file is generated for each configuration, storing latency measurements and processing metadata.

---

## 🔧 Environment Setup

### 1. **Create Conda Environment**

If you don't already have the environment, you can create it from the `full_environment.yml` file:

```bash
conda env create -f full_environment.yml -n rtse_env
conda activate rtse_env
pip install -r rtse_requirements.txt
```

## 🚀 Running the RTSE and Metrics
### 1. **Basic Execution with Diezmation and Progressive Startup**
```bash
python3 metrics/rtse_latency.py
```
### 2. **Execution with Warm-Up Factor**
Used to reset GPU cache and simulate warm-up behavior:
```bash
python3 metrics/rtse_latency.py ./metrics/configs/config-3.json
python3 metrics/rtse_latency.py ./metrics/configs/config-4.json
python3 metrics/rtse_latency.py ./metrics/configs/config-5.json
```
### 3. Zip latency results
```bash
zip -r latency_results.zip time_profiling
```

🔧 Quick Start
To fully set up the environment, install dependencies, run RTSE latency profiling, and compress the results, you can use the provided script:

```bash
chmod +x cmd.sh
./cmd.sh
```

This script performs the following actions automatically:


1. Creates a fresh environment from rtse_environment.yml.

2. Installs additional pip dependencies from rtse_requirements.txt.

3. Executes the latency measurement script with multiple configuration files.

4. Zips the profiling results into latency_results.zip.

5. Deactivates and remove the Conda environment when done.
