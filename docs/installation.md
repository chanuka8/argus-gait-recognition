# Installation Guide

This document walks through setting up the ARGUS Gait Recognition system on a local machine, configuring parameters, and running system diagnostic checks.

---

## 1. Prerequisites

Before installing, ensure the system environment meets the following conditions:

- **Operating System:** Windows 10/11 (fully supported), Linux, or macOS.
- **Python Version:** Python 3.10 or 3.11 (Python 3.11.9 is verified and recommended).
- **C++ Compilers:** Visual Studio Build Tools with the "Desktop development with C++" workload installed (needed to compile specific libraries like `psutil` or `scikit-learn` from source if pre-built binaries are unavailable).
- **Git:** Installed and available on the system path.

---

## 2. Installation Steps

Execute the following commands in the terminal (Command Prompt, PowerShell, or Bash) to install the system:

1. **Clone the Repository:**
   ```bash
   git clone <repository_url>
   cd ARGUS_AI
   ```

2. **Initialize a Virtual Environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment:**
   - **Windows (PowerShell):**
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Windows (CMD):**
     ```cmd
     .\venv\Scripts\activate.bat
     ```
   - **Linux/macOS:**
     ```bash
     source venv/bin/activate
     ```

4. **Install System Dependencies:**
   Install both runtime and development tools using the configuration files:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
   *Alternative:* Use the project `Makefile`:
   ```bash
   make install
   make install-dev
   ```

---

## 3. Configuration Setup

Configuration settings reside inside the [configs/](file:///e:/ARGUS_AI/configs/) folder as YAML files:

- **[configs/base.yaml](file:///e:/ARGUS_AI/configs/base.yaml):** Set directory paths, image width/height parameters, and diagnostic parameters.
- **[configs/inference.yaml](file:///e:/ARGUS_AI/configs/inference.yaml):** Define camera index indices, target classification thresholds, confidence ranges, and tracking queues:
  - `live_threshold`: Target match floor (defaults to `0.85`).
  - `security_threshold`: Security alert ceiling (defaults to `0.90`).
  - `matching_policy`: Details similarity scores ranges (`confirmed_threshold`, `verify_low`, `verify_high`, etc.).
- **[configs/cameras.yaml](file:///e:/ARGUS_AI/configs/cameras.yaml):** List multiple camera devices, RTSP feeds, or local video files to process in parallel multi-camera mode.

---

## 4. Run Diagnostic Checks

ARGUS includes a system checking script to verify that directories, configuration structures, Python packages, and hardware acceleration drivers (CUDA) are properly configured.

Run the health diagnostics check:
```bash
python cli.py --mode health
```
*Alternative (Makefile):*
```bash
make health
```

The script will print out:
- CPU cores count and system memory details.
- CUDA accessibility (GPU) status.
- Necessary sub-directories verification (`data/`, `models/`, `outputs/`).
- Package validation outcomes.
- Config parser status.
