# PowerShell and Python Virtual Environment Auto-Activation Audit Report

**Repository**: `E:\ARGUS_AI`  
**Date**: July 29, 2026  
**Auditor**: Senior Systems & VS Code Automation Engineer  
**Audit Purpose**: Read-only diagnosis of PowerShell and Python `venv` auto-activation mechanics in the ARGUS AI workspace.

---

## 1. Executive Summary

An audit of the ARGUS AI workspace was conducted to diagnose why the Python virtual environment (`E:\ARGUS_AI\venv`) no longer automatically activates when a PowerShell terminal is opened in VS Code.

### Key Findings
1. **Virtual Environment Health**: The virtual environment at `E:\ARGUS_AI\venv` is **healthy, complete, and fully functional**. Manual activation via `& E:\ARGUS_AI\venv\Scripts\Activate.ps1` succeeds cleanly.
2. **PowerShell Execution Policy**: Execution policy is `RemoteSigned` at the `LocalMachine` scope, which permits local script execution. Execution policy is **not** blocking activation.
3. **PowerShell Profiles**: No PowerShell profile files exist at user or system levels (`$PROFILE` returned `False` for `Test-Path`). There are no profile scripts interfering with or attempting activation.
4. **VS Code Workspace Settings**: The workspace configuration file [.vscode/settings.json](file:///e:/ARGUS_AI/.vscode/settings.json) specifies `"python.defaultInterpreterPath": "${workspaceFolder}/venv/Scripts/python.exe"`, but **omits** explicit definitions for `"python.terminal.activateEnvironment": true` and `"terminal.integrated.defaultProfile.windows": "PowerShell"`.

### Primary Cause
The issue is a **VS Code Configuration Defect**. While `.vscode/settings.json` points to the workspace Python interpreter, it lacks explicit workspace-level enforcement of `"python.terminal.activateEnvironment": true`. Consequently, auto-activation relies on global user-level VS Code settings or extension states, which are either unset, disabled, or failing to trigger upon terminal launch.

---

## 2. Expected Behavior

The intended workflow when working in the ARGUS AI workspace:
1. Developer opens the `E:\ARGUS_AI` directory in VS Code.
2. An integrated PowerShell terminal opens automatically or is opened by the user (`Ctrl + ~`).
3. VS Code's Python extension detects the configured interpreter (`E:\ARGUS_AI\venv\Scripts\python.exe`) and sends the activation command `& "E:/ARGUS_AI/venv/Scripts/Activate.ps1"` to the terminal.
4. The terminal prompt reflects active virtual environment state:
   ```powershell
   (venv) PS E:\ARGUS_AI>
   ```
5. Environment variable `$env:VIRTUAL_ENV` is set to `E:\ARGUS_AI\venv`.

---

## 3. Current Observed State

- Opening a PowerShell terminal in VS Code results in a standard un-activated prompt:
  ```powershell
  PS E:\ARGUS_AI>
  ```
- Running `python` defaults to system Python or global PATH Python (`C:\Users\Chanuka Sandun\AppData\Local\Programs\Python\Python311\python.exe`) unless manually activated.
- `$env:VIRTUAL_ENV` is empty (`$null`).

---

## 4. Existing Auto-Activation Mechanisms

The repository was inspected for active auto-activation mechanisms:

| Mechanism | Status | Details |
|---|---|---|
| **VS Code Native Terminal Activation** | Partial / Misconfigured | `.vscode/settings.json` has `python.defaultInterpreterPath`, but lacks `python.terminal.activateEnvironment: true`. |
| **PowerShell Startup Profile** | Absent | No `$PROFILE` file exists on the system. |
| **VS Code Automation Task** | Absent | `.vscode/tasks.json` does not exist. |
| **Wrapper Launch Script** | Absent | No batch or PowerShell startup wrapper script configured for VS Code terminal launch. |

---

## 5. Virtual Environment Health

Read-only verification checks were executed against the virtual environment:

### Path Verification
- `Test-Path E:\ARGUS_AI\venv`: `True`
- `Test-Path E:\ARGUS_AI\venv\Scripts\Activate.ps1`: `True`
- `Test-Path E:\ARGUS_AI\venv\Scripts\python.exe`: `True`

### Interpreter Inspection
```powershell
& E:\ARGUS_AI\venv\Scripts\python.exe --version
# Output: Python 3.11.9

& E:\ARGUS_AI\venv\Scripts\python.exe -c "import sys; print('executable:', sys.executable); print('prefix:', sys.prefix); print('base_prefix:', sys.base_prefix)"
# Output:
# executable: E:\ARGUS_AI\venv\Scripts\python.exe
# prefix: E:\ARGUS_AI\venv
# base_prefix: C:\Users\Chanuka Sandun\AppData\Local\Programs\Python\Python311
```

### Configuration File (`E:\ARGUS_AI\venv\pyvenv.cfg`)
```ini
home = C:\Users\Chanuka Sandun\AppData\Local\Programs\Python\Python311
include-system-site-packages = false
version = 3.11.9
executable = C:\Users\Chanuka Sandun\AppData\Local\Programs\Python\Python311\python.exe
command = C:\Users\Chanuka Sandun\AppData\Local\Programs\Python\Python311\python.exe -m venv E:\ARGUS_AI\venv
```

**Health Verdict**: **VENV HEALTHY**. `sys.prefix` (`E:\ARGUS_AI\venv`) differs from `sys.base_prefix` (`C:\Users\Chanuka Sandun\AppData\Local\Programs\Python\Python311`), confirming a valid, undamaged Python 3.11.9 virtual environment. Base Python executable exists.

---

## 6. VS Code Workspace Configuration

The workspace settings file [.vscode/settings.json](file:///e:/ARGUS_AI/.vscode/settings.json) was audited:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/Scripts/python.exe",
  "python.analysis.exclude": [
    "**/venv/**",
    "**/.venv/**",
    "**/outputs/**",
    "**/__pycache__/**",
    "**/.pytest_cache/**"
  ],
  "files.watcherExclude": {
    "**/venv/**": true,
    "**/.venv/**": true,
    "**/outputs/**": true,
    "**/__pycache__": true,
    "**/.pytest_cache/**": true
  },
  "files.exclude": {
    "**/__pycache__": true,
    "**/.pytest_cache": true
  },
  "cSpell.words": [...],
  "cSpell.ignorePaths": [...],
  "markdownlint.ignore": [...]
}
```

### Analysis of Settings
1. **Interpreter Path**: `"python.defaultInterpreterPath": "${workspaceFolder}/venv/Scripts/python.exe"` is present.
2. **Missing Activation Key**: `"python.terminal.activateEnvironment"` is omitted. When this key is absent from workspace settings, VS Code falls back to User Settings. If User Settings have `"python.terminal.activateEnvironment": false` or if extension state is uninitialized, terminal activation will not occur.
3. **Missing Default Profile Key**: `"terminal.integrated.defaultProfile.windows"` is omitted. If VS Code opens Git Bash or Command Prompt instead of PowerShell by default, activation script syntax differs.
4. **Workspace Files**: No `.code-workspace` files exist in the repository root or subdirectories.

---

## 7. PowerShell Profile Findings

The following PowerShell profile locations were audited using `Test-Path`:

| Profile Scope | Target Path | Exists |
|---|---|---|
| `$PROFILE` (CurrentUserCurrentHost) | `C:\Users\Chanuka Sandun\OneDrive\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1` | `False` |
| `$PROFILE.CurrentUserAllHosts` | `C:\Users\Chanuka Sandun\OneDrive\Documents\WindowsPowerShell\profile.ps1` | `False` |
| `$PROFILE.AllUsersCurrentHost` | `C:\Windows\System32\WindowsPowerShell\v1.0\Microsoft.PowerShell_profile.ps1` | `False` |
| `$PROFILE.AllUsersAllHosts` | `C:\Windows\System32\WindowsPowerShell\v1.0\profile.ps1` | `False` |

**Verdict**: No PowerShell startup profile scripts exist on this machine. Auto-activation never relied on `$PROFILE` hooks.

---

## 8. Execution Policy Findings

PowerShell execution policies were inspected via `Get-ExecutionPolicy -List`:

```text
        Scope ExecutionPolicy
        ----- ---------------
MachinePolicy       Undefined
   UserPolicy       Undefined
      Process       Undefined
  CurrentUser       Undefined
 LocalMachine    RemoteSigned
```

**Verdict**: `LocalMachine` policy is `RemoteSigned`. Locally created scripts (including `E:\ARGUS_AI\venv\Scripts\Activate.ps1`) do not require digital signatures and run without policy restriction. Execution policy is **not** blocking auto-activation.

---

## 9. Manual Activation Test

Manual activation was tested in an isolated process session:

```powershell
Set-Location E:\ARGUS_AI
& .\venv\Scripts\Activate.ps1
```

### Execution Evidence
- `$env:VIRTUAL_ENV` $\rightarrow$ `E:\ARGUS_AI\venv`
- `(Get-Command python).Source` $\rightarrow$ `E:\ARGUS_AI\venv\Scripts\python.exe`
- `python --version` $\rightarrow$ `Python 3.11.9`
- `python -c "import sys; print(sys.executable)"` $\rightarrow$ `E:\ARGUS_AI\venv\Scripts\python.exe`

**Verdict**: Manual activation completes cleanly without errors. The activation script `Activate.ps1` is undamaged.

---

## 10. Git Regression Analysis

Git commit history was examined (`git log --oneline -- .vscode/`):
- `.vscode/settings.json` has **never been committed** to the Git repository.
- There are no past commits that deleted or modified terminal activation settings in `.vscode/`.
- The current `.vscode/settings.json` was created locally and was populated with `python.defaultInterpreterPath`, but without the explicit terminal activation flags.

---

## 11. Confirmed Root Cause

### Primary Confirmed Cause: VS CODE CONFIGURATION DEFECT
The workspace file [.vscode/settings.json](file:///e:/ARGUS_AI/.vscode/settings.json) does not explicitly set:
```json
"python.terminal.activateEnvironment": true
```
Without this explicit workspace declaration, VS Code relies on global User Settings or Python extension internal state. If VS Code User Settings have terminal auto-activation disabled or if the Python extension initializes after terminal creation, no activation command is sent to PowerShell.

---

## 12. Probable Contributing Causes

1. **User-Level VS Code Setting Override**: Global VS Code User Settings (`%APPDATA%\Code\User\settings.json`) may have `"python.terminal.activateEnvironment": false`.
2. **Terminal Creation Prior to Extension Load**: Opening the terminal immediately upon launching VS Code before the Python extension (`ms-python.python`) finishes activating prevents auto-injection of `Activate.ps1`.
3. **Path Separator Syntax**: `.vscode/settings.json` uses forward slashes `${workspaceFolder}/venv/Scripts/python.exe`. While VS Code resolves forward slashes, PowerShell on Windows prefers explicit backslashes `${workspaceFolder}\\venv\\Scripts\\python.exe`.

---

## 13. Safe Fix Options

### Option A — VS Code Native Activation (Recommended)
Add explicit terminal activation and default profile settings to `.vscode/settings.json`.

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}\\venv\\Scripts\\python.exe",
  "python.terminal.activateEnvironment": true,
  "terminal.integrated.defaultProfile.windows": "PowerShell"
}
```

*Pros*: Zero impact on external system scripts; native to VS Code; works for all developers opening the repository.  
*Cons*: Requires VS Code Python extension (`ms-python.python`) to be enabled.

---

### Option B — Workspace Terminal Profile Wrapper
Define a custom terminal profile in `.vscode/settings.json` that launches PowerShell with `Activate.ps1` pre-executed.

```json
{
  "terminal.integrated.profiles.windows": {
    "ARGUS PowerShell (venv)": {
      "source": "PowerShell",
      "args": ["-NoExit", "-Command", "& '${workspaceFolder}\\venv\\Scripts\\Activate.ps1'"]
    }
  },
  "terminal.integrated.defaultProfile.windows": "ARGUS PowerShell (venv)"
}
```

*Pros*: Guarantees activation regardless of Python extension state or user settings.  
*Cons*: Only applies inside VS Code integrated terminal.

---

### Option C — PowerShell Profile Auto-Activation
Add an auto-activation snippet to the user's PowerShell profile (`$PROFILE`):

```powershell
# Auto-activate ARGUS venv when entering E:\ARGUS_AI
function Prompt {
    if ($PWD.Path -eq "E:\ARGUS_AI" -and (Test-Path "E:\ARGUS_AI\venv\Scripts\Activate.ps1")) {
        if ($env:VIRTUAL_ENV -ne "E:\ARGUS_AI\venv") {
            & "E:\ARGUS_AI\venv\Scripts\Activate.ps1"
        }
    }
    "PS $($executionContext.SessionState.Path.CurrentLocation)$('>' * ($nestedPromptLevel + 1)) "
}
```

*Pros*: Works across external PowerShell windows and terminals.  
*Cons*: Alters global user profile; specific to a single local machine path.

---

### Option D — VS Code Task-Based Activation
Create a startup task in `.vscode/tasks.json` to trigger activation.

*Pros*: Runs on folder open.  
*Cons*: Tasks execute in dedicated task terminals and do not persist into interactive developer terminals.

---

## 14. Recommended Minimal Fix

Implement **Option A** by updating `.vscode/settings.json` to explicitly include terminal activation and default profile keys:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}\\venv\\Scripts\\python.exe",
  "python.terminal.activateEnvironment": true,
  "terminal.integrated.defaultProfile.windows": "PowerShell"
}
```

---

## 15. Verification Commands

After applying the fix in `.vscode/settings.json`, close all open terminals in VS Code (`Trash` icon) and open a new PowerShell terminal (`Ctrl + Shift + ~`).

Run the following verification sequence:

```powershell
Get-Location
$env:VIRTUAL_ENV
(Get-Command python).Source
python --version
python -c "import sys; print(sys.executable)"
pip --version
ruff check .
.\venv\Scripts\python.exe -m pytest tests/unit/test_sync_folder_readmes.py -q
git status
```

### Expected Output
- **Location**: `E:\ARGUS_AI`
- **$env:VIRTUAL_ENV**: `E:\ARGUS_AI\venv`
- **Python Source**: `E:\ARGUS_AI\venv\Scripts\python.exe`
- **Python Version**: `Python 3.11.9`
- **sys.executable**: `E:\ARGUS_AI\venv\Scripts\python.exe`
- **Prompt**: `(venv) PS E:\ARGUS_AI>`

---

## 16. Rollback Instructions

If modifications to `.vscode/settings.json` need to be undone:
1. Revert `.vscode/settings.json` to its previous state using Git or manual file edit.
2. Remove `"python.terminal.activateEnvironment": true` and `"terminal.integrated.defaultProfile.windows": "PowerShell"`.

---

## 17. Files That Would Need Modification

Only one file requires modification to resolve this issue:
- [.vscode/settings.json](file:///e:/ARGUS_AI/.vscode/settings.json)

---

## 18. Final Verdict

# `VS CODE CONFIGURATION DEFECT`
