#!/usr/bin/env node

/**
 * ARGUS AI - Unified Development Process Orchestrator
 *
 * Automatically launches and manages both the Python FastAPI backend
 * (Uvicorn @ 127.0.0.1:8000) and the React Vite frontend (@ localhost:5173).
 * Synchronizes backend readiness before opening frontend traffic and guarantees
 * clean process tree teardown on Ctrl+C (preventing orphaned camera handles).
 */

import { spawn, execSync } from 'child_process';
import http from 'http';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');
const frontendDir = path.join(rootDir, 'frontend');

// ANSI Color formatting
const RESET = '\x1b[0m';
const BOLD = '\x1b[1m';
const CYAN = '\x1b[36m';
const GREEN = '\x1b[32m';
const YELLOW = '\x1b[33m';
const RED = '\x1b[31m';
const MAGENTA = '\x1b[35m';

function logArgus(msg) {
    console.log(`${BOLD}${GREEN}[ARGUS]${RESET} ${msg}`);
}

function logBackend(msg) {
    console.log(`${CYAN}[BACKEND]${RESET} ${msg}`);
}

function logFrontend(msg) {
    console.log(`${MAGENTA}[FRONTEND]${RESET} ${msg}`);
}

function logError(msg) {
    console.error(`${BOLD}${RED}[ARGUS ERROR]${RESET} ${msg}`);
}

/**
 * Locate Python executable within the project's virtual environment.
 */
function resolvePythonPath() {
    const isWin = process.platform === 'win32';
    const venvPythonWin = path.join(rootDir, 'venv', 'Scripts', 'python.exe');
    const venvPythonPosix = path.join(rootDir, 'venv', 'bin', 'python');

    if (isWin && fs.existsSync(venvPythonWin)) {
        return venvPythonWin;
    }
    if (!isWin && fs.existsSync(venvPythonPosix)) {
        return venvPythonPosix;
    }

    if (process.env.VIRTUAL_ENV) {
        const customVenv = isWin
            ? path.join(process.env.VIRTUAL_ENV, 'Scripts', 'python.exe')
            : path.join(process.env.VIRTUAL_ENV, 'bin', 'python');
        if (fs.existsSync(customVenv)) return customVenv;
    }

    // Fallback to system python
    return isWin ? 'python.exe' : 'python3';
}

/**
 * Verify backend health via HTTP request.
 */
function checkBackendHealth(timeoutMs = 1000) {
    return new Promise((resolve) => {
        const req = http.get(
            {
                hostname: '127.0.0.1',
                port: 8000,
                path: '/api/v1/health',
                timeout: timeoutMs,
            },
            (res) => {
                if (res.statusCode === 200) {
                    resolve(true);
                } else {
                    resolve(false);
                }
            }
        );

        req.on('error', () => resolve(false));
        req.on('timeout', () => {
            req.destroy();
            resolve(false);
        });
    });
}

/**
 * Cleanly kill a child process and all its sub-processes.
 */
function killProcessTree(proc) {
    if (!proc || !proc.pid) return;
    try {
        if (process.platform === 'win32') {
            execSync(`taskkill /pid ${proc.pid} /T /F`, { stdio: 'ignore' });
        } else {
            process.kill(-proc.pid, 'SIGKILL');
        }
    } catch {
        try {
            proc.kill('SIGKILL');
        } catch {
            // Already exited
        }
    }
}

async function main() {
    const pythonExe = resolvePythonPath();

    logArgus(`${BOLD}ARGUS AI Unified Development Environment${RESET}`);
    logArgus(`Project Root: ${rootDir}`);
    logArgus(`Python Path:  ${pythonExe}`);

    if (!fs.existsSync(pythonExe) && !pythonExe.includes('python')) {
        logError(`Python virtual environment not found at: ${pythonExe}`);
        process.exit(1);
    }

    let backendProc = null;
    let frontendProc = null;
    let isShuttingDown = false;

    const cleanup = () => {
        if (isShuttingDown) return;
        isShuttingDown = true;
        logArgus(`${YELLOW}Shutting down ARGUS AI services...${RESET}`);

        if (frontendProc) {
            logArgus('Stopping frontend dev server...');
            killProcessTree(frontendProc);
            frontendProc = null;
        }

        if (backendProc) {
            logArgus('Stopping FastAPI backend server...');
            killProcessTree(backendProc);
            backendProc = null;
        }

        logArgus(`${GREEN}ARGUS AI services stopped cleanly.${RESET}`);
        process.exit(0);
    };

    process.on('SIGINT', cleanup);
    process.on('SIGTERM', cleanup);
    if (process.platform === 'win32') {
        process.on('SIGBREAK', cleanup);
    }

    // -------------------------------------------------------------------------
    // 1. Start Backend Process (FastAPI / Uvicorn)
    // -------------------------------------------------------------------------
    logArgus('Starting FastAPI backend server on http://127.0.0.1:8000 ...');

    const backendArgs = [
        '-m',
        'uvicorn',
        'api.server:app',
        '--host',
        '127.0.0.1',
        '--port',
        '8000',
        '--reload',
    ];

    backendProc = spawn(pythonExe, backendArgs, {
        cwd: rootDir,
        env: {
            ...process.env,
            PYTHONUNBUFFERED: '1',
            PYTHONPATH: rootDir,
        },
        shell: false,
    });

    let backendStartupFailed = false;
    let backendStderrBuffer = '';
    let backendImportError = false;

    backendProc.stdout.on('data', (data) => {
        const text = data.toString();
        text.split(/\r?\n/).forEach((line) => {
            if (line.trim()) logBackend(line);
        });
    });

    backendProc.stderr.on('data', (data) => {
        const text = data.toString();
        backendStderrBuffer += text;

        // Detect fatal import errors early so we can fast-fail
        if (
            text.includes('ModuleNotFoundError') ||
            text.includes('ImportError') ||
            text.includes('Error loading ASGI app')
        ) {
            backendImportError = true;
        }

        text.split(/\r?\n/).forEach((line) => {
            if (line.trim()) logBackend(line);
        });
    });

    backendProc.on('exit', (code, signal) => {
        if (!isShuttingDown) {
            logError(`Backend server exited unexpectedly with code ${code} (${signal})`);
            if (backendStderrBuffer) {
                console.error(backendStderrBuffer);
            }
            backendStartupFailed = true;
            cleanup();
        }
    });

    // -------------------------------------------------------------------------
    // 2. Wait for Backend Health Synchronization
    // -------------------------------------------------------------------------
    logArgus('Waiting for backend service readiness...');
    logArgus('(ML model loading may take 30-90s on first run with CUDA)');

    // Allow up to 120s for CUDA/PyTorch model initialization on cold starts.
    // Warm starts typically complete within 10-15s.
    const maxWaitMs = 120000;
    const progressIntervalMs = 5000;
    const startWait = Date.now();
    let backendReady = false;
    let lastProgressLog = startWait;

    while (Date.now() - startWait < maxWaitMs) {
        if (backendStartupFailed || isShuttingDown) break;

        // Fast-fail on import/module errors instead of waiting for timeout
        if (backendImportError) {
            logError('Backend failed with an import error (see output above).');
            break;
        }

        const healthy = await checkBackendHealth(500);
        if (healthy) {
            backendReady = true;
            break;
        }

        // Log progress every 5 seconds so the user knows we are still waiting
        const now = Date.now();
        if (now - lastProgressLog >= progressIntervalMs) {
            const elapsed = Math.round((now - startWait) / 1000);
            logArgus(`${YELLOW}Still waiting for backend... (${elapsed}s elapsed)${RESET}`);
            lastProgressLog = now;
        }

        await new Promise((r) => setTimeout(r, 250));
    }

    if (!backendReady) {
        const elapsed = Math.round((Date.now() - startWait) / 1000);
        if (backendImportError) {
            logError('Backend failed to start due to an import error.');
        } else if (backendStartupFailed) {
            logError('Backend process exited before becoming healthy.');
        } else {
            logError(`FastAPI backend failed to become healthy within ${elapsed}s timeout.`);
        }
        if (backendStderrBuffer) {
            console.error(backendStderrBuffer);
        }
        cleanup();
        return;
    }

    logArgus(`${BOLD}${GREEN}✓ Backend API is ready and healthy at http://127.0.0.1:8000${RESET}`);

    // -------------------------------------------------------------------------
    // 3. Start Frontend Process (Vite)
    // -------------------------------------------------------------------------
    logArgus('Starting Vite frontend dev server on http://localhost:5173 ...');

    const isWin = process.platform === 'win32';
    const frontendCommand = isWin ? 'cmd.exe' : 'npm';
    const frontendArgs = isWin
        ? ['/d', '/s', '/c', 'npm run dev:frontend']
        : ['run', 'dev:frontend'];

    frontendProc = spawn(frontendCommand, frontendArgs, {
        cwd: frontendDir,
        env: {
            ...process.env,
        },
        shell: false,
    });

    frontendProc.stdout.on('data', (data) => {
        const text = data.toString();
        text.split(/\r?\n/).forEach((line) => {
            if (line.trim()) logFrontend(line);
        });
    });

    frontendProc.stderr.on('data', (data) => {
        const text = data.toString();
        text.split(/\r?\n/).forEach((line) => {
            if (line.trim()) logFrontend(line);
        });
    });

    frontendProc.on('exit', (code, signal) => {
        if (!isShuttingDown) {
            logArgus(`Frontend dev server stopped with code ${code} (${signal})`);
            cleanup();
        }
    });

    logArgus(`${BOLD}${GREEN}====================================================${RESET}`);
    logArgus(`${BOLD}${GREEN}ARGUS AI IS READY!${RESET}`);
    logArgus(`• Web Portal:  ${BOLD}${CYAN}http://localhost:5173/${RESET}`);
    logArgus(`• API Backend: ${BOLD}${CYAN}http://127.0.0.1:8000/docs${RESET}`);
    logArgus(`• Camera Zone: ${BOLD}${CYAN}http://localhost:5173/cctv-network${RESET}`);
    logArgus(`${BOLD}${GREEN}====================================================${RESET}`);
}

main().catch((err) => {
    logError(`Fatal startup error: ${err.message}`);
    process.exit(1);
});
