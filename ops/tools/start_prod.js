#!/usr/bin/env node

/**
 * ARGUS AI - Production Launcher
 *
 * Builds the frontend once and serves it as static assets straight out of
 * FastAPI (app/api/server.py already mounts frontend/dist when present) --
 * no Vite dev server, no HMR, no file-watcher polling. The backend runs
 * without --reload / --reload-dir, so WatchFiles never scans the tree.
 *
 * Env overrides: ARGUS_HOST (default 127.0.0.1), ARGUS_PORT (default 8000),
 * ARGUS_LOG_LEVEL (default info), ARGUS_SKIP_BUILD=1 to reuse an existing
 * frontend/dist instead of rebuilding it.
 */

import { spawn, execSync } from 'child_process';
import http from 'http';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..', '..');
const frontendDir = path.join(rootDir, 'frontend');
const distDir = path.join(frontendDir, 'dist');

const RESET = '\x1b[0m';
const BOLD = '\x1b[1m';
const CYAN = '\x1b[36m';
const GREEN = '\x1b[32m';
const YELLOW = '\x1b[33m';
const RED = '\x1b[31m';

const HOST = process.env.ARGUS_HOST || '127.0.0.1';
const PORT = process.env.ARGUS_PORT || '8000';
const LOG_LEVEL = process.env.ARGUS_LOG_LEVEL || 'info';
const SKIP_BUILD = process.env.ARGUS_SKIP_BUILD === '1';

function logArgus(msg) {
    console.log(`${BOLD}${GREEN}[ARGUS]${RESET} ${msg}`);
}
function logBackend(msg) {
    console.log(`${CYAN}[BACKEND]${RESET} ${msg}`);
}
function logError(msg) {
    console.error(`${BOLD}${RED}[ARGUS ERROR]${RESET} ${msg}`);
}

function resolvePythonPath() {
    const isWin = process.platform === 'win32';
    const dotVenvPython = isWin
        ? path.join(rootDir, '.venv', 'Scripts', 'python.exe')
        : path.join(rootDir, '.venv', 'bin', 'python');
    const legacyVenvPython = isWin
        ? path.join(rootDir, 'venv', 'Scripts', 'python.exe')
        : path.join(rootDir, 'venv', 'bin', 'python');

    if (fs.existsSync(dotVenvPython)) return dotVenvPython;
    if (fs.existsSync(legacyVenvPython)) return legacyVenvPython;

    if (process.env.VIRTUAL_ENV) {
        const customVenv = isWin
            ? path.join(process.env.VIRTUAL_ENV, 'Scripts', 'python.exe')
            : path.join(process.env.VIRTUAL_ENV, 'bin', 'python');
        if (fs.existsSync(customVenv)) return customVenv;
    }

    return isWin ? 'python.exe' : 'python3';
}

function checkBackendHealth(timeoutMs = 1000) {
    return new Promise((resolve) => {
        const req = http.get(
            { hostname: HOST, port: PORT, path: '/api/v1/health', timeout: timeoutMs },
            (res) => resolve(res.statusCode === 200)
        );
        req.on('error', () => resolve(false));
        req.on('timeout', () => {
            req.destroy();
            resolve(false);
        });
    });
}

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

function runBuild() {
    return new Promise((resolve, reject) => {
        logArgus('Building production frontend bundle (vite build)...');
        const isWin = process.platform === 'win32';
        const cmd = isWin ? 'cmd.exe' : 'npm';
        const args = isWin ? ['/d', '/s', '/c', 'npm run build'] : ['run', 'build'];
        const build = spawn(cmd, args, { cwd: frontendDir, stdio: 'inherit', shell: false });
        build.on('exit', (code) => {
            if (code === 0) resolve();
            else reject(new Error(`frontend build failed with exit code ${code}`));
        });
        build.on('error', reject);
    });
}

async function main() {
    const pythonExe = resolvePythonPath();

    logArgus(`${BOLD}ARGUS AI Production Launcher${RESET}`);
    logArgus(`Project Root: ${rootDir}`);
    logArgus(`Python Path:  ${pythonExe}`);

    if (!fs.existsSync(pythonExe) && !pythonExe.includes('python')) {
        logError(`Python virtual environment not found at: ${pythonExe}`);
        process.exit(1);
    }

    if (SKIP_BUILD) {
        if (!fs.existsSync(path.join(distDir, 'index.html'))) {
            logError(`ARGUS_SKIP_BUILD=1 but no build found at ${distDir}. Run "npm run build" first.`);
            process.exit(1);
        }
        logArgus('ARGUS_SKIP_BUILD=1: reusing existing frontend/dist.');
    } else {
        try {
            await runBuild();
        } catch (err) {
            logError(err.message);
            process.exit(1);
        }
    }

    let backendProc = null;
    let isShuttingDown = false;

    const cleanup = () => {
        if (isShuttingDown) return;
        isShuttingDown = true;
        logArgus(`${YELLOW}Shutting down ARGUS AI...${RESET}`);
        if (backendProc) {
            killProcessTree(backendProc);
            backendProc = null;
        }
        logArgus(`${GREEN}ARGUS AI stopped cleanly.${RESET}`);
        process.exit(0);
    };

    process.on('SIGINT', cleanup);
    process.on('SIGTERM', cleanup);
    if (process.platform === 'win32') {
        process.on('SIGBREAK', cleanup);
    }

    logArgus(`Starting FastAPI backend (production, no reload) on http://${HOST}:${PORT} ...`);

    const backendArgs = [
        '-m', 'uvicorn',
        'app.api.server:app',
        '--host', HOST,
        '--port', String(PORT),
        '--log-level', LOG_LEVEL,
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
        data.toString().split(/\r?\n/).forEach((line) => {
            if (line.trim()) logBackend(line);
        });
    });

    backendProc.stderr.on('data', (data) => {
        const text = data.toString();
        backendStderrBuffer += text;
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
            if (backendStderrBuffer) console.error(backendStderrBuffer);
            backendStartupFailed = true;
            cleanup();
        }
    });

    logArgus('Waiting for backend service readiness...');
    const maxWaitMs = 120000;
    const progressIntervalMs = 5000;
    const startWait = Date.now();
    let backendReady = false;
    let lastProgressLog = startWait;

    while (Date.now() - startWait < maxWaitMs) {
        if (backendStartupFailed || isShuttingDown) break;
        if (backendImportError) {
            logError('Backend failed with an import error (see output above).');
            break;
        }

        const healthy = await checkBackendHealth(500);
        if (healthy) {
            backendReady = true;
            break;
        }

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
        if (!backendImportError && !backendStartupFailed) {
            logError(`FastAPI backend failed to become healthy within ${elapsed}s timeout.`);
        }
        if (backendStderrBuffer) console.error(backendStderrBuffer);
        cleanup();
        return;
    }

    logArgus(`${BOLD}${GREEN}====================================================${RESET}`);
    logArgus(`${BOLD}${GREEN}ARGUS AI IS READY (production mode)${RESET}`);
    logArgus(`• Web Portal + API: ${BOLD}${CYAN}http://${HOST}:${PORT}/${RESET}`);
    logArgus(`• API Docs:         ${BOLD}${CYAN}http://${HOST}:${PORT}/docs${RESET}`);
    logArgus(`${BOLD}${GREEN}====================================================${RESET}`);
    logArgus('Note: ML models still warm up in the background after this point; check /api/v1/readiness.');
}

main().catch((err) => {
    logError(`Fatal startup error: ${err.message}`);
    process.exit(1);
});
