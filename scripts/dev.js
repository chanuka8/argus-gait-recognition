#!/usr/bin/env node
/**
 * ARGUS AI - Backward-compatibility shim for scripts/dev.js.
 * Forwards execution to tools/dev.js.
 */
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const targetPath = path.resolve(__dirname, '..', 'tools', 'dev.js');

import(targetPath);
