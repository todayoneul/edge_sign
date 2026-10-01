// V8 CPU profile of ORT-Web WASM inference, step 1 of the Mac WASM INT8 analysis (docs/ROADMAP.md).
//
// For each model and round: a fresh headless Chrome opens wasm_kernel_profile.html (served by
// `runtime_matrix.py serve`), the page warms up, then the V8 sampling profiler runs around window.profileRun(n)
// only. Writes <run>.cpuprofile.gz (DevTools format, gzip) and <run>.json (latencies, Chrome/ORT versions, the .wasm
// that was loaded). wasm_kernel_breakdown.py maps the samples to wasm functions.
//
// The profiler samples the page's main thread, so run with 1 WASM thread (all inference on that thread); the
// Mac shows the same effect at 1 thread (FP32/INT8 0.91) as at 4.
//
// Usage (server running on --server-url):
//   node scripts/paper/wasm_kernel_profile.mjs --out paper_evidence/extra/wasm_int8/mac \
//     [--models v4_fp32 v4_int8_head_excl] [--rounds 2] [--iterations 60] [--warmup 20] [--interval-us 100]
import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import zlib from 'node:zlib';

const CHROME = {
  darwin: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  win32: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
}[process.platform] ?? 'google-chrome';

function parseArgs(argv) {
  const args = { serverUrl: 'http://127.0.0.1:8791', models: ['v4_fp32', 'v4_int8_head_excl'], rounds: 2,
    iterations: 60, warmup: 20, threads: 1, intervalUs: 100, ort: '1.30.0', chrome: CHROME, out: null, timeoutS: 600 };
  const keys = { '--server-url': 'serverUrl', '--rounds': 'rounds', '--iterations': 'iterations', '--warmup': 'warmup',
    '--threads': 'threads', '--interval-us': 'intervalUs', '--ort': 'ort', '--chrome': 'chrome', '--out': 'out',
    '--timeout-s': 'timeoutS' };
  for (let i = 2; i < argv.length; i++) {
    if (argv[i] === '--models') {
      args.models = [];
      while (argv[i + 1] && !argv[i + 1].startsWith('--')) args.models.push(argv[++i]);
    } else if (keys[argv[i]]) {
      const key = keys[argv[i]], value = argv[++i];
      args[key] = typeof args[key] === 'number' ? Number(value) : value;
    } else {
      throw new Error(`unknown argument ${argv[i]}`);
    }
  }
  if (!args.out) throw new Error('--out is required');
  return args;
}

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

class Cdp {
  constructor(url) {
    this.ws = new WebSocket(url);
    this.next = 1;
    this.pending = new Map();
    this.ws.onmessage = event => {
      const msg = JSON.parse(event.data);
      const entry = msg.id && this.pending.get(msg.id);
      if (!entry) return;
      this.pending.delete(msg.id);
      if (msg.error) entry.reject(new Error(`${entry.method}: ${msg.error.message}`));
      else entry.resolve(msg.result);
    };
  }
  open() {
    return new Promise((resolve, reject) => { this.ws.onopen = resolve; this.ws.onerror = reject; });
  }
  send(method, params = {}) {
    const id = this.next++;
    this.ws.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => this.pending.set(id, { resolve, reject, method }));
  }
  async evaluate(expression, awaitPromise = false) {
    const r = await this.send('Runtime.evaluate', { expression, awaitPromise, returnByValue: true });
    if (r.exceptionDetails) throw new Error(`${expression}: ${JSON.stringify(r.exceptionDetails).slice(0, 400)}`);
    return r.result.value;
  }
  close() { this.ws.close(); }
}

function killTree(child) {
  try {
    if (process.platform === 'win32') spawnSync('taskkill', ['/PID', String(child.pid), '/T', '/F']);
    else process.kill(-child.pid, 'SIGKILL');
  } catch { /* already gone */ }
}

function summarize(xs) {
  const s = [...xs].sort((a, b) => a - b), mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  const p = f => { const k = (s.length - 1) * f, lo = Math.floor(k), hi = Math.min(lo + 1, s.length - 1); return s[lo] + (s[hi] - s[lo]) * (k - lo); };
  return { n: xs.length, mean_ms: mean, p50_ms: p(0.5), p90_ms: p(0.9) };
}

async function profileOnce(args, model, run) {
  const userDir = fs.mkdtempSync(path.join(os.tmpdir(), 'edge_sign_profile_'));
  const query = new URLSearchParams({ model, ort: args.ort, warmup: args.warmup, threads: args.threads });
  const url = `${args.serverUrl}/wasm_kernel_profile.html?${query}`;
  const child = spawn(args.chrome, ['--headless=new', '--remote-debugging-port=0', `--user-data-dir=${userDir}`,
    '--no-first-run', '--no-default-browser-check', '--disable-background-networking', url],
  { stdio: 'ignore', detached: process.platform !== 'win32' });
  const result = { run, model, status: 'error', iterations: args.iterations, sampling_interval_us: args.intervalUs,
    load_average_before: os.loadavg(), platform: `${process.platform} ${os.arch()} ${os.cpus()[0]?.model ?? ''}` };
  let cdp;
  try {
    const portFile = path.join(userDir, 'DevToolsActivePort');
    const deadline = Date.now() + args.timeoutS * 1000;
    while (!fs.existsSync(portFile)) {
      if (Date.now() > deadline) throw new Error('DevToolsActivePort not written');
      await sleep(200);
    }
    const port = fs.readFileSync(portFile, 'utf8').split('\n')[0].trim();
    result.browser = (await (await fetch(`http://127.0.0.1:${port}/json/version`)).json()).Browser;
    let target;
    while (!target) {
      const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
      target = targets.find(t => t.type === 'page' && t.url.includes('wasm_kernel_profile'));
      if (!target) await sleep(200);
    }
    cdp = new Cdp(target.webSocketDebuggerUrl);
    await cdp.open();
    let state;
    do {
      if (Date.now() > deadline) throw new Error('page did not become ready');
      await sleep(500);
      state = await cdp.evaluate('JSON.stringify(window.profileState ?? {status: "loading"})').then(JSON.parse);
    } while (state.status === 'loading');
    if (state.status !== 'ready') throw new Error(state.reason ?? `page status ${state.status}`);
    await cdp.send('Profiler.enable');
    await cdp.send('Profiler.setSamplingInterval', { interval: args.intervalUs });
    await cdp.send('Profiler.start');
    const times = await cdp.evaluate(`window.profileRun(${args.iterations})`, true);
    const { profile } = await cdp.send('Profiler.stop');
    fs.writeFileSync(path.join(args.out, `${run}.cpuprofile.gz`), zlib.gzipSync(JSON.stringify(profile), { level: 9 }));
    Object.assign(result, { status: 'completed', page: state, inference_ms: summarize(times), times_ms: times,
      profile_samples: profile.samples.length, profile_span_ms: (profile.endTime - profile.startTime) / 1000 });
  } catch (error) {
    result.reason = String(error?.stack ?? error);
  } finally {
    cdp?.close();
    killTree(child);
    await sleep(1000);
    fs.rmSync(userDir, { recursive: true, force: true });
  }
  return result;
}

async function main() {
  const args = parseArgs(process.argv);
  fs.mkdirSync(args.out, { recursive: true });
  const tag = `wasmt${args.threads}_ort${args.ort.replaceAll('.', '')}`;
  for (let round = 1; round <= args.rounds; round++) {
    const order = round % 2 ? args.models : [...args.models].reverse();  // alternate which model goes first
    for (const model of order) {
      const run = `profile_${tag}_${model}_r${round}`;
      const target = path.join(args.out, `${run}.json`);
      if (fs.existsSync(target)) { console.log(`skip existing ${run}`); continue; }
      const result = await profileOnce(args, model, run);
      fs.writeFileSync(target, JSON.stringify(result, null, 2) + '\n');
      const m = result.inference_ms;
      console.log(`${run}: ${result.status}${m ? ` mean ${m.mean_ms.toFixed(2)} ms p50 ${m.p50_ms.toFixed(2)} samples ${result.profile_samples}` : ''} ${result.reason ?? ''}`);
    }
  }
}

main().catch(error => { console.error(error); process.exit(1); });
