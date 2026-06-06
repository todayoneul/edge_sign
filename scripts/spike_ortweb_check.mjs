// export된 ONNX를 onnxruntime-web(node)로 로드해 1회 추론하고 출력 shape를 출력한다.
// ORT-Web이 모델을 파싱/실행하지 못하면(미지원 op 등) Plan B의 WebGPU 경로가 불가하므로
// 여기서 조기에 잡는다. (WebGPU EP는 브라우저 전용 → node에선 wasm으로 로드 가능성만 확인.)
import * as ort from "onnxruntime-web";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
// node 환경에서 wasm 바이너리 경로를 패키지 dist로 지정
const distDir = require.resolve("onnxruntime-web").replace(/[^\\/]+$/, "");
ort.env.wasm.wasmPaths = distDir;
ort.env.wasm.numThreads = 1;

const path = process.argv[2];
if (!path) {
  console.error("usage: node spike_ortweb_check.mjs <onnx_path>");
  process.exit(1);
}

const buf = readFileSync(path);
const sess = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
console.log("inputs:", sess.inputNames, "outputs:", sess.outputNames);

const x = new ort.Tensor("float32", new Float32Array(1 * 3 * 640 * 640), [1, 3, 640, 640]);
const feeds = { [sess.inputNames[0]]: x };
const out = await sess.run(feeds);
for (const k of sess.outputNames) console.log("out", k, out[k].dims);
console.log("[ortweb] OK — ORT-Web이 모델을 로드·실행함");
