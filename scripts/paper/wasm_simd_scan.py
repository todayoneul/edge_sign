"""Static scan of a WebAssembly module: which SIMD instructions each function uses.

Step 0 of the Mac WASM INT8 analysis (docs/ROADMAP.md): does the ORT-Web build that the paper measured
use fixed or relaxed SIMD, and which instructions carry its INT8 dot products?

The decoder walks every function body of the code section instruction by instruction (MVP, sign-extension,
saturating conversions, bulk memory, reference types, exceptions, threads/atomics, fixed and relaxed SIMD).
Self-check: every body must end exactly at its declared size with an `end` opcode; the report counts the
bodies that do not (`decode_mismatches`, expected 0).

The module has no name section, so functions are identified by their index in the function index space
(imported functions first) and by the byte offset of their body, which is what wasm_kernel_breakdown.py
matches against V8 CPU-profile frames.

Usage: python scripts/paper/wasm_simd_scan.py <module.wasm> [--output summary.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

SIMD_NAMES = {
    0xBA: "i32x4.dot_i16x8_s",
    0x87: "i16x8.extend_low_i8x16_s", 0x88: "i16x8.extend_high_i8x16_s",
    0x89: "i16x8.extend_low_i8x16_u", 0x8A: "i16x8.extend_high_i8x16_u",
    0x9C: "i16x8.extmul_low_i8x16_s", 0x9D: "i16x8.extmul_high_i8x16_s",
    0x9E: "i16x8.extmul_low_i8x16_u", 0x9F: "i16x8.extmul_high_i8x16_u",
    0xBC: "i32x4.extmul_low_i16x8_s", 0xBD: "i32x4.extmul_high_i16x8_s",
    0x7C: "i16x8.extadd_pairwise_i8x16_s", 0x7D: "i16x8.extadd_pairwise_i8x16_u",
    0x7E: "i32x4.extadd_pairwise_i16x8_s",
    0xAE: "i32x4.add", 0xE4: "f32x4.add", 0xE6: "f32x4.mul", 0xE7: "f32x4.div", 0x95: "i16x8.mul", 0xB5: "i32x4.mul",
    0xF8: "i32x4.trunc_sat_f32x4_s", 0xFA: "f32x4.convert_i32x4_s",
    0x65: "i8x16.narrow_i16x8_s", 0x66: "i8x16.narrow_i16x8_u",
    0x85: "i16x8.narrow_i32x4_s", 0x86: "i16x8.narrow_i32x4_u",
    0x100: "i8x16.relaxed_swizzle", 0x101: "i32x4.relaxed_trunc_f32x4_s", 0x102: "i32x4.relaxed_trunc_f32x4_u",
    0x103: "i32x4.relaxed_trunc_f64x2_s_zero", 0x104: "i32x4.relaxed_trunc_f64x2_u_zero",
    0x105: "f32x4.relaxed_madd", 0x106: "f32x4.relaxed_nmadd", 0x107: "f64x2.relaxed_madd",
    0x108: "f64x2.relaxed_nmadd", 0x109: "i8x16.relaxed_laneselect", 0x10A: "i16x8.relaxed_laneselect",
    0x10B: "i32x4.relaxed_laneselect", 0x10C: "i64x2.relaxed_laneselect", 0x10D: "f32x4.relaxed_min",
    0x10E: "f32x4.relaxed_max", 0x10F: "f64x2.relaxed_min", 0x110: "f64x2.relaxed_max",
    0x111: "i16x8.relaxed_q15mulr_s", 0x112: "i16x8.relaxed_dot_i8x16_i7x16_s",
    0x113: "i32x4.relaxed_dot_i8x16_i7x16_add_s",
}
RELAXED = range(0x100, 0x114)

_MEM = set(range(0x28, 0x3F))  # loads/stores: memarg
_ONE_IDX = {0x07, 0x08, 0x09, 0x0C, 0x0D, 0x10, 0x12, 0x18, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0xD2}
_NO_IMM = set(range(0x45, 0xC5)) | {0x00, 0x01, 0x05, 0x0B, 0x0F, 0x19, 0x1A, 0x1B, 0xD1, 0xD4}
_SIMD_MEM = set(range(0x00, 0x0C)) | {0x5C, 0x5D}
_SIMD_MEM_LANE = set(range(0x54, 0x5C))
_SIMD_LANE = set(range(0x15, 0x23))


def _uleb(b: bytes, i: int) -> tuple[int, int]:
    r = s = 0
    while True:
        x = b[i]
        i += 1
        r |= (x & 0x7F) << s
        s += 7
        if not x & 0x80:
            return r, i


def _sleb(b: bytes, i: int) -> tuple[int, int]:
    r = s = 0
    while True:
        x = b[i]
        i += 1
        r |= (x & 0x7F) << s
        s += 7
        if not x & 0x80:
            return (r - (1 << s) if x & 0x40 else r), i


def _blocktype(b: bytes, i: int) -> int:
    if b[i] in (0x40, 0x7F, 0x7E, 0x7D, 0x7C, 0x7B, 0x70, 0x6F):
        return i + 1
    return _sleb(b, i)[1]  # type index (s33)


def _memarg(b: bytes, i: int) -> int:
    align, i = _uleb(b, i)
    if align & 0x40:  # multi-memory: explicit memory index
        i = _uleb(b, i)[1]
    return _uleb(b, i)[1]


def _decode(b: bytes, i: int, end: int, simd: Counter) -> int:
    while i < end:
        op = b[i]
        i += 1
        if op in _NO_IMM:
            continue
        if op in (0x02, 0x03, 0x04, 0x06):
            i = _blocktype(b, i)
        elif op in _ONE_IDX or op in (0x3F, 0x40):
            i = _uleb(b, i)[1]
        elif op == 0x0E:  # br_table
            n, i = _uleb(b, i)
            for _ in range(n + 1):
                i = _uleb(b, i)[1]
        elif op in (0x11, 0x13):  # call_indirect / return_call_indirect
            i = _uleb(b, _uleb(b, i)[1])[1]
        elif op == 0x1C:  # select t*
            n, i = _uleb(b, i)
            i += n
        elif op in _MEM:
            i = _memarg(b, i)
        elif op in (0x41, 0x42):
            i = _sleb(b, i)[1]
        elif op == 0x43:
            i += 4
        elif op == 0x44:
            i += 8
        elif op == 0xD0:
            i += 1
        elif op == 0x1F:  # try_table
            i = _blocktype(b, i)
            n, i = _uleb(b, i)
            for _ in range(n):
                kind = b[i]
                i += 1
                if kind in (0, 1):
                    i = _uleb(b, i)[1]
                i = _uleb(b, i)[1]
        elif op == 0xFC:
            sub, i = _uleb(b, i)
            if sub in (8, 10, 12, 14):  # memory.init, memory.copy, table.init, table.copy
                i = _uleb(b, _uleb(b, i)[1])[1]
            elif sub in (9, 11, 13, 15, 16, 17):
                i = _uleb(b, i)[1]
            elif sub > 7:
                raise ValueError(f"unknown 0xFC {sub}")
        elif op == 0xFE:  # threads/atomics
            sub, i = _uleb(b, i)
            i = i + 1 if sub == 0x03 else _memarg(b, i)
        elif op == 0xFD:
            sub, i = _uleb(b, i)
            simd[sub] += 1
            if sub in _SIMD_MEM:
                i = _memarg(b, i)
            elif sub in _SIMD_MEM_LANE:
                i = _memarg(b, i) + 1
            elif sub in (0x0C, 0x0D):  # v128.const / i8x16.shuffle
                i += 16
            elif sub in _SIMD_LANE:
                i += 1
        else:
            raise ValueError(f"unknown opcode 0x{op:02X} at byte {i - 1}")
    return i


@dataclass
class Function:
    index: int  # in the function index space (imports first)
    body_start: int  # byte offset of the body (its size field) in the module
    body_end: int
    simd: Counter = field(default_factory=Counter)

    def has(self, opcode: int) -> bool:
        return self.simd.get(opcode, 0) > 0


def scan(module: bytes) -> tuple[list[Function], int]:
    """All defined functions with their SIMD opcode counts, and the number of bodies that failed the self-check."""
    if module[:4] != b"\0asm":
        raise ValueError("not a wasm module")
    i, imported_funcs, code = 8, 0, None
    while i < len(module):
        sid = module[i]
        size, j = _uleb(module, i + 1)
        if sid == 2:  # import section: count imported functions
            n, p = _uleb(module, j)
            for _ in range(n):
                for _ in range(2):  # module and field names
                    ln, p = _uleb(module, p)
                    p += ln
                kind = module[p]
                p += 1
                if kind == 0:
                    imported_funcs += 1
                    p = _uleb(module, p)[1]
                elif kind == 1:  # table: reftype + limits
                    p += 1
                    flags, p = _uleb(module, p)
                    p = _uleb(module, p)[1]
                    if flags & 1:
                        p = _uleb(module, p)[1]
                elif kind == 2:  # memory: limits
                    flags, p = _uleb(module, p)
                    p = _uleb(module, p)[1]
                    if flags & 1:
                        p = _uleb(module, p)[1]
                elif kind == 3:  # global: valtype + mutability
                    p += 2
                elif kind == 4:  # tag
                    p = _uleb(module, p + 1)[1]
        elif sid == 10:
            code = (j, j + size)
        i = j + size
    if code is None:
        raise ValueError("no code section")
    p = code[0]
    n, p = _uleb(module, p)
    functions, bad = [], 0
    for k in range(n):
        start = p
        size, p = _uleb(module, p)
        end = p + size
        q = p
        groups, q = _uleb(module, q)
        for _ in range(groups):
            q = _uleb(module, q)[1] + 1
        simd: Counter = Counter()
        if _decode(module, q, end, simd) != end or module[end - 1] != 0x0B:
            bad += 1
        functions.append(Function(imported_funcs + k, start, end, simd))
        p = end
    return functions, bad


def summary(path: Path) -> dict:
    module = path.read_bytes()
    functions, bad = scan(module)
    total: Counter = Counter()
    for f in functions:
        total.update(f.simd)
    return {
        "module": path.name, "sha256": hashlib.sha256(module).hexdigest(), "bytes": len(module),
        "defined_functions": len(functions), "decode_mismatches": bad,
        "functions_with_simd": sum(1 for f in functions if f.simd),
        "simd_instructions": sum(total.values()),
        "relaxed_simd_instructions": sum(total[k] for k in RELAXED),
        "relaxed_simd": {SIMD_NAMES[k]: total[k] for k in RELAXED if total[k]},
        "functions_with_i32x4_dot_i16x8_s": sum(1 for f in functions if f.has(0xBA)),
        "selected": {SIMD_NAMES[k]: total[k] for k in sorted(SIMD_NAMES) if k < 0x100},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("module", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summary(args.module)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
