# Handoff: Moon repositioning + mitigation/calibration experiments (2026-09-27)

Worktree: `C:\Users\leegy\.codex\worktrees\tiis-evidence-revalidation\CNN_Quant`, branch `main`. Push convention: commit on main without a Co-Authored-By line, then `git branch -f paper/tiis-evidence-revalidation HEAD`, then push both.

## Done (this round)
- [x] **Paper (Korean draft).**
  - Moon et al. repositioned: Intro (5 contributions), 2.2 (accurate summary plus difference bullets), Table 1 "혼합 범위 실패" column, 4.2 attribution, 5.1, Appendix A.
  - 4.2 has the inequality eq. (6); DFL is now eq. (7).
  - 4.2 has new paragraphs on normalization, size bins and calibration; Table 5 has new rows.
  - Table 6 has YOLOv8s end-to-end rows.
  - Size figure added as Fig. 3; later figures renumbered.
  - Abstract, 4.6, 5.2 and 5.3 updated.
  - Captions restyled to "**Table N. Title.** text", with bullet notes moved below tables as "주.".
- [x] **Evidence** in `paper_evidence/extra/` (index in its README):
  - `size_bins/`: the pre-declared hypothesis was supported.
  - `mitigation/`: normalization INT8 retains 34–55%.
  - `calibration/`: full INT8 collapses under every method. Head-excluded: v4 Percentile 98.4% (fail), v3 Percentile 98.99% [98.6–99.3] (inconclusive), Entropy lower.
  - `end_to_end_v3/`: YOLOv8s end-to-end 98.2%.
- [x] **Figures.** Shared style `scripts/paper/paper_style.py`. Fig. 1, 2 and 4–6 re-rendered; new Fig. 3 (`fig13_size_retention`). PDF plus PNG.

## Next
- [ ] The user reviews the report: results, claims to change, the 21-page plan, whether more experiments are needed.
- [ ] Apply the agreed page cuts. Then do the English conversion and the TIIS DOC template (DOC only since 2026-09-15; author profiles required; 13–21 pages).
- [ ] Optional: the Mac quiet WASM re-check (guide §8), and YOLOv8s-COCO. The user decides after reviewing P1.
