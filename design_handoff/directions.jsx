/* directions.jsx — Edge-Sign redesign: 4 console directions + logo set.
   Markup is injected as HTML (canonical) for compact, directly-editable frames.
   Exposes window.EdgeSignCanvas (mounted by the host html). */

/* ── shared SVG atoms ─────────────────────────────────────── */
const MARK = {
  // A — rounded warning-diamond + centre detection cell
  cell: `<svg viewBox="0 0 40 40" fill="none">
    <rect x="9" y="9" width="22" height="22" rx="5.5" transform="rotate(45 20 20)" stroke="currentColor" stroke-width="2"/>
    <rect x="16" y="16" width="8" height="8" rx="1.6" fill="var(--sign)"/></svg>`,
  // B — round regulatory sign + crosshair ticks (instrument)
  scope: `<svg viewBox="0 0 40 40" fill="none">
    <circle cx="20" cy="20" r="13" stroke="currentColor" stroke-width="2"/>
    <path d="M20 3.5v6M20 30.5v6M3.5 20h6M30.5 20h6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
    <circle cx="20" cy="20" r="3.4" fill="var(--sign)"/></svg>`,
  // C — concentric diamonds, minimal (figure)
  fig: `<svg viewBox="0 0 40 40" fill="none">
    <rect x="7.5" y="7.5" width="25" height="25" rx="3" transform="rotate(45 20 20)" stroke="currentColor" stroke-width="2"/>
    <rect x="15.5" y="15.5" width="9" height="9" rx="1.5" transform="rotate(45 20 20)" fill="none" stroke="var(--sign)" stroke-width="2"/></svg>`,
  // D — diamond + aperture lens
  lens: `<svg viewBox="0 0 40 40" fill="none">
    <rect x="9" y="9" width="22" height="22" rx="5.5" transform="rotate(45 20 20)" stroke="currentColor" stroke-width="2"/>
    <circle cx="20" cy="20" r="5" stroke="var(--sign)" stroke-width="2"/>
    <circle cx="20" cy="20" r="1.7" fill="var(--sign)"/></svg>`,
};
const IC = {
  scan: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 8V5.5A1.5 1.5 0 0 1 5.5 4H8M16 4h2.5A1.5 1.5 0 0 1 20 5.5V8M20 16v2.5a1.5 1.5 0 0 1-1.5 1.5H16M8 20H5.5A1.5 1.5 0 0 1 4 18.5V16M7 12h10"/></svg>`,
  cam: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="2.5" y="6.5" width="13" height="11" rx="2"/><path d="M15.5 10.5l6-3.5v10l-6-3.5"/></svg>`,
  sun: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>`,
  help: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9.2"/><path d="M9.4 9.4a2.7 2.7 0 0 1 5.2 1c0 1.8-2.5 2.1-2.5 3.8"/><circle cx="12" cy="17" r="0.6" fill="currentColor"/></svg>`,
  send: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5M5 12l7-7 7 7"/></svg>`,
  pen: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4 12.5-12.5Z"/></svg>`,
};

/* shared right-side header cluster */
const headRight = (extra = '') => `
  <span class="es-engine"><span class="dot"></span>온디바이스 · WebGPU</span>
  ${extra}
  <button class="es-iconbtn" title="라이트/다크">${IC.sun}</button>
  <button class="es-iconbtn" title="단축키">${IC.help}</button>`;

/* shared Scene-Q&A rail block (surfaced + labelled) */
const qaBlock = `
  <div class="es-qa">
    <div class="title"><b>장면 질의</b><span class="en">Scene Q&amp;A</span></div>
    <p class="desc">분석 중인 장면을 자연어로 질문하세요. 검출된 표지판·신호등·간판을 근거로 답합니다.</p>
    <div class="es-chips">
      <span class="es-chip">제한속도는?</span>
      <span class="es-chip">앞에 신호등 있나요?</span>
      <span class="es-chip">간판 텍스트 읽어줘</span>
    </div>
    <div class="es-ask"><span class="ph">장면에 대해 물어보기…</span><span class="send">${IC.send}</span></div>
  </div>`;

const legendBlock = `
  <div class="es-legend">
    <div class="row"><span class="sw sign"></span><span class="nm">교통표지판</span><span class="en">Sign</span></div>
    <div class="row"><span class="sw light"></span><span class="nm">신호등</span><span class="en">Light</span></div>
    <div class="row"><span class="sw board"></span><span class="nm">간판</span><span class="en">Signboard</span></div>
  </div>`;

/* ════════════════════════ A — 랩 노트북 ════════════════════════ */
const DIR_A = `
<div class="es-head">
  <div class="es-brand"><span class="es-mark">${MARK.cell}</span>
    <span class="es-word"><b>Edge<span class="acc">·</span>Sign</b><span class="sub">온디바이스 주행 인지</span></span></div>
  <span class="spacer"></span>
  ${headRight('<span class="es-ready"><span class="dot"></span>분석 준비됨</span>')}
</div>
<div class="es-main">
  <div class="es-stage">
    <div class="es-viewport">
      <div class="es-ticks"><i></i><i></i><i></i><i></i></div>
      <div class="es-hero">
        <span class="es-mark" style="width:54px;height:54px;color:rgba(255,255,255,0.85)">${MARK.cell}</span>
        <h2><span class="en">On-device perception</span>주행 장면을<br>엣지에서 직접 분석합니다</h2>
        <p class="lead">표지판 · 신호등 · 간판을 <b>브라우저(WebGPU)</b>에서 동시에 검출·추적·인식합니다. <span class="acc">서버 전송 없이</span>, 영상은 기기 밖으로 나가지 않습니다.</p>
        <div class="es-actions">
          <button class="es-btn es-btn-primary">${IC.scan}영상 분석 시작</button>
          <button class="es-btn es-btn-ghost">${IC.cam}웹캠으로 보기</button>
        </div>
        <p class="es-hint">영상 파일을 이 영역에 끌어다 놓아도 됩니다</p>
      </div>
    </div>
  </div>
  <div class="es-rail">
    <div class="es-rsec">검출 범례 <span class="tag">3 classes</span></div>
    ${legendBlock}
    <div class="es-rsec">장면 질의 <span class="tag">Scene Q&amp;A</span></div>
    ${qaBlock}
  </div>
</div>
<div class="es-foot">
  <span class="item"><b>대기</b> · 입력 소스를 선택하세요</span>
  <span class="item">검출기 <b>YOLOv8s-signs</b></span>
  <span class="item">추론 <b>INT8</b> · 13 MB</span>
  <span class="right">v3 · WebGPU EP</span>
</div>`;

/* ════════════════════════ B — 계측기 ════════════════════════ */
const DIR_B = `
<div class="es-head">
  <div class="es-brand"><span class="es-mark">${MARK.scope}</span>
    <span class="es-word"><b>Edge<span class="acc">·</span>Sign</b><span class="sub">Perception Instrument</span></span></div>
  <span class="spacer"></span>
  ${headRight('<span class="es-ready"><span class="dot"></span>STANDBY</span>')}
</div>
<div class="es-main">
  <div class="es-stage"><div class="es-stage-inner">
    <div class="es-viewport">
      <div class="es-ruler top"></div><div class="es-ruler left"></div>
      <div class="es-ticks"><i></i><i></i><i></i><i></i></div>
      <div class="es-hero">
        <span class="es-mark" style="width:52px;height:52px;color:rgba(255,255,255,0.85)">${MARK.scope}</span>
        <h2><span class="en">Real-time perception</span>주행 인지 계측을<br>온디바이스로 실행</h2>
        <p class="lead">검출 · 추적 · 인식 파이프라인을 <b>WebGPU</b>에서 직접 계측합니다. 입력을 연결하면 채널별 지연과 신뢰도가 아래 베이스라인에 표시됩니다.</p>
        <div class="es-actions">
          <button class="es-btn es-btn-primary">${IC.scan}영상 연결</button>
          <button class="es-btn es-btn-ghost">${IC.cam}웹캠 연결</button>
        </div>
      </div>
    </div>
    <div class="es-readout">
      <div class="ch"><span class="lab">처리율 Throughput</span><span class="val standby">— <span class="unit">fps</span></span></div>
      <div class="ch"><span class="lab">추론 지연 Latency</span><span class="val standby">— <span class="unit">ms</span></span></div>
      <div class="ch"><span class="lab"><span class="sw" style="background:var(--sign)"></span>SIGN</span><span class="val standby">—</span></div>
      <div class="ch"><span class="lab"><span class="sw" style="background:var(--light)"></span>LIGHT</span><span class="val standby">—</span></div>
      <div class="ch"><span class="lab"><span class="sw" style="background:var(--board)"></span>BOARD</span><span class="val standby">—</span></div>
    </div>
  </div></div>
  <div class="es-rail">
    <div class="es-rsec">채널 Channels <span class="tag">3</span></div>
    ${legendBlock}
    <div class="es-rsec">장면 질의 Query</div>
    <div class="es-qa" style="flex:1">
      <p class="desc">장면을 질의어로 조회합니다. 검출 결과를 근거로 응답합니다.</p>
      <div class="es-chips"><span class="es-chip">제한속도</span><span class="es-chip">신호 상태</span><span class="es-chip">간판 OCR</span></div>
    </div>
    <div class="es-cmd">
      <div class="cmdline"><span class="prompt">query&nbsp;&rsaquo;</span><span>장면에 대해 질문…</span><span class="cur"></span></div>
    </div>
  </div>
</div>
<div class="es-foot">
  <span class="item"><b>STANDBY</b> · awaiting input</span>
  <span class="item">model <b>yolov8s-signs-v3</b></span>
  <span class="item">quant <b>INT8</b> · 13 MB</span>
  <span class="right">EP: WebGPU</span>
</div>`;

/* ════════════════════════ C — 페이퍼 / figure ════════════════════════ */
const DIR_C = `
<div class="es-head">
  <div class="es-brand"><span class="es-mark">${MARK.fig}</span>
    <span class="es-word"><b>Edge<span class="acc">·</span>Sign</b><span class="sub">On-device perception</span></span></div>
  <span class="spacer"></span>
  ${headRight('<span class="es-ready"><span class="dot"></span>준비됨 Ready</span>')}
</div>
<div class="es-main">
  <div class="es-stage">
    <div class="secno"><span>01</span><b>입력 — Input</b><span class="en">video / webcam</span></div>
    <div class="es-viewport">
      <div class="es-ticks"><i></i><i></i><i></i><i></i></div>
      <div class="es-hero">
        <h2><span class="en">Figure 1 · perception pipeline</span>주행 장면 인지,<br>엣지에서 재현 가능하게</h2>
        <p class="lead">표지판 · 신호등 · 간판을 <b>브라우저(WebGPU)</b>에서 검출 → 추적 → 인식. 모든 처리는 기기 내에서 끝나며 원본 영상은 전송되지 않습니다.</p>
        <div class="es-actions">
          <button class="es-btn es-btn-primary">${IC.scan}영상 분석</button>
          <button class="es-btn es-btn-ghost">${IC.cam}웹캠</button>
        </div>
      </div>
    </div>
    <div class="figcap"><b>Fig. 1</b><span>입력 소스를 연결하면 검출 결과가 색상 키에 따라 오버레이됩니다. 좌측 위 기준 마크는 정합(registration) 표식입니다.</span></div>
  </div>
  <div class="es-rail">
    <div class="es-rsec">02 · 인지 — Perception</div>
    <div class="figcap" style="padding:13px 18px 4px"><span>검출 클래스 색상 키 — Colour key</span></div>
    ${legendBlock}
    <div class="es-rsec">03 · 질의 — Query</div>
    ${qaBlock}
  </div>
</div>
<div class="es-foot">
  <span class="item"><b>Ready</b> · 입력 대기</span>
  <span class="item">§ detector <b>YOLOv8s-signs v3</b></span>
  <span class="item">INT8 · 13 MB</span>
  <span class="right">WebGPU execution provider</span>
</div>`;

/* ════════════════════════ D — 정제된 콘솔 ════════════════════════ */
const DIR_D = `
<div class="es-head">
  <div class="es-brand"><span class="es-mark">${MARK.lens}</span>
    <span class="es-word"><b>Edge<span class="acc">·</span>Sign</b><span class="sub">주행 인지 분석</span></span></div>
  <span class="spacer"></span>
  ${headRight('<span class="es-ready"><span class="dot"></span>분석 준비됨</span>')}
</div>
<div class="es-main">
  <div class="es-stage">
    <div class="es-viewport">
      <div class="es-source-bar">
        <div class="es-seg">
          <button data-on>${IC.scan}영상</button>
          <button>${IC.cam}웹캠</button>
        </div>
      </div>
      <div class="es-ticks"><i></i><i></i><i></i><i></i></div>
      <div class="es-hero">
        <span class="es-mark" style="width:52px;height:52px;color:rgba(255,255,255,0.85)">${MARK.lens}</span>
        <h2><span class="en">On-device · WebGPU</span>주행 장면을<br>실시간으로 분석</h2>
        <p class="lead">표지판 · 신호등 · 간판을 <b>브라우저에서 직접</b> 검출·추적·인식합니다. 오른쪽 위에서 <b>영상</b> 또는 <span class="acc">웹캠</span>을 선택해 시작하세요.</p>
        <div class="es-actions">
          <button class="es-btn es-btn-primary">${IC.scan}영상 분석 시작</button>
        </div>
        <p class="es-hint">영상을 이 영역에 끌어다 놓아도 됩니다</p>
      </div>
    </div>
  </div>
  <div class="es-rail">
    <div class="es-rtabs">
      <button data-on>검출 트랙</button>
      <button>장면 질의 <span class="pip"></span></button>
    </div>
    <div class="es-rsec">검출 범례 <span class="tag">3 classes</span></div>
    ${legendBlock}
    ${qaBlock}
  </div>
</div>
<div class="es-foot">
  <span class="item idle"><b>분석 대기</b> · 소스 선택</span>
  <span class="item idle">처리율 <b>—</b> fps</span>
  <span class="item idle">지연 <b>—</b> ms</span>
  <span class="right">YOLOv8s · INT8 · WebGPU</span>
</div>`;

/* ── logo lockup set ──────────────────────────────────────── */
const LOGO = (mark, name, ko) => `
  <div class="logo-card">
    <span class="logo-mark">${mark}</span>
    <div class="logo-word"><b>Edge<span style="color:var(--sign)">·</span>Sign</b><span>${name}</span></div>
    <span class="logo-note">${ko}</span>
  </div>`;

function htmlBoard(id, label, w, h, html) {
  return React.createElement(window.DCArtboard, { id, label, width: w, height: h },
    React.createElement('div', { className: 'es ' + id, dangerouslySetInnerHTML: { __html: html } })
  );
}

function EdgeSignCanvas() {
  const A = React.createElement;
  return A(window.DesignCanvas, null,
    A(window.DCSection, { id: 'logo', title: '로고 / 브랜드 마크', subtitle: '표지판(다이아몬드·원) 추상화 — 보라 그라데이션 블롭 대체' },
      A(window.DCArtboard, { id: 'logos', label: '마크 4종 + 워드마크', width: 1180, height: 300 },
        A('div', { className: 'es', style: { width: '100%', height: 'auto', display: 'block', padding: '0' }, dangerouslySetInnerHTML: { __html: `
          <div class="logo-grid">
            ${LOGO(MARK.cell, 'Detection cell', 'A · 경고 다이아몬드 + 검출 셀')}
            ${LOGO(MARK.scope, 'Sighting scope', 'B · 원형 표지 + 조준 십자선')}
            ${LOGO(MARK.fig, 'Concentric', 'C · 동심 다이아몬드 (figure)')}
            ${LOGO(MARK.lens, 'Aperture lens', 'D · 다이아몬드 + 렌즈 조리개')}
          </div>` } })
      )
    ),
    A(window.DCSection, { id: 'console', title: '콘솔 시안 · 4 방향', subtitle: '시작(idle) 화면 — 빈 0.0 KPI 제거 · URL 삭제 · 소스/질의 재배치 · 관제 → 분석 재프레이밍' },
      htmlBoard('dir-a', 'A · 랩 노트북 — 미니멀 화이트', 1360, 860, DIR_A),
      htmlBoard('dir-b', 'B · 계측기 — 채널 리드아웃', 1360, 860, DIR_B),
      htmlBoard('dir-c', 'C · 페이퍼/figure — 에디토리얼', 1360, 860, DIR_C),
      htmlBoard('dir-d', 'D · 정제된 콘솔 — 컨트롤 재배치', 1360, 860, DIR_D)
    )
  );
}
window.EdgeSignCanvas = EdgeSignCanvas;
