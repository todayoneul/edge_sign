/**
 * LogoMark.tsx — "Detection Cell" 브랜드 마크 (Direction A).
 * 회전한 둥근 사각(표지판 기하) + 중앙 검출 셀. 헤더·히어로 공용.
 */

interface Props {
  size?: number;
  /** 외곽선 색 (기본: 본문 잉크). 어두운 히어로에선 흰 계열 전달. */
  color?: string;
}

export default function LogoMark({ size = 38, color = "currentColor" }: Props) {
  return (
    <svg width={size} height={size} viewBox="0 0 40 40" fill="none" aria-hidden="true">
      <rect
        x="9"
        y="9"
        width="22"
        height="22"
        rx="5.5"
        transform="rotate(45 20 20)"
        stroke={color}
        strokeWidth="2"
      />
      <rect x="16" y="16" width="8" height="8" rx="1.6" fill="var(--c-sign)" />
    </svg>
  );
}
