import React from "react";
import { useCurrentFrame, spring, useVideoConfig } from "remotion";

export const AnimatedNumber: React.FC<{
  target: number;
  delay?: number;
  suffix?: string;
  fontSize?: number;
  color?: string;
  decimals?: number;
  style?: React.CSSProperties;
}> = ({
  target,
  delay = 0,
  suffix = "%",
  fontSize = 48,
  color = "#E6EDF3",
  decimals = 1,
  style,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const progress = spring({
    frame: frame - delay,
    fps,
    config: { damping: 30, stiffness: 80 },
  });
  const value = (target * progress).toFixed(decimals);

  return (
    <span
      style={{
        fontSize,
        fontWeight: 700,
        color,
        fontVariantNumeric: "tabular-nums",
        ...style,
      }}
    >
      {value}
      {suffix}
    </span>
  );
};
