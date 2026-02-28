import React from "react";
import { useCurrentFrame, spring, useVideoConfig } from "remotion";
import { fonts } from "../styles/theme";

export const CircularProgress: React.FC<{
  value: number;
  color: string;
  label: string;
  delay?: number;
  size?: number;
}> = ({ value, color, label, delay = 0, size = 180 }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const progress = spring({
    frame: frame - delay,
    fps,
    config: { damping: 25, stiffness: 50 },
  });

  const radius = (size - 16) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference * (1 - (value / 100) * progress);
  const displayValue = (value * progress).toFixed(1);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
      <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#30363D"
          strokeWidth={12}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={12}
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          style={{
            filter: `drop-shadow(0 0 8px ${color})`,
          }}
        />
      </svg>
      <div
        style={{
          position: "relative",
          marginTop: -size / 2 - 20,
          fontFamily: fonts.primary,
          textAlign: "center",
          marginBottom: size / 2 - 40,
        }}
      >
        <div style={{ fontSize: 42, fontWeight: 700, color: "#E6EDF3" }}>
          {displayValue}%
        </div>
      </div>
      <div
        style={{
          fontFamily: fonts.primary,
          fontSize: 22,
          fontWeight: 600,
          color,
          marginTop: 8,
        }}
      >
        {label}
      </div>
    </div>
  );
};
