import React from "react";
import { useCurrentFrame, spring, useVideoConfig, interpolate } from "remotion";
import { fonts } from "../styles/theme";

export const AnimatedBar: React.FC<{
  value: number;
  maxValue?: number;
  color: string;
  label: string;
  delay?: number;
  width?: number;
  maxHeight?: number;
  showValue?: boolean;
  highlight?: boolean;
  highlightDelay?: number;
}> = ({
  value,
  maxValue = 100,
  color,
  label,
  delay = 0,
  width = 60,
  maxHeight = 350,
  showValue = true,
  highlight = false,
  highlightDelay = 0,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const growth = spring({
    frame: frame - delay,
    fps,
    config: { damping: 20, stiffness: 60 },
  });

  const barHeight = (value / maxValue) * maxHeight * growth;

  const highlightOpacity = highlight
    ? interpolate(
        frame,
        [highlightDelay, highlightDelay + 15],
        [0, 1],
        { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
      )
    : 0;

  const displayValue = (value * growth).toFixed(1);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        width,
        position: "relative",
      }}
    >
      {showValue && (
        <div
          style={{
            fontSize: 14,
            fontWeight: 600,
            color: "#E6EDF3",
            marginBottom: 4,
            fontFamily: fonts.primary,
            fontVariantNumeric: "tabular-nums",
            opacity: growth,
          }}
        >
          {displayValue}%
        </div>
      )}
      <div
        style={{
          width: "100%",
          height: barHeight,
          background: color,
          borderRadius: "4px 4px 0 0",
          position: "relative",
          boxShadow: highlight
            ? `0 0 ${20 * highlightOpacity}px ${color}`
            : "none",
        }}
      />
      {highlight && (
        <div
          style={{
            position: "absolute",
            top: -30,
            left: "50%",
            transform: "translateX(-50%)",
            background: "#EF5350",
            color: "#fff",
            padding: "2px 8px",
            borderRadius: 4,
            fontSize: 12,
            fontWeight: 700,
            opacity: highlightOpacity,
            fontFamily: fonts.primary,
          }}
        >
          -20pp
        </div>
      )}
      <div
        style={{
          fontSize: 11,
          color: "#8B949E",
          marginTop: 6,
          textAlign: "center",
          fontFamily: fonts.primary,
          lineHeight: 1.2,
          maxWidth: width + 10,
        }}
      >
        {label}
      </div>
    </div>
  );
};
