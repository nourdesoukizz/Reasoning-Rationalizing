import React from "react";
import { useCurrentFrame, interpolate } from "remotion";
import { TypewriterText } from "../components/TypewriterText";
import { colors, fonts } from "../styles/theme";

export const TitleScene: React.FC = () => {
  const frame = useCurrentFrame();

  const subtitleOpacity = interpolate(frame, [100, 130], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const glowIntensity = interpolate(frame, [0, 90, 180], [0, 15, 8], {
    extrapolateRight: "clamp",
  });

  const lineWidth = interpolate(frame, [60, 120], [0, 600], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        background: colors.background,
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        padding: 80,
      }}
    >
      <div
        style={{
          textShadow: `0 0 ${glowIntensity}px ${colors.accent}`,
        }}
      >
        <TypewriterText
          text="Reasoning or Rationalizing?"
          startFrame={10}
          speed={2}
          fontSize={82}
          fontWeight={800}
          style={{ textAlign: "center" }}
        />
      </div>

      <div
        style={{
          width: lineWidth,
          height: 3,
          background: `linear-gradient(90deg, transparent, ${colors.accent}, transparent)`,
          marginTop: 30,
          marginBottom: 30,
        }}
      />

      <div
        style={{
          fontFamily: fonts.primary,
          fontSize: 28,
          color: colors.textMuted,
          opacity: subtitleOpacity,
          textAlign: "center",
          maxWidth: 800,
          lineHeight: 1.5,
        }}
      >
        Testing Model Robustness Against Misleading Information
      </div>
    </div>
  );
};
