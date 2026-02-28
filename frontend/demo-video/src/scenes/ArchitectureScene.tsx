import React from "react";
import { useCurrentFrame, interpolate } from "remotion";
import { TokenFlow } from "../components/TokenFlow";
import { FadeIn } from "../components/FadeIn";
import { colors, fonts } from "../styles/theme";

export const ArchitectureScene: React.FC = () => {
  const frame = useCurrentFrame();

  const titleOpacity = interpolate(frame, [0, 20], [0, 1], {
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
        padding: "60px 100px",
        gap: 50,
      }}
    >
      <div
        style={{
          fontFamily: fonts.primary,
          fontSize: 38,
          fontWeight: 700,
          color: colors.text,
          opacity: titleOpacity,
        }}
      >
        Why Position Matters
      </div>

      <div
        style={{
          fontFamily: fonts.primary,
          fontSize: 18,
          color: colors.textMuted,
          opacity: titleOpacity,
          maxWidth: 700,
          lineHeight: 1.6,
        }}
      >
        Autoregressive models generate tokens left-to-right. Each token is
        conditioned on all previous tokens:
      </div>

      <FadeIn delay={15} duration={20}>
        <div
          style={{
            fontFamily: fonts.mono,
            fontSize: 20,
            color: colors.accent,
            background: colors.cardBg,
            border: `1px solid ${colors.cardBorder}`,
            borderRadius: 12,
            padding: "16px 24px",
            display: "inline-block",
          }}
        >
          P(tₙ) = P(tₙ | t₁, t₂, ... , tₙ₋₁)
        </div>
      </FadeIn>

      <div style={{ display: "flex", gap: 60 }}>
        <TokenFlow mode="before" delay={30} />
        <TokenFlow mode="after" delay={60} />
      </div>
    </div>
  );
};
