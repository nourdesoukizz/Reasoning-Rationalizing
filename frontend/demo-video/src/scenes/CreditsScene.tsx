import React from "react";
import { useCurrentFrame, interpolate } from "remotion";
import { FadeIn } from "../components/FadeIn";
import { colors, fonts } from "../styles/theme";

export const CreditsScene: React.FC = () => {
  const frame = useCurrentFrame();

  const bgOpacity = interpolate(frame, [0, 20], [0, 1], {
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
        gap: 30,
        opacity: bgOpacity,
      }}
    >
      <FadeIn delay={5} duration={25}>
        <div
          style={{
            fontFamily: fonts.primary,
            fontSize: 52,
            fontWeight: 800,
            color: colors.text,
            textAlign: "center",
          }}
        >
          Reasoning or Rationalizing?
        </div>
      </FadeIn>

      <FadeIn delay={20} duration={25}>
        <div
          style={{
            width: 200,
            height: 3,
            background: `linear-gradient(90deg, transparent, ${colors.accent}, transparent)`,
          }}
        />
      </FadeIn>

      <FadeIn delay={30} duration={25}>
        <div
          style={{
            fontFamily: fonts.primary,
            fontSize: 28,
            color: colors.textMuted,
            textAlign: "center",
          }}
        >
          By Nour Desouki
        </div>
      </FadeIn>

      <FadeIn delay={50} duration={25}>
        <div
          style={{
            fontFamily: fonts.mono,
            fontSize: 20,
            color: colors.accent,
            background: colors.cardBg,
            border: `1px solid ${colors.cardBorder}`,
            borderRadius: 10,
            padding: "12px 28px",
            marginTop: 20,
          }}
        >
          github.com/nourdesouki/Reasoning-Rationalizing
        </div>
      </FadeIn>

      <FadeIn delay={70} duration={25}>
        <div
          style={{
            fontFamily: fonts.primary,
            fontSize: 16,
            color: colors.textMuted,
            marginTop: 20,
          }}
        >
          2026
        </div>
      </FadeIn>
    </div>
  );
};
