import React from "react";
import { useCurrentFrame, interpolate } from "remotion";
import { FadeIn } from "../components/FadeIn";
import { colors, fonts } from "../styles/theme";

export const ConclusionScene: React.FC = () => {
  const frame = useCurrentFrame();

  // "Reasoning" with strikethrough animation
  const strikeWidth = interpolate(frame, [30, 60], [0, 100], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // "Rationalizing" highlight
  const highlightOpacity = interpolate(frame, [50, 75], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const bullets = [
    "Models generate plausible justifications, not logical derivations",
    "Early context anchors reasoning — models cannot unsee misinformation",
    "Current benchmarks are blind to this fundamental vulnerability",
  ];

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
        gap: 50,
      }}
    >
      {/* Main text */}
      <div
        style={{
          display: "flex",
          gap: 30,
          alignItems: "center",
          fontFamily: fonts.primary,
          fontSize: 72,
          fontWeight: 800,
        }}
      >
        <FadeIn delay={0} duration={20}>
          <div style={{ position: "relative", display: "inline-block" }}>
            <span style={{ color: colors.textMuted }}>Reasoning</span>
            <div
              style={{
                position: "absolute",
                top: "50%",
                left: 0,
                width: `${strikeWidth}%`,
                height: 4,
                background: colors.danger,
                transform: "translateY(-50%)",
              }}
            />
          </div>
        </FadeIn>

        <FadeIn delay={40} duration={20}>
          <span style={{ color: colors.textMuted, fontSize: 48 }}>→</span>
        </FadeIn>

        <FadeIn delay={50} duration={20}>
          <span
            style={{
              color: colors.danger,
              textShadow: `0 0 ${30 * highlightOpacity}px ${colors.danger}60`,
            }}
          >
            Rationalizing
          </span>
        </FadeIn>
      </div>

      {/* Bullet points */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 20,
          maxWidth: 900,
        }}
      >
        {bullets.map((bullet, i) => (
          <FadeIn key={i} delay={80 + i * 20} duration={20}>
            <div
              style={{
                display: "flex",
                alignItems: "flex-start",
                gap: 16,
                fontFamily: fonts.primary,
                fontSize: 24,
                color: colors.text,
                lineHeight: 1.5,
              }}
            >
              <span style={{ color: colors.danger, fontWeight: 700, fontSize: 20, marginTop: 4 }}>
                ///
              </span>
              {bullet}
            </div>
          </FadeIn>
        ))}
      </div>
    </div>
  );
};
