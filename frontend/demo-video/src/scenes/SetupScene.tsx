import React from "react";
import { useCurrentFrame, interpolate } from "remotion";
import { CircularProgress } from "../components/CircularProgress";
import { FadeIn } from "../components/FadeIn";
import { colors, fonts } from "../styles/theme";
import { accuracyData, models, totalQuestions } from "../data/researchData";

export const SetupScene: React.FC = () => {
  const frame = useCurrentFrame();

  const questionCardOpacity = interpolate(frame, [0, 20], [0, 1], {
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
        gap: 50,
      }}
    >
      {/* Question count badge */}
      <FadeIn delay={0} duration={20}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 20,
            fontFamily: fonts.primary,
          }}
        >
          <div
            style={{
              background: colors.cardBg,
              border: `1px solid ${colors.cardBorder}`,
              borderRadius: 12,
              padding: "16px 32px",
              fontSize: 26,
              color: colors.text,
              fontWeight: 600,
            }}
          >
            {totalQuestions} Questions
          </div>
          <div style={{ fontSize: 22, color: colors.textMuted }}>
            60 Math + 60 Science
          </div>
        </div>
      </FadeIn>

      {/* Model cards with circular progress */}
      <div
        style={{
          display: "flex",
          gap: 120,
          opacity: questionCardOpacity,
        }}
      >
        <FadeIn delay={30} duration={25}>
          <div
            style={{
              background: colors.cardBg,
              border: `1px solid ${colors.cardBorder}`,
              borderRadius: 20,
              padding: "40px 50px",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 10,
            }}
          >
            <CircularProgress
              value={accuracyData.gemini.baseline}
              color={models.gemini.color}
              label={models.gemini.name}
              delay={40}
              size={200}
            />
            <div
              style={{
                fontFamily: fonts.primary,
                fontSize: 16,
                color: colors.textMuted,
                marginTop: 4,
              }}
            >
              Baseline Accuracy
            </div>
          </div>
        </FadeIn>

        <FadeIn delay={50} duration={25}>
          <div
            style={{
              background: colors.cardBg,
              border: `1px solid ${colors.cardBorder}`,
              borderRadius: 20,
              padding: "40px 50px",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 10,
            }}
          >
            <CircularProgress
              value={accuracyData.openai.baseline}
              color={models.openai.color}
              label={models.openai.name}
              delay={60}
              size={200}
            />
            <div
              style={{
                fontFamily: fonts.primary,
                fontSize: 16,
                color: colors.textMuted,
                marginTop: 4,
              }}
            >
              Baseline Accuracy
            </div>
          </div>
        </FadeIn>
      </div>
    </div>
  );
};
