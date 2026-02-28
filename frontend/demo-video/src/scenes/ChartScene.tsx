import React from "react";
import { useCurrentFrame, interpolate, spring, useVideoConfig } from "remotion";
import { AnimatedBar } from "../components/AnimatedBar";
import { FadeIn } from "../components/FadeIn";
import { colors, fonts } from "../styles/theme";
import { accuracyData, conditions, models } from "../data/researchData";

export const ChartScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const titleOpacity = interpolate(frame, [0, 20], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Phase 2: highlight incorrect before (starts at ~frame 240)
  const highlightPhase = frame > 240;
  const highlightOpacity = interpolate(frame, [240, 260], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // 4x callout
  const calloutScale = highlightPhase
    ? spring({
        frame: frame - 280,
        fps,
        config: { damping: 10, stiffness: 100 },
      })
    : 0;

  const conditionKeys = [
    "baseline",
    "correctAfter",
    "correctBefore",
    "incorrectAfter",
    "incorrectBefore",
  ] as const;

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        background: colors.background,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        padding: "50px 80px",
        gap: 30,
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
        Accuracy Under Each Condition
      </div>

      {/* Chart area */}
      <div style={{ display: "flex", gap: 80, alignItems: "flex-end" }}>
        {/* Gemini group */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 16 }}>
          <div
            style={{
              fontFamily: fonts.primary,
              fontSize: 20,
              fontWeight: 600,
              color: models.gemini.color,
              opacity: titleOpacity,
            }}
          >
            {models.gemini.name}
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "flex-end", height: 380 }}>
            {conditionKeys.map((key, i) => {
              const isIBefore = key === "incorrectBefore";
              return (
                <AnimatedBar
                  key={key}
                  value={accuracyData.gemini[key]}
                  color={
                    isIBefore && highlightPhase
                      ? colors.danger
                      : models.gemini.color
                  }
                  label={conditions[i].short}
                  delay={20 + i * 8}
                  width={55}
                  maxHeight={350}
                  highlight={isIBefore && highlightPhase}
                  highlightDelay={250}
                />
              );
            })}
          </div>
        </div>

        {/* OpenAI group */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 16 }}>
          <div
            style={{
              fontFamily: fonts.primary,
              fontSize: 20,
              fontWeight: 600,
              color: models.openai.color,
              opacity: titleOpacity,
            }}
          >
            {models.openai.name}
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "flex-end", height: 380 }}>
            {conditionKeys.map((key, i) => {
              const isIBefore = key === "incorrectBefore";
              return (
                <AnimatedBar
                  key={key}
                  value={accuracyData.openai[key]}
                  color={
                    isIBefore && highlightPhase
                      ? colors.danger
                      : models.openai.color
                  }
                  label={conditions[i].short}
                  delay={40 + i * 8}
                  width={55}
                  maxHeight={350}
                  highlight={isIBefore && highlightPhase}
                  highlightDelay={250}
                />
              );
            })}
          </div>
        </div>
      </div>

      {/* 4x callout */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 20,
          opacity: calloutScale,
          transform: `scale(${calloutScale})`,
        }}
      >
        <div
          style={{
            fontFamily: fonts.primary,
            fontSize: 64,
            fontWeight: 800,
            color: colors.danger,
            textShadow: `0 0 30px ${colors.danger}`,
          }}
        >
          4x
        </div>
        <div
          style={{
            fontFamily: fonts.primary,
            fontSize: 22,
            color: colors.textMuted,
            maxWidth: 400,
          }}
        >
          worse when misinformation comes{" "}
          <span style={{ color: colors.danger, fontWeight: 700 }}>before</span>{" "}
          the question
        </div>
      </div>
    </div>
  );
};
