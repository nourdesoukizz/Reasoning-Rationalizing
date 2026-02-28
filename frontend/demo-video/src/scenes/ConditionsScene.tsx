import React from "react";
import { useCurrentFrame, interpolate, spring, useVideoConfig } from "remotion";
import { colors, fonts } from "../styles/theme";
import { conditions } from "../data/researchData";

const ConditionCard: React.FC<{
  condition: (typeof conditions)[number];
  index: number;
}> = ({ condition, index }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const delay = 20 + index * 12;

  const slideUp = spring({
    frame: frame - delay,
    fps,
    config: { damping: 15, stiffness: 80 },
  });

  const isIncorrectBefore = condition.key === "incorrectBefore";
  const borderColor = isIncorrectBefore ? colors.danger : colors.cardBorder;

  return (
    <div
      style={{
        background: colors.cardBg,
        border: `2px solid ${borderColor}`,
        borderRadius: 16,
        padding: "24px 28px",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 12,
        width: 190,
        opacity: slideUp,
        transform: `translateY(${(1 - slideUp) * 50}px)`,
        boxShadow: isIncorrectBefore
          ? `0 0 20px ${colors.danger}30`
          : "none",
      }}
    >
      <div style={{ fontSize: 36 }}>{condition.icon}</div>
      <div
        style={{
          fontFamily: fonts.primary,
          fontSize: 16,
          fontWeight: 600,
          color: isIncorrectBefore ? colors.danger : colors.text,
          textAlign: "center",
        }}
      >
        {condition.label}
      </div>
    </div>
  );
};

export const ConditionsScene: React.FC = () => {
  const frame = useCurrentFrame();

  const titleOpacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const diagramOpacity = interpolate(frame, [120, 145], [0, 1], {
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
        padding: 60,
        gap: 40,
      }}
    >
      <div
        style={{
          fontFamily: fonts.primary,
          fontSize: 42,
          fontWeight: 700,
          color: colors.text,
          opacity: titleOpacity,
        }}
      >
        5 Experimental Conditions
      </div>

      <div style={{ display: "flex", gap: 20, flexWrap: "wrap", justifyContent: "center" }}>
        {conditions.map((c, i) => (
          <ConditionCard key={c.key} condition={c} index={i} />
        ))}
      </div>

      {/* Before/After diagram */}
      <div
        style={{
          display: "flex",
          gap: 60,
          opacity: diagramOpacity,
          fontFamily: fonts.mono,
          fontSize: 18,
          color: colors.textMuted,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ color: colors.danger }}>HINT</span>
          <span>→</span>
          <span>QUESTION</span>
          <span style={{ color: colors.textMuted, fontSize: 14 }}>
            (Before)
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span>QUESTION</span>
          <span>→</span>
          <span style={{ color: colors.success }}>HINT</span>
          <span style={{ color: colors.textMuted, fontSize: 14 }}>
            (After)
          </span>
        </div>
      </div>
    </div>
  );
};
