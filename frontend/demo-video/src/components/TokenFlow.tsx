import React from "react";
import { useCurrentFrame, interpolate, spring, useVideoConfig } from "remotion";
import { colors, fonts } from "../styles/theme";

const Token: React.FC<{
  text: string;
  index: number;
  contaminated: boolean;
  delay: number;
}> = ({ text, index, contaminated, delay }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const appear = spring({
    frame: frame - delay - index * 4,
    fps,
    config: { damping: 15, stiffness: 100 },
  });

  const bgColor = contaminated ? colors.danger : colors.cardBg;
  const borderColor = contaminated ? colors.dangerDark : colors.cardBorder;

  return (
    <div
      style={{
        display: "inline-flex",
        padding: "8px 14px",
        margin: 4,
        background: bgColor,
        border: `2px solid ${borderColor}`,
        borderRadius: 6,
        fontFamily: fonts.mono,
        fontSize: 16,
        color: colors.text,
        opacity: appear,
        transform: `scale(${appear})`,
        boxShadow: contaminated ? `0 0 12px ${colors.danger}40` : "none",
      }}
    >
      {text}
    </div>
  );
};

export const TokenFlow: React.FC<{
  mode: "before" | "after";
  delay?: number;
}> = ({ mode, delay = 0 }) => {
  const frame = useCurrentFrame();

  const hintTokens = ["[HINT:", "deriv", "of", "x³", "is", "x²]"];
  const questionTokens = ["What", "is", "the", "derivative", "of", "x³+2x?"];
  const responseTokens = ["Answer:", "x²", "+", "2"];

  const beforeTokens = [
    ...hintTokens.map((t) => ({ text: t, contaminated: true })),
    ...questionTokens.map((t) => ({ text: t, contaminated: false })),
    ...responseTokens.map((t) => ({ text: t, contaminated: true })),
  ];

  const afterTokens = [
    ...questionTokens.map((t) => ({ text: t, contaminated: false })),
    ...hintTokens.map((t) => ({ text: t, contaminated: true })),
    ...responseTokens.map((t) => ({ text: t, contaminated: false })),
  ];

  const tokens = mode === "before" ? beforeTokens : afterTokens;

  const labelOpacity = interpolate(frame, [delay, delay + 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div
        style={{
          fontFamily: fonts.primary,
          fontSize: 18,
          fontWeight: 600,
          color: mode === "before" ? colors.danger : colors.success,
          opacity: labelOpacity,
        }}
      >
        {mode === "before" ? "Hint BEFORE question:" : "Hint AFTER question:"}
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", maxWidth: 700 }}>
        {tokens.map((token, i) => (
          <Token
            key={i}
            text={token.text}
            index={i}
            contaminated={token.contaminated}
            delay={delay + 10}
          />
        ))}
      </div>
      <div
        style={{
          fontFamily: fonts.primary,
          fontSize: 14,
          color: colors.textMuted,
          opacity: interpolate(
            frame,
            [delay + 70, delay + 85],
            [0, 1],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
          ),
          marginTop: 4,
        }}
      >
        {mode === "before"
          ? "→ Hint contaminates reasoning → Wrong answer"
          : "→ Reasoning already formed → Correct answer"}
      </div>
    </div>
  );
};
