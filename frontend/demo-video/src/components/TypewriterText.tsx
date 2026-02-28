import React from "react";
import { useCurrentFrame, interpolate } from "remotion";
import { fonts, colors } from "../styles/theme";

export const TypewriterText: React.FC<{
  text: string;
  startFrame?: number;
  speed?: number;
  fontSize?: number;
  color?: string;
  fontWeight?: number;
  style?: React.CSSProperties;
}> = ({
  text,
  startFrame = 0,
  speed = 2,
  fontSize = 72,
  color = colors.text,
  fontWeight = 700,
  style,
}) => {
  const frame = useCurrentFrame();
  const charsToShow = Math.floor(
    interpolate(
      frame,
      [startFrame, startFrame + text.length * speed],
      [0, text.length],
      { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
    )
  );
  const visibleText = text.slice(0, charsToShow);
  const showCursor = frame >= startFrame && charsToShow < text.length;

  return (
    <div
      style={{
        fontFamily: fonts.primary,
        fontSize,
        fontWeight,
        color,
        letterSpacing: "-0.02em",
        ...style,
      }}
    >
      {visibleText}
      {showCursor && (
        <span
          style={{
            opacity: Math.sin(frame * 0.3) > 0 ? 1 : 0,
            color: colors.accent,
          }}
        >
          |
        </span>
      )}
    </div>
  );
};
