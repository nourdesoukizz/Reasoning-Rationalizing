import React from "react";
import { Audio, Series, staticFile } from "remotion";
import { TitleScene } from "./scenes/TitleScene";
import { SetupScene } from "./scenes/SetupScene";
import { ConditionsScene } from "./scenes/ConditionsScene";
import { ChartScene } from "./scenes/ChartScene";
import { ArchitectureScene } from "./scenes/ArchitectureScene";
import { ConclusionScene } from "./scenes/ConclusionScene";
import { CreditsScene } from "./scenes/CreditsScene";

export const DemoVideo: React.FC = () => {
  return (
    <div style={{ flex: 1, background: "#0D1117" }}>
      <Audio src={staticFile("audio/narration.mp3")} />
      <Series>
        {/* Scene 1: Title Hook — 0:00-0:06 (180 frames) */}
        <Series.Sequence durationInFrames={180}>
          <TitleScene />
        </Series.Sequence>

        {/* Scene 2: Setup — 0:06-0:14 (240 frames) */}
        <Series.Sequence durationInFrames={240}>
          <SetupScene />
        </Series.Sequence>

        {/* Scene 3: Conditions — 0:14-0:22 (240 frames) */}
        <Series.Sequence durationInFrames={240}>
          <ConditionsScene />
        </Series.Sequence>

        {/* Scene 4: Chart Reveal — 0:22-0:38 (480 frames) */}
        <Series.Sequence durationInFrames={480}>
          <ChartScene />
        </Series.Sequence>

        {/* Scene 5: Architecture — 0:38-0:48 (300 frames) */}
        <Series.Sequence durationInFrames={300}>
          <ArchitectureScene />
        </Series.Sequence>

        {/* Scene 6: Conclusion — 0:48-0:55 (210 frames) */}
        <Series.Sequence durationInFrames={210}>
          <ConclusionScene />
        </Series.Sequence>

        {/* Scene 7: Credits — 0:55-1:00 (150 frames) */}
        <Series.Sequence durationInFrames={150}>
          <CreditsScene />
        </Series.Sequence>
      </Series>
    </div>
  );
};
