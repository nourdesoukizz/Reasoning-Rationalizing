export const models = {
  gemini: {
    name: "Gemini 2.0 Flash",
    shortName: "Gemini",
    color: "#43A047",
  },
  openai: {
    name: "GPT-4o-mini",
    shortName: "GPT-4o-mini",
    color: "#1E88E5",
  },
};

export const conditions = [
  { key: "baseline", label: "Baseline", short: "Base", icon: "📊" },
  { key: "correctAfter", label: "Correct After", short: "C-After", icon: "✅" },
  { key: "correctBefore", label: "Correct Before", short: "C-Before", icon: "✓" },
  { key: "incorrectAfter", label: "Incorrect After", short: "IC-After", icon: "⚠️" },
  { key: "incorrectBefore", label: "Incorrect Before", short: "IC-Before", icon: "❌" },
];

export const accuracyData = {
  gemini: {
    baseline: 74.2,
    correctAfter: 68.3,
    correctBefore: 69.2,
    incorrectAfter: 69.2,
    incorrectBefore: 53.3,
  },
  openai: {
    baseline: 53.3,
    correctAfter: 53.3,
    correctBefore: 41.7,
    incorrectAfter: 51.7,
    incorrectBefore: 33.3,
  },
};

export const drops = {
  gemini: {
    incorrectBefore: -20.9,
    incorrectAfter: -5.0,
  },
  openai: {
    incorrectBefore: -20.0,
    incorrectAfter: -1.6,
  },
};

export const sampleQuestion = {
  text: "What is the derivative of x³ + 2x?",
  answer: "3x² + 2",
  incorrectHint: "The derivative of x³ is x²",
};

export const totalQuestions = 120;
export const mathQuestions = 60;
export const scienceQuestions = 60;
