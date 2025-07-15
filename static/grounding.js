let heartbeatAudio = new Audio("/static/heartbeat.mp3");

heartbeatAudio.loop = true;
heartbeatAudio.volume = 0.3;

function showGroundingUI() {
  heartbeatAudio.play();

  const container = document.createElement("div");
  container.id = "grounding-container";
  container.innerHTML = `
  <div class="grounding-box">
    <h2>✨ Grounding Mode</h2>
    <p id="grounding-step">Let's begin by taking a slow breath. Inhale deeply…</p>
    <div class="breath-circle"></div>
    <div id="affirmation" class="affirmation"></div>
    <div class="grounding-buttons">
      <button id="grounding-next">Next Step</button>
      <button id="grounding-done">I'm Feeling Better</button>
    </div>
  </div>
`;


  document.body.classList.add("grounding-mode");
  document.body.appendChild(container);
  startGroundingSteps();
}

function hideGroundingUI() {
    heartbeatAudio.pause();
    heartbeatAudio.currentTime = 0;
  
    const container = document.getElementById("grounding-container");
    if (container) {
      container.style.display = "none";
      container.remove();
    }
  
    document.body.classList.remove("grounding-mode");
    document.body.style.overflow = "auto"; // Optional: re-enable scroll
  
    // 🧘 Toggle button visibility
    const startBtn = document.getElementById("start-grounding-btn");
    const stopBtn = document.getElementById("stop-grounding-btn");
  
    if (startBtn) startBtn.style.display = "inline-block";
    if (stopBtn) stopBtn.style.display = "none";
  }
  
  
const steps = [
  "Let's begin by taking a slow breath. Inhale deeply…",
  "Exhale slowly… Let your shoulders relax.",
  "Name 5 things you can see around you.",
  "Name 4 things you can touch right now.",
  "Name 3 things you can hear.",
  "Name 2 things you can smell.",
  "Name 1 thing you can taste.",
  "Feel your feet on the ground. Notice the texture, the weight.",
  "Put your hands together and press gently. You are here.",
  "Imagine putting your worries in a safe box you can revisit later.",
  "Say something kind to yourself. Maybe: 'I am doing the best I can.'"
];

const affirmations = [
  "You're doing really well. Stay with me in this moment.",
  "It's okay to pause and just be here.",
  "You are safe. You are grounded.",
  "This moment will pass. You're not alone.",
  "Keep breathing — you’re doing the best you can."
];

let stepIndex = 0;

function startGroundingSteps() {
  stepIndex = 0;
  updateStep();

  const nextBtn = document.getElementById("grounding-next");
  const doneBtn = document.getElementById("grounding-done");

  nextBtn.onclick = () => {
    stepIndex++;
    if (stepIndex < steps.length) {
      updateStep();
    } else {
      document.getElementById("grounding-step").textContent =
        "You've completed the grounding steps. Take a breath. You're doing okay.";
      nextBtn.style.display = "none";
      document.getElementById("affirmation").textContent =
        "If you’d like, you can check in with yourself now: how are you feeling?";
    }
  };

  doneBtn.onclick = () => {
    hideGroundingUI();
  };
}

function updateStep() {
    const stepEl = document.getElementById("grounding-step");
    const affirmationEl = document.getElementById("affirmation");
  
    if (stepEl) {
      stepEl.style.opacity = 0;
      setTimeout(() => {
        stepEl.textContent = steps[stepIndex];
        stepEl.style.opacity = 1;
      }, 300);
    }
  
    if (affirmationEl) {
      affirmationEl.style.opacity = 0;
      if (stepIndex % 2 === 0) {
        const quote = affirmations[Math.floor(Math.random() * affirmations.length)];
        setTimeout(() => {
          affirmationEl.textContent = quote;
          affirmationEl.style.animation = "fadeAffirmation 2s forwards";
        }, 1000);
      } else {
        affirmationEl.textContent = "";
      }
    }
  }
  
// ESC key exits grounding
document.addEventListener("keydown", function (e) {
  if (e.key === "Escape") {
    hideGroundingUI();
  }
});

window.showGroundingUI = showGroundingUI;
window.hideGroundingUI = hideGroundingUI;
