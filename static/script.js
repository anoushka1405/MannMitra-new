window.addEventListener("load", () => {
  const { jsPDF } = window.jspdf;
  window.jsPDF = jsPDF;  // ✅ Save it globally so you can use it in any function
});


const { jsPDF } = window.jspdf;


const chatForm = document.getElementById('chat-form');
const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const typingIndicator = document.getElementById('typing-indicator');
const voiceSelect = document.getElementById('voiceSelect');
const ttsToggle = document.getElementById('ttsToggle');
let voices = [];
let synth = window.speechSynthesis;
let currentUtterance = null;

const emotionResponses = {
  joy: "I'm truly happy with you — moments like this are precious.",
  sadness: "I'm holding space for all the heaviness you're carrying right now.",
  anger: "Your frustration is valid and understandable.",
  fear: "It’s okay to feel scared — I’m here and you’re not alone in this.",
  surprise: "That unexpected moment can really shake us.",
  love: "That warm feeling is a beautiful part of your day.",
  neutral: "I’m here, right with you — whatever you’re feeling is okay."
};

document.querySelectorAll('.emotion-swatch').forEach(swatch => {
  swatch.addEventListener('click', () => {
    const emotion = swatch.getAttribute('data-emotion');
    document.getElementById('emotion-name').textContent =
      emotion.charAt(0).toUpperCase() + emotion.slice(1);
    document.getElementById('emotion-response').textContent =
      emotionResponses[emotion];
    document.querySelectorAll('.emotion-swatch').forEach(el => el.classList.remove('active'));
    swatch.classList.add('active');
  });
});

document.getElementById("kindness-btn").addEventListener("click", () => {
  const quotes = [
    "You’re doing the best you can — and that’s something to honor.",
    "It’s okay to rest. You deserve kindness, even from yourself.",
    "Whatever you’re feeling, it’s human. And you’re not alone.",
    "You don’t have to be ‘on’ all the time. Just being is enough.",
    "Let’s slow down. One deep breath, right here with me."
  ];
  document.getElementById("kindness-quote").textContent =
    quotes[Math.floor(Math.random() * quotes.length)];
});

chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  synth.cancel(); // 🔇 stop speaking right away
  const msg = userInput.value.trim();
  if (!msg) return;

  chatBox.innerHTML += `<div class="chat-msg user"><strong>You:</strong> ${msg}</div>`;
  chatBox.scrollTop = chatBox.scrollHeight;
  userInput.value = '';
  typingIndicator.style.display = 'block';

  try {
    const res = await fetch('/get', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ msg })
    });
    

    const data = await res.json();
    typingIndicator.style.display = 'none';

    if (data.trigger_grounding) {
      showGroundingUI();
      return;
    }
    

    chatBox.innerHTML += `<div class="chat-msg bot"><strong>Aasha:</strong> ${data.reply}</div>`;
    chatBox.scrollTop = chatBox.scrollHeight;

    updateChatboxEmotionColor(data.emotion || "neutral");

    // 💥 Celebration trigger
if (data.celebration_type === "balloons") {
    console.log("🎈 Balloons celebration triggered");
    triggerBalloons();
  } else if (data.celebration_type === "hearts") {
    console.log("💖 Hearts celebration triggered");
    triggerHearts();
  }else if (data.celebration_type === "confetti") {
    console.log("🎉 Confetti celebration triggered");
    triggerConfetti();
  }

    if (ttsToggle.checked) {
      speak(data.reply);
    }
  } catch (error) {
    typingIndicator.style.display = 'none';
    chatBox.innerHTML += `<div class="chat-msg bot"><strong>Aasha:</strong> Oops! Something went wrong.</div>`;
  }
});

function updateChatboxEmotionColor(emotion) {
  const emotionColors = {
    sadness: "#6baffd",
    joy: "#ffe96e",
    anger: "#f36565",
    fear: "#c36cfa",
    surprise: "#f8c66a",
    love: "#f64e86",
    neutral: "#ffe4c4"
  };
  chatBox.style.backgroundColor = emotionColors[emotion] || emotionColors.neutral;
}

// 🎤 Speech Recognition
let recognition;
if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SpeechRecognition();
  recognition.lang = 'en-IN';
  recognition.interimResults = false;
  recognition.onresult = function (event) {
    const transcript = event.results[0][0].transcript;
    userInput.value = transcript;
    chatForm.requestSubmit();
  };
}

function startListening() {
  if (recognition) {
    recognition.start();
  } else {
    alert("Speech recognition not supported in this browser.");
  }
}

function populateVoiceList() {
  voices = speechSynthesis.getVoices();
  if (!voices.length) return false;

  voiceSelect.innerHTML = "";

  // Allowed languages (lowercase)
  const allowedLangs = ["en-in", "en-us", "en-gb", "en-au"];

  // Filter voices with allowed langs only
  const filteredVoices = voices.filter(v =>
    allowedLangs.includes(v.lang.toLowerCase())
  );

  // Find preferred Indian female voice by name (Lekha or Heera)
  let preferredIndex = -1;
  for (let i = 0; i < voices.length; i++) {
    const name = voices[i].name.toLowerCase();
    const lang = voices[i].lang.toLowerCase();

    if (
      (lang === "en-in" || lang === "hi-in") &&
      (name.includes("lekha") || name.includes("heera"))
    ) {
      preferredIndex = i;
      break;
    }
  }

  // If no preferred Indian female voice found, pick first voice from filtered list
  if (preferredIndex === -1 && filteredVoices.length > 0) {
    preferredIndex = voices.indexOf(filteredVoices[0]);
  }

  // Add only filtered voices to dropdown
  filteredVoices.forEach((voice) => {
    const i = voices.indexOf(voice);
    const option = document.createElement("option");
    option.value = i;
    option.textContent = `${voice.name} (${voice.lang})${voice.default ? " — Default" : ""}`;
    voiceSelect.appendChild(option);
  });

  voiceSelect.value = preferredIndex !== -1 ? preferredIndex : 0;

  return true;
}

speechSynthesis.onvoiceschanged = populateVoiceList;
populateVoiceList();


function loadVoicesWithRetry(retries = 10, delay = 200) {
  if (!populateVoiceList() && retries > 0) {
    setTimeout(() => loadVoicesWithRetry(retries - 1, delay), delay);
  }
}

function speak(text) {
  if (synth.speaking) {
    synth.cancel(); // stop any ongoing speech first
  }

  if (ttsToggle.checked) {
    currentUtterance = new SpeechSynthesisUtterance(text);
    const selectedIndex = voiceSelect.value;
    if (voices[selectedIndex]) {
      currentUtterance.voice = voices[selectedIndex];
    }
    synth.speak(currentUtterance);
  }
}

function initializeVoices() {
  loadVoicesWithRetry();
  window.speechSynthesis.onvoiceschanged = () => {
    populateVoiceList();
  };
}

// 🌐 On Page Load
window.addEventListener("load", () => {
  initializeVoices();

  const savedTTS = localStorage.getItem("ttsEnabled");
  if (savedTTS !== null) {
    ttsToggle.checked = savedTTS === "true";
  }

  ttsToggle.addEventListener("change", () => {
    localStorage.setItem("ttsEnabled", ttsToggle.checked);
    if (!ttsToggle.checked) {
      synth.cancel();  // stop speech immediately
    };
  });
})

window.addEventListener("DOMContentLoaded", () => {
  const rememberToggle = document.getElementById("rememberToggle");
  const savedPref = localStorage.getItem("rememberSession") === "true";
  rememberToggle.checked = savedPref;

  if (!savedPref) {
    fetch('/clear_session', { method: 'POST' });
    document.cookie = "session=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";

  }

  rememberToggle.addEventListener("change", () => {
    localStorage.setItem("rememberSession", rememberToggle.checked);
  });
});

const toggle = document.getElementById('darkModeSwitch');
  const prefersDark = localStorage.getItem('darkMode') === 'true';

  if (prefersDark) {
    document.body.classList.add('dark-mode');
    toggle.checked = true;
  }

  toggle.addEventListener('change', () => {
    document.body.classList.toggle('dark-mode');
    localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
  });

  function restartChat() {
    synth.cancel();  // stop voice immediately

    const chatBox = document.getElementById('chat-box');
    chatBox.innerHTML = `<div class="chat-msg bot"><strong>Aasha:</strong> Hi, I’m Aasha. What’s on your mind today?</div>`;
    document.getElementById('user-input').placeholder = "Type your message...";
    sessionStorage.clear(); // optional: clears sessionStorage if you use it
  }

  function triggerBalloons() {
    for (let i = 0; i < 10; i++) {
      const balloon = document.createElement('div');
      balloon.className = `balloon color${(i % 4) + 1}`;
      balloon.style.left = `${Math.random() * 100}vw`;
      balloon.style.animationDuration = `${Math.random() * 3 + 4}s`;
      document.body.appendChild(balloon);
  
      setTimeout(() => {
        balloon.remove();
      }, 7000);
    }
  }
  
  function triggerHearts() {
    for (let i = 0; i < 15; i++) {
      const heart = document.createElement('div');
      heart.className = 'heart';
      heart.textContent = '💖';
      heart.style.left = `${Math.random() * 100}vw`;
      heart.style.animationDuration = `${Math.random() * 2 + 2}s`;
      document.body.appendChild(heart);
  
      setTimeout(() => {
        heart.remove();
      }, 5000);
    }
  }

  function triggerConfetti() {
    confetti({
      particleCount: 150,
      spread: 70,
      origin: { y: 0.6 },
      angle: 90,
      zIndex: 9999
    });
  }
  
  window.addEventListener("beforeunload", () => {
    const remember = localStorage.getItem("rememberSession") === "true";
    if (!remember) {
      navigator.sendBeacon('/clear_session');
      document.cookie = "session=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
    }
  });
  window.addEventListener("beforeunload", () => {
    synth.cancel();
  });
  
  function downloadChat() {
    const chatBox = document.getElementById('chat-box');
    let text = '';
  
    chatBox.querySelectorAll('.chat-msg').forEach(msg => {
      text += msg.textContent.trim() + '\n\n';
    });
  
    const blob = new Blob([text], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'chat.txt';
    a.click();
  }
  
  function downloadChatAsPDF() {
    if (typeof jsPDF === "undefined") {
      alert("PDF library failed to load. Please try again later.");
      return;
    }
  
    const chatBox = document.getElementById('chat-box');
    let text = '';
  
    chatBox.querySelectorAll('.chat-msg').forEach(msg => {
      text += msg.textContent.trim() + '\n\n';
    });
  
    const doc = new jsPDF();
    const lines = doc.splitTextToSize(text, 180);
    doc.text(lines, 10, 10);
    doc.save('chat.pdf');
  }
  // 🌟 Intro.js Guided Tour Logic
  window.addEventListener('load', () => {
    if (!localStorage.getItem('seenTutorial')) {
      introJs().start();
      localStorage.setItem('seenTutorial', 'true');
    }
  });