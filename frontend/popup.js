const messagesEl = document.getElementById("messages");
const form = document.getElementById("chatForm");
const input = document.getElementById("text");
const sendBtn = document.getElementById("sendBtn");
const typingEl = document.getElementById("typing");
const clearBtn = document.getElementById("clearBtn");

const BACKEND_URL = "http://localhost:8000/chat";

function nowTime() {
  const d = new Date();
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function addMessage(role, text, extras) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + (role === "user" ? "me" : "bot");

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  if (extras) {
    if (extras.source_1) {
      const s = document.createElement("div");
      s.style.marginTop = "6px";
      s.style.fontSize = "11px";
      s.style.color = "rgba(255,255,255,0.7)";
      s.textContent = `Source: ${extras.source_1}`;
      bubble.appendChild(s);
    }
    if (extras.analysis) {
      const a = document.createElement("div");
      a.style.marginTop = "4px";
      a.style.fontSize = "11px";
      a.style.color = "rgba(255,255,255,0.7)";
      a.textContent = `Analysis: ${extras.analysis}`;
      bubble.appendChild(a);
    }
    if (extras.citation) {
      const c = document.createElement("div");
      c.style.marginTop = "4px";
      c.style.fontSize = "11px";
      c.style.color = "rgba(255,255,255,0.7)";
      c.textContent = `Citation: ${extras.citation}`;
      bubble.appendChild(c);
    }
  }

  wrap.appendChild(bubble);

  if (role === "user") {
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = nowTime();
    bubble.appendChild(meta);
  }

  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setTyping(on) {
  typingEl.classList.toggle("hidden", !on);
  if (on) messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setSending(on) {
  sendBtn.disabled = on;
  input.disabled = on;
}

async function botReply(userText) {
  setTyping(true);
  setSending(true);
  try {
    const res = await fetch(BACKEND_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: userText }),
    });

    if (!res.ok) throw new Error(`Backend responded ${res.status}`);

    const data = await res.json();
    setTyping(false);
    addMessage("assistant", data.answer ?? "(no answer)", {
      source_1: data.source_1,
      analysis: data.source_analysis,
      citation: data.citation,
    });
  } catch (err) {
    setTyping(false);
    addMessage(
      "assistant",
      `Couldn't reach backend at ${BACKEND_URL}.\n${err.message}`
    );
  } finally {
    setSending(false);
    input.focus();
  }
}

function clearChat() {
  messagesEl.innerHTML = "";
  setTyping(false);
  input.focus();
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;

  addMessage("user", text);
  input.value = "";
  botReply(text);
});

clearBtn.addEventListener("click", clearChat);

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") clearChat();
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
    e.preventDefault();
    form.requestSubmit();
  }
});

addMessage("assistant", "Hey — I'm ready. Ask me anything.");
input.focus();
