const API_URL = window.location.protocol.startsWith("http")
  ? `${window.location.protocol}//${window.location.hostname}:8000/chat`
  : "http://127.0.0.1:8000/chat";

const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const messagesElement = document.querySelector("#messages");
const welcome = document.querySelector("#welcome");
const sendButton = document.querySelector("#send-button");
const chatScroll = document.querySelector("#chat-scroll");
const connectionStatus = document.querySelector("#connection-status");
const statusLabel = document.querySelector("#status-label");
const conversation = [];
let busy = false;

function setStatus(state, label) {
  connectionStatus.dataset.state = state;
  statusLabel.textContent = label;
}

function scrollToLatest() {
  chatScroll.scrollTo({ top: chatScroll.scrollHeight, behavior: "smooth" });
}

function addMessage(role, content, options = {}) {
  const row = document.createElement("article");
  row.className = `message-row ${role}${options.error ? " error" : ""}`;
  if (options.pending) row.dataset.pending = "true";

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = role === "user" ? "YOU" : "F";

  const body = document.createElement("div");
  body.className = "message-body";
  const heading = document.createElement("div");
  heading.className = "message-heading";
  const name = document.createElement("strong");
  name.textContent = role === "user" ? "You" : "Fieldnote";
  const time = document.createElement("time");
  time.textContent = new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" }).format(new Date());
  heading.append(name, time);

  const text = document.createElement("div");
  text.className = "message-text";
  if (options.pending) {
    const dots = document.createElement("span");
    dots.className = "typing-dots";
    dots.setAttribute("aria-label", "Thinking");
    dots.innerHTML = "<span></span><span></span><span></span>";
    text.append(dots);
  } else {
    text.textContent = content;
  }

  body.append(heading, text);
  row.append(avatar, body);
  messagesElement.append(row);
  scrollToLatest();
  return row;
}

function extractAnswer(data) {
  if (typeof data === "string") return data;
  if (typeof data?.answer === "string") return data.answer;
  if (typeof data?.response === "string") return data.response;
  if (typeof data?.message === "string") return data.message;
  if (typeof data?.content === "string") return data.content;
  if (typeof data?.choices?.[0]?.message?.content === "string") return data.choices[0].message.content;
  throw new Error("The API response did not contain an answer, response, or message field.");
}

async function sendMessage(rawMessage) {
  const message = rawMessage.trim();
  if (!message || busy) return;

  welcome.hidden = true;
  addMessage("user", message);
  input.value = "";
  input.style.height = "auto";
  busy = true;
  sendButton.disabled = true;
  setStatus("working", "Waiting for answer");
  const pending = addMessage("assistant", "", { pending: true });

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history: conversation }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = data.detail || data.error || `Request failed (${response.status}).`;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }

    const answer = extractAnswer(data);
    pending.remove();
    addMessage("assistant", answer);
    conversation.push({ role: "user", content: message }, { role: "assistant", content: answer });
    setStatus("online", "API connected");
  } catch (error) {
    pending.remove();
    const detail = error instanceof TypeError
      ? `Could not reach ${API_URL}. Check that your FastAPI server is running, the route is correct, and CORS allows this frontend.`
      : error.message;
    addMessage("assistant", detail, { error: true });
    setStatus("error", "Connection issue");
  } finally {
    busy = false;
    sendButton.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(input.value);
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.dataset.prompt;
    form.requestSubmit();
  });
});

document.querySelector("#new-chat").addEventListener("click", () => {
  conversation.length = 0;
  messagesElement.replaceChildren();
  welcome.hidden = false;
  setStatus("idle", "API not checked");
  input.value = "";
  input.focus();
});