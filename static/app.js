"use strict";

// Токен хранится только в поле ввода: не записывается в localStorage, cookie или URL.
// Ответ сервера выводится только через textContent (без innerHTML).

const form = document.getElementById("predict-form");
const tokenInput = document.getElementById("token");
const textInput = document.getElementById("text");
const submitButton = document.getElementById("submit");
const statusBox = document.getElementById("status");
const resultBox = document.getElementById("result");
const counter = document.getElementById("count");

const LABELS = {
  spam: "Спам",
  ham: "Не спам",
  undetermined: "Недостаточно данных для решения (большинство слов незнакомо модели)",
};

function showStatus(kind, message) {
  statusBox.hidden = false;
  statusBox.className = "status " + kind;
  statusBox.textContent = message;
}

function setField(id, value) {
  document.getElementById(id).textContent = value === null || value === undefined ? "—" : String(value);
}

function errorMessage(status, data, retryAfter) {
  const serverMessage = data && data.error && typeof data.error.message === "string" ? data.error.message : "";
  switch (status) {
    case 401: return "Нет доступа: токен отсутствует или неверен.";
    case 403: return "Недостаточно прав: этот токен не предназначен для предсказаний.";
    case 413: return "Запрос слишком большой.";
    case 415: return "Неподдерживаемый формат запроса.";
    case 400:
    case 422: return "Ошибка ввода: " + serverMessage;
    case 429: return "Превышен лимит запросов. Повторите через " + (retryAfter || "несколько") + " с.";
    case 503: return "Сервис временно недоступен или занят. Повторите позже.";
    case 504: return "Превышено время обработки запроса.";
    default: return "Ошибка сервиса (код " + status + ").";
  }
}

textInput.addEventListener("input", () => {
  counter.textContent = String(textInput.value.length);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  resultBox.hidden = true;
  const token = tokenInput.value.trim();
  if (!/^[A-Za-z0-9._~+\/=-]+$/.test(token)) {
    showStatus("error", "Введите API-токен (латинские буквы, цифры и символы ._~+/=-).");
    return;
  }

  submitButton.disabled = true;
  showStatus("pending", "Выполняется запрос…");
  let response;
  try {
    response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Authorization": "Bearer " + token },
      body: JSON.stringify({ text: textInput.value }),
      cache: "no-store",
      credentials: "omit",
      referrerPolicy: "no-referrer",
    });
  } catch (e) {
    showStatus("error", "Сервис недоступен: нет соединения.");
    submitButton.disabled = false;
    return;
  }

  let data = null;
  try {
    data = await response.json();
  } catch (e) {
    data = null;
  }
  submitButton.disabled = false;

  if (!response.ok || !data) {
    const requestId = data && data.request_id ? " Идентификатор запроса: " + data.request_id : "";
    showStatus("error", errorMessage(response.status, data, response.headers.get("Retry-After")) + requestId);
    return;
  }

  showStatus("ok", "Готово.");
  setField("r-prediction", LABELS[data.prediction] || data.prediction);
  setField("r-score", data.spam_score);
  setField("r-coverage", data.known_token_share);
  setField("r-version", data.model_version);
  setField("r-request", data.request_id);
  resultBox.hidden = false;
});
