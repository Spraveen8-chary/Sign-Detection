(() => {
  const DETECT_INTERVAL_MS = 800;
  const SPEAK_CONFIDENCE_THRESHOLD = 70;
  const MAX_HISTORY_ITEMS = 20;

  const state = {
    isDetecting: false,
    timerId: null,
    requestInFlight: false,
    abortController: null,
    lastGesture: "",
    lastSpokenText: "",
  };

  const el = {
    detectBtn: document.getElementById("detectBtn"),
    speakBtn: document.getElementById("speakBtn"),
    liveCamera: document.getElementById("liveCamera"),
    cameraPlaceholder: document.getElementById("cameraPlaceholder"),
    cameraStatus: document.getElementById("cameraStatus"),
    language: document.getElementById("language"),
    gestureText: document.getElementById("gestureText"),
    translation: document.getElementById("translation"),
    confidenceText: document.getElementById("confidenceText"),
    history: document.getElementById("history"),
    uploadBtn: document.getElementById("uploadBtn"),
    fileInput: document.getElementById("fileInput"),
    uploadPreview: document.getElementById("uploadPreview"),
    previewCard: document.getElementById("previewCard"),
  };

  if (!el.detectBtn || !el.liveCamera) {
    return;
  }

  el.detectBtn.addEventListener("click", onToggleDetection);
  if (el.speakBtn) {
    el.speakBtn.addEventListener("click", speakCurrentTranslation);
  }
  if (el.uploadBtn && el.fileInput) {
    el.uploadBtn.addEventListener("click", () => el.fileInput.click());
    el.fileInput.addEventListener("change", handleFileUpload);
  }

  el.liveCamera.addEventListener("load", () => {
    if (state.isDetecting && el.cameraStatus) {
      el.cameraStatus.textContent = "Camera stream is live.";
    }
  });

  el.liveCamera.addEventListener("error", () => {
    if (el.cameraStatus) {
      el.cameraStatus.textContent = "Could not load video stream. Check webcam permissions.";
    }
  });

  window.addEventListener("beforeunload", handlePageExit);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      handlePageExit();
    }
  });

  function onToggleDetection() {
    if (state.isDetecting) {
      void stopDetection();
      return;
    }
    void startDetection();
  }

  async function startDetection() {
    if (state.isDetecting) {
      return;
    }

    state.isDetecting = true;
    state.lastGesture = "";
    state.lastSpokenText = "";

    updateDetectButton();
    attachStream();

    if (el.cameraStatus) {
      el.cameraStatus.textContent = "Connecting camera...";
    }

    await runDetection();
    state.timerId = window.setInterval(runDetection, DETECT_INTERVAL_MS);
  }

  async function stopDetection() {
    if (!state.isDetecting && !state.timerId) {
      return;
    }

    state.isDetecting = false;

    if (state.timerId) {
      window.clearInterval(state.timerId);
      state.timerId = null;
    }

    if (state.abortController) {
      state.abortController.abort();
      state.abortController = null;
    }

    detachStream();
    updateDetectButton();

    if (el.cameraStatus) {
      el.cameraStatus.textContent = "Camera is off.";
    }

    resetDetectionOutput();
    await notifyCameraStop();
  }

  function attachStream() {
    if (el.liveCamera) {
      el.liveCamera.src = `/video?ts=${Date.now()}`;
      el.liveCamera.classList.remove("camera-hidden");
    }

    if (el.cameraPlaceholder) {
      el.cameraPlaceholder.classList.add("camera-hidden");
    }
  }

  function detachStream() {
    if (el.liveCamera) {
      el.liveCamera.removeAttribute("src");
      el.liveCamera.classList.add("camera-hidden");
    }

    if (el.cameraPlaceholder) {
      el.cameraPlaceholder.classList.remove("camera-hidden");
    }
  }

  function updateDetectButton() {
    if (!el.detectBtn) {
      return;
    }
    el.detectBtn.textContent = state.isDetecting ? "Stop Detection" : "Start Detection";
  }

  async function runDetection() {
    if (!state.isDetecting || state.requestInFlight) {
      return;
    }

    const lang = el.language ? el.language.value : "en";

    state.requestInFlight = true;
    state.abortController = new AbortController();

    try {
      const response = await fetch(`/detect/${lang}`, {
        method: "GET",
        cache: "no-store",
        signal: state.abortController.signal,
      });

      if (!response.ok) {
        throw new Error(`Detection failed: ${response.status}`);
      }

      const data = await response.json();
      updateDetectionUI(data);
    } catch (error) {
      if (error && error.name !== "AbortError") {
        if (el.gestureText) {
          el.gestureText.textContent = "Detection error";
        }
        if (el.cameraStatus) {
          el.cameraStatus.textContent = "Detection request failed.";
        }
      }
    } finally {
      state.requestInFlight = false;
      state.abortController = null;
    }
  }

  async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
      if (el.uploadPreview && el.previewCard) {
        el.uploadPreview.src = e.target.result;
        el.previewCard.classList.remove("camera-hidden");
      }
    };
    reader.readAsDataURL(file);

    // Stop live detection if running
    if (state.isDetecting) {
      await stopDetection();
    }

    const lang = el.language ? el.language.value : "en";
    const formData = new FormData();
    formData.append("file", file);

    try {
      if (el.gestureText) el.gestureText.textContent = "Analyzing image...";
      const response = await fetch(`/upload/${lang}`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Upload failed");

      const data = await response.json();
      updateDetectionUI(data);
    } catch (error) {
      console.error(error);
      if (el.gestureText) el.gestureText.textContent = "Upload error";
    } finally {
      // Reset file input
      el.fileInput.value = "";
    }
  }

  function updateDetectionUI(data) {
    const gesture = data.gesture || "No gesture detected";
    const translation = data.translation || "No gesture detected";
    const confidence = Number(data.confidence || 0);

    if (el.gestureText) {
      el.gestureText.textContent = gesture;
    }

    if (el.translation) {
      el.translation.textContent = translation;
    }

    updateConfidence(confidence);

    if (gesture !== state.lastGesture && gesture !== "No gesture detected") {
      addHistory(`${gesture} (${confidence}%)`);
      state.lastGesture = gesture;
    }

    if (confidence >= SPEAK_CONFIDENCE_THRESHOLD && translation !== state.lastSpokenText) {
      speakText(translation);
      state.lastSpokenText = translation;
    }
  }

  function updateConfidence(value) {
    const safeValue = Math.max(0, Math.min(100, Number(value) || 0));
    if (el.confidenceFill) {
      el.confidenceFill.style.width = `${safeValue}%`;
    }
    if (el.confidenceText) {
      el.confidenceText.textContent = `${safeValue}%`;
    }
  }

  function addHistory(text) {
    if (!el.history) {
      return;
    }

    const item = document.createElement("li");
    item.textContent = text;
    el.history.prepend(item);

    while (el.history.children.length > MAX_HISTORY_ITEMS) {
      el.history.removeChild(el.history.lastChild);
    }
  }

  function resetDetectionOutput() {
    if (el.gestureText) {
      el.gestureText.textContent = "Waiting for gesture...";
    }
    if (el.translation) {
      el.translation.textContent = "Waiting for translation...";
    }
    updateConfidence(0);
  }

  function speakCurrentTranslation() {
    if (!el.translation) {
      return;
    }
    speakText(el.translation.textContent || "");
  }

  function speakText(text) {
    if (!text) {
      return;
    }

    const language = el.language ? el.language.value : "en";
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = language;

    const voices = window.speechSynthesis.getVoices();
    const voice = voices.find(v => v.lang.toLowerCase().includes(language.toLowerCase()));
    if (voice) {
      utterance.voice = voice;
    }

    window.speechSynthesis.speak(utterance);
  }

  async function notifyCameraStop() {
    try {
      await fetch("/camera/stop", {
        method: "POST",
        keepalive: true,
      });
    } catch (_) {
      // Ignore network errors during page teardown.
    }
  }

  function handlePageExit() {
    if (!state.isDetecting) {
      return;
    }
    void stopDetection();
  }
})();
