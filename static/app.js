/**
 * app.js
 * Sketch-to-Story Frontend Logic — Kid-Friendly "Magic Sketchbook" Edition
 * Handles canvas drawing, bitmap export, API calls, and story display.
 */

// ── State ──────────────────────────────────────────────────────────
const state = {
    canvases: [],       // { canvas, ctx, isDrawing, hasDrawn }
    brushSize: 12,
    selectedColor: "#000000", // Current selected ink color
    predictions: [null, null, null], // Individual results
    story: null,
    wordLimit: "medium",  // short | medium | long
    genre: "Fantasy",     // selected genre string
};

// ── Initialisation ─────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    initParticles();
    initCanvases();
    initBrushControls();
    initButtons();
    initStoryOptions();
});

// ── Background Particles ───────────────────────────────────────────
function initParticles() {
    const container = document.getElementById("bg-particles");
    // Vibrant pastel colors for kids
    const colors = [
        "rgba(108, 92, 231, 0.4)",  // Pastel Purple
        "rgba(116, 185, 255, 0.4)", // Pastel Blue
        "rgba(0, 184, 148, 0.3)",   // Mint Green
        "rgba(253, 203, 110, 0.4)", // Sunshine Yellow
        "rgba(255, 118, 117, 0.4)", // Soft Coral
    ];

    for (let i = 0; i < 40; i++) {
        const particle = document.createElement("div");
        particle.classList.add("particle");
        const size = Math.random() * 12 + 6;
        particle.style.width = `${size}px`;
        particle.style.height = `${size}px`;
        particle.style.left = `${Math.random() * 100}%`;
        particle.style.background = colors[Math.floor(Math.random() * colors.length)];
        particle.style.animationDuration = `${Math.random() * 10 + 8}s`;
        particle.style.animationDelay = `${Math.random() * 5}s`;
        container.appendChild(particle);
    }
}

// ── Canvas Management ───────────────────────────────────────────────
function initCanvases() {
    for (let i = 1; i <= 3; i++) {
        const canvas = document.getElementById(`canvas-${i}`);
        const ctx = canvas.getContext("2d");

        // White background (Sketchbook feel)
        ctx.fillStyle = "#FFFFFF";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const canvasState = {
            canvas,
            ctx,
            isDrawing: false,
            hasDrawn: false,
            tool: "pen", 
            index: i,
        };

        // Mouse events
        canvas.addEventListener("mousedown", (e) => startDrawing(e, canvasState));
        canvas.addEventListener("mousemove", (e) => draw(e, canvasState));
        canvas.addEventListener("mouseup", () => stopDrawing(canvasState));
        canvas.addEventListener("mouseleave", () => stopDrawing(canvasState));

        // Touch events
        canvas.addEventListener("touchstart", (e) => {
            e.preventDefault();
            startDrawing(e.touches[0], canvasState);
        }, { passive: false });
        canvas.addEventListener("touchmove", (e) => {
            e.preventDefault();
            draw(e.touches[0], canvasState);
        }, { passive: false });
        canvas.addEventListener("touchend", (e) => {
            e.preventDefault();
            stopDrawing(canvasState);
        });

        state.canvases.push(canvasState);
    }
}

function getCanvasPos(e, canvas) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY,
    };
}

function startDrawing(e, canvasState) {
    canvasState.isDrawing = true;
    const pos = getCanvasPos(e, canvasState.canvas);
    canvasState.ctx.beginPath();
    canvasState.ctx.moveTo(pos.x, pos.y);

    // Visual feedback: Add glow class to the card while drawing
    const card = document.getElementById(`canvas-card-${canvasState.index}`);
    if (card) card.classList.add("active-glow");

    // Hide placeholder
    if (!canvasState.hasDrawn) {
        canvasState.hasDrawn = true;
        const placeholder = document.getElementById(`placeholder-${canvasState.index}`);
        placeholder.classList.add("hidden");
    }
}

function draw(e, canvasState) {
    if (!canvasState.isDrawing) return;
    const pos = getCanvasPos(e, canvasState.canvas);
    const ctx = canvasState.ctx;
    const isEraser = canvasState.tool === "eraser";

    // Black ink on White paper, or child's selected color
    ctx.strokeStyle = isEraser ? "#FFFFFF" : state.selectedColor;
    ctx.lineWidth = isEraser ? state.brushSize * 2 : state.brushSize;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
}

function stopDrawing(canvasState) {
    if (canvasState.isDrawing) {
        canvasState.isDrawing = false;
        canvasState.ctx.beginPath();
        
        // Remove glow class when done drawing
        const card = document.getElementById(`canvas-card-${canvasState.index}`);
        if (card) card.classList.remove("active-glow");
    }
}

function clearCanvas(index) {
    const cs = state.canvases[index];
    cs.ctx.fillStyle = "#FFFFFF";
    cs.ctx.fillRect(0, 0, cs.canvas.width, cs.canvas.height);
    cs.hasDrawn = false;

    // Reset this specific prediction
    state.predictions[index] = null;

    // Show placeholder again
    const placeholder = document.getElementById(`placeholder-${cs.index}`);
    placeholder.classList.remove("hidden");

    // Hide prediction badge
    const badge = document.getElementById(`prediction-${cs.index}`);
    badge.classList.add("hidden");
}

// ── Image Inversion for AI ─────────────────────────────────────────

/**
 * Inverts canvas colors for the AI model.
 * The model expects white strokes on a black background.
 * The UI uses black strokes on a white background.
 */
function getInvertedImageData(canvas) {
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
    const tempCtx = tempCanvas.getContext("2d");

    // Copy original canvas to temp
    tempCtx.drawImage(canvas, 0, 0);

    // Get pixel data
    const imageData = tempCtx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;

    // Thresholding: Convert colored sketches to Pure B&W for the AI
    // Boundary: Any pixel that is NOT the background (white) becomes the stroke (white for model)
    for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        
        // If the pixel is not white (i.e., it's drawn)
        if (r < 250 || g < 250 || b < 250) {
            data[i]     = 255; // White R
            data[i + 1] = 255; // White G
            data[i + 2] = 255; // White B
        } else {
            data[i]     = 0;   // Black R
            data[i + 1] = 0;   // Black G
            data[i + 2] = 0;   // Black B
        }
        data[i + 3] = 255; // Fully opaque
    }

    tempCtx.putImageData(imageData, 0, 0);
    return tempCanvas.toDataURL("image/png");
}

// ── Tool Management ────────────────────────────────────────────────
function initToolControls() {
    for (let i = 1; i <= 3; i++) {
        const penBtn = document.getElementById(`tool-pen-${i}`);
        const eraserBtn = document.getElementById(`tool-eraser-${i}`);
        const cs = state.canvases[i - 1];

        penBtn.addEventListener("click", () => {
            cs.tool = "pen";
            penBtn.classList.add("active");
            eraserBtn.classList.remove("active");
        });

        eraserBtn.addEventListener("click", () => {
            cs.tool = "eraser";
            eraserBtn.classList.add("active");
            penBtn.classList.remove("active");
        });
    }
}

// ── Brush Controls ──────────────────────────────────────────────────
function initBrushControls() {
    const slider = document.getElementById("brush-size");
    const display = document.getElementById("brush-size-value");

    slider.addEventListener("input", () => {
        state.brushSize = parseInt(slider.value);
        display.textContent = slider.value;
    });
}

// ── Button Actions ──────────────────────────────────────────────────
function initButtons() {
    // Clear individual canvases
    for (let i = 0; i < 3; i++) {
        document.getElementById(`clear-${i + 1}`).addEventListener("click", () => {
            clearCanvas(i);
        });
    }

    initToolControls();

    // Clear All
    document.getElementById("btn-clear-all").addEventListener("click", () => {
        for (let i = 0; i < 3; i++) clearCanvas(i);
        hideStory();
        resetSteps();
    });

    // Process sketches → predict → generate story
    document.getElementById("btn-recognize").addEventListener("click", generateMagicStory);

    // Color Palette selection
    document.querySelectorAll(".color-swatch").forEach(swatch => {
        swatch.addEventListener("click", () => {
            document.querySelectorAll(".color-swatch").forEach(s => s.classList.remove("active"));
            swatch.classList.add("active");
            state.selectedColor = swatch.dataset.color;
        });
    });

    // New Story (same objects, re-generate with current options)
    document.getElementById("btn-new-story").addEventListener("click", () => {
        if (state.predictions.every(p => p !== null)) {
            generateStory(state.predictions.map((p) => p.label));
        }
    });

    // Draw Again
    document.getElementById("btn-draw-again").addEventListener("click", () => {
        for (let i = 0; i < 3; i++) clearCanvas(i);
        hideStory();
        resetSteps();
        window.scrollTo({ top: 0, behavior: "smooth" });
    });
}

// ── Story Options (word length + genre) ────────────────────────────
function initStoryOptions() {
    // ── Word Length pills ──
    document.querySelectorAll("#length-pills .pill").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll("#length-pills .pill").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            state.wordLimit = btn.dataset.value;
        });
    });

    // ── Genre chips ──
    document.querySelectorAll("#genre-grid .genre-chip").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll("#genre-grid .genre-chip").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            if (btn.dataset.genre === "__other__") {
                document.getElementById("custom-genre-wrap").classList.remove("hidden");
                const input = document.getElementById("custom-genre-input");
                input.focus();
                // Update state live as user types
                input.oninput = () => {
                    state.genre = input.value.trim() || "Fantasy";
                };
                state.genre = input.value.trim() || "Fantasy";
            } else {
                document.getElementById("custom-genre-wrap").classList.add("hidden");
                state.genre = btn.dataset.genre;
            }
        });
    });
}

// ── Recognition (on demand) ─────────────────────────────────────────
/**
 * Sends all 3 sketches to /predict and updates state.predictions.
 * Returns true if all 3 are successfully identified.
 */
async function predictAllCanvases() {
    const sketches = state.canvases.map(cs => {
        return cs.hasDrawn ? getInvertedImageData(cs.canvas) : null;
    });

    if (sketches.some(s => s === null)) {
        alert("Please draw something in all three canvases first! ✨");
        return false;
    }

    const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sketches }),
    });

    if (!response.ok) return false;
    const data = await response.json();

    let allOk = true;
    data.predictions.forEach((pred, idx) => {
        const i = idx + 1;
        const badge = document.getElementById(`prediction-${i}`);
        const labelEl = badge.querySelector(".prediction-label");

        if (pred && pred.label !== "unknown") {
            state.predictions[idx] = pred;
            labelEl.textContent = pred.label;
            badge.classList.remove("hidden");
        } else {
            allOk = false;
        }
    });

    return allOk;
}

async function generateMagicStory() {
    const btn = document.getElementById("btn-recognize");
    const btnText = btn.querySelector(".btn-text");
    const btnLoading = btn.querySelector(".btn-loading");

    const progressContainer = document.getElementById("story-progress-container");
    const progressBar = document.getElementById("story-progress-bar");
    const progressStatus = document.getElementById("story-progress-status");

    // Show loading state immediately
    btn.disabled = true;
    btnText.classList.add("hidden");
    btnLoading.classList.remove("hidden");
    progressContainer.style.display = "block";
    progressBar.style.width = "5%";
    progressStatus.textContent = "Magician is looking at your sketches...";

    try {
        updateStep(2);

        const success = await predictAllCanvases();
        if (!success) {
            progressContainer.style.display = "none";
            updateStep(1);
            return;
        }

        progressBar.style.width = "40%";
        progressStatus.textContent = "Recognition complete! Now for the story...";
        updateStep(3);

        // Safe extraction — skip any nulls
        const objects = state.predictions
            .filter(p => p !== null && p.label)
            .map(p => p.label);

        if (objects.length !== 3) {
            throw new Error("Could not identify all 3 sketches. Please try drawing more clearly.");
        }

        await generateStory(objects);

    } catch (error) {
        progressContainer.style.display = "none";
        console.error("generateMagicStory error:", error);
        alert(`Something went wrong:\n${error.message}`);
    } finally {
        btn.disabled = false;
        btnText.classList.remove("hidden");
        btnLoading.classList.add("hidden");
    }
}

async function generateStory(objects) {
    const progressBar = document.getElementById("story-progress-bar");
    const progressStatus = document.getElementById("story-progress-status");
    const progressContainer = document.getElementById("story-progress-container");

    // Declare outside try so it can be cleared in finally
    let progressInterval = null;

    try {
        progressBar.style.width = "50%";
        progressStatus.textContent = "Magician is gathering ingredients...";

        // Simulate progress ticks while waiting for the AI
        progressInterval = setInterval(() => {
            const currentWidth = parseFloat(progressBar.style.width);
            if (currentWidth < 95) {
                progressBar.style.width = (currentWidth + Math.random() * 2) + "%";
                if (currentWidth > 60) progressStatus.textContent = "Mixing the magic words...";
                if (currentWidth > 80) progressStatus.textContent = "Writing the final chapter...";
            }
        }, 1000);

        const storyResponse = await fetch("/generate-story", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                objects,
                word_limit: state.wordLimit,
                genre: state.genre,
            }),
        });

        // Read the body ONCE
        const storyData = await storyResponse.json();

        if (!storyResponse.ok) {
            throw new Error(storyData.error || "Story generation failed");
        }

        progressBar.style.width = "100%";
        progressStatus.textContent = "🪄 Magic complete!";
        state.story = storyData;

        // Make all steps green (Draw, Recognize, Story)
        updateStep(4);

        setTimeout(() => {
            progressContainer.style.display = "none";
            showStory(storyData, objects);
            setupAudioPlayer(storyData.story);
        }, 800);

    } finally {
        // Always clear the interval, whether success or error
        if (progressInterval) clearInterval(progressInterval);
    }
}

// ── Audio Player (Web Speech API) ─────────────────────────────────

let currentUtterance = null;
let isSpeaking = false;
let isPaused = false;

function stopSpeech() {
    if (currentUtterance) {
        window.speechSynthesis.cancel();
        currentUtterance = null;
    }
    isSpeaking = false;
    isPaused = false;
    clearWordHighlight();
}

function clearWordHighlight() {
    document.querySelectorAll(".story-word.speaking").forEach(el => el.classList.remove("speaking"));
}

function highlightWord(charIndex) {
    clearWordHighlight();
    // Find which word span contains this character offset
    const spans = document.querySelectorAll("#story-text .story-word");
    // charIndex from SpeechSynthesis maps to the utterance text
    // We track cumulative char position via data-start attribute set in showStory
    for (const span of spans) {
        const start = parseInt(span.dataset.start, 10);
        const end   = parseInt(span.dataset.end,   10);
        if (charIndex >= start && charIndex < end) {
            span.classList.add("speaking");
            span.scrollIntoView({ behavior: "smooth", block: "nearest" });
            break;
        }
    }
}

function setupAudioPlayer(text) {
    const playBtn = document.getElementById("btn-play-audio");
    playBtn.style.display = "inline-flex";
    setPlayBtnState("idle");

    // Remove any old listener by replacing the node
    const freshBtn = playBtn.cloneNode(true);
    playBtn.parentNode.replaceChild(freshBtn, playBtn);
    setPlayBtnState("idle", freshBtn);

    freshBtn.onclick = () => {
        if (!isSpeaking) {
            // ─ Start fresh ─
            stopSpeech(); // cancel any lingering speech

            currentUtterance = new SpeechSynthesisUtterance(text);
            currentUtterance.rate  = 0.92;
            currentUtterance.pitch = 1.05;

            currentUtterance.onboundary = (e) => {
                if (e.name === "word") highlightWord(e.charIndex);
            };

            currentUtterance.onstart = () => {
                isSpeaking = true;
                isPaused   = false;
                setPlayBtnState("playing", freshBtn);
            };

            currentUtterance.onend = () => {
                isSpeaking = false;
                isPaused   = false;
                clearWordHighlight();
                setPlayBtnState("idle", freshBtn);
            };

            currentUtterance.onerror = () => {
                isSpeaking = false;
                isPaused   = false;
                clearWordHighlight();
                setPlayBtnState("idle", freshBtn);
            };

            window.speechSynthesis.speak(currentUtterance);

        } else if (!isPaused) {
            // ─ Pause ─
            window.speechSynthesis.pause();
            isPaused = true;
            setPlayBtnState("paused", freshBtn);

        } else {
            // ─ Resume ─
            window.speechSynthesis.resume();
            isPaused = false;
            setPlayBtnState("playing", freshBtn);
        }
    };
}

function setPlayBtnState(btnState, btn) {
    const el = btn || document.getElementById("btn-play-audio");
    if (!el) return;
    const icons = { idle: "🔊 Play Audio", playing: "⏸️ Pause", paused: "▶️ Resume" };
    el.textContent = icons[btnState] || icons.idle;
}

function showStory(storyData, objects) {
    const section   = document.getElementById("story-section");
    const genreEl   = document.getElementById("story-genre");
    const objectsEl = document.getElementById("story-objects");
    const textEl    = document.getElementById("story-text");

    // Stop any in-progress speech from a previous story
    stopSpeech();

    // Genre badge
    genreEl.textContent = storyData.genre;

    // Object tags
    objectsEl.innerHTML = objects
        .map((obj) => `<span class="story-object-tag">${obj}</span>`)
        .join('<span class="arrow" style="color: var(--accent-secondary); font-size: 1.5rem; margin: 0 10px;">✨</span>');

    // Build word spans — track character offsets for speech boundary highlighting
    textEl.innerHTML = "";
    section.classList.remove("hidden");

    const fullText = storyData.story;
    const vocab = storyData.vocabulary || {};
    const vocabNames = Object.keys(vocab);

    console.log(`[*] Rendering story with ${vocabNames.length} vocabulary words.`);
    if (vocabNames.length > 0) {
        console.log(`[*] Vocabulary keys: ${vocabNames.join(", ")}`);
    }

    // Get lowercase, trimmed keys for robust matching
    const vocabKeys = vocabNames.map(k => k.trim().toLowerCase());

    // Split preserving spaces and punctuation
    const tokens = fullText.match(/(\S+|\s+)/g) || [];
    const wordCount = tokens.filter(t => /\S/.test(t)).length;
    const delay = wordCount > 50 ? 30 : 50;
    
    let charPos = 0;
    let wordIdx = 0;

    tokens.forEach((token) => {
        // Clean token: lowercase and remove most punctuation, including trailing 's' for simple plural matching
        const cleanToken = token.trim().toLowerCase().replace(/[.,!?;:\"'()]/g, "");
        
        if (/\S/.test(token)) {
            // Word token
            const span = document.createElement("span");
            span.classList.add("story-word");
            
            // Vocabulary check - robust match (exact or plural-aware)
            let matchedKey = null;
            if (cleanToken) {
                // 1. Exact match
                if (vocabKeys.includes(cleanToken)) {
                    matchedKey = vocabNames.find(k => k.trim().toLowerCase() === cleanToken);
                } 
                // 2. Simple plural match (e.g., 'forests' -> 'forest')
                else if (cleanToken.endsWith('s') && vocabKeys.includes(cleanToken.slice(0, -1))) {
                    matchedKey = vocabNames.find(k => k.trim().toLowerCase() === cleanToken.slice(0, -1));
                }
            }

            if (matchedKey) {
                span.classList.add("vocab-word");
                span.setAttribute("data-definition", vocab[matchedKey]);
            }

            span.textContent = token;
            span.dataset.start = charPos;
            span.dataset.end   = charPos + token.length;
            span.style.animationDelay = `${wordIdx * delay}ms`;
            textEl.appendChild(span);
            wordIdx++;
        } else {
            // Whitespace
            textEl.appendChild(document.createTextNode(token));
        }
        charPos += token.length;
    });

    // Smooth scroll to story
    setTimeout(() => {
        section.scrollIntoView({ behavior: "smooth", block: "center" });
    }, 500);
}


function hideStory() {
    stopSpeech();
    const playBtn = document.getElementById("btn-play-audio");
    if (playBtn) playBtn.style.display = "none";
    document.getElementById("story-section").classList.add("hidden");
}

function updateStep(activeStep) {
    for (let i = 1; i <= 3; i++) {
        const stepEl = document.getElementById(`step-${i}-indicator`);
        stepEl.classList.remove("active", "completed");

        if (i < activeStep) {
            stepEl.classList.add("completed");
        } else if (i === activeStep) {
            stepEl.classList.add("active");
        }
    }

    // Update connectors
    const connectors = document.querySelectorAll(".step-connector");
    connectors.forEach((conn, idx) => {
        if (idx + 1 < activeStep) {
            conn.classList.add("active");
        } else {
            conn.classList.remove("active");
        }
    });
}

function resetSteps() {
    updateStep(1);
    state.predictions = [null, null, null];
    state.story = null;
}
