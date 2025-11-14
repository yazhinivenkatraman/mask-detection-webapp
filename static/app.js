const statusEl = document.getElementById("status");
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const startBtn = document.getElementById("startBtn");
const stopBtn  = document.getElementById("stopBtn");

let stream = null;       // current MediaStream
let detectTimer = null;  // interval ID for sendFrame loop

console.log("app.js loaded");

// ---- Start camera and detection ----
async function startCamera() {
    try {
        // If already running, stop first (safety)
        await stopCamera(false);

        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 }
        });
        video.srcObject = stream;

        statusEl.textContent = "Webcam started. Running detection...";
        startBtn.disabled = true;
        stopBtn.disabled = false;

        // Start detection loop (every 500 ms)
        detectTimer = setInterval(sendFrame, 500);

    } catch (err) {
        console.error("Error accessing webcam:", err);
        statusEl.textContent = "Error accessing webcam. Check permissions.";
    }
}

// ---- Stop camera and detection ----
async function stopCamera(updateStatus = true) {
    // Stop detection timer
    if (detectTimer !== null) {
        clearInterval(detectTimer);
        detectTimer = null;
    }

    // Stop video tracks
    if (stream) {
        stream.getTracks().forEach(t => t.stop());
        stream = null;
    }

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (updateStatus) {
        statusEl.textContent = "Camera stopped. Click 'Start Camera' to run again.";
    }
    startBtn.disabled = false;
    stopBtn.disabled = true;
}

// ---- Send frame to backend ----
async function sendFrame() {
    if (!video.videoWidth || !video.videoHeight) {
        // video not ready
        return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    canvas.toBlob(async (blob) => {
        if (!blob) return;

        const formData = new FormData();
        formData.append("file", blob, "frame.jpg");

        try {
            const res = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                console.error("Non-200 from /predict:", res.status);
                statusEl.textContent = "Server error from /predict";
                return;
            }

            const data = await res.json();

            const bytes = new Uint8Array(
                data.image.match(/.{1,2}/g).map(b => parseInt(b, 16))
            );
            const img = new Image();
            img.src = URL.createObjectURL(new Blob([bytes], { type: "image/jpeg" }));

            img.onload = () => {
                ctx.drawImage(img, 0, 0);
            };
        } catch (e) {
            console.error("Error calling /predict:", e);
            statusEl.textContent = "Error calling /predict. See console.";
        }
    }, "image/jpeg", 0.8);
}

// ---- Wire up buttons ----
startBtn.addEventListener("click", () => {
    startCamera();
});

stopBtn.addEventListener("click", () => {
    stopCamera();
});

// No auto-start now; user must click Start
