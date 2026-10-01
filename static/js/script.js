const video = document.getElementById("webcam");
const result = document.getElementById("result");
const canvas = document.getElementById("canvas");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");

const overlay = document.getElementById("overlay");
const status = document.getElementById("status");

let stream = null;
let running = false;
let processing = false;


/* ================= CAMERA ================= */

async function startCamera() {

    try {

        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: {
                    ideal: 640
                },

                height: {
                    ideal: 480
                },

                facingMode: "user"
            },

            audio: false
        });

        video.srcObject = stream;

        running = true;

        startBtn.disabled = true;
        stopBtn.disabled = false;

        overlay.classList.add("hidden");

        updateStatus("Camera Active", true);

        processFrame();

    } catch (error) {

        console.error(error);

        updateStatus("Camera Error", false);

        alert(
            "Unable to access your camera. Please allow camera permission and try again."
        );
    }
}


/* ================= STOP CAMERA ================= */

function stopCamera() {

    running = false;

    if (stream) {

        stream.getTracks().forEach(track => {
            track.stop();
        });

        stream = null;
    }

    video.srcObject = null;

    result.src = "";

    startBtn.disabled = false;
    stopBtn.disabled = true;

    overlay.classList.remove("hidden");

    updateStatus("Ready", false);
}


/* ================= STATUS ================= */

function updateStatus(text, active) {

    if (!status) return;

    status.innerHTML = `
        <span class="status-dot"
            style="background:${active ? "#22c55e" : "#64748b"};
            ${active ? "box-shadow:0 0 10px #22c55e;" : ""}">
        </span>
        ${text}
    `;
}


/* ================= PROCESS FRAME ================= */

async function processFrame() {

    if (!running) {
        return;
    }

    if (!processing) {

        processing = true;

        try {

            if (video.readyState >= 2) {

                canvas.width = video.videoWidth || 640;
                canvas.height = video.videoHeight || 480;

                const context = canvas.getContext("2d");

                context.drawImage(
                    video,
                    0,
                    0,
                    canvas.width,
                    canvas.height
                );

                const blob = await new Promise(resolve => {

                    canvas.toBlob(
                        resolve,
                        "image/jpeg",
                        0.8
                    );

                });

                const formData = new FormData();

                formData.append(
                    "frame",
                    blob,
                    "frame.jpg"
                );

                const response = await fetch(
                    "/predict",
                    {
                        method: "POST",
                        body: formData
                    }
                );

                if (!response.ok) {
                    throw new Error("Prediction request failed");
                }

                const imageBlob = await response.blob();

                const imageUrl = URL.createObjectURL(imageBlob);

                const oldUrl = result.dataset.url;

                result.src = imageUrl;

                result.dataset.url = imageUrl;

                if (oldUrl) {
                    URL.revokeObjectURL(oldUrl);
                }
            }

        } catch (error) {

            console.error(
                "Detection error:",
                error
            );

        } finally {

            processing = false;
        }
    }

    setTimeout(
        processFrame,
        120
    );
}


/* ================= BUTTONS ================= */

startBtn.addEventListener(
    "click",
    startCamera
);

stopBtn.addEventListener(
    "click",
    stopCamera
);


/* ================= NAVBAR ================= */

const navLinks =
    document.querySelectorAll(".nav-link");

const sections =
    document.querySelectorAll("section[id]");


window.addEventListener(
    "scroll",
    () => {

        let current = "";

        sections.forEach(section => {

            const sectionTop =
                section.offsetTop - 150;

            if (
                window.scrollY >= sectionTop
            ) {
                current = section.getAttribute("id");
            }

        });

        navLinks.forEach(link => {

            link.classList.remove("active");

            if (
                link.getAttribute("href") ===
                `#${current}`
            ) {

                link.classList.add("active");
            }
        });
    }
);


/* ================= CLEANUP ================= */

window.addEventListener(
    "beforeunload",
    () => {

        if (stream) {

            stream.getTracks().forEach(track => {
                track.stop();
            });
        }
    }
);