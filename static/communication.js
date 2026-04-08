let peer;
let dataConn;
let mediaConn;
let localStream;
let detectionInterval;
let transcriptionActive = false;
let signDetectionActive = false;
let ttsEnabled = true;

const localVideo = document.getElementById('localVideo');
const remoteVideo = document.getElementById('remoteVideo');
const joinBtn = document.getElementById('joinBtn');
const roomIdInput = document.getElementById('roomIdInput');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const localCaption = document.getElementById('localCaption');
const remoteCaption = document.getElementById('remoteCaption');
const toggleSignBtn = document.getElementById('toggleSignBtn');
const toggleTransBtn = document.getElementById('toggleTransBtn');
const toggleTTSBtn = document.getElementById('toggleTTSBtn');
const leaveBtn = document.getElementById('leaveBtn');

// Initialize Local Video
async function initLocalStream() {
    try {
        localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        localVideo.srcObject = localStream;
    } catch (err) {
        console.error("Error accessing media devices:", err);
        alert("Microphone and Camera access required for this feature.");
    }
}

// PeerJS Setup
joinBtn.onclick = () => {
    const roomId = roomIdInput.value.trim();
    if (!roomId) {
        alert("Please enter a Room ID or Name.");
        return;
    }

    // We use the roomId as the Peer ID for one of the users
    // For simplicity, we'll try to connect to 'roomId-host' or 'roomId-guest'
    // A better way is a signaling server, but for now we'll just try to join
    const suffix = prompt("Are you joining as 'host' or 'guest'?", "host");
    const myId = `${roomId}-${suffix}`;
    const peerId = `${roomId}-${suffix === 'host' ? 'guest' : 'host'}`;

    peer = new Peer(myId);

    peer.on('open', (id) => {
        statusText.innerText = `Connected as ${suffix}`;
        statusDot.className = 'status-dot online';
        
        // Try to connect to peer
        if (suffix === 'guest') {
            connectToPeer(peerId);
        }
    });

    peer.on('connection', (conn) => {
        setupDataConnection(conn);
    });

    peer.on('call', (call) => {
        call.answer(localStream);
        setupMediaConnection(call);
    });

    peer.on('error', (err) => {
        console.error("PeerJS Error:", err);
        statusText.innerText = "Error: " + err.type;
        statusDot.className = 'status-dot';
    });
};

function connectToPeer(peerId) {
    const conn = peer.connect(peerId);
    setupDataConnection(conn);

    const call = peer.call(peerId, localStream);
    setupMediaConnection(call);
}

function setupDataConnection(conn) {
    dataConn = conn;
    dataConn.on('open', () => {
        console.log("Data connection established");
    });
    dataConn.on('data', (data) => {
        handleIncomingData(data);
    });
}

function setupMediaConnection(call) {
    mediaConn = call;
    mediaConn.on('stream', (stream) => {
        remoteVideo.srcObject = stream;
    });
}

function handleIncomingData(data) {
    if (data.type === 'gesture') {
        showCaption(remoteCaption, data.text, 'gesture');
        if (ttsEnabled) speak(data.text);
    } else if (data.type === 'transcription') {
        showCaption(remoteCaption, data.text, 'speech');
    }
}

function showCaption(element, text, type) {
    element.innerText = text;
    element.style.display = 'block';
    element.style.borderColor = type === 'gesture' ? '#2ed573' : '#ff4757';
    
    // Auto hide after 3 seconds
    clearTimeout(element.hideTimeout);
    element.hideTimeout = setTimeout(() => {
        element.style.display = 'none';
    }, 3000);
}

// Sign Detection Loop
toggleSignBtn.onclick = () => {
    signDetectionActive = !signDetectionActive;
    toggleSignBtn.classList.toggle('active', signDetectionActive);
    
    if (signDetectionActive) {
        startDetectionLoop();
    } else {
        stopDetectionLoop();
    }
};

async function startDetectionLoop() {
    detectionInterval = setInterval(async () => {
        if (!signDetectionActive) return;

        const canvas = document.createElement('canvas');
        canvas.width = localVideo.videoWidth;
        canvas.height = localVideo.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(localVideo, 0, 0);
        
        canvas.toBlob(async (blob) => {
            const formData = new FormData();
            formData.append('file', blob, 'frame.jpg');

            try {
                const response = await fetch('/upload/en', { method: 'POST', body: formData });
                const data = await response.json();
                
                if (data.gesture && data.gesture !== "No gesture detected") {
                    showCaption(localCaption, data.gesture, 'gesture');
                    if (dataConn && dataConn.open) {
                        dataConn.send({ type: 'gesture', text: data.gesture });
                    }
                }
            } catch (err) {
                console.error("Detection error:", err);
            }
        }, 'image/jpeg', 0.8);
    }, 1000); // Check every second
}

function stopDetectionLoop() {
    clearInterval(detectionInterval);
    localCaption.style.display = 'none';
}

// Transcription handling
let mediaRecorder;
let audioChunks = [];

toggleTransBtn.onclick = async () => {
    transcriptionActive = !transcriptionActive;
    toggleTransBtn.classList.toggle('active', transcriptionActive);

    if (transcriptionActive) {
        startTranscription();
    } else {
        stopTranscription();
    }
};

async function startTranscription() {
    mediaRecorder = new MediaRecorder(localStream);
    mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
    mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        audioChunks = [];
        
        const formData = new FormData();
        formData.append('file', audioBlob, 'record.wav');

        try {
            const response = await fetch('/api/transcribe', { method: 'POST', body: formData });
            const data = await response.json();
            if (data.transcription) {
                showCaption(localCaption, data.transcription, 'speech');
                if (dataConn && dataConn.open) {
                    dataConn.send({ type: 'transcription', text: data.transcription });
                }
            }
            // Restart if still active
            if (transcriptionActive) mediaRecorder.start();
        } catch (err) {
            console.error("Transcription error:", err);
        }
    };
    
    // We'll record in 5-second chunks for communication
    mediaRecorder.start();
    let transcriptionLoop = setInterval(() => {
        if (mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
        }
    }, 5000);
}

function stopTranscription() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    clearInterval(transcriptionLoop);
    localCaption.style.display = 'none';
}

// TTS
toggleTTSBtn.onclick = () => {
    ttsEnabled = !ttsEnabled;
    toggleTTSBtn.classList.toggle('active', ttsEnabled);
};

function speak(text) {
    if (!ttsEnabled) return;
    const utterance = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(utterance);
}

// Disconnect
leaveBtn.onclick = () => {
    if (peer) peer.destroy();
    location.reload();
};

initLocalStream();
