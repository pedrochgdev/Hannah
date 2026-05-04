/**
 * Hannah Chat - Frontend Logic
 * Handles chat interface, history, backend communication, STT (Whisper), and TTS (Kokoro).
 */

// ---- State ----
let currentConversation = [];
let conversationId = null;
let isWaiting = false;
let isVoiceEnabled = true;
let isContinuousVoiceMode = false;
// STT State
let mediaRecorder;
let audioChunks = [];
let isRecording = false;

let sessionId = localStorage.getItem('hannah_session_id');
if (!sessionId) {
    sessionId = (''+[1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g,c=>(c^crypto.getRandomValues(new Uint8Array(1))[0]&15>>c/4).toString(16));
    localStorage.setItem('hannah_session_id', sessionId);
}

// --- Variables para Detección de Silencio ---
let audioContext;
let analyser;
let microphone;
let silenceTimer = null;
const SILENCE_THRESHOLD = 15; // Nivel de volumen mínimo (0-255). Súbelo si hay ruido de fondo.
const SILENCE_DURATION = 2500; // Milisegundos de silencio necesarios para enviar el mensaje (1.5s).

// ---- DOM Elements ----
const welcomeScreen = document.getElementById('welcomeScreen');
const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
const btnSend = document.getElementById('btnSend');
const typingIndicator = document.getElementById('typingIndicator');
const historyPanel = document.getElementById('historyPanel');
const historyList = document.getElementById('historyList');
const historyEmpty = document.getElementById('historyEmpty');
const modelInfoModal = document.getElementById('modelInfoModal');
const modelInfoBody = document.getElementById('modelInfoBody');

// Voice DOM
const btnToggleVoice = document.getElementById('btnToggleVoice');
const voiceIcon = document.getElementById('voiceIcon');
const btnMic = document.getElementById('btnMic');

// ---- Init ----
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    loadHistory();
});

function setupEventListeners() {
    btnSend.addEventListener('click', () => {
        isContinuousVoiceMode = false; // Apagar modo voz si escribes
        sendMessage();
    });
    
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            isContinuousVoiceMode = false; // Apagar modo voz si escribes
            sendMessage();
        }
    });
    messageInput.addEventListener('input', () => {
        btnSend.disabled = messageInput.value.trim() === '';
        autoResizeTextarea();
    });

    document.querySelectorAll('.quick-action').forEach(btn => {
        btn.addEventListener('click', () => {
            const msg = btn.getAttribute('data-message');
            messageInput.value = msg;
            btnSend.disabled = false;
            sendMessage();
        });
    });

    document.getElementById('btnHistory').addEventListener('click', openHistory);
    document.getElementById('btnCloseHistory').addEventListener('click', closeHistory);
    document.getElementById('btnClearHistory').addEventListener('click', clearHistory);
    document.getElementById('btnNewChat').addEventListener('click', startNewChat);

    document.getElementById('btnModelInfo').addEventListener('click', openModelInfo);
    document.getElementById('btnCloseModal').addEventListener('click', () => {
        modelInfoModal.classList.remove('open');
    });
    modelInfoModal.addEventListener('click', (e) => {
        if (e.target === modelInfoModal) modelInfoModal.classList.remove('open');
    });

    // Voice & Audio Event Listeners
    btnToggleVoice.addEventListener('click', toggleVoiceState);
    btnMic.addEventListener('click', toggleRecording);
}

// ---- Audio Recording (STT with Faster-Whisper) ----
async function toggleRecording() {
    if (isRecording) {
        isContinuousVoiceMode = false; // Apagamos el modo continuo si tú lo detienes a mano
        stopRecording();
    } else {
        isContinuousVoiceMode = true; // Encendemos el modo continuo al tocar el micro
        await startRecordingWithVAD();
    }
}

function stopRecording() {
    if (!isRecording) return;
    
    mediaRecorder.stop();
    btnMic.classList.remove('recording');
    btnMic.querySelector('.material-icons-round').textContent = 'mic';
    isRecording = false;
    
    // Limpiamos los analizadores de audio
    if (audioContext) {
        audioContext.close();
    }
    if (silenceTimer) {
        clearTimeout(silenceTimer);
        silenceTimer = null;
    }
}

async function startRecordingWithVAD() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        // Configuramos el Analizador de Audio
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        microphone = audioContext.createMediaStreamSource(stream);
        microphone.connect(analyser);
        
        analyser.fftSize = 512;
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        // Configuramos el MediaRecorder (igual que antes)
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        mediaRecorder.ondataavailable = e => {
            if (e.data.size > 0) audioChunks.push(e.data);
        };
        mediaRecorder.onstop = sendAudioForTranscription; // Se ejecuta al llamar a stopRecording()

        mediaRecorder.start();
        isRecording = true;
        btnMic.classList.add('recording');
        btnMic.querySelector('.material-icons-round').textContent = 'stop';
        messageInput.placeholder = "Habla, te estoy escuchando...";

	let hasSpoken = false;

        // Bucle de detección de silencio
        function detectSilence() {
            if (!isRecording) return;
            
            analyser.getByteFrequencyData(dataArray);
            let sum = 0;
            for(let i = 0; i < bufferLength; i++) {
                sum += dataArray[i];
            }
            let averageVolume = sum / bufferLength;

            if (averageVolume > SILENCE_THRESHOLD) {
                // Hay ruido/voz: cancelamos el temporizador de silencio
		hasSpoken = true;
                if (silenceTimer) {
                    clearTimeout(silenceTimer);
                    silenceTimer = null;
                }
            } else {
                // Hay silencio: iniciamos el temporizador si no existe
                if (hasSpoken && !silenceTimer) {
                    silenceTimer = setTimeout(() => {
                        console.log("Silencio detectado. Deteniendo grabación...");
                        stopRecording();
                    }, SILENCE_DURATION);
                }
            }
            
            // Seguimos analizando en el siguiente frame
            requestAnimationFrame(detectSilence);
        }
        
        detectSilence(); // Iniciamos el bucle

    } catch (err) {
        console.error("No se pudo acceder al micrófono:", err);
        alert("Por favor, permite el acceso al micrófono.");
    }
}

async function sendAudioForTranscription() {
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    const formData = new FormData();
    formData.append('audio', audioBlob, 'voice.webm');

    messageInput.placeholder = "Transcribiendo...";
    
    try {
        // Llama a tu endpoint backend donde corre faster-whisper
        const response = await fetch('/api/v1/transcribe', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.text) {
            // Añade el texto al input
            messageInput.value = data.text;
            btnSend.disabled = false;
            autoResizeTextarea();
	    sendMessage()
        }
    } catch (err) {
        console.error("Error en la transcripción:", err);
    } finally {
        messageInput.placeholder = "Type your message or use the mic...";
    }
}

// ---- Text-to-Speech (TTS with Kokoro) ----
function toggleVoiceState() {
    isVoiceEnabled = !isVoiceEnabled;
    voiceIcon.textContent = isVoiceEnabled ? 'volume_up' : 'volume_off';
    btnToggleVoice.classList.toggle('active', isVoiceEnabled);
}

async function playHannahVoice(text) {
    if (!isVoiceEnabled) return;

    try {
        const response = await fetch('/api/v1/tts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });

        if (response.ok) {
            const blob = await response.blob();
            const audioUrl = URL.createObjectURL(blob);
            const audio = new Audio(audioUrl);
	    audio.onended = () => {
    // Cuando Hannah termina de hablar, volvemos a encender el micrófono automáticamente
    		if (isContinuousVoiceMode && !isRecording) {
        		startRecordingWithVAD();
  	   	}	
	    };
            audio.play();
        }
    } catch (error) {
        console.error("Error reproduciendo la voz de Hannah:", error);
    }
}

// ---- Chat Core ----
async function sendMessage() {
    const text = messageInput.value.trim();
    if (!text || isWaiting) return;

    if (currentConversation.length === 0) {
        welcomeScreen.classList.add('hidden');
        chatMessages.classList.add('active');
        conversationId = Date.now().toString();
    }

    const userMsg = { role: 'user', content: text, time: new Date().toISOString() };
    currentConversation.push(userMsg);
    appendMessage('user', text);

    messageInput.value = '';
    btnSend.disabled = true;
    autoResizeTextarea();

    isWaiting = true;
    typingIndicator.classList.add('active');
    scrollToBottom();

    try {
        const response = await fetch(`/api/v1/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                prompt: text,
            }),
        });

        const data = await response.json();

        if (data.error) {
            appendMessage('bot', 'Sorry, there was an error: ' + data.error);
        } else {
            const botMsg = { role: 'assistant', content: data.response, time: new Date().toISOString() };
            currentConversation.push(botMsg);
            appendMessage('bot', data.response);
            
            // REPRODUCIR AUDIO DE HANNAH
            playHannahVoice(data.response);
        }
    } catch (err) {
        appendMessage('bot', "Couldn't connect to the server. Make sure it's running.");
    }

    typingIndicator.classList.remove('active');
    isWaiting = false;

    saveCurrentConversation();
    scrollToBottom();
}

function appendMessage(role, text) {
    const now = new Date();
    const timeStr = now.toLocaleTimeString('en', { hour: '2-digit', minute: '2-digit' });

    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role === 'user' ? 'user' : 'bot'}`;

    const avatarIcon = role === 'user' ? 'person' : 'favorite';

    const bubbleContent = role === 'user' 
    ? escapeHtml(text)           // usuario: texto plano
    : marked.parse(text);        // bot: renderiza markdown

    msgDiv.innerHTML = `
        <div class="message-avatar">
            <span class="material-icons-round">${avatarIcon}</span>
        </div>
        <div class="message-content">
            <div class="bubble ${role === 'bot' ? 'bubble-markdown' : ''}">${bubbleContent}</div>
            <span class="message-time">${timeStr}</span>
        </div>
    `;

    chatMessages.appendChild(msgDiv);
    scrollToBottom();
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    });
}

function autoResizeTextarea() {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 100) + 'px';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ---- History (localStorage) ----
const STORAGE_KEY = 'hannah_chat_history';

function getStoredHistory() {
    try {
        return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    } catch {
        return [];
    }
}

function saveCurrentConversation() {
    if (currentConversation.length === 0) return;

    const history = getStoredHistory();
    const firstUserMsg = currentConversation.find(m => m.role === 'user');
    const title = firstUserMsg ? firstUserMsg.content.substring(0, 50) : 'Conversation';

    const existing = history.findIndex(h => h.id === conversationId);
    const entry = {
        id: conversationId,
        title: title,
        preview: currentConversation[currentConversation.length - 1].content.substring(0, 80),
        messages: currentConversation,
        updatedAt: new Date().toISOString(),
    };

    if (existing >= 0) {
        history[existing] = entry;
    } else {
        history.unshift(entry);
    }

    if (history.length > 50) history.pop();

    localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
}

function loadHistory() {
    const history = getStoredHistory();
    renderHistory(history);
}

function renderHistory(history) {
    historyList.innerHTML = '';

    if (history.length === 0) {
        historyEmpty.classList.add('visible');
        return;
    }

    historyEmpty.classList.remove('visible');

    const groups = {};
    const today = new Date().toDateString();
    const yesterday = new Date(Date.now() - 86400000).toDateString();

    history.forEach(entry => {
        const date = new Date(entry.updatedAt);
        const dateStr = date.toDateString();
        let label;

        if (dateStr === today) {
            label = 'Today';
        } else if (dateStr === yesterday) {
            label = 'Yesterday';
        } else {
            label = date.toLocaleDateString('en', {
                weekday: 'long',
                day: 'numeric',
                month: 'long',
                year: 'numeric'
            });
        }

        if (!groups[label]) groups[label] = [];
        groups[label].push(entry);
    });

    Object.entries(groups).forEach(([label, entries]) => {
        const groupDiv = document.createElement('div');
        groupDiv.className = 'history-date-group';
        groupDiv.innerHTML = `<div class="history-date-label">${label}</div>`;

        entries.forEach(entry => {
            const time = new Date(entry.updatedAt).toLocaleTimeString('en', {
                hour: '2-digit', minute: '2-digit'
            });

            const item = document.createElement('div');
            item.className = 'history-item';
            item.innerHTML = `
                <div class="history-item-icon">
                    <span class="material-icons-round">chat_bubble</span>
                </div>
                <div class="history-item-text">
                    <div class="history-item-title">${escapeHtml(entry.title)}</div>
                    <div class="history-item-preview">${escapeHtml(entry.preview)}</div>
                </div>
                <span class="history-item-time">${time}</span>
            `;

            item.addEventListener('click', () => loadConversation(entry));
            groupDiv.appendChild(item);
        });

        historyList.appendChild(groupDiv);
    });
}

function loadConversation(entry) {
    currentConversation = [...entry.messages];
    conversationId = entry.id;

    welcomeScreen.classList.add('hidden');
    chatMessages.classList.add('active');
    chatMessages.innerHTML = '';

    currentConversation.forEach(msg => {
        appendMessage(msg.role === 'user' ? 'user' : 'bot', msg.content);
    });

    closeHistory();
    scrollToBottom();
}

function openHistory() {
    loadHistory();
    historyPanel.classList.add('open');
}

function closeHistory() {
    historyPanel.classList.remove('open');
}

function clearHistory() {
    if (confirm('Clear all chat history?')) {
        localStorage.removeItem(STORAGE_KEY);
        loadHistory();
    }
}

function startNewChat() {
    sessionId = (''+[1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g,c=>(c^crypto.getRandomValues(new Uint8Array(1))[0]&15>>c/4).toString(16));
    localStorage.setItem('hannah_session_id', sessionId);
    currentConversation = [];
    conversationId = null;
    chatMessages.innerHTML = '';
    chatMessages.classList.remove('active');
    welcomeScreen.classList.remove('hidden');
    closeHistory();
}

// ---- Model Info ----
async function openModelInfo() {
    modelInfoModal.classList.add('open');
    modelInfoBody.innerHTML = '<p style="text-align:center;color:#999;">Loading...</p>';

    try {
        const res = await fetch('/api/model-info');
        const data = await res.json();

        if (data.loaded) {
            modelInfoBody.innerHTML = `
                <div class="model-info-row">
                    <span class="model-info-label">Status</span>
                    <span class="model-status-badge loaded">Loaded</span>
                </div>
                <div class="model-info-row">
                    <span class="model-info-label">File</span>
                    <span class="model-info-value">${data.model_file}</span>
                </div>
                <div class="model-info-row">
                    <span class="model-info-label">Parameters</span>
                    <span class="model-info-value">${data.parameters}</span>
                </div>
                <div class="model-info-row">
                    <span class="model-info-label">Device</span>
                    <span class="model-info-value">${data.device.toUpperCase()}</span>
                </div>
            `;
        } else {
            modelInfoBody.innerHTML = `
                <div class="model-info-row">
                    <span class="model-info-label">Status</span>
                    <span class="model-status-badge error">No model</span>
                </div>
                <p style="margin-top:12px;font-size:13px;color:#999;">
                    Place a .pt file in the <strong>models/</strong> folder and restart the server.
                </p>
            `;
        }
    } catch {
        modelInfoBody.innerHTML = '<p style="color:#c62828;">Could not connect to the server.</p>';
    }
}
