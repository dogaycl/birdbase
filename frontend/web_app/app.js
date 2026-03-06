const fileInput = document.getElementById('file-input');
const dropZone = document.getElementById('drop-zone');
const previewSection = document.getElementById('preview-section');
const imagePreview = document.getElementById('image-preview');
const loader = document.getElementById('loader');
const resultsSection = document.getElementById('results-section');

// API ENDPOINT
const API_URL = 'http://localhost:8000/predict/';

// Drag and drop events
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
});

dropZone.addEventListener('drop', (e) => {
    let dt = e.dataTransfer;
    let files = dt.files;
    handleFiles(files);
});

fileInput.addEventListener('change', function() {
    handleFiles(this.files);
});

function handleFiles(files) {
    if (files.length > 0) {
        const file = files[0];
        
        if (!file.type.startsWith('image/')) return;

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZone.classList.add('hidden');
            previewSection.classList.remove('hidden');
            
            // Start Upload and Analysis
            analyzeImage(file);
        }
        reader.readAsDataURL(file);
    }
}

async function analyzeImage(file) {
    loader.classList.remove('hidden');
    resultsSection.classList.add('hidden');

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('API Error');

        const data = await response.json();
        
        if(data.detected) {
            displayResults(data);
        } else {
            alert(data.message || "No bird detected.");
            resetApp();
        }

    } catch (error) {
        console.error("Error:", error);
        alert("Failed to connect to the backend API. Ensure it is running on port 8000.");
        resetApp();
    } finally {
        loader.classList.add('hidden');
    }
}

function displayResults(data) {
    resultsSection.classList.remove('hidden');
    
    document.getElementById('species-name').innerText = data.species;
    document.getElementById('confidence-score').innerText = Math.round(data.confidence * 100) + '%';
    
    const info = data.info;
    document.getElementById('scientific-name').innerText = info.scientific_name || "N/A";
    document.getElementById('lifespan').innerText = info.lifespan || "N/A";
    document.getElementById('habitat').innerText = info.habitat || "N/A";
    document.getElementById('description-text').innerText = info.description || "N/A";
}

function resetApp() {
    fileInput.value = "";
    dropZone.classList.remove('hidden');
    previewSection.classList.add('hidden');
    resultsSection.classList.add('hidden');
    loader.classList.add('hidden');
}
