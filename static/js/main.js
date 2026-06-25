// ============================================
// FILE UPLOAD HANDLING
// ============================================

const fileInput = document.getElementById('file-input');
const uploadArea = document.getElementById('upload-area');
const imagePreviewContainer = document.getElementById('image-preview-container');
const imagePreview = document.getElementById('image-preview');
const btnRemove = document.getElementById('btn-remove');
const btnGenerate = document.getElementById('btn-generate');
const resultPlaceholder = document.getElementById('result-placeholder');
const loadingContainer = document.getElementById('loading-container');
const resultImageContainer = document.getElementById('result-image-container');
const resultImage = document.getElementById('result-image');
const resultActions = document.getElementById('result-actions');
const btnDownload = document.getElementById('btn-download');
const btnNewUpload = document.getElementById('btn-new-upload');

let uploadedFile = null;

// Click to upload
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// Drag and drop handlers
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// File input change
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// Handle file selection
function handleFile(file) {
    // Validate file type
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg'];
    if (!allowedTypes.includes(file.type)) {
        alert('Please upload a PNG or JPG image.');
        return;
    }

    // Validate file size (16MB)
    if (file.size > 16 * 1024 * 1024) {
        alert('File size must be less than 16MB.');
        return;
    }

    uploadedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadArea.style.display = 'none';
        imagePreviewContainer.style.display = 'block';
        btnGenerate.disabled = false;
        updateFlowStep('upload', true);
    };
    reader.readAsDataURL(file);
}

// Remove image
btnRemove.addEventListener('click', () => {
    uploadedFile = null;
    fileInput.value = '';
    imagePreviewContainer.style.display = 'none';
    uploadArea.style.display = 'flex';
    btnGenerate.disabled = true;
    resetResult();
    updateFlowStep('upload', false);
});

// ============================================
// GENERATE RESULT
// ============================================

btnGenerate.addEventListener('click', async () => {
    if (!uploadedFile) return;

    // Update UI - Show loading
    btnGenerate.disabled = true;
    btnGenerate.textContent = 'Processing...';
    resultPlaceholder.style.display = 'none';
    resultImageContainer.style.display = 'none';
    resultActions.style.display = 'none';
    loadingContainer.style.display = 'flex';
    updateFlowStep('process', true);

    // Upload file
    const formData = new FormData();
    formData.append('file', uploadedFile);

    try {
        const uploadResponse = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const uploadData = await uploadResponse.json();

        if (!uploadResponse.ok) {
            throw new Error(uploadData.error || 'Upload failed');
        }

        // Process image
        const processResponse = await fetch('/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ filename: uploadData.filename })
        });

        const processData = await processResponse.json();

        if (!processResponse.ok) {
            throw new Error(processData.error || 'Processing failed');
        }

        // Show the actual segmented result returned by the model
        showResult(processData.result_filename);
        showProbabilities(processData);
        updateFlowStep('download', true);

    } catch (error) {
        alert('Error: ' + error.message);
        btnGenerate.disabled = false;
        btnGenerate.textContent = 'Generate Result';
        loadingContainer.style.display = 'none';
        resultPlaceholder.style.display = 'flex';
        updateFlowStep('process', false);
    }
});

// Show result
function showResult(resultFilename) {
    // Hide loading, show result
    loadingContainer.style.display = 'none';

    // Load the actual segmented image from the server
    resultImage.src = `/result/${resultFilename}`;
    resultPlaceholder.style.display = 'none';
    resultImageContainer.style.display = 'block';
    resultActions.style.display = 'flex';
    btnGenerate.textContent = 'Generate Result';
    btnGenerate.disabled = false;

    // Scroll to results section
    const resultsSection = document.getElementById('results');
    if (resultsSection) {
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 300);
    }
}

// Reset result
function resetResult() {
    resultPlaceholder.style.display = 'flex';
    loadingContainer.style.display = 'none';
    resultImageContainer.style.display = 'none';
    resultActions.style.display = 'none';
    resultImage.src = '';
    updateFlowStep('process', false);
    updateFlowStep('download', false);
}

// ============================================
// DOWNLOAD RESULT
// ============================================

btnDownload.addEventListener('click', () => {
    if (!resultImage.src) return;

    // Create download link
    const link = document.createElement('a');
    link.href = resultImage.src;
    link.download = 'segmented_result.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
});

// ============================================
// UPLOAD NEW IMAGE
// ============================================

btnNewUpload.addEventListener('click', () => {
    btnRemove.click(); // Reset everything
});

// ============================================
// FLOW STEP INDICATOR
// ============================================

function updateFlowStep(step, active) {
    const stepElement = document.getElementById(`step-${step}`);
    if (active) {
        stepElement.classList.add('active');
    } else {
        stepElement.classList.remove('active');
    }
}

// ============================================
// SMOOTH SCROLL FOR NAVIGATION LINKS
// ============================================

document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const href = this.getAttribute('href');
        if (href === '#') return; // Don't scroll for # links
        
        const target = document.querySelector(href);
        if (target) {
            // Update active nav link
            document.querySelectorAll('.nav-link').forEach(link => {
                link.classList.remove('active');
            });
            this.classList.add('active');
            
            // Scroll to target
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});
function showProbabilities(data) {
    const container = document.getElementById('probability-container');
    
    document.getElementById('liver-bar').style.width = data.liver_prob + '%';
    document.getElementById('tumor-bar').style.width = data.tumor_prob + '%';
    document.getElementById('liver-prob-text').textContent = data.liver_prob + '%';
    document.getElementById('tumor-prob-text').textContent = data.tumor_prob + '%';
    document.getElementById('liver-dsc').textContent = data.liver_dsc ?? 'N/A';
    document.getElementById('tumor-dsc').textContent = data.tumor_dsc ?? 'N/A';
    
    const badge = document.getElementById('overall-confidence');
    badge.textContent = data.tumor_prob > 50 ? '⚠️ Tumor Detected' : '✅ No Tumor';
    badge.style.background = data.tumor_prob > 50 ? '#ef4444' : '#22c55e';
    
    container.style.display = 'block';
}
