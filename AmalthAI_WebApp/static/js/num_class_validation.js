document.addEventListener("DOMContentLoaded", () => {
    // Elements
    const input     = document.getElementById("numClassesInput");
    const error     = document.getElementById("num_classes_error");
    const submitBtn = document.getElementById("submitBtn");
    const zipInput  = document.getElementById("datasetZipInput");
    const loader    = document.getElementById("zipUploadLoader");
    const statusEl  = document.getElementById("zipUploadStatus");

    // State flags
    const hasNumClasses = !!input;
    let numClassesValid = !hasNumClasses; 
    let uploadComplete   = false;
    let uploadInProgress = false;

    // Validate number of classes
    function validateNumClasses() {
        if (!input) return false;
        const min = parseInt(input.min);
        const max = parseInt(input.max);
        let val = parseInt(input.value);

        // Clear previous state
        input.classList.remove("is-invalid");
        error.classList.add("d-none");

        let valid = true;

        if (isNaN(val)) valid = false;
        if (valid && val < min) {
            val = min;
            input.value = String(min);
        }
        if (valid && val > max) {
            val = max;
            input.value = String(max);
        }
        if (!isNaN(val) && (val < min || val > max)) valid = false;

        if (!valid) {
            input.classList.add("is-invalid");
            error.classList.remove("d-none");
        }
        numClassesValid = valid;
        setSubmitState();
        return valid;
    }

    // Enable/disable submit based on combined conditions
    function setSubmitState() {
        // Submit enabled only when classes valid, upload finished successfully, and not currently uploading
        submitBtn.disabled = !(numClassesValid && uploadComplete && !uploadInProgress);
    }

    // Handle async zip upload
    async function handleZipSelection() {
        const file = zipInput.files && zipInput.files[0];
        if (!file) {
            uploadComplete = false;
            statusEl.textContent = "";
            setSubmitState();
            return;
        }
        if (!file.name.toLowerCase().endsWith('.zip')) {
            uploadComplete = false;
            statusEl.textContent = "Only .zip files are allowed";
            statusEl.className = "small text-danger mt-2 text-center";
            setSubmitState();
            return;
        }

        // Start upload
        uploadInProgress = true;
        uploadComplete = false;
        setSubmitState();
        statusEl.textContent = "";
        statusEl.className = "small mt-2";
        if (loader) loader.style.display = 'block';

        const formData = new FormData();
        formData.append('dataset_zip', file, file.name);

        try {
            const resp = await fetch('/upload_dataset_zip', {
                method: 'POST',
                body: formData
            });
            const data = await resp.json().catch(() => ({}));
            if (resp.ok && data.status === 'ok') {
                uploadComplete = true;
                statusEl.textContent = `Upload successful: ${data.filename}`;
                statusEl.className = "small text-success mt-2 text-center";

                document.getElementById("datasetZipFilename").value = data.filename;
            } else {
                uploadComplete = false;
                statusEl.textContent = data.message || 'Upload failed';
                statusEl.className = "small text-danger mt-2 text-center";
            }
        } catch (e) {
            uploadComplete = false;
            statusEl.textContent = 'Upload error';
            statusEl.className = "small text-danger mt-2 text-center";
        } finally {
            uploadInProgress = false;
            if (loader) loader.style.display = 'none';
            setSubmitState();
        }
    }

    // Event wiring
    if (input) {
        input.addEventListener("input", validateNumClasses);
        input.addEventListener("change", validateNumClasses);
    }
    if (zipInput) {
        zipInput.addEventListener('change', handleZipSelection);
    }

    // Initial validation
    validateNumClasses();
    setSubmitState();
});