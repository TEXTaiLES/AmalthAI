/**
 * Loader spinning wheel JS and upload-button wiring.
 */
document.addEventListener('DOMContentLoaded', function () {
    // show loader and disable submit on form submit
    const forms = document.getElementsByTagName('form');
    for (let form of forms) {
        form.addEventListener('submit', function () {
            const submitButton = form.querySelector('button[type="submit"]');
            const loader = form.querySelector('.loader');
            if (submitButton) {
                submitButton.disabled = true;
            }
            if (loader) {
                loader.style.display = 'inline-block';
            }
        });
    }

    // wire Upload Images button to hidden file input
    const uploadBtn = document.getElementById('upload-btn');
    const fileInput = document.getElementById('file-input');
    const fileCount = document.getElementById('file-count');

    if (uploadBtn && fileInput) {
        uploadBtn.addEventListener('click', () => fileInput.click());
    }

    // update file count display and enable/disable submit button accordingly
    if (fileInput && fileCount) {
        const formOfInput = fileInput.closest('form');
        const submitButton = formOfInput ? formOfInput.querySelector('button[type="submit"]') : null;

        const updateCount = () => {
            const n = fileInput.files ? fileInput.files.length : 0;
            fileCount.textContent = n === 0 ? 'No files selected' : (n === 1 ? '1 file selected' : `${n} files selected`);
            if (submitButton) {
                submitButton.disabled = (n === 0);
            }
        };
        fileInput.addEventListener('change', updateCount);
        // initialize display and submit button state
        updateCount();
    }
});
