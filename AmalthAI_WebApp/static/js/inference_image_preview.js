const inferenceImagePreview = document.getElementById('inferenceImagePreview');

if (inferenceImagePreview) {
    const previewImage = inferenceImagePreview.querySelector('.inference-preview-image');
    const previewTitle = inferenceImagePreview.querySelector('.modal-title');

    inferenceImagePreview.addEventListener('show.bs.modal', (event) => {
        const trigger = event.relatedTarget;
        previewImage.src = trigger.dataset.previewSrc;
        previewImage.alt = trigger.dataset.previewAlt;
        previewTitle.textContent = trigger.dataset.previewAlt;
    });

    inferenceImagePreview.addEventListener('hidden.bs.modal', () => {
        previewImage.removeAttribute('src');
    });
}
