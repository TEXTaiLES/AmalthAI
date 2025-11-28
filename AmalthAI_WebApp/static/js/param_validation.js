document.addEventListener("DOMContentLoaded", () => {

    const trainBtn = document.getElementById("trainBtn");

    const modelSelect = document.getElementById("modelSelect");
    const collectionSelect = document.getElementById("collectionSelect");

    const mode = document.getElementById("mode").value;

    // Collect all advanced inputs (note: template uses classes 'adv-lower' / 'adv-upper')
    const lowerInputs = [...document.querySelectorAll(".adv-lower")];
    const upperInputs = [...document.querySelectorAll(".adv-upper")];

    // augmentation controls (template uses aug-enable / aug-prob / aug-maxval)
    const augEnables = [...document.querySelectorAll(".aug-enable")];
    const augProbs   = [...document.querySelectorAll(".aug-prob")];
    const augMaxvals = [...document.querySelectorAll(".aug-maxval")];

    // Ensure clicking augmentations doesn't inadvertently block validation; re-run validate on toggle/value change
    augEnables.forEach(cb => cb.addEventListener("change", validate));
    augProbs.forEach(s => s.addEventListener("change", validate));
    augMaxvals.forEach(s => s.addEventListener("change", validate));


    //----------------------------------------------------------------------
    // ADVANCED PARAM HELPERS (merged from advanced_params.js)
    //----------------------------------------------------------------------

    function clampContinuous(input) {
        if (!input || input.type !== "number") return;

        const min = parseFloat(input.min);
        const max = parseFloat(input.max);
        let val = parseFloat(input.value);

        if (isNaN(min) || isNaN(max)) return;

        if (!isNaN(val)) {
            if (val < min) input.value = String(min);
            if (val > max) input.value = String(max);
        }
    }

    function validateParam(param) {
        const lower = document.querySelector(`.adv-lower[data-param="${param}"]`);
        const upper = document.querySelector(`.adv-upper[data-param="${param}"]`);
        const error = document.getElementById(`${param}_error`);

        if (!lower || !upper || !error) return;

        // Always clamp continuous ranges first
        clampContinuous(lower);
        clampContinuous(upper);

        const lowVal = parseFloat(lower.value);
        const upVal = parseFloat(upper.value);

        // Clear previous invalid state
        lower.classList.remove("is-invalid");
        upper.classList.remove("is-invalid");
        error.classList.add("d-none");
        error.classList.remove("d-block");

        const lowIsNumber = Number.isFinite(lowVal);
        const upIsNumber = Number.isFinite(upVal);

        // Only show error when both bounds are valid numbers and lower > upper
        if (lowIsNumber && upIsNumber && lowVal > upVal) {
            lower.classList.add("is-invalid");
            upper.classList.add("is-invalid");
            error.classList.remove("d-none");
            error.classList.add("d-block");
        }
    }

    //----------------------------------------------------------------------
    // VALIDATION FUNCTION (uses merged per-param checks)
    //----------------------------------------------------------------------

    function validate() {

        let valid = true;

        // Model + Collection must be chosen
        if (!modelSelect || modelSelect.value === "Choose model" || modelSelect.value === "" ||
            modelSelect.selectedIndex === 0) {
            valid = false;
        }

        if (!collectionSelect || collectionSelect.value === "Choose collection" || collectionSelect.value === "" ||
            collectionSelect.selectedIndex === 0) {
            valid = false;
        }

        // Check each param pair: lower ≤ upper (and clamp continuous)
        lowerInputs.forEach(lowerEl => {
            const param = lowerEl.dataset.param;
            const upperEl = document.getElementById(`${param}_right`);
            const errorEl = document.getElementById(`${param}_error`);

            // perform per-param clamping/validation
            validateParam(param);

            const lowVal = parseFloat(lowerEl.value);
            const upVal = parseFloat(upperEl ? upperEl.value : NaN);

            if (isNaN(lowVal) || isNaN(upVal) || lowVal > upVal) {
                valid = false;
                if (errorEl) errorEl.classList.remove("d-none");
            } else {
                if (errorEl) errorEl.classList.add("d-none");
            }
        });

        trainBtn.disabled = !valid;
        return valid;
    }


    //----------------------------------------------------------------------
    // EVENT LISTENERS
    //----------------------------------------------------------------------

    if (modelSelect) modelSelect.addEventListener("change", validate);
    if (collectionSelect) collectionSelect.addEventListener("change", validate);

    lowerInputs.forEach(el => {
        // run both per-param validation and overall validate
        el.addEventListener("input", () => { validateParam(el.dataset.param); validate(); });
        el.addEventListener("change", () => { validateParam(el.dataset.param); validate(); });
    });
    upperInputs.forEach(el => {
        el.addEventListener("input", () => { validateParam(el.dataset.param); validate(); });
        el.addEventListener("change", () => { validateParam(el.dataset.param); validate(); });
    });

    // initial validation for every parameter (ensures errors are hidden unless invalid)
    const params = new Set();
    lowerInputs.forEach(el => params.add(el.dataset.param));
    params.forEach(p => validateParam(p));
    validate();


    //----------------------------------------------------------------------
    // SUBMIT — only allowed when valid
    //----------------------------------------------------------------------

    trainBtn.addEventListener("click", () => {

        if (!validate()) return;

        // Build POST data
        const formData = new FormData();

        formData.append("mode", mode);
        formData.append("model", modelSelect.value);
        formData.append("collection", collectionSelect.value);

        // Add all advanced fields
        lowerInputs.forEach(lowerEl => {
            const param = lowerEl.dataset.param;
            const upperEl = document.getElementById(`${param}_right`);

            formData.append(`${param}_left`, lowerEl.value);
            formData.append(`${param}_right`, upperEl.value);
        });

        // Add augmentations: for each augmentation present append enabled flag,
        // and if prob/maxval inputs exist append them as well.
        augEnables.forEach(cb => {
            const aug = cb.dataset.aug;
            const enabled = cb.checked;
            formData.append(`${aug}_enabled`, String(enabled));

            const probEl = document.querySelector(`.aug-prob[data-aug="${aug}"]`);
            const maxEl  = document.querySelector(`.aug-maxval[data-aug="${aug}"]`);
            if (probEl) formData.append(`${aug}_prob`, probEl.value);
            if (maxEl)  formData.append(`${aug}_maxval`, maxEl.value);
        });

        const spinner = document.getElementById("trainSpinner");

        // Disable UI and show spinner to prevent double-submit
        spinner && spinner.classList.remove("d-none");
        trainBtn.disabled = true;
        // disable other controls while submitting
        const controls = [
            ...document.querySelectorAll(".adv-lower, .adv-upper, .aug-enable, .aug-prob, .aug-maxval, #modelSelect, #collectionSelect")
        ];
        controls.forEach(el => el.disabled = true);

        // Send POST request
        fetch("/train_model_submit", {
            method: "POST",
            body: formData
        })
        .then(res => {
            if (res.redirected) {
                // follow redirect (will unload page)
                window.location.href = res.url;
                return;
            }

            // Not redirected — hide spinner and re-enable controls so user can fix issues
            spinner && spinner.classList.add("d-none");
            trainBtn.disabled = false;
            controls.forEach(el => el.disabled = false);

            // Optionally handle non-redirect responses here (show message, etc.)
            return res.text();
        })
        .catch(err => {
            console.error("Training POST error:", err);
            spinner && spinner.classList.add("d-none");
            trainBtn.disabled = false;
            const controls = [
                ...document.querySelectorAll(".adv-lower, .adv-upper, .aug-enable, .aug-prob, .aug-maxval, #modelSelect, #collectionSelect")
            ];
            controls.forEach(el => el.disabled = false);
        });
    });

});
