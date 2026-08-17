/**
 * app.js
 * ------------------------
 * STEP 10 - User-friendly frontend logic.
 *
 * Talks to the existing, unmodified Step 8 Flask API. This file does
 * not implement any prediction logic itself -- it only collects the
 * 8 form answers, sends them to POST /api/predict, and renders
 * whatever the real API returns.
 */

(function () {
  "use strict";

  // ---------------------------------------------------------------
  // Single place to change the API location. Nothing else in this
  // file hardcodes the host/port or the endpoint path.
  // ---------------------------------------------------------------
  var CONFIG = {
    API_BASE_URL: "http://127.0.0.1:5000",
    PREDICT_ENDPOINT: "/api/predict",
  };

  var TOTAL_STEPS = 8;

  // Field name -> input type, used to know how to read/validate it.
  var STEP_FIELDS = {
    1: { name: "age", type: "number" },
    2: { name: "gender", type: "radio" },
    3: { name: "hypertension", type: "radio" },
    4: { name: "heart_disease", type: "radio" },
    5: { name: "ever_married", type: "radio" },
    6: { name: "work_type", type: "radio" },
    7: { name: "Residence_type", type: "radio" },
    8: { name: "smoking_status", type: "radio" },
  };

  var currentStep = 1;

  // ---------------------------------------------------------------
  // Element references
  // ---------------------------------------------------------------
  var form = document.getElementById("risk-form");
  var formCard = document.getElementById("form-card");
  var resultCard = document.getElementById("result-card");

  var progressFill = document.getElementById("progress-fill");
  var progressLabel = document.getElementById("progress-label");
  var progressBar = document.querySelector(".progress");

  var btnBack = document.getElementById("btn-back");
  var btnNext = document.getElementById("btn-next");
  var btnSubmit = document.getElementById("btn-submit");
  var loadingText = document.getElementById("loading-text");
  var apiErrorEl = document.getElementById("form-api-error");

  var btnRestart = document.getElementById("btn-restart");

  var GAUGE_ARC_LENGTH = 314.159; // matches stroke-dasharray in CSS

  // ---------------------------------------------------------------
  // Step navigation
  // ---------------------------------------------------------------

  function showStep(stepNumber) {
    var steps = form.querySelectorAll(".step");
    steps.forEach(function (stepEl) {
      var isActive = Number(stepEl.getAttribute("data-step")) === stepNumber;
      if (isActive) {
        stepEl.setAttribute("data-active", "true");
      } else {
        stepEl.removeAttribute("data-active");
      }
    });

    var pct = (stepNumber / TOTAL_STEPS) * 100;
    progressFill.style.width = pct + "%";
    progressLabel.textContent = "Step " + stepNumber + " of " + TOTAL_STEPS;
    progressBar.setAttribute("aria-valuenow", String(stepNumber));

    btnBack.hidden = stepNumber === 1;
    btnNext.hidden = stepNumber === TOTAL_STEPS;
    btnSubmit.hidden = stepNumber !== TOTAL_STEPS;

    clearApiError();

    // Move focus to the new question for keyboard/screen-reader users.
    var activeStep = form.querySelector('.step[data-active="true"]');
    if (activeStep) {
      var focusTarget = activeStep.querySelector("input");
      if (focusTarget) {
        focusTarget.focus({ preventScroll: true });
      }
    }
  }

  function validateStep(stepNumber) {
    var field = STEP_FIELDS[stepNumber];
    var errorEl = document.getElementById("error-" + field.name);

    if (field.type === "number") {
      var input = form.querySelector('[name="' + field.name + '"]');
      var raw = input.value.trim();

      if (raw === "") {
        setError(errorEl, "Please enter your age.");
        return false;
      }
      var value = Number(raw);
      if (Number.isNaN(value) || !Number.isFinite(value)) {
        setError(errorEl, "Age must be a number.");
        return false;
      }
      if (value < 0 || value > 120) {
        setError(errorEl, "Age must be between 0 and 120.");
        return false;
      }
      setError(errorEl, "");
      return true;
    }

    // radio group
    var checked = form.querySelector('[name="' + field.name + '"]:checked');
    if (!checked) {
      setError(errorEl, "Please choose an option to continue.");
      return false;
    }
    setError(errorEl, "");
    return true;
  }

  function setError(errorEl, message) {
    if (!errorEl) return;
    errorEl.textContent = message;
  }

  function clearApiError() {
    apiErrorEl.textContent = "";
  }

  btnNext.addEventListener("click", function () {
    if (!validateStep(currentStep)) {
      return;
    }
    if (currentStep < TOTAL_STEPS) {
      currentStep += 1;
      showStep(currentStep);
    }
  });

  btnBack.addEventListener("click", function () {
    if (currentStep > 1) {
      currentStep -= 1;
      showStep(currentStep);
    }
  });

  // ---------------------------------------------------------------
  // Submission
  // ---------------------------------------------------------------

  function collectPayload() {
    var payload = {};
    Object.keys(STEP_FIELDS).forEach(function (key) {
      var field = STEP_FIELDS[key];
      if (field.type === "number") {
        var input = form.querySelector('[name="' + field.name + '"]');
        payload[field.name] = Number(input.value.trim());
      } else {
        var checked = form.querySelector('[name="' + field.name + '"]:checked');
        var raw = checked.value;
        // hypertension / heart_disease are sent as integers 0/1;
        // everything else is a string category the API already expects.
        if (field.name === "hypertension" || field.name === "heart_disease") {
          payload[field.name] = Number(raw);
        } else {
          payload[field.name] = raw;
        }
      }
    });
    return payload;
  }

  function setLoading(isLoading) {
    btnSubmit.disabled = isLoading;
    btnNext.disabled = isLoading;
    btnBack.disabled = isLoading;
    loadingText.hidden = !isLoading;
    if (isLoading) {
      btnSubmit.classList.add("btn--loading");
    } else {
      btnSubmit.classList.remove("btn--loading");
    }
  }

  form.addEventListener("submit", function (event) {
    event.preventDefault();

    if (!validateStep(TOTAL_STEPS)) {
      return;
    }

    clearApiError();
    setLoading(true);

    var payload = collectPayload();

    fetch(CONFIG.API_BASE_URL + CONFIG.PREDICT_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then(function (response) {
        return response
          .json()
          .catch(function () {
            // Response wasn't valid JSON at all.
            throw new Error("BAD_RESPONSE");
          })
          .then(function (body) {
            if (!response.ok || !body || body.success !== true) {
              var message =
                (body && body.error && body.error.message) ||
                "We couldn't process your information. Please check your answers and try again.";
              throw new Error(message);
            }
            return body;
          });
      })
      .then(function (body) {
        setLoading(false);
        renderResult(body.data);
      })
      .catch(function (err) {
        setLoading(false);
        if (err && err.message === "BAD_RESPONSE") {
          showApiError(
            "We received an unexpected response from the server. Please try again in a moment."
          );
        } else if (err instanceof TypeError) {
          // fetch() throws a TypeError for network-level failures
          // (server down, wrong port, CORS blocked, offline, etc.)
          showApiError(
            "We couldn't reach the prediction server. Please make sure it's running and try again."
          );
        } else {
          showApiError(err.message || "Something went wrong. Please try again.");
        }
      });
  });

  function showApiError(message) {
    apiErrorEl.textContent = message;
  }

  // ---------------------------------------------------------------
  // Result rendering
  // ---------------------------------------------------------------

  function renderResult(data) {
    formCard.hidden = true;
    resultCard.hidden = false;
    resultCard.scrollIntoView({ behavior: "smooth", block: "start" });

    var percentage = typeof data.risk_percentage === "number" ? data.risk_percentage : 0;
    var category = data.risk_category || "—";
    var predictedClass = data.predicted_class || "—";

    var percentageEl = document.getElementById("result-percentage");
    var categoryEl = document.getElementById("result-category");
    var classEl = document.getElementById("result-class");
    var arc = document.getElementById("gauge-arc");

    percentageEl.textContent = percentage.toFixed(2) + "%";
    categoryEl.textContent = category;
    categoryEl.setAttribute("data-level", category);
    classEl.textContent = predictedClass + " prediction group";

    var arcColor = "var(--risk-moderate)";
    if (category === "LOW") arcColor = "var(--risk-low)";
    else if (category === "HIGH") arcColor = "var(--risk-high)";
    arc.style.stroke = arcColor;

    // Animate the gauge fill from 0 to the returned percentage.
    var clamped = Math.max(0, Math.min(100, percentage));
    var offset = GAUGE_ARC_LENGTH - (GAUGE_ARC_LENGTH * clamped) / 100;

    arc.style.strokeDashoffset = String(GAUGE_ARC_LENGTH);
    // Force reflow so the transition re-triggers from full offset.
    // eslint-disable-next-line no-unused-expressions
    arc.getBoundingClientRect();
    requestAnimationFrame(function () {
      arc.style.strokeDashoffset = String(offset);
    });
  }

  // ---------------------------------------------------------------
  // Reset
  // ---------------------------------------------------------------

  btnRestart.addEventListener("click", function () {
    form.reset();
    form.querySelectorAll(".field__error").forEach(function (el) {
      el.textContent = "";
    });
    clearApiError();

    resultCard.hidden = true;
    formCard.hidden = false;
    formCard.scrollIntoView({ behavior: "smooth", block: "start" });

    currentStep = 1;
    showStep(currentStep);
  });

  // ---------------------------------------------------------------
  // Init
  // ---------------------------------------------------------------
  showStep(currentStep);
})();
