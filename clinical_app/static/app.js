const $ = (selector) => document.querySelector(selector);
let authenticated = false;
let passwordRequired = true;
let maxImagesPerSession = 300;
let selectedFiles = [];
let activeSubmissionId = null;
let toastTimer = null;
let submitting = false;

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", "\"": "&quot;"
  })[character]);
}

function apiHeaders(json = false) {
  return json ? {"Content-Type": "application/json"} : {};
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try { detail = (await response.json()).detail || detail; } catch (_) {}
    const error = new Error(detail);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

function toast(message, {error = false, sticky = false} = {}) {
  const element = $("#toast");
  if (toastTimer) window.clearTimeout(toastTimer);
  element.textContent = message;
  element.className = error ? "toast toast-error" : "toast";
  element.hidden = false;
  if (!sticky) toastTimer = window.setTimeout(() => { element.hidden = true; }, 5000);
}

function currentContext() {
  return document.querySelector('input[name="useContext"]:checked')?.value || "pathologist_self_review";
}

function isPatientContext() {
  return currentContext() === "patient_self_check";
}

function percent(value) {
  return value === null || value === undefined ? "Not available" : `${(Number(value) * 100).toFixed(1)}%`;
}

function durationText(value) {
  if (value === null || value === undefined) return "Not recorded";
  const seconds = Number(value) / 1000;
  return seconds < 60 ? `${seconds.toFixed(1)} sec` : `${(seconds / 60).toFixed(1)} min`;
}

function formatDate(value) {
  if (!value) return "Not recorded";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}

function labelText(label) {
  if (Number(label) === 1) return "MIP-associated morphology present";
  if (Number(label) === 0) return "MIP-associated morphology absent";
  return "Not recorded";
}

function ratingText(image) {
  if (!image.accepted) return "General · Specialist review requested";
  return Number(image.predicted_label) === Number(image.ground_truth_label)
    ? "Effective · Result agreed with the initial assessment"
    : "No help · Result disagreed with the initial assessment";
}

function setAuthenticatedState(value) {
  authenticated = Boolean(value);
  document.querySelectorAll(".protected-section input, .protected-section select, .protected-section button, #caseForm input, #caseForm button")
    .forEach((element) => { element.disabled = !authenticated; });
  document.querySelectorAll("[data-export]").forEach((element) => {
    element.classList.toggle("is-disabled", !authenticated);
    element.setAttribute("aria-disabled", authenticated ? "false" : "true");
  });
  $("#accessPassword").disabled = authenticated || !passwordRequired;
  $("#loginAccess").hidden = authenticated || !passwordRequired;
  $("#logoutAccess").hidden = !authenticated || !passwordRequired;
  $("#authStatus").textContent = authenticated
    ? "Authorized workspace is open on this browser."
    : "The information page is public. An authorized session is required to upload images or view records.";
  if (!authenticated) {
    $("#queue").innerHTML = '<p class="subtle">Sign in to access protected image review records.</p>';
    $("#metrics").innerHTML = '<p class="subtle">Sign in to view prospective monitoring.</p>';
    $("#physicianMetrics").innerHTML = '<p class="subtle">Sign in to view reviewer summaries.</p>';
    $("#emptyResult").innerHTML = '<span class="empty-symbol" aria-hidden="true"></span><strong>Workspace locked</strong><p>Sign in with the authorized password to analyze pathology images.</p>';
    $("#resultSummary").hidden = true;
    $("#emptyResult").hidden = false;
    $("#reportsPanel").hidden = true;
  } else {
    $("#emptyResult").innerHTML = '<span class="empty-symbol" aria-hidden="true"></span><strong>No images analyzed yet</strong><p>Each submitted image will receive its own interpretation. A group of images is never combined into a patient diagnosis.</p>';
  }
  updateContextView();
}

async function loadPublicSystem() {
  const status = await api("/api/v1/public/system");
  passwordRequired = status.password_required;
  const version = `v${status.app_version}`;
  $("#appVersion").textContent = version;
  $("#footerVersion").textContent = `${version} · image-level decision support · expert review required`;
  const badge = $("#modeBadge");
  badge.className = status.service_available ? "status-chip status-ready" : "status-chip status-unavailable";
  badge.textContent = status.service_available ? "Service available" : "Service unavailable";
}

async function loadAuthStatus() {
  const status = await api("/api/v1/auth/status");
  passwordRequired = status.password_required;
  setAuthenticatedState(status.authenticated);
  return status.authenticated;
}

async function loginAccess() {
  const password = $("#accessPassword").value;
  if (!password) return toast("Enter the authorized password.", {error: true});
  $("#loginAccess").disabled = true;
  try {
    await api("/api/v1/auth/login", {
      method: "POST",
      headers: apiHeaders(true),
      body: JSON.stringify({password})
    });
    $("#accessPassword").value = "";
    toast("Authorized workspace opened.");
    await refreshAll();
  } catch (error) {
    toast(error.message, {error: true, sticky: true});
  } finally {
    $("#loginAccess").disabled = false;
  }
}

async function logoutAccess() {
  try { await api("/api/v1/auth/logout", {method: "POST"}); } finally {
    clearSelectedFiles();
    activeSubmissionId = null;
    setAuthenticatedState(false);
    await loadPublicSystem();
  }
}

async function loadSystem() {
  const status = await api("/api/v1/system");
  maxImagesPerSession = status.max_images_per_case || 300;
  $("#uploadEmpty").textContent = `PNG, JPEG, TIFF, or BMP · up to ${maxImagesPerSession} images per session`;
  $("#privacyPill").textContent = status.image_persistence_enabled
    ? "Pseudonymous records retained on host"
    : "Images processed without retention";
  const badge = $("#modeBadge");
  if (!status.ready) {
    badge.className = "status-chip status-unavailable";
    badge.textContent = "Service unavailable";
  } else if (status.mode === "demo") {
    badge.className = "status-chip status-caution";
    badge.textContent = "Demonstration mode";
  } else {
    badge.className = "status-chip status-ready";
    badge.textContent = "Image review ready";
  }
}

function updateContextView() {
  const patient = isPatientContext();
  const studyMetadata = $("#studyMetadata");
  $("#identityLabel").textContent = patient ? "Pseudonymous user ID" : "Pathologist ID";
  $("#identityId").placeholder = patient ? "e.g. USER-2026-001" : "e.g. PATH-07";
  $("#identityNotice").textContent = patient
    ? "Use a non-identifying code. Results are educational image information, not a pathology diagnosis."
    : "Use a study-specific clinician code. Your initial assessment remains hidden until submitted.";
  $("#submitButton").textContent = patient ? "View image information" : "Analyze images for self-review";
  studyMetadata.hidden = patient;
  $("#evaluationCohort").required = !patient;
  $("#evaluationCohort").disabled = patient || !authenticated;
  $("#evidenceRole").disabled = patient || !authenticated;
  $("#reviewSection").hidden = patient;
  $("#metricsSection").hidden = patient;
  if (authenticated && !patient) {
    loadQueue();
    loadMetrics();
  }
}

function rememberIdentity() {
  const value = $("#identityId").value.trim();
  if (value) sessionStorage.setItem(`mipIdentity:${currentContext()}`, value);
}

function restoreIdentity() {
  $("#identityId").value = sessionStorage.getItem(`mipIdentity:${currentContext()}`) || "";
}

function clearSelectedFiles() {
  selectedFiles.forEach((item) => URL.revokeObjectURL(item.url));
  selectedFiles = [];
  $("#imageInput").value = "";
  $("#folderInput").value = "";
  renderSelectedFiles();
}

function addSelectedFiles(fileList) {
  const supported = new Set(["image/png", "image/jpeg", "image/tiff", "image/bmp"]);
  const existing = new Set(selectedFiles.map((item) => `${item.file.name}:${item.file.size}:${item.file.lastModified}`));
  let ignored = 0;
  for (const file of Array.from(fileList || [])) {
    const extensionOk = /\.(png|jpe?g|tiff?|bmp)$/i.test(file.name);
    if (!(supported.has(file.type) || extensionOk)) { ignored += 1; continue; }
    const signature = `${file.name}:${file.size}:${file.lastModified}`;
    if (existing.has(signature)) { ignored += 1; continue; }
    if (selectedFiles.length >= maxImagesPerSession) { ignored += 1; continue; }
    selectedFiles.push({file, url: URL.createObjectURL(file)});
    existing.add(signature);
  }
  renderSelectedFiles();
  if (ignored) toast(`${ignored} unsupported, duplicate, or excess file(s) were not added.`, {error: true});
}

function renderSelectedFiles() {
  const grid = $("#uploadPreviewGrid");
  const empty = $("#uploadEmpty");
  const totalBytes = selectedFiles.reduce((sum, item) => sum + item.file.size, 0);
  const sizeText = totalBytes >= 1024 * 1024
    ? `${(totalBytes / 1024 / 1024).toFixed(1)} MB`
    : `${Math.max(0, Math.round(totalBytes / 1024))} KB`;
  $("#imageCounter").textContent = selectedFiles.length
    ? `${selectedFiles.length} image${selectedFiles.length === 1 ? "" : "s"} · ${sizeText}`
    : "0 images";
  empty.hidden = selectedFiles.length > 0;
  grid.replaceChildren(...selectedFiles.map((item, index) => {
    const card = document.createElement("article");
    card.className = "upload-preview-card";
    const preview = document.createElement("img");
    preview.src = item.url;
    preview.alt = `Preview of ${item.file.name}`;
    const copy = document.createElement("div");
    copy.innerHTML = `<strong>Image ${index + 1}</strong><span title="${escapeHtml(item.file.name)}">${escapeHtml(item.file.name)}</span><small>${(item.file.size / 1024 / 1024).toFixed(2)} MB</small>`;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "preview-remove";
    remove.setAttribute("aria-label", `Remove ${item.file.name}`);
    remove.textContent = "×";
    remove.addEventListener("click", () => {
      URL.revokeObjectURL(selectedFiles[index].url);
      selectedFiles.splice(index, 1);
      renderSelectedFiles();
    });
    card.append(preview, copy, remove);
    return card;
  }));
}

function decisionContent(image, modelName = "HistoNexa-MIP") {
  const clinicianRecorded = image.ground_truth_label !== null && image.ground_truth_label !== undefined;
  const clinicianPositive = clinicianRecorded && Number(image.ground_truth_label) === 1;
  const systemPositive = Number(image.predicted_label) === 1;
  const positiveScore = percent(image.p_mip);
  const reviewScore = image.selection_score === null || image.selection_score === undefined
    ? "Not available"
    : percent(1 - Number(image.selection_score));
  const scoreSentence = `${modelName} produced an image-level MIP morphology score of ${positiveScore} and a review score of ${reviewScore}. These values describe model evidence and uncertainty in this image; they are not patient-level disease probabilities.`;
  const limitations = "This interpretation is limited to the submitted image. It does not assess the full slide or specimen, quantify a final micropapillary component, establish grade or TNM stage, predict prognosis, or determine treatment. Final reporting must integrate complete morphology, sampling, and clinical information.";

  if (!image.accepted && clinicianRecorded && clinicianPositive) {
    return {
      scenario: "pathologist-positive-model-review",
      tone: "report-inconclusive",
      result: "REVIEW",
      short: "Model deferred · pathologist positive",
      title: "Positive microscopic assessment with model deferral",
      alignment: "Pathologist positive · model requested review",
      priority: "Expert confirmation required · retain the pathologist finding",
      finding: `${scoreSentence} The pathologist recorded MIP-associated morphology as present, while the model withheld a binary conclusion. Deferral is not a negative result and must not overturn directly observed morphology.`,
      reviewPlan: "Confirm image focus, staining, magnification, and the suspected papillary or floret-like clusters without a central fibrovascular core. Retain the manual observation for targeted whole-slide review, quantify the component across representative blocks, exclude detached-cell or processing artifact, and obtain second-pathologist adjudication when appropriate.",
      limitations
    };
  }

  if (!image.accepted && clinicianRecorded && !clinicianPositive) {
    return {
      scenario: "pathologist-negative-model-review",
      tone: "report-inconclusive",
      result: "REVIEW",
      short: "Model deferred · pathologist negative",
      title: "Negative microscopic assessment with model deferral",
      alignment: "Pathologist negative · model requested review",
      priority: "Technical and morphologic review advised",
      finding: `${scoreSentence} The pathologist recorded no MIP-associated morphology in the submitted field, while the model withheld a binary conclusion. The manual image-level assessment remains primary, but absence cannot be inferred beyond this field.`,
      reviewPlan: "Check focus, staining, magnification, tissue representation, and whether the selected field is adequate. Continue routine systematic review of the complete slide set; reacquire this field when technically inadequate and seek expert review if suspicious architecture or uncertainty persists.",
      limitations
    };
  }

  if (clinicianRecorded && clinicianPositive && systemPositive) {
    return {
      scenario: "pathologist-positive-model-positive",
      tone: "report-positive",
      result: "POSITIVE",
      short: "Concordant MIP-associated morphology",
      title: "Concordant positive image assessment",
      alignment: "Pathologist positive · model positive",
      priority: "Concordant finding · complete specimen review remains required",
      finding: `${scoreSentence} Both assessments support MIP-associated morphology in this field. The relevant pattern includes compact papillary or floret-like tumor-cell clusters without an evident central fibrovascular core. Agreement strengthens the image-level observation but does not establish extent, predominant pattern, grade, or stage.`,
      reviewPlan: "Map and quantify the suspected component across the complete H&E slide set and relevant blocks. Exclude detached tumor clusters, spread through air spaces, tangential sectioning, and processing artifact before recording the final proportion in the institution's standard reporting workflow.",
      limitations
    };
  }

  if (clinicianRecorded && !clinicianPositive && !systemPositive) {
    return {
      scenario: "pathologist-negative-model-negative",
      tone: "report-negative",
      result: "NEGATIVE",
      short: "Concordant absence in this image",
      title: "Concordant negative image assessment",
      alignment: "Pathologist negative · model negative",
      priority: "Concordant finding · continue standard sampling",
      finding: `${scoreSentence} Neither assessment identifies sufficiently supported free-floating papillary tumor tufts lacking fibrovascular cores in this field. This is an image-level absence finding, not evidence that the pattern is absent elsewhere.`,
      reviewPlan: "Maintain routine whole-slide review and confirm adequate tumor sampling. Suspicious architecture, poor preservation, or detached clusters elsewhere should prompt additional levels, representative blocks, or expert review. This image result must not be used to reduce standard sampling.",
      limitations
    };
  }

  if (clinicianRecorded && !clinicianPositive && systemPositive) {
    return {
      scenario: "pathologist-negative-model-positive",
      tone: "report-positive",
      result: "POSITIVE",
      short: "Model-positive discordance",
      title: "Model-positive discordant image assessment",
      alignment: "Pathologist negative · model positive",
      priority: "Discordant finding · targeted re-review required",
      finding: `${scoreSentence} The model flags possible MIP-associated morphology, while the initial pathologist assessment is negative. Compact clusters, tangentially sectioned glands, spread through air spaces, detached tumor cells, or processing artifact may account for the discrepancy. The model result alone is insufficient to establish a micropapillary component.`,
      reviewPlan: "Re-examine the flagged field at suitable magnification, compare adjacent levels and representative blocks, and document the source of discordance. Obtain second-pathologist adjudication if disagreement persists before changing the final pathology report.",
      limitations
    };
  }

  if (clinicianRecorded && clinicianPositive && !systemPositive) {
    return {
      scenario: "pathologist-positive-model-negative",
      tone: "report-negative",
      result: "NEGATIVE",
      short: "Model-negative discordance",
      title: "Model-negative discordant image assessment",
      alignment: "Pathologist positive · model negative",
      priority: "Potential model miss · pathologist finding takes precedence",
      finding: `${scoreSentence} The pathologist identifies MIP-associated morphology, while the model does not detect sufficient support in the same field. This is a potential model miss; a negative model result must not negate directly observed morphology.`,
      reviewPlan: "Confirm the suspected clusters at appropriate magnification, assess reproducibility across adjacent levels, document the discordance, and exclude mimics or artifact. Retain the manual finding for expert adjudication and obtain second-pathologist review when appropriate.",
      limitations
    };
  }

  if (!image.accepted) {
    return {
      scenario: "model-review-no-pathologist",
      tone: "report-inconclusive",
      result: "REVIEW",
      short: "Specialist assessment required",
      title: "Model deferred image interpretation",
      alignment: "No model conclusion issued",
      priority: "No automated conclusion · specialist review required",
      finding: `${scoreSentence} The model withheld a positive or negative conclusion because the image evidence was insufficient for reliable automated interpretation. Deferral is not a negative result.`,
      reviewPlan: "Obtain specialist microscopic review. Check staining, focus, magnification, tissue representation, and region selection; repeat image capture when technically appropriate and assess the complete slide set through the standard pathology workflow.",
      limitations
    };
  }

  return systemPositive ? {
    scenario: "model-positive-no-pathologist",
    tone: "report-positive",
    result: "POSITIVE",
    short: "MIP-associated morphology flagged",
    title: "Positive model image assessment",
    alignment: "Model positive · pathologist assessment not recorded",
    priority: "Pathologist confirmation required",
    finding: `${scoreSentence} The model identifies support for compact papillary or floret-like tumor-cell clusters without an evident central fibrovascular core. This is an image-level flag and does not establish a final diagnosis.`,
    reviewPlan: "Request qualified review of the submitted field, complete slide set, and representative tumor blocks. Confirm the architecture, exclude mimics and artifact, and quantify any micropapillary component only within the standard specimen-level workflow.",
    limitations
  } : {
    scenario: "model-negative-no-pathologist",
    tone: "report-negative",
    result: "NEGATIVE",
    short: "No MIP pattern detected in this image",
    title: "Negative model image assessment",
    alignment: "Model negative · pathologist assessment not recorded",
    priority: "Routine expert review remains required",
    finding: `${scoreSentence} The model does not identify sufficiently supported MIP-associated morphology in this submitted field. This finding does not exclude the pattern elsewhere in the slide or specimen.`,
    reviewPlan: "Continue qualified whole-slide and specimen review and confirm adequate tumor sampling. Do not use this image result to reduce sampling, bypass specialist interpretation, or alter the standard diagnostic pathway.",
    limitations
  };
}

function imagePreviewUrl(data, image) {
  if (data.submission_id === activeSubmissionId) {
    return selectedFiles[Number(image.image_index) - 1]?.url || image.image_url || "";
  }
  return image.image_url || "";
}

function createImageReport(data, image) {
  const article = document.createElement("article");
  const previewUrl = imagePreviewUrl(data, image);
  const contextText = data.use_context === "patient_self_check" ? "Patient image information" : "Pathologist self-review";
  const modelName = data.model_display_name || "HistoNexa-MIP";
  const decision = decisionContent(image, modelName);
  article.className = `image-report ${decision.tone}`;
  article.dataset.reportScenario = decision.scenario;
  article.setAttribute("aria-label", `${decision.result} report for ${image.image_name || `image ${image.image_index}`}`);
  const contextGuidance = data.use_context === "patient_self_check"
    ? "Patient information context: discuss this report with a qualified pathologist or treating clinician. Do not use it to diagnose yourself or change treatment or follow-up."
    : "Pathologist self-review context: reconcile this secondary result with the complete slide set, specimen findings, and the institution's standard diagnostic workflow.";
  const rejectionScore = image.selection_score === null || image.selection_score === undefined
    ? null
    : 1 - Number(image.selection_score);
  article.innerHTML = `
    <header class="image-report-header">
      <div>
        <p class="report-kicker">${escapeHtml(modelName.toUpperCase())} · PATHOLOGY IMAGE INTERPRETATION</p>
        <h3>${escapeHtml(decision.title)}</h3>
      </div>
      <span class="report-status"><small>IMAGE RESULT</small><strong>${escapeHtml(decision.result)}</strong><em>${escapeHtml(decision.short)}</em></span>
    </header>
    <div class="report-body">
      <div class="report-preview">
        ${previewUrl ? `<img src="${escapeHtml(previewUrl)}" alt="Submitted pathology image ${Number(image.image_index)}">` : '<div class="preview-unavailable">Preview unavailable</div>'}
        <div class="score-panel">
          <div class="score-line"><span>MIP morphology score</span><strong>${percent(image.p_mip)}</strong></div>
          <div class="score-line"><span>Review score</span><strong>${percent(rejectionScore)}</strong></div>
        </div>
        <p class="score-explanation">The morphology score reflects model support in this image. A higher review score indicates greater uncertainty and a stronger reason for specialist review. Neither is a patient-level disease probability.</p>
      </div>
      <div class="report-content">
        <dl class="report-identifiers">
          <div><dt>Report ID</dt><dd>${escapeHtml(`${data.case_ref}-${String(image.image_index).padStart(2, "0")}`)}</dd></div>
          <div><dt>Upload reference</dt><dd>${escapeHtml(data.case_ref)}</dd></div>
          <div><dt>Image</dt><dd>${escapeHtml(image.image_name || `Image ${image.image_index}`)}</dd></div>
          <div><dt>Generated</dt><dd>${escapeHtml(formatDate(data.submitted_at))}</dd></div>
          <div><dt>Review context</dt><dd>${escapeHtml(contextText)}</dd></div>
          <div><dt>Analysis system</dt><dd>${escapeHtml(modelName)}</dd></div>
          <div><dt>Model release</dt><dd>${escapeHtml(data.model_version || "Recorded in audit log")}</dd></div>
          <div><dt>Decision profile</dt><dd>${escapeHtml(data.policy_version || "Recorded in audit log")}</dd></div>
          <div><dt>Clinical purpose</dt><dd>Secondary assessment of possible MIP morphology</dd></div>
          ${image.ground_truth_label === null || image.ground_truth_label === undefined ? "" : `<div><dt>Pathologist assessment</dt><dd>${escapeHtml(labelText(image.ground_truth_label))}</dd></div>`}
        </dl>
        <div class="report-alignment"><span>Interpretive alignment</span><strong>${escapeHtml(decision.alignment)}</strong><em>${escapeHtml(decision.priority)}</em></div>
        <section class="report-section"><span class="report-section-index">01</span><div><h4>Image finding + morphologic context</h4><p>${escapeHtml(decision.finding)}</p></div></section>
        <section class="report-section report-section-review"><span class="report-section-index">02</span><div><h4>Recommended expert review pathway</h4><p>${escapeHtml(decision.reviewPlan)}</p></div></section>
        <section class="report-section report-section-limit"><span class="report-section-index">03</span><div><h4>Scope + limitations</h4><p>${escapeHtml(decision.limitations)} ${escapeHtml(contextGuidance)}</p></div></section>
      </div>
    </div>
    <footer class="report-footer"><span>Research-use image interpretation</span><span>Final diagnosis remains the responsibility of qualified clinicians</span></footer>`;
  return article;
}

function renderSubmission(data) {
  const images = data.images || [];
  $("#emptyResult").hidden = true;
  $("#resultSummary").hidden = false;
  $("#summaryReference").textContent = data.case_ref;
  $("#summaryImages").textContent = `${images.length}`;
  $("#summaryLatency").textContent = durationText(data.inference_ms);
  $("#clinicalNotice").textContent = data.clinical_notice || "";
  const decision = $("#decision");

  if (!data.ai_result_visible) {
    decision.className = "decision-card decision-secured";
    decision.innerHTML = "<strong>Analysis complete</strong><span>Results are secured until the initial image assessments are submitted.</span>";
    $("#summaryAvailable").textContent = "After initial review";
    $("#summaryDeferred").textContent = "Available after review";
    $("#reportsPanel").hidden = true;
    return;
  }

  const deferred = images.filter((image) => !image.accepted).length;
  const positive = images.filter((image) => image.accepted && Number(image.predicted_label) === 1).length;
  const negative = images.filter((image) => image.accepted && Number(image.predicted_label) === 0).length;
  decision.className = "decision-card decision-ready";
  decision.innerHTML = `<strong>${images.length} image report${images.length === 1 ? "" : "s"} ready</strong><span>${positive} detected · ${negative} not detected · ${deferred} inconclusive</span>`;
  $("#summaryAvailable").textContent = `${images.length}`;
  $("#summaryDeferred").textContent = `${deferred}`;
  $("#imageReports").replaceChildren(...images.map((image) => createImageReport(data, image)));
  $("#reportsPanel").hidden = false;
}

async function submitImages(event) {
  event.preventDefault();
  if (!authenticated) return toast("Sign in before uploading images.", {error: true});
  const identity = $("#identityId").value.trim();
  const caseRef = $("#caseRef").value.trim();
  const evaluationCohort = $("#evaluationCohort").value.trim();
  if (!identity) return toast(isPatientContext() ? "Enter a pseudonymous user ID." : "Enter the pathologist ID.", {error: true});
  if (!caseRef) return toast("Enter an upload reference.", {error: true});
  if (!isPatientContext() && !evaluationCohort) return toast("Enter the evaluation cohort ID.", {error: true});
  if (!selectedFiles.length) return toast("Add at least one pathology image.", {error: true});
  rememberIdentity();
  const form = new FormData();
  form.append("case_ref", caseRef);
  form.append("submitted_by", identity);
  form.append("use_context", currentContext());
  form.append("interaction_mode", isPatientContext() ? "self_service" : "direct_on_device");
  form.append("evaluation_cohort", isPatientContext() ? "PATIENT_INFORMATION" : evaluationCohort);
  form.append("evidence_role", isPatientContext() ? "patient_information" : $("#evidenceRole").value);
  if (!isPatientContext()) form.append("diagnosing_pathologist_id", identity);
  selectedFiles.forEach((item) => form.append("images", item.file, item.file.name));
  const button = $("#submitButton");
  const status = $("#submissionStatus");
  submitting = true;
  button.disabled = true;
  button.dataset.originalText = button.textContent;
  button.textContent = `Processing ${selectedFiles.length} image${selectedFiles.length === 1 ? "" : "s"}…`;
  $("#caseForm").setAttribute("aria-busy", "true");
  status.hidden = false;
  status.className = "submission-status submission-running";
  status.innerHTML = `<span class="status-spinner" aria-hidden="true"></span><span>Analyzing ${selectedFiles.length} image${selectedFiles.length === 1 ? "" : "s"}. Large sessions may take several minutes; keep this page open.</span>`;
  try {
    const data = await api("/api/v1/cases", {method: "POST", body: form});
    activeSubmissionId = data.submission_id;
    status.className = "submission-status submission-success";
    status.textContent = data.ai_result_visible
      ? "Individual image reports are ready."
      : "Analysis is complete. Record the initial image assessments to reveal the reports.";
    renderSubmission(data);
    if (!isPatientContext()) {
      await loadQueue();
      await loadMetrics();
    }
    $("#resultSummary").scrollIntoView({behavior: "smooth", block: "nearest"});
  } catch (error) {
    status.className = "submission-status submission-error";
    status.textContent = error.message;
    toast(error.message, {error: true, sticky: true});
  } finally {
    submitting = false;
    button.disabled = false;
    button.textContent = button.dataset.originalText || "Analyze images";
    button.removeAttribute("data-original-text");
    $("#caseForm").removeAttribute("aria-busy");
  }
}

function imageReference(image) {
  const wrapper = document.createElement("div");
  wrapper.className = "review-image-reference";
  const preview = document.createElement("img");
  preview.src = image.image_url || "";
  preview.alt = image.image_url ? `Preview of ${image.image_name}` : "Preview unavailable";
  const text = document.createElement("div");
  text.innerHTML = `<strong>Image ${Number(image.image_index)}</strong><span title="${escapeHtml(image.image_name)}">${escapeHtml(image.image_name)}</span>`;
  wrapper.append(preview, text);
  return wrapper;
}

function workflowHeader(item, phase) {
  const header = document.createElement("header");
  header.className = "workflow-header";
  header.innerHTML = `<div><strong>${escapeHtml(item.case_ref)}</strong><span>${item.image_count} image${item.image_count === 1 ? "" : "s"} · ${escapeHtml(formatDate(item.submitted_at))}</span></div><span class="workflow-phase">${escapeHtml(phase)}</span>`;
  return header;
}

function renderDiagnosisCard(item) {
  const card = document.createElement("article");
  card.className = "workflow-card";
  card.dataset.submissionId = item.submission_id;
  card.append(workflowHeader(item, "Initial assessment"));
  if (!item.diagnosis_started_at) {
    const startPanel = document.createElement("div");
    startPanel.className = "start-panel";
    startPanel.innerHTML = "<div><strong>Analysis reports are hidden</strong><span>Start when you are ready to assess each image.</span></div>";
    const start = document.createElement("button");
    start.type = "button";
    start.className = "button button-primary";
    start.textContent = "Start image review";
    start.addEventListener("click", () => startDiagnosis(item, start));
    startPanel.append(start);
    card.append(startPanel);
    return card;
  }
  const list = document.createElement("div");
  list.className = "diagnosis-list";
  item.images.forEach((image) => {
    const row = document.createElement("section");
    row.className = "diagnosis-row";
    row.dataset.caseId = image.case_id;
    row.append(imageReference(image));
    const choices = document.createElement("fieldset");
    choices.className = "binary-review";
    choices.innerHTML = `<legend>Initial image assessment</legend><label><input type="radio" name="image-${escapeHtml(image.case_id)}" value="1"> MIP-associated morphology present</label><label><input type="radio" name="image-${escapeHtml(image.case_id)}" value="0"> MIP-associated morphology absent</label>`;
    const notes = document.createElement("textarea");
    notes.className = "image-diagnosis-notes";
    notes.maxLength = 1000;
    notes.rows = 2;
    notes.placeholder = "Optional image findings";
    row.append(choices, notes);
    list.append(row);
  });
  const bulkTools = document.createElement("div");
  bulkTools.className = "bulk-assessment-tools";
  bulkTools.innerHTML = "<div><strong>Batch label helper</strong><span>For a pre-labelled validation folder, prefill every image and verify the labels before submission.</span></div>";
  const setAllNegative = document.createElement("button");
  setAllNegative.type = "button";
  setAllNegative.className = "button button-secondary";
  setAllNegative.textContent = "Set all negative";
  setAllNegative.addEventListener("click", () => {
    list.querySelectorAll('input[type="radio"][value="0"]').forEach((input) => {
      input.checked = true;
    });
  });
  const setAllPositive = document.createElement("button");
  setAllPositive.type = "button";
  setAllPositive.className = "button button-secondary";
  setAllPositive.textContent = "Set all positive";
  setAllPositive.addEventListener("click", () => {
    list.querySelectorAll('input[type="radio"][value="1"]').forEach((input) => {
      input.checked = true;
    });
  });
  bulkTools.append(setAllNegative, setAllPositive);
  card.append(bulkTools);
  card.append(list);
  const footer = document.createElement("div");
  footer.className = "workflow-footer";
  const notes = document.createElement("textarea");
  notes.className = "session-diagnosis-notes";
  notes.maxLength = 2000;
  notes.rows = 2;
  notes.placeholder = "Optional review-session notes";
  const submit = document.createElement("button");
  submit.type = "button";
  submit.className = "button button-primary";
  submit.textContent = "Submit image assessments and reveal reports";
  submit.addEventListener("click", () => submitDiagnosis(item, card, submit));
  footer.append(notes, submit);
  card.append(footer);
  return card;
}

async function startDiagnosis(item, button) {
  const reviewer = $("#identityId").value.trim();
  if (!reviewer) return toast("Enter the pathologist ID first.", {error: true});
  button.disabled = true;
  try {
    await api(`/api/v1/cases/${item.submission_id}/workflow/start-diagnosis`, {
      method: "POST",
      headers: apiHeaders(true),
      body: JSON.stringify({reviewer_id: reviewer, operator_id: reviewer})
    });
    rememberIdentity();
    await loadQueue();
  } catch (error) {
    toast(error.message, {error: true, sticky: true});
  } finally {
    button.disabled = false;
  }
}

async function submitDiagnosis(item, card, button) {
  const reviewer = $("#identityId").value.trim();
  if (!reviewer) return toast("Enter the pathologist ID first.", {error: true});
  const rows = Array.from(card.querySelectorAll(".diagnosis-row"));
  const images = [];
  for (const row of rows) {
    const chosen = row.querySelector("input:checked");
    if (!chosen) return toast("Complete an initial assessment for every image.", {error: true});
    images.push({
      case_id: row.dataset.caseId,
      ground_truth_label: Number(chosen.value),
      notes: row.querySelector(".image-diagnosis-notes").value.trim() || null
    });
  }
  button.disabled = true;
  try {
    const data = await api(`/api/v1/cases/${item.submission_id}/diagnosis`, {
      method: "POST",
      headers: apiHeaders(true),
      body: JSON.stringify({
        reviewer_id: reviewer,
        operator_id: reviewer,
        source_attested: false,
        images,
        patient_notes: card.querySelector(".session-diagnosis-notes").value.trim() || null,
        surgical_procedure: null
      })
    });
    renderSubmission(data);
    toast("Initial image assessments saved. Analysis reports are now visible.");
    await loadQueue();
    await loadMetrics();
  } catch (error) {
    toast(error.message, {error: true, sticky: true});
  } finally {
    button.disabled = false;
  }
}

function renderFeedbackCard(item) {
  const card = document.createElement("article");
  card.className = "workflow-card";
  card.dataset.submissionId = item.submission_id;
  card.append(workflowHeader(item, "Result comparison"));
  const list = document.createElement("div");
  list.className = "feedback-list";
  item.images.forEach((image) => {
    const row = document.createElement("section");
    row.className = "feedback-row";
    row.dataset.caseId = image.case_id;
    row.append(imageReference(image));
    const comparison = document.createElement("div");
    comparison.className = "comparison-result";
    const decision = decisionContent(image);
    comparison.innerHTML = `<span>Pathologist: ${escapeHtml(labelText(image.ground_truth_label))}</span><strong>${escapeHtml(decision.title)}</strong><span>MIP morphology score ${percent(image.p_mip)}</span>`;
    const evaluation = document.createElement("div");
    evaluation.className = "evaluation-control";
    evaluation.innerHTML = `<div class="objective-rating"><span>Recorded evaluation</span><strong>${escapeHtml(ratingText(image))}</strong></div><textarea class="image-feedback-notes" maxlength="1000" rows="2" placeholder="Optional comparison note"></textarea>`;
    row.append(comparison, evaluation);
    list.append(row);
  });
  card.append(list);
  const footer = document.createElement("div");
  footer.className = "feedback-footer";
  footer.innerHTML = `<fieldset class="changed-review"><legend>Did any image-level diagnosis change after result review?</legend><label><input type="radio" name="changed-${escapeHtml(item.submission_id)}" value="false"> No</label><label><input type="radio" name="changed-${escapeHtml(item.submission_id)}" value="true"> Yes</label></fieldset><textarea class="session-feedback-notes" maxlength="2000" rows="2" placeholder="Optional session feedback"></textarea>`;
  const submit = document.createElement("button");
  submit.type = "button";
  submit.className = "button button-primary";
  submit.textContent = "Save image review feedback";
  submit.addEventListener("click", () => submitFeedback(item, card, submit));
  footer.append(submit);
  card.append(footer);
  return card;
}

function renderCollapsedWorkflowCard(item) {
  const card = document.createElement("article");
  card.className = "workflow-card workflow-card-collapsed";
  card.dataset.submissionId = item.submission_id;
  const phase = item.review_status === "reviewed" ? "Result comparison" : "Initial assessment";
  card.append(workflowHeader(item, phase));
  const body = document.createElement("div");
  body.className = "collapsed-workflow-body";
  body.innerHTML = `<div><strong>Large image session</strong><span>${Number(item.image_count)} image rows are collapsed to keep the workspace responsive.</span></div>`;
  const open = document.createElement("button");
  open.type = "button";
  open.className = "button button-secondary";
  open.textContent = `Open ${Number(item.image_count)}-image session`;
  open.addEventListener("click", () => {
    const expanded = item.review_status === "reviewed"
      ? renderFeedbackCard(item)
      : renderDiagnosisCard(item);
    card.replaceWith(expanded);
    expanded.scrollIntoView({behavior: "smooth", block: "start"});
  });
  body.append(open);
  card.append(body);
  return card;
}

async function submitFeedback(item, card, button) {
  const reviewer = $("#identityId").value.trim();
  if (!reviewer) return toast("Enter the pathologist ID first.", {error: true});
  const changed = card.querySelector(".changed-review input:checked");
  if (!changed) return toast("Record whether any image-level diagnosis changed.", {error: true});
  const images = Array.from(card.querySelectorAll(".feedback-row")).map((row) => ({
    case_id: row.dataset.caseId,
    notes: row.querySelector(".image-feedback-notes").value.trim() || null
  }));
  button.disabled = true;
  try {
    const data = await api(`/api/v1/cases/${item.submission_id}/feedback`, {
      method: "POST",
      headers: apiHeaders(true),
      body: JSON.stringify({
        reviewer_id: reviewer,
        operator_id: reviewer,
        images,
        patient_notes: card.querySelector(".session-feedback-notes").value.trim() || null,
        diagnosis_changed_after_ai: changed.value === "true"
      })
    });
    renderSubmission(data);
    toast("Image review feedback saved.");
    await loadQueue();
    await loadMetrics();
  } catch (error) {
    toast(error.message, {error: true, sticky: true});
  } finally {
    button.disabled = false;
  }
}

async function loadQueue() {
  if (!authenticated || isPatientContext()) return;
  try {
    const data = await api("/api/v1/cases?limit=100");
    const items = data.items.filter((item) =>
      item.use_context === "pathologist_self_review" &&
      (item.review_status !== "reviewed" || item.feedback_status === "pending")
    );
    const queue = $("#queue");
    if (!items.length) {
      queue.innerHTML = '<div class="queue-empty"><strong>Queue is clear</strong><span>No image review sessions are awaiting assessment or feedback.</span></div>';
      return;
    }
    queue.replaceChildren(...items.map((item) => {
      if (Number(item.image_count) > 24) return renderCollapsedWorkflowCard(item);
      return item.review_status === "reviewed"
        ? renderFeedbackCard(item)
        : renderDiagnosisCard(item);
    }));
  } catch (error) {
    $("#queue").innerHTML = `<p class="subtle">${escapeHtml(error.message)}</p>`;
  }
}

function metricCard(label, value, note = "") {
  const article = document.createElement("article");
  article.className = "metric-card";
  article.innerHTML = `<span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong>${note ? `<small>${escapeHtml(note)}</small>` : ""}`;
  return article;
}

function renderReviewerMetrics(summaries) {
  const container = $("#physicianMetrics");
  if (!summaries?.length) {
    container.innerHTML = '<p class="subtle">No completed image reviews yet.</p>';
    return;
  }
  container.replaceChildren(...summaries.map((reviewer) => {
    const card = document.createElement("article");
    card.className = "reviewer-card";
    card.innerHTML = `<strong>${escapeHtml(reviewer.reviewer_id)}</strong><span>${reviewer.image_reviews} images across ${reviewer.review_sessions} sessions</span><span>Effective result: ${percent(reviewer.image_effective_rate)}</span><span>Diagnosis changed: ${percent(reviewer.diagnosis_changed_rate)}</span><span>Median initial review: ${durationText(reviewer.median_independent_review_ms)}</span>`;
    return card;
  }));
}

async function loadMetrics() {
  if (!authenticated || isPatientContext()) return;
  try {
    const data = await api("/api/v1/metrics/summary");
    $("#metrics").replaceChildren(
      metricCard("Images in operational dashboard", String(data.all_images), "Before manuscript independence-scope filtering"),
      metricCard("Images reviewed before result", String(data.reviewed_images)),
      metricCard("Image result coverage", percent(data.image_operational_coverage)),
      metricCard("Image-level false-negative rate", percent(data.image_system_fnr), `${data.image_system_fnr_failures} misses / ${data.image_fnr_denominator_positive_images} positive images`),
      metricCard("One-sided FNR upper bound", percent(data.image_system_fnr_upper_bound)),
      metricCard("Accepted-image error rate", percent(data.image_accepted_risk)),
      metricCard("Inconclusive reviewed images", String(data.image_deferred_reviewed_count)),
      metricCard("Effective result rate", percent(data.image_effective_rate), `${data.image_feedback_count} evaluated images`),
      metricCard("Review sessions completed", String(data.feedback_completed_sessions)),
      metricCard("Sessions with a changed diagnosis", percent(data.diagnosis_changed_rate)),
      metricCard("Median initial review", durationText(data.median_independent_review_ms)),
      metricCard("Patient-information images excluded", String(data.patient_information_images))
    );
    $("#metricsNotice").textContent = `${data.interpretation} The Paper Validation Table applies the registered independence-audit scopes and is the manuscript-facing result.`;
    renderReviewerMetrics(data.physician_summaries || []);
  } catch (error) {
    $("#metrics").innerHTML = `<p class="subtle">${escapeHtml(error.message)}</p>`;
  }
}

async function refreshAll() {
  await loadPublicSystem();
  const signedIn = await loadAuthStatus();
  if (!signedIn) return;
  await loadSystem();
  await Promise.all([loadQueue(), loadMetrics()]);
}

$("#loginAccess").addEventListener("click", loginAccess);
$("#logoutAccess").addEventListener("click", logoutAccess);
$("#accessPassword").addEventListener("keydown", (event) => {
  if (event.key === "Enter") loginAccess();
});
document.querySelectorAll('input[name="useContext"]').forEach((input) => {
  input.addEventListener("change", () => {
    restoreIdentity();
    updateContextView();
  });
});
$("#identityId").addEventListener("change", rememberIdentity);
$("#imageInput").addEventListener("change", (event) => addSelectedFiles(event.target.files));
$("#folderInput").addEventListener("change", (event) => addSelectedFiles(event.target.files));
$("#clearImages").addEventListener("click", clearSelectedFiles);
$("#uploadBuilder").addEventListener("dragover", (event) => {
  event.preventDefault();
  if (authenticated && !submitting) $("#uploadBuilder").classList.add("is-dragging");
});
$("#uploadBuilder").addEventListener("dragleave", () => $("#uploadBuilder").classList.remove("is-dragging"));
$("#uploadBuilder").addEventListener("drop", (event) => {
  event.preventDefault();
  $("#uploadBuilder").classList.remove("is-dragging");
  if (!authenticated || submitting) return;
  addSelectedFiles(event.dataTransfer?.files);
});
$("#caseForm").addEventListener("submit", submitImages);
$("#refreshQueue").addEventListener("click", loadQueue);
$("#refreshMetrics").addEventListener("click", loadMetrics);
$("#printReports").addEventListener("click", () => window.print());
document.querySelectorAll("[data-export]").forEach((link) => {
  link.addEventListener("click", (event) => {
    if (!authenticated) event.preventDefault();
  });
});

restoreIdentity();
renderSelectedFiles();
refreshAll().catch((error) => {
  toast(error.message, {error: true, sticky: true});
});
