const form = document.querySelector('#analysis-form');
const result = document.querySelector('#result');
const csrf = document.querySelector('[name=csrfmiddlewaretoken]').value;
const escapeHtml = value => String(value).replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
form.addEventListener('submit', async event => {
  event.preventDefault(); result.classList.remove('hidden'); result.textContent = 'Analyzing label...';
  try {
    const response = await fetch('/api/analyze/', {method: 'POST', body: new FormData(form), headers: {'X-CSRFToken': csrf}}); const data = await response.json();
    if (!response.ok) { result.innerHTML = `<strong>Could not analyze:</strong> ${escapeHtml(data.error)}${data.hint ? `<p>${escapeHtml(data.hint)}</p>` : ''}`; return; }
    const analysis = data.analysis;
    const warnings = analysis.warnings.length ? analysis.warnings.map(item => `<div class="warning"><strong>${escapeHtml(item.name)}</strong>: DNN risk confidence ${escapeHtml(item.confidence)}%</div>`).join('') : '<p>The DNN did not predict risk for the selected profiles.</p>';
    const unavailable = Object.entries(data.model_errors || {}).map(([name, message]) => `<p class="model-error"><strong>${escapeHtml(name.replaceAll('_', ' '))} unavailable:</strong> ${escapeHtml(message)}</p>`).join('');
    const unsupported = analysis.unsupported?.length ? `<p class="model-error">Not trained for: ${escapeHtml(analysis.unsupported.join(', '))}</p>` : '';
    result.innerHTML = `<h2>Final DNN dietary prediction</h2>${unavailable}<div class="status ${analysis.status === 'AVOID' ? 'avoid' : 'safe'}">${escapeHtml(analysis.status.replaceAll('_', ' '))}</div><p><strong>Risk score:</strong> ${escapeHtml(analysis.risk_score)}/100</p>${warnings}${unsupported}<p>${escapeHtml(analysis.recommendation)}</p><details><summary>Extracted text</summary><p>${escapeHtml(data.ocr_text)}</p></details>`;
  } catch (error) { result.innerHTML = '<strong>Could not contact the analysis service.</strong>'; }
});
