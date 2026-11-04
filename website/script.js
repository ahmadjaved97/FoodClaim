const copyButton = document.querySelector('#copy-citation');
const citation = document.querySelector('#bibtex');
const copyStatus = document.querySelector('#copy-status');

if (navigator.clipboard?.writeText) {
  copyButton.hidden = false;
  copyButton.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(citation.textContent);
      copyStatus.textContent = 'BibTeX copied to clipboard.';
    } catch {
      copyStatus.textContent = 'Could not access the clipboard. Select the citation above to copy it manually.';
    }
  });
}
