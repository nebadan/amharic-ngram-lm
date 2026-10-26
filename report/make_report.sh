#!/bin/bash
# Two-pass build: render once to find each heading's page, then rebuild with a filled-in contents page.
set -e
cd "$(dirname "$0")"
rm -f .toc_pages.json
node build_report.js
soffice --headless --convert-to pdf NLP_Assignment1_Report.docx >/dev/null 2>&1
../.venv/bin/python - <<'PY'
import json, subprocess
heads = json.load(open(".headings.json"))
n = int([l for l in subprocess.run(["pdfinfo", "NLP_Assignment1_Report.pdf"], capture_output=True, text=True).stdout.splitlines() if l.startswith("Pages")][0].split()[-1])
texts = [subprocess.run(["pdftotext", "-layout", "-f", str(i), "-l", str(i), "NLP_Assignment1_Report.pdf", "-"], capture_output=True, text=True).stdout for i in range(1, n + 1)]
pages, cur = {}, 3   # body starts after title + contents
for _, h in heads:
    for i in range(cur, n + 1):
        if any(l.strip().startswith(h) for l in texts[i - 1].splitlines()):
            pages[h] = i; cur = i; break
missing = [h for _, h in heads if h not in pages]
if missing: print("not found:", missing)
json.dump(pages, open(".toc_pages.json", "w"))
PY
node build_report.js
soffice --headless --convert-to pdf NLP_Assignment1_Report.docx >/dev/null 2>&1
echo "pages: $(pdfinfo NLP_Assignment1_Report.pdf | awk '/Pages/{print $2}')"
