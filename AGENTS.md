# Operational Guardrails

- Never issue multiple `view_image` calls in the same turn. Inspect rendered page images one at a time and wait for each result before calling the next. Use `render_docx.py`, `sips`, or the image-reader skill for deterministic layout and text checks.
- If `view_image` returns `[Unsupported Image]` or no output, fall back to OCR or `pdfplumber` instead of retrying several images in parallel.
- When resuming this thread from the CLI, run from inside this project directory and include `--skip-git-repo-check`. This directory is not a Git repository, and resuming from `/Users/yellowich` will fail with a trusted-directory error.
