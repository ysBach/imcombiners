DOCS_DIR := docs/quarto
DOCS_PORT ?= 8000
DOCS_ENV = MPLBACKEND=module://matplotlib_inline.backend_inline QUARTO_PYTHON="$$(uv run --extra docs python -c 'import sys; print(sys.executable)')"

.PHONY: docs-api docs-tutorials docs-render docs-build docs-preview docs-serve docs-clean

docs-api:  ## Regenerate docstring/API reference qmd files only.
	cd $(DOCS_DIR) && uv run --extra docs python -m quartodoc build

docs-tutorials:  ## Render tutorial qmd notebooks only, leaving generated API sources alone.
	cd $(DOCS_DIR) && $(DOCS_ENV) quarto render --profile tutorials --no-clean

docs-render:  ## Render the whole Quarto website from existing qmd files.
	cd $(DOCS_DIR) && $(DOCS_ENV) quarto render

docs-build: docs-api docs-render  ## Regenerate API qmd files, then render the whole website.

docs-preview: docs-build docs-serve  ## Build the site, then serve docs/quarto/_site locally.

docs-serve:  ## Serve the existing rendered site without rebuilding.
	cd $(DOCS_DIR) && python -m http.server $(DOCS_PORT) -d _site

docs-clean:
	rm -rf docs/quarto/_site docs/quarto/.quarto docs/quarto/api docs/quarto/objects.json
	find docs -name .DS_Store -delete
