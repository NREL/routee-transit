# Contributing

Guidelines for contributing to RouteE Transit.

- How to submit issues and pull requests
- Coding standards
- Testing instructions

## Documentation

### Install Documentation Dependencies
To install the documentation dependencies, run:

```bash
poetry install --with docs
```

### Serve Documentation Locally
To preview the documentation site locally, run:

```bash
poetry run mkdocs serve
```

This will start a local server (usually at http://127.0.0.1:8000/) where you can view the docs as you edit them.

### Edit Documentation
- All documentation source files are in the `docs/` directory as Markdown files (e.g., `index.md`, `usage.md`).
- Edit or add Markdown files as needed. The navigation is controlled by `mkdocs.yml` at the project root.
- For API documentation, use docstrings in your Python code. The `mkdocstrings` plugin will automatically include them in the docs if referenced in `api.md`.

After editing, refresh your browser to see changes reflected immediately.
