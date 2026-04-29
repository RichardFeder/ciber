**AGENTS — ciber**

This file documents the recommended environment and activation steps for working in the `ciber` codebase.

- **Use this environment:** Before running scripts or notebooks, activate the Conda environment named `ciber`:

```
conda activate ciber
```

- **Jupyter / Notebooks:** If you run notebooks, ensure a kernel is available for this environment:

```
python -m ipykernel install --user --name=ciber --display-name "ciber"
```

- **VS Code:** Select the `ciber` interpreter via the Command Palette: `Python: Select Interpreter` → choose the `ciber` environment. For workspace-default behavior, configure the interpreter in `.vscode/settings.json` if desired.

- **Shell startup (optional):** To have the environment active automatically when opening a project terminal, add `conda activate ciber` to your shell startup file (for example `~/.zshrc`) or use `conda init` as preferred.

- **Why:** Using the `ciber` Conda environment ensures required packages, Python versions, and kernels are consistent for running tests, notebooks, and scripts in this repository.

If you'd like, I can also add a `.vscode/settings.json` snippet that pins the interpreter to this environment.
