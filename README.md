# VAR_SOICT Figure 10 / Parti Evaluation Bundle

This folder contains the full Figure 10 style-prompt preparation bundle for arXiv:2507.04482 using the local FineStyle/Parti prompt subset.

- `styles/`: 21 Figure 10 style reference images used by the evaluation bundle.
- `manifest/styles_21.csv`: metadata for all 21 style references.
- `prompts/content_prompts_190.csv`: the 190 filtered Parti content prompts with superclass labels.
- `prompts/eval_cases_190x21.csv`: all 3,990 content/style prompt combinations.
- `prompts/infinity_metadata_190x21.jsonl`: compact generation metadata with `prompt`, `h_div_w`, and `case_id`.
- `src/var_soict/`: reusable Python modules used by `notebooks/infinity2b.ipynb`.
- `notebooks/infinity2b.ipynb`: thin notebook runner that imports the modules and calls the experiment flow.

Paths in the CSV and JSONL files are relative to this `VAR_SOICT` folder.

Infinity runtime layout:

- The official `FoundationVision/Infinity` source is vendored in `Infinity/` at the `VAR_SOICT` project root and used as the local source base by the notebooks.
- `Infinity/` is a normal folder, not a nested git repository or submodule, so it can be pushed with this repo.
- The GGUF loader files, patched GGUF source copies, Infinity-2B GGUF weights, T5 GGUF weights, VAE checkpoint, and generated outputs are kept outside the project-local source tree: `/content/...` on Colab, or the system temp directory for local dry runs.
- At setup time, the code copies `Infinity/` to a runtime folder, applies the GGUF patches there, and imports from that patched runtime copy. This keeps the vendored `Infinity/` source clean.
- Runtime folders are ignored by git. If a required GGUF/runtime file is already present in the runtime cache, setup prints `Using cached file:`; otherwise the first setup run downloads the missing file.
- To force a no-download run after setup, set `download_missing_model_files=False` in `ExperimentConfig`; the setup cell will fail clearly if any required runtime file is missing.

Prompt template:

```text
{content_prompt}, {superclass}, in {style_descriptor} style
```

For descriptors already ending in `style`, the duplicated trailing word is omitted, e.g. `in oil painting style`.

Counts:

- Content prompts: 190
- Style references: 21
- Evaluation cases: 3990
