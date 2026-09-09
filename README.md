# VAR_SOICT Figure 10 / Parti Evaluation Bundle

This folder contains the full Figure 10 style-prompt preparation bundle for arXiv:2507.04482 using the local FineStyle/Parti prompt subset.

- `styles/`: 21 Figure 10 style reference images copied from `VAR_Style_Transfer_Workspace/style_figure10`.
- `manifest/styles_21.csv`: metadata for all 21 style references.
- `prompts/content_prompts_190.csv`: the 190 filtered Parti content prompts with superclass labels.
- `prompts/eval_cases_190x21.csv`: all 3,990 content/style prompt combinations.
- `prompts/infinity_metadata_190x21.jsonl`: compact generation metadata with `prompt`, `h_div_w`, and `case_id`.
- `src/var_soict/`: reusable Python modules used by `notebooks/infinity2b.ipynb`.
- `notebooks/infinity2b.ipynb`: thin notebook runner that imports the modules and calls the experiment flow.

Paths in the CSV and JSONL files are relative to this `VAR_SOICT` folder.

Prompt template:

```text
{content_prompt}, {superclass}, in {style_descriptor} style
```

For descriptors already ending in `style`, the duplicated trailing word is omitted, e.g. `in oil painting style`.

Counts:

- Content prompts: 190
- Style references: 21
- Evaluation cases: 3990
