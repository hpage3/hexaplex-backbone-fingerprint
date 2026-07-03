\# Codex Prompt Template



Use this template for future Codex tasks in this repository.



```text

We are working in:



C:\\Users\\hpage3\\research\\hexaplex-backbone-fingerprint



First read:



\- AGENTS.md

\- docs/current\_research\_checkpoint.md



Task:



\[Describe the single focused task here.]



Constraints:



\- Do not modify raw input files.

\- Do not commit unless explicitly asked.

\- Keep generated outputs under outputs/.

\- Prefer small deterministic scripts.

\- Prefer reading files from disk instead of pasting file contents.

\- Do not print full PDB, XYZ, CSV, radial-profile, DataFrame, or coordinate-array contents.

\- Do not print long git diffs unless needed to diagnose a failure.

\- Use small synthetic fixtures in tests.

\- Keep tests fast and deterministic.

\- Use PowerShell-compatible commands in any instructions.



Expected outputs:



\[Describe exact files to create or modify.]



Tests:



\[Describe exact tests to add or run.]



Validation commands:



.\\.venv\\Scripts\\python.exe -m pytest



Report back only:



\- pytest result;

\- files changed;

\- files created;

\- top-level metrics or short scientific result;

\- output paths;

\- git status --short.



Do not commit unless asked.

