# Contributing

Thanks for taking an interest in the project. Changes that improve reproducibility, clarify the evaluation, or add meaningful comparisons are welcome.

## Development setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the reproducible pipeline from the repository root:

```powershell
python DataProcessing\pipeline.py
```

For interactive work, launch Jupyter from the repository root. Keep generated model and report files out of commits; they can be recreated by the pipeline.

## Pull request guidance

- Keep the canonical label mapping: `0 = computer`, `1 = human`.
- Keep paths relative to the repository, not to a local machine.
- Describe the data split and positive class when reporting metrics.
- State which computer generator was used; results from one generator do not generalize automatically to others.
- Do not add participant data or derived attributes without confirming their sharing permissions.
- Include a concise summary of the change and the command or notebook steps used to verify it.

## Data and responsible reporting

The participant data may have restrictions that are not documented in this repository. Confirm the source citation and terms before redistributing it. Avoid presenting this exploratory model as a universal human-versus-computer detector.
