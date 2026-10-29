# CLAUDE.md — Fast Failure Detection in LLM-generated Patches

This file gives you (Claude Code) the full context of this project and the rules for working with me.
Read it fully at the start of every session.

---

## 1. Who I am and how we work

I am Nayo, a Master's student in Data Science at EURECOM. This is my semester project, and **it is a
learning project**: I must understand and be able to explain every line of code.

**Working rules (always follow these):**

1. **One step at a time.** Only do the single step I ask for (see the roadmap in section 6). Never
   jump ahead to later steps, never "also" add extra features, files or refactors.
2. **Explain first, then wait.** Before writing or changing code, explain in plain, simple words
   what you plan to do and why. Wait for my OK before you edit files.
3. **Keep code short, simple and commented.** Prefer clear code over clever code. When you use a
   library function I may not know, explain what it does in one sentence.
4. **After each step, stop.** Tell me: what you did, what files changed, what I should look at in the
   output, and what the next step would be. Then wait for my review. I may ask for changes.
5. **Inspect before assuming.** Never guess the structure of the data. Print it and look first.
6. **Do not commit or push to git.** I commit myself. You may suggest a commit message.
7. **Ask when unsure** instead of making a big decision on your own.

---

## 2. The project

**Goal:** predict, *without running any unit test*, whether a patch produced by an LLM coding agent
is correct (`resolved = True`) or wrong (`resolved = False`).

**Why:** running Continuous Integration (CI) on every agent patch costs time and money. If a fast,
cheap classifier can spot the patches that are clearly wrong, those can be thrown away **before** CI.
Good or uncertain patches still go to CI. So the most costly mistake is throwing away a patch that
was actually correct.

**Important constraints from my supervisor:**

- We do **not** build the agent. The agent runs are already in a Hugging Face dataset; we only analyse them.
- Use **small, fast, traditional ML models** (scikit-learn): Logistic Regression first, then Random
  Forest and SVM. Speed is the main point of the project.
- **First build a small dataset** (about 500 rows) and get the whole pipeline working on it. Only then
  run the same code on the full dataset.
- **No Google Colab.** Everything is code in this Git repository, run locally in VS Code.
- The final deliverable must be **reusable**: my supervisor wants to run our pipeline and get the
  metrics on **his own dataset** (same format). So feature extraction + model + evaluation must be a
  script he can point at a different data file.

**From the supervisor's slides:**

- Background: an LLM agent gets a natural-language description of a bug and acts in a loop (think →
  run one shell command → read the output → repeat) until it believes the bug is solved, then submits
  a patch (a git diff).
- Initial steps:
  - Understand how the dataset is built. Print statistics, look for null / problematic values.
    Assess whether we can safely use it for our classifier.
  - Think about key features we can extract to build a traditional ML classifier.
  - Train a simple ML model as a starting point.
  - Input features: extracted from the trajectory (messages) and/or the patch.
  - Target label: `resolved` — predicted without running any unit test.
- Next steps: statistics on the dataset → extract features into a table (one row per patch, columns
  `feat_1 … feat_N`, `resolved`) → train a simple Logistic Regression and show results on
  train / validation / test.

---

## 3. The dataset

- Hugging Face: `SWE-bench/SWE-smith-trajectories`
  (https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories)
- Runs of SWE-agent + Claude 3.7 Sonnet on bug tasks from the SWE-smith dataset.
- **We use only the `ticks` split** (about 25.8k rows). In this split the agent writes its commands
  inside ```` ```bash ```` blocks. Do not mix in the `tool` or `xml` splits (they are separate runs in
  other formats; mixing them risks hidden duplicates of the same bug across train/test).
- Columns:
  - `messages` — the full trajectory: list of turns with `role` (system / user / assistant) and
    `content`. The first user message contains the bug description in `<pr_description>`. It may be
    stored as a list or as a JSON string — check and handle it.
  - `instance_id` — the bug task, e.g. `mahmoud__boltons.3bfcfdd0.lm_rewrite__or6ab7bk`
    (repo owner__repo . commit . bug-type__id). The same bug can appear several times.
  - `resolved` — the target (bool).
  - `model` — the LLM used (one model is the large majority).
  - `traj_id` — the unique id of this run.
  - `patch` — the final git diff (can be empty).

**Known things to check (do not skip):**

- **Class balance:** how many True vs False.
- **Patch / instance mismatch:** in the Hugging Face preview, patches seem to edit files from a
  *different* repository than the one in `instance_id` (e.g. a boltons task with a patch editing dask
  files). Verify this on the real data. If true, patch-based features may be unreliable — report it to
  me, I will discuss it with my supervisor.
- Empty patches, missing values, duplicates, very long or truncated trajectories.

---

## 4. Decisions we already made

**Sampling the mini dataset**
- Random sample (not the first rows), with a fixed random seed so it is reproducible.
- Make sure both classes are well represented (roughly balanced if possible).
- Keep all attempts at the same `instance_id` together.

**Cleaning (preprocessing)**
- Turn `resolved` into a clean 0/1; drop rows where it is missing.
- Remove exact duplicates.
- Empty patch is *information*, not an error: keep the row, flag it as a feature.
- Drop rows with an empty or broken trajectory.
- Parse `messages` into structure: agent turns, the command in each turn, the shell outputs and their
  return codes. Inspect the real format first.
- Remove pure noise (e.g. the system prompt, identical in every row).
- **No leakage:** only use information that exists *before* CI runs (the agent's own actions and the
  patch). Never use anything coming from the official test evaluation.
- One cleaning script that works unchanged on the mini dataset, the full split, and the supervisor's data.

**Features (hand-made first)** — every feature must be computable for every row without knowing the answer:
- From the task: bug type taken from `instance_id` (e.g. `lm_rewrite`, `combine_file`,
  `func_pm_...`), length of the PR description, whether it contains a traceback / code example.
- From the trajectory: number of assistant turns, number of commands with non-zero return code,
  number of outputs containing error signs (`Traceback`, `Error`, `command not found`), whether the
  agent created a reproduce script, whether it ran tests, whether the last test run looked like pass or
  fail, number of file edits, repeated identical commands (stuck in a loop), whether it ended with a
  clean submit or hit a limit, total conversation length.
- From the patch: lines added, lines removed, files changed, hunks, empty patch (0/1), touches test
  files, contains leftover helper files (e.g. `reproduce.py`), contains leftover `print(` debug lines.
- Be careful with the `model` column (may not carry over to the supervisor's data): test with and without it.
- Later, optional: TF-IDF on the patch text, compared against hand-made features only.

**Splitting and training**
- Group split by `instance_id` into train / validation / test (no bug on both sides).
- Fit any scaling / TF-IDF / encoding on the **training set only** (use sklearn `Pipeline`).
- Standardise features for Logistic Regression and SVM; log-transform very large counts if useful.
- Handle class imbalance with `class_weight="balanced"` if needed.
- Logistic Regression is the baseline. Then Random Forest and SVM, same features, same splits.
- With small data use cross-validation (grouped) to compare models.
- Use the test set only once, at the end.

**Feature selection**
- Write a one-line reason for each feature; drop constant / mostly-missing ones; be suspicious of any
  feature that is "too perfect" (probably leakage).
- Compare each feature between resolved and failed rows (plots), single-feature ROC-AUC.
- Remove near-duplicate (highly correlated) features.
- Look at Logistic Regression weights, L1 regularisation, Random Forest feature importance.
- Ablation: remove a feature or a group and see if the validation score drops.
- Prefer fewer, meaningful features.

**Metrics**
- Not only accuracy. Confusion matrix, precision, recall, F1, ROC-AUC.
- Especially: precision on the "fail" class (when we reject, are we right?) and recall on it (how many
  bad patches we catch = CI time saved). Consider tuning the decision threshold.
- Also report prediction speed.

---

## 5. Environment rules (keep it clean, do not break it)

- My machine is Windows, VS Code, PowerShell, Python 3.14.
- The repo lives inside OneDrive (`C:\Users\nayon\OneDrive\Documents\Final ResProj\Fast-patch-failure-detection`).
  Therefore the virtual environment is kept **outside** the repo, at `C:\Users\nayon\venvs\fastpatch`,
  so OneDrive does not sync it. **Never create a `.venv` inside the repo.**
- Activate it with `C:\Users\nayon\venvs\fastpatch\Scripts\Activate.ps1`. VS Code's interpreter must be
  `C:\Users\nayon\venvs\fastpatch\Scripts\python.exe`.
- **Never** `pip install` into the global/system Python. Always inside the activated `fastpatch` environment.
- Never upgrade or install a library without telling me first. Every new library goes into
  `requirements.txt`.
- If a library fails to install on Python 3.14 ("failed building wheel", "Microsoft Visual C++ is
  required"), stop and tell me. The fallback is to recreate the `fastpatch` environment with Python 3.12 — ask me before doing it.
- Data and models are never committed: `data/`, `models/`, `*.parquet`, `*.joblib` are in `.gitignore`.
  The Hugging Face cache stays outside the repo.
- Fixed random seed everywhere (`random_state=42`) so results are reproducible.

**Repository layout**

```
Fast-patch-failure-detection/
├── CLAUDE.md
├── README.md            what the project is + how to run it
├── requirements.txt
├── .gitignore
├── data/                not committed, created by scripts
├── notebooks/           quick exploration only
├── src/
│   ├── load_data.py     download ticks split, build the mini dataset
│   ├── explore.py       statistics and data-quality checks
│   ├── preprocess.py    cleaning, parsing trajectories
│   ├── features.py      hand-made features → one row of numbers per patch
│   ├── train.py         train LogReg / RF / SVM, save model
│   └── evaluate.py      run a saved model on ANY dataset in the same format → metrics
└── results/             metrics tables and plots
```

---

## 6. Roadmap (one step per session/prompt — I tell you which one)

0. **Environment check:** Python version, git status, `.gitignore` correct, verify the `fastpatch`
   environment (outside the repo), install `requirements.txt` into it, check imports, confirm VS Code interpreter.
1. **First look:** load the `ticks` split, print columns, types, and 2–3 example rows (shortened).
2. **Statistics on the full split:** row count, True/False balance, `model` distribution, nulls,
   empty patches, duplicates, repeated `instance_id`, trajectory and patch lengths, bug types,
   patch/instance repo mismatch check.
3. **Mini dataset:** random ~500 rows, fixed seed, both classes represented, grouped by
   `instance_id`, saved to `data/`.
4. **Cleaning:** the rules from section 4, on the mini dataset.
5. **Feature extraction:** a first small set of hand-made features → feature table.
6. **Split** train / validation / test grouped by `instance_id`.
7. **Logistic Regression baseline** + metrics.
8. **Feature analysis and selection.**
9. **Random Forest and SVM**, compared with grouped cross-validation.
10. **(Optional) TF-IDF on the patch**, compared with hand-made features only.
11. **`evaluate.py`** — reusable script for the supervisor's dataset.
12. **Run everything on the full ticks split** and write up results.
