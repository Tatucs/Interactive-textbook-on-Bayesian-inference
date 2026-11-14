# Interactive textbook chapter on Bayesian inference

A short interactive chapter on Bayesian inference. All Python code runs **in the browser**
(Pyodide), so no server or install is needed.

**Live site:** <https://tatucs.github.io/Interactive-textbook-on-Bayesian-inference/>

| Page | Interactive figure | Auto-checked exercise |
|---|---|---|
| 1. Priors | Beta prior with α, β sliders | design a prior (mean 0.7, worth 10 flips) |
| 2. Posteriors | prior × likelihood → posterior | compute a grid posterior |
| 3. Conjugacy | flip-by-flip Beta update vs. grid | conjugate update by hand |
| 4. MCMC | Metropolis trace + histogram, step-size tuning | write the accept/reject step |
| 5. Variational inference | fit a mean-field q to a 2-D Gaussian, CAVI | optimal mean-field σ |

The running example is a coin with an unknown heads probability θ (Beta-Binomial).

## Usage

Open the live site. The first load takes about 30 s while Python downloads.
Move the sliders to update the figures. In an exercise, fill in the blanks and press
**Run Code** to have your answer checked. **Show Hint** is there if you get stuck.

## How it works

- **Quarto + [quarto-live](https://r-wasm.github.io/quarto-live/)** turns `.qmd` pages into a static
  site with in-browser Python cells, slider inputs and graded exercises.
- **`bayes/`** is a small Python package used by every page: the model math
  (`models.py`, `mcmc.py`, `vi.py`) and the exercise checkers (`checks.py`).
- **GitHub Actions** (`.github/workflows/publish.yml`) runs the tests, then builds the site and
  deploys it to GitHub Pages. If a test fails, nothing is deployed.

## Experiences and lessons learned

**Tools:** Claude Code

**What worked well:**
- The agent tested pages in a headless browser (moving sliders, submitting right and wrong answers),
  not only with unit tests.
- Keeping the math and checkers in one tested package made the exercises reliable.

**What had to be fixed or decided by hand:**
- The project-specific decisions: stack, examples, scope
- Quarto silently used my local Jupyter kernel, so `engine: markdown` was needed on every page.
- A mistaken `git add` led to a commit that contained only part of the changes. It was caught from the git output and fixed with a follow-up commit.
- The visual style I chose needed multiple iterations and adjustments in a few rounds.
