# jaxon — JNE manuscript

LaTeX source for the J Neural Eng submission of **jaxon**: differentiable
peripheral nerve fiber simulation in JAX for scalable selectivity
optimisation.

## Layout

```
manuscript/
├── main.tex                   master document (iopart class)
├── abstract.tex
├── sections/
│   ├── 01_introduction.tex
│   ├── 02_methods.tex
│   ├── 03_results.tex
│   └── 04_discussion.tex
├── refs.bib
├── figures/                   ← populated by `make figs`
├── Makefile                   one-shot build
└── README.md                  this file
```

## Build

```bash
make figs && make            # full build (latexmk → bibtex → latexmk → pdf)
make watch                   # latexmk -pvc, rebuilds on save
make clean                   # remove intermediates
```

Requires a TeX Live distribution with `iopart` (in `texlive-publishers`).

## What's drafted vs. stubbed

| Section | Status |
|---|---|
| Abstract | drafted, two placeholders for sweep numbers |
| Intro | drafted in full |
| Methods | drafted in full; a handful of `\needdata{}` notes for version pins |
| Results §validation | drafted, awaiting validation table numbers |
| Results §scaling | drafted, awaiting final benchmark numbers |
| Results §selectivity single-diam | drafted, awaiting Phase 3 sweep stats |
| Results §selectivity mixed-diam | drafted as honest negative result |
| Discussion | drafted in full |

Anything wrapped in `\needdata{...}` produces an inline yellow callout in
the PDF and a per-section entry in the `\listoftodos` at the end — so
you can see at a glance what still needs filling in. Strip the
`\listoftodos` line in `main.tex` before camera-ready.

## Editing conventions

- **One sentence per line** in the source. Makes diffs and PR review
  much cleaner than reflowing paragraphs.
- **Hyphenation:** `multi-fascicle`, `multi-start`, `single-cuff` etc.
  with explicit hyphens. UK English (J Neural Eng house style).
- **Citation keys:** `lastnameYEAR` (e.g. `mcintyre2002`) or
  `lastnameYEARtopic` if a single author has multiple cited works
  (e.g. `hussain2024`, `musselman2023`).
- **Math:** use `\mathbf{}` for vectors, `\boldsymbol{}` only when a
  bold Greek letter is needed.
- **Figures:** copy/link the source PNG/PDF into `figures/` rather than
  referencing `../outputs/...` from the LaTeX. The `Makefile` does this
  for the locked figures.

## Pre-submission checklist

- [ ] All `\needdata{}` callouts resolved.
- [ ] `\listoftodos` commented out in `main.tex`.
- [ ] Final version pins in Methods (jax, jaxley, CUDA).
- [ ] Author ORCID inserted.
- [ ] Funding statement and acknowledgements completed.
- [ ] Data availability section points to the correct git tag.
- [ ] `iopart-num.bst` available locally (TeX Live ships it).
- [ ] PDF compiles cleanly with no `??` undefined references.
