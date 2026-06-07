# jaxon JNE manuscript build
#
# Targets:
#   make            full build (latexmk -pdf, runs bibtex automatically)
#   make figs       copy locked figures from ../outputs into figures/
#   make clean      remove all latexmk intermediates
#   make distclean  also remove the PDF

LATEXMK = latexmk -pdf -interaction=nonstopmode -halt-on-error
MAIN    = main
FIGDIR  = figures
OUTROOT = ../outputs

# Locked-content figure copies. Phase 3 figures (selectivity violin,
# example cross-section) are added by the user once the sweep finishes.
FIG_SCALING = $(OUTROOT)/scaling/fig_scaling.png
FIG_SEED0_RECT_X = $(OUTROOT)/selectivity_sweep_phase3_manuscript/fig_seed_0000_rect_xsection.png

.PHONY: all figs clean distclean watch

all: figs
	$(LATEXMK) $(MAIN).tex

figs:
	@mkdir -p $(FIGDIR)
	@cp -u $(FIG_SCALING)        $(FIGDIR)/fig_scaling.png        2>/dev/null || \
	  echo "[figs] WARN: scaling fig not found at $(FIG_SCALING)"
	@cp -u $(FIG_SEED0_RECT_X)   $(FIGDIR)/fig_seed_0000_rect_xsection.png 2>/dev/null || \
	  echo "[figs] WARN: seed 0 rect xsection not found at $(FIG_SEED0_RECT_X)"

watch:
	$(LATEXMK) -pvc $(MAIN).tex

clean:
	$(LATEXMK) -c $(MAIN).tex

distclean:
	$(LATEXMK) -C $(MAIN).tex
	rm -f $(FIGDIR)/*.png $(FIGDIR)/*.pdf
