# jaxon JNE manuscript build (portable: Linux + Windows MSYS/Git Bash)
#
# Targets:
#   make            full build of the article-class preview (main.pdf)
#   make iop        full build of the IOP/JNE submission PDF (main_iop.pdf)
#   make all-pdfs   build both main.pdf AND main_iop.pdf in one pass
#   make figs       (re)generate figures via copy_figs.py + make_figures.py
#   make watch      latexmk -pvc, rebuild main.pdf on save
#   make watch-iop  latexmk -pvc, rebuild main_iop.pdf on save
#   make clean      remove latexmk intermediates for both variants
#   make distclean  also remove the PDFs + copied figures
#
# main.tex and main_iop.tex share the same content: main_iop.tex is a
# 2-line wrapper that sets \jneSubmission so main.tex selects
# iopjournal.cls instead of article class.

LATEXMK = latexmk -pdf -interaction=nonstopmode -f
MAIN    = main
IOP     = main_iop

.PHONY: all iop all-pdfs figs watch watch-iop clean distclean

all: figs
	$(LATEXMK) $(MAIN).tex

iop: figs
	$(LATEXMK) $(IOP).tex

all-pdfs: figs
	$(LATEXMK) $(MAIN).tex
	$(LATEXMK) $(IOP).tex

figs:
	python copy_figs.py

watch:
	$(LATEXMK) -pvc $(MAIN).tex

watch-iop:
	$(LATEXMK) -pvc $(IOP).tex

clean:
	$(LATEXMK) -c $(MAIN).tex
	$(LATEXMK) -c $(IOP).tex

distclean:
	$(LATEXMK) -C $(MAIN).tex
	$(LATEXMK) -C $(IOP).tex
