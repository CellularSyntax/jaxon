# jaxon JNE manuscript build (portable: Linux + Windows MSYS/Git Bash)
#
# Targets:
#   make            full build (figs + latexmk -pdf)
#   make figs       copy locked figures from ../outputs into figures/
#   make watch      latexmk -pvc, rebuild on save
#   make clean      remove latexmk intermediates
#   make distclean  also remove the PDF + copied figures

LATEXMK = latexmk -pdf -interaction=nonstopmode -f
MAIN    = main

.PHONY: all figs watch clean distclean

all: figs
	$(LATEXMK) $(MAIN).tex

figs:
	python copy_figs.py

watch:
	$(LATEXMK) -pvc $(MAIN).tex

clean:
	$(LATEXMK) -c $(MAIN).tex

distclean:
	$(LATEXMK) -C $(MAIN).tex
