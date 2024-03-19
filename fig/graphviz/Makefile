DOTFILES=$(wildcard *.dot) #for directed graphs
PDFOUTPUTS=$(DOTFILES:.dot=.pdf)
PNGOUTPUTS=$(DOTFILES:.dot=.png)
SVGOUTPUTS=$(DOTFILES:.dot=.svg)

all : $(PDFOUTPUTS) $(SVGOUTPUTS) $(PNGOUTPUTS)

%.pdf : %.dot
	@dot -Tpdf -o$@ $<

%.png : %.dot
	@dot -Tpng -o$@ $<

%.svg : %.dot
	@dot -Tsvg -o$@ $<
