# Generating Infographics

this folder contains the code to generate infographics. For the time being and lacking artistic talent, `graphviz` is used to render simple charts. For more information on graphviz, see [graphviz.org](https://graphviz.org/).

# Building the charts

I assume you have the `dot` utility available on your command line. If not, consider [installing it](https://graphviz.org/download/). To build all charts, do

```
$ make
```

This should produce 3 rendered versions of every chart: `png`, `svg` and `pdf`. I suggest to use `png` in the rendered website.
