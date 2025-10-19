#!/bin/bash

pandoc report.md \
  -o report.pdf \
  --pdf-engine=xelatex \
  -V colorlinks=false \
  -V linkcolor=black \
  -V urlcolor=black \
  -V toccolor=black \
  --toc \
  --toc-depth=2 \
  -V geometry:margin=1in \
  -V mainfont="Times New Roman" \
  -V CJKmainfont="Songti SC" \
  --metadata title="Digital Image Processing - Homework 2 Report"

echo "PDF exported successfully to report.pdf"

