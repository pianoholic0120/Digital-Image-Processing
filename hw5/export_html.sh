#!/bin/bash
pandoc report.md \
    -o report.html \
    --standalone \
    --mathjax=https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js \
    --embed-resources \
    -c style.css

echo "Successfully generated report.html"
echo "To export PDF, please open report.html in browser and print it as PDF"

