#!/bin/bash

pandoc report.md \
  -o report.html \
  --standalone \
  --embed-resources \
  -c style.css \
  --metadata title="Digital Image Processing - Homework 2 Report"

echo "Successfully generated report.html"
echo "To export PDF, please open report.html in browser and print it as PDF"

