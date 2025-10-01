#!/bin/bash

# Script to compile the research paper LaTeX document
# Usage: ./compile_paper.sh

echo "Compiling research paper..."

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "Error: pdflatex is not installed."
    echo "Please install LaTeX:"
    echo "  macOS: brew install --cask mactex"
    echo "  Ubuntu: sudo apt-get install texlive-full"
    echo "  Windows: Install MiKTeX or TeX Live"
    exit 1
fi

# Compile the LaTeX document
echo "Running pdflatex (first pass)..."
pdflatex research_paper.tex

echo "Running bibtex..."
bibtex research_paper

echo "Running pdflatex (second pass)..."
pdflatex research_paper.tex

echo "Running pdflatex (final pass)..."
pdflatex research_paper.tex

# Clean up auxiliary files
echo "Cleaning up auxiliary files..."
rm -f *.aux *.log *.bbl *.blg *.out *.toc

echo "Research paper compiled successfully!"
echo "Output: research_paper.pdf"
