#!/bin/bash

# Compilation script for the Advanced DQN Portfolio Optimization paper
echo "🚀 Compiling Advanced DQN Portfolio Optimization Paper..."

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "❌ Error: pdflatex is not installed. Please install LaTeX."
    echo "   On macOS: brew install --cask mactex"
    echo "   On Ubuntu: sudo apt-get install texlive-full"
    exit 1
fi

# Check if bibtex is available
if ! command -v bibtex &> /dev/null; then
    echo "❌ Error: bibtex is not installed. Please install LaTeX."
    exit 1
fi

# Compile the paper
echo "📝 Running pdflatex (first pass)..."
pdflatex Advanced_DQN_Portfolio_Optimization.tex

echo "📚 Running bibtex..."
bibtex Advanced_DQN_Portfolio_Optimization

echo "📝 Running pdflatex (second pass)..."
pdflatex Advanced_DQN_Portfolio_Optimization.tex

echo "📝 Running pdflatex (final pass)..."
pdflatex Advanced_DQN_Portfolio_Optimization.tex

# Clean up auxiliary files
echo "🧹 Cleaning up auxiliary files..."
rm -f *.aux *.log *.out *.toc *.bbl *.blg *.fdb_latexmk *.fls *.synctex.gz

# Check if PDF was created successfully
if [ -f "Advanced_DQN_Portfolio_Optimization.pdf" ]; then
    echo "✅ Paper compiled successfully!"
    echo "📄 Output: Advanced_DQN_Portfolio_Optimization.pdf"
    
    # Get file size
    size=$(ls -lh Advanced_DQN_Portfolio_Optimization.pdf | awk '{print $5}')
    echo "📊 File size: $size"
    
    # Count pages
    pages=$(pdfinfo Advanced_DQN_Portfolio_Optimization.pdf 2>/dev/null | grep Pages | awk '{print $2}')
    if [ ! -z "$pages" ]; then
        echo "📄 Pages: $pages"
    fi
    
else
    echo "❌ Error: PDF compilation failed!"
    echo "📋 Check the log files for errors:"
    ls -la *.log
    exit 1
fi

echo "🎉 Compilation complete!"
