#!/usr/bin/env python3
"""
Script to convert the research paper from LaTeX to Word format.
"""

import subprocess
import os
import sys

def install_pandoc():
    """Install pandoc if not available."""
    try:
        subprocess.run(['pandoc', '--version'], check=True, capture_output=True)
        print("Pandoc is already installed.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Pandoc not found. Installing via Homebrew...")
        try:
            subprocess.run(['brew', 'install', 'pandoc'], check=True)
            print("Pandoc installed successfully.")
            return True
        except subprocess.CalledProcessError:
            print("Failed to install pandoc via Homebrew.")
            return False

def convert_latex_to_word():
    """Convert LaTeX file to Word format."""
    input_file = "research_paper.tex"
    output_file = "research_paper.docx"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return False
    
    try:
        # Convert LaTeX to Word using pandoc
        cmd = [
            'pandoc',
            input_file,
            '-o', output_file,
            '--bibliography=references.bib',  # If you have a .bib file
            '--citeproc',
            '--reference-doc=template.docx'  # Optional: use a custom template
        ]
        
        # Remove bibliography and citeproc if no .bib file exists
        if not os.path.exists('references.bib'):
            cmd = [arg for arg in cmd if arg not in ['--bibliography=references.bib', '--citeproc']]
        
        subprocess.run(cmd, check=True)
        print(f"Successfully converted {input_file} to {output_file}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error converting to Word: {e}")
        return False

def create_simple_word_version():
    """Create a simple Word version by converting the PDF."""
    pdf_file = "research_paper.pdf"
    output_file = "research_paper.docx"
    
    if not os.path.exists(pdf_file):
        print(f"Error: {pdf_file} not found. Please compile the LaTeX first.")
        return False
    
    try:
        # Try to convert PDF to Word using pandoc
        cmd = ['pandoc', pdf_file, '-o', output_file]
        subprocess.run(cmd, check=True)
        print(f"Successfully converted {pdf_file} to {output_file}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error converting PDF to Word: {e}")
        print("Note: PDF to Word conversion may not preserve all formatting.")
        return False

def main():
    """Main conversion function."""
    print("Converting research paper to Word format...")
    
    # Try to install pandoc if needed
    if not install_pandoc():
        print("Cannot proceed without pandoc. Please install it manually:")
        print("brew install pandoc")
        return False
    
    # Try direct LaTeX to Word conversion first
    if convert_latex_to_word():
        return True
    
    # Fallback: convert PDF to Word
    print("Direct LaTeX conversion failed. Trying PDF to Word conversion...")
    if create_simple_word_version():
        return True
    
    print("All conversion methods failed.")
    return False

if __name__ == "__main__":
    success = main()
    if success:
        print("Word document created successfully!")
    else:
        print("Failed to create Word document.")
