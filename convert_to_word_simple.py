#!/usr/bin/env python3
"""
Simple script to convert LaTeX to Word format using pandoc.
"""

import subprocess
import os

def convert_to_word():
    """Convert LaTeX to Word using pandoc."""
    input_file = "research_paper.tex"
    output_file = "research_paper.docx"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return False
    
    try:
        # Simple pandoc conversion without custom template
        cmd = [
            'pandoc',
            input_file,
            '-o', output_file,
            '--filter', 'pandoc-crossref',  # For cross-references
            '--number-sections',  # Add section numbers
            '--toc'  # Add table of contents
        ]
        
        # Remove filter if not available
        try:
            subprocess.run(['pandoc-crossref', '--version'], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            cmd = [arg for arg in cmd if arg not in ['--filter', 'pandoc-crossref']]
        
        subprocess.run(cmd, check=True)
        print(f"Successfully converted {input_file} to {output_file}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error converting to Word: {e}")
        
        # Try even simpler conversion
        try:
            simple_cmd = ['pandoc', input_file, '-o', output_file]
            subprocess.run(simple_cmd, check=True)
            print(f"Successfully converted {input_file} to {output_file} (simple mode)")
            return True
        except subprocess.CalledProcessError as e2:
            print(f"Simple conversion also failed: {e2}")
            return False

if __name__ == "__main__":
    if convert_to_word():
        print("Word document created successfully!")
    else:
        print("Failed to create Word document.")
