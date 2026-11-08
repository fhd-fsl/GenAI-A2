@echo off
REM ============================================================================
REM One-Click Report Compilation Script (MiKTeX pdflatex + bibtex)
REM ============================================================================

echo ============================================================================
echo   Compiling IEEE Research Paper: TrialMatch
echo ============================================================================

cd /d "%~dp0"

echo [*] Step 1: First pdflatex pass...
pdflatex -interaction=nonstopmode main.tex

echo [*] Step 2: Processing bibliography with bibtex...
bibtex main

echo [*] Step 3: Second pdflatex pass (resolving citations)...
pdflatex -interaction=nonstopmode main.tex

echo [*] Step 4: Final pdflatex pass (cross-references and page numbers)...
pdflatex -interaction=nonstopmode main.tex

if exist main.pdf (
    copy /Y main.pdf GenAI_Assignment2_Report.pdf
    echo ============================================================================
    echo   [SUCCESS] Report compiled successfully to:
    echo   %~dp0GenAI_Assignment2_Report.pdf
    echo ============================================================================
) else (
    echo ============================================================================
    echo   [ERROR] Compilation failed. Check main.log for details.
    echo ============================================================================
)

pause
