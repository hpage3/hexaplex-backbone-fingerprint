@echo off
:: ---------------------------------------------------------
:: getpdb.bat — Download PDB + box + arrows from Apollo2 server
:: Usage:  getpdb <PDBID>
:: Example: getpdb 9l47
:: ---------------------------------------------------------

if "%~1"=="" (
    echo Usage: getpdb PDBID
    exit /b 1
)

set PDBID=%~1
set SERVER=hpage3@apollo2.chemistry.gatech.edu
set REMOTE_BASE=/home/hpage3/theta_pp/peptide_box

echo Downloading files for %PDBID% ...

:: Try downloading PDB and CIF
scp %SERVER%:%REMOTE_BASE%/input_data/%PDBID%.pdb %PDBID%.pdb
scp %SERVER%:%REMOTE_BASE%/input_data/%PDBID%.cif %PDBID%.cif

:: Download boxing visualization
scp %SERVER%:%REMOTE_BASE%/output/%PDBID%_boxes.pdb %PDBID%_boxes.pdb

:: Download arrow overlay if present
scp %SERVER%:%REMOTE_BASE%/output/%PDBID%_arrows.pml %PDBID%_arrows.pml

:: Generate PyMOL loader script
echo load %PDBID%.pdb, protein > load_%PDBID%.pml
echo load %PDBID%_boxes.pdb, boxes >> load_%PDBID%.pml
echo run %PDBID%_arrows.pml >> load_%PDBID%.pml
echo show cartoon, protein >> load_%PDBID%.pml
echo show sticks, boxes >> load_%PDBID%.pml
echo util.cbc >> load_%PDBID%.pml
echo color red, boxes >> load_%PDBID%.pml

echo.
echo ---------------------------------------------------------
echo ✅ Files downloaded. To load in PyMOL, type:
echo
echo @load_%PDBID%.pml
echo ---------------------------------------------------------
pause
