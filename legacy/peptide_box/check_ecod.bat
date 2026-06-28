@echo off
setlocal enabledelayedexpansion

REM === Adjust this path to point to your ECOD domains file ===
set ECOD_FILE=..\HHSearch\ecod.latest.domains_2023.txt

echo Checking architectures for IDs in ids.txt...
echo.

for /f %%i in (ids.txt) do (
    echo ==== %%i ====
    REM Search for PDB in ECOD file
    findstr /i "%%i" "%ECOD_FILE%" > tmp_match.txt

    REM Now filter for alpha/beta
    findstr /i "alpha beta" tmp_match.txt > tmp_ab.txt

    if exist tmp_ab.txt (
        for /f "usebackq delims=" %%l in (`type tmp_ab.txt`) do (
            echo %%l
        )
    )

    REM If no alpha/beta found but there were hits, report
    if %errorlevel% neq 0 (
        if exist tmp_match.txt (
            echo [WARNING] %%i has ECOD entries but none contain "alpha" or "beta"
        ) else (
            echo [MISSING] %%i not found in ECOD file
        )
    )

    echo.
)

del tmp_match.txt 2>nul
del tmp_ab.txt 2>nul

echo Done.
pause
