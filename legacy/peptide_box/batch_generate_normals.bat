@echo off
setlocal enabledelayedexpansion

REM Read list from ids.txt
set count=0
for /f "usebackq tokens=* delims=" %%A in ("ids.txt") do (
    set /a count+=1
    set id[!count!]=%%A
)

echo Found %count% IDs in ids.txt
echo.

REM Main processing loop
for /L %%i in (1,1,%count%) do (
    set "pid=!id[%%i]!"
    set "boxes=output\!pid!_boxes.pdb"
    set "src=input_data\!pid!.pdb"

    echo [%%i/%count%] Processing !pid!

    if exist "!boxes!" (
        if exist "!src!" (
            python generate_normals_pml.py ^
                --boxes "!boxes!" ^
                --src "!src!" ^
				--out "qc\!pid!_overlay.pml" ^
                --angle-thresh 8
        ) else (
            echo   [MISSING SOURCE] !src!
        )
    ) else (
        echo   [MISSING BOXES] !boxes!
    )
    echo.
)

echo All %count% IDs processed!
pause
