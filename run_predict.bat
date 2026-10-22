@echo off
set DATA=%~1
if "%DATA%"=="" set DATA=.
python predict.py --data "%DATA%" --weights weights --out submission_reproduced
pause
