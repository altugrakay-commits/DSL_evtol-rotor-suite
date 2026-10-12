@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
del rotor.dll rotor.exp rotor.lib src\*.obj 2>nul
cl /nologo /std:c17 /O2 /W3 /Isrc /DROTOR_API_BUILD /D_CRT_SECURE_NO_WARNINGS /LD src\bem.c src\noise.c src\structure.c src\rotor_api.c /Fe:rotor.dll
dumpbin /headers rotor.dll | findstr machine
dumpbin /exports rotor.dll | findstr rotor_eval