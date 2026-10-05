@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (echo VCVARS_FAILED & exit /b 1)
"C:\Program Files\CMake\bin\cmake.exe" -G Ninja -DCMAKE_MAKE_PROGRAM="C:\Users\aryan\miniconda3\Scripts\ninja.exe" -S "D:\code\Strata" -B "D:\code\Strata\build" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe" -DSTRATA_GGML_DIR="D:\code\Strata\third_party\llama.cpp"
if errorlevel 1 (echo CONFIGURE_FAILED & exit /b 1)
"C:\Program Files\CMake\bin\cmake.exe" --build "D:\code\Strata\build" --target strata -j 6
if errorlevel 1 (echo BUILD_FAILED & exit /b 1)
echo BUILD_OK
