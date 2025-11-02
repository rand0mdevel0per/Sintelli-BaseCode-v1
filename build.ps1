# PowerShell script to automate the build process using CMake and Visual Studio

Write-Host "Starting build process..."
Write-Host "Cleaning previous build artifacts..."
Remove-Item -Recurse -Force E:\Sintelli\src\cmake-build-debug-visual-studio
Remove-Item -Force E:\Sintelli\compile.log
Write-Host "Configuring the project with CMake..."
Write-Host "# D:\CMake\bin\cmake.exe -G "Visual Studio 17 2022" -DCMAKE_TOOLCHAIN_FILE=D:\VisualStudio2026\VC\vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DPython_EXECUTABLE=D:/Python313/python.exe -DPython3_EXECUTABLE=D:/Python313/python.exe -S E:\Sintelli\src -B E:\Sintelli\src\cmake-build-debug-visual-studio"
D:\CMake\bin\cmake.exe -G "Visual Studio 17 2022" -DCMAKE_TOOLCHAIN_FILE=D:\VisualStudio2026\VC\vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DPython_EXECUTABLE=D:/Python313/python.exe -DPython3_EXECUTABLE=D:/Python313/python.exe -S E:\Sintelli\src -B E:\Sintelli\src\cmake-build-debug-visual-studio
Write-Host "Building the project..."
Write-Host "Build output will be logged to compile.log"
Write-Host "# D:\CMake\bin\cmake.exe --build E:\Sintelli\src\cmake-build-debug-visual-studio --target src --config Debug"
D:\CMake\bin\cmake.exe --build E:\Sintelli\src\cmake-build-debug-visual-studio --target src --config Debug >> E:\Sintelli\compile.log 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed. Check compile.log for details."
    Write-Debug "Errors:"
    Get-Content ./compile.log | findstr -i error
    Write-Host "Exiting..."
    exit $LASTEXITCODE
} else {
    Write-Host "Build succeeded."
    exit 0
}