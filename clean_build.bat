@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

set LIBTORCH_USE_PYTORCH=1
set PYO3_PYTHON=C:\Users\Admin\desktop\claude\klondike-solver\.venv\Scripts\python.exe
set PATH=C:\Users\Admin\desktop\claude\klondike-solver\.venv\Scripts;C:\Users\Admin\.cargo\bin;C:\Users\Admin\desktop\claude\klondike-solver\.venv\Lib\site-packages\torch\lib;%PATH%

cd /d C:\Users\Admin\desktop\claude\klondike-solver\klondike-rs

echo === Clean rebuild ===
cargo clean 2>&1
cargo build --release -vv 2>&1 | findstr /i "cuda torch-sys cargo:rustc"
echo === Build done ===

echo === CUDA test ===
cargo test --release -p klondike-nn test_auto_device -- --nocapture 2>&1
