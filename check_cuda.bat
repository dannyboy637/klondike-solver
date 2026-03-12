@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

set LIBTORCH_USE_PYTORCH=1
set LIBTORCH_BYPASS_VERSION_CHECK=1
set PATH=C:\Users\Admin\desktop\klondike-solver\.venv\Scripts;C:\Users\Admin\.cargo\bin;C:\Users\Admin\desktop\klondike-solver\.venv\Lib\site-packages\torch\lib;%PATH%

cd /d C:\Users\Admin\desktop\klondike-solver\klondike-rs
cargo test --release -p klondike-nn test_auto_device -- --nocapture 2>&1
