@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

set LIBTORCH_USE_PYTORCH=1
set LIBTORCH_BYPASS_VERSION_CHECK=1
set PYO3_PYTHON=C:\Users\Admin\desktop\claude\klondike-solver\.venv\Scripts\python.exe
set PATH=C:\Users\Admin\desktop\claude\klondike-solver\.venv\Scripts;C:\Users\Admin\.cargo\bin;C:\Users\Admin\desktop\claude\klondike-solver\.venv\Lib\site-packages\torch\lib;%PATH%

cd /d C:\Users\Admin\desktop\claude\klondike-solver\klondike-rs
echo === Building ===
cargo build --release --bin benchmark 2>&1
echo === Build done, running eval ===
cargo run --release --bin benchmark -- --neural-pimc --checkpoint checkpoints/pc_heavy_512/model_best.safetensors --hidden-dim 512 --n-blocks 4 --det 20 --sims 200 --games 200 --log-every 50 2>&1
echo === Done ===
