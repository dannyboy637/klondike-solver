@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

set LIBTORCH_USE_PYTORCH=1
set LIBTORCH_BYPASS_VERSION_CHECK=1
set PYO3_PYTHON=C:\Users\Admin\desktop\klondike-solver\.venv\Scripts\python.exe
set PATH=C:\Users\Admin\desktop\klondike-solver\.venv\Scripts;C:\Users\Admin\.cargo\bin;C:\Users\Admin\desktop\klondike-solver\.venv\Lib\site-packages\torch\lib;%PATH%

cd /d C:\Users\Admin\desktop\klondike-solver\klondike-rs
cargo run --release --bin train -- ^
  --iterations 10 ^
  --games-per-iter 50 ^
  --det 5 ^
  --sims 200 ^
  --hidden-dim 256 ^
  --n-blocks 2 ^
  --batch-size 512 ^
  --buffer-capacity 200000 ^
  --eval-every 3 ^
  --eval-games 50 ^
  --min-buffer 2000 ^
  --checkpoint-dir checkpoints/pc_heavy
