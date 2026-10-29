@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

set LIBTORCH_USE_PYTORCH=1
set PATH=C:\Users\Admin\claude\_archive\klondike-solver\.venv\Scripts;C:\Users\Admin\.cargo\bin;C:\Users\Admin\claude\_archive\klondike-solver\.venv\Lib\site-packages\torch\lib;%PATH%

echo === Checking torch lib DLLs ===
dir "C:\Users\Admin\claude\_archive\klondike-solver\.venv\Lib\site-packages\torch\lib\torch_cuda*.dll"

echo === Checking CUDA DLLs ===
dir "C:\Users\Admin\claude\_archive\klondike-solver\.venv\Lib\site-packages\torch\lib\cudart*.dll"
dir "C:\Users\Admin\claude\_archive\klondike-solver\.venv\Lib\site-packages\torch\lib\cublas*.dll"
dir "C:\Users\Admin\claude\_archive\klondike-solver\.venv\Lib\site-packages\torch\lib\nvrtc*.dll"

echo === Python CUDA check ===
C:\Users\Admin\claude\_archive\klondike-solver\.venv\Scripts\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo === Running tch-rs CUDA test ===
cd /d C:\Users\Admin\claude\_archive\klondike-solver\klondike-rs
set TORCH_CUDA_ARCH_LIST=8.9
target\release\deps\klondike_nn-721774db80068b3e.exe --test-threads=1 test_auto_device --nocapture 2>&1
