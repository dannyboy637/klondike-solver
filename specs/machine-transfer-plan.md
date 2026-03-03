# Machine Transfer Plan — MacBook <-> PC

## Setup: GitHub Private Repo (Do This Once)

Git is the cleanest way to sync between machines. Everything stays versioned and you can work on either machine seamlessly.

### Step 1: Initialize repo on MacBook

```bash
cd ~/claude/solitaire

# Initialize git
git init

# Create .gitignore FIRST
cat > .gitignore << 'EOF'
# Build artifacts
lonelybot/target/
klondike-env/target/

# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/
*.egg-info/
dist/
build/

# Training data & checkpoints (too large for git)
data/
checkpoints/
logs/
*.npy
*.npz
*.pt
*.pth

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/

# Rust
Cargo.lock

# Maturin build
klondike_env/*.so
klondike_env/*.dylib
klondike_env/*.pyd
EOF

# Add everything
git add .
git commit -m "Initial project: specs, research docs, Lonelybot submodule"
```

### Step 2: Add Lonelybot as a git submodule (cleaner than committing the whole repo)

```bash
# Remove the cloned lonelybot directory
rm -rf lonelybot

# Re-add as submodule
git submodule add https://github.com/vuonghy2442/lonelybot.git lonelybot
git commit -m "Add Lonelybot as git submodule"
```

### Step 3: Create private GitHub repo

```bash
# Using GitHub CLI (install: brew install gh)
gh repo create klondike-solver --private --source=. --push

# Or manually:
# 1. Go to github.com → New Repository → Private
# 2. git remote add origin git@github.com:YOUR_USERNAME/klondike-solver.git
# 3. git push -u origin main
```

### Step 4: Clone on PC

```bash
# On PC
git clone --recurse-submodules git@github.com:YOUR_USERNAME/klondike-solver.git
cd klondike-solver
```

---

## Daily Workflow: Switching Machines

### Leaving MacBook → Going to PC

```bash
# On MacBook: push your work
cd ~/claude/solitaire
git add -A
git commit -m "WIP: describe what you were working on"
git push
```

### Starting work on PC

```bash
# On PC: pull latest
cd /path/to/klondike-solver
git pull
git submodule update --init --recursive

# Rebuild anything that needs it:
cd lonelybot && cargo build --release && cd ..
cd klondike-env && maturin develop --release && cd ..
```

### Leaving PC → Going to MacBook

```bash
# On PC: push your work
git add -A
git commit -m "WIP: describe what you were working on"
git push
```

### Starting work on MacBook

```bash
# On MacBook: pull latest
cd ~/claude/solitaire
git pull
git submodule update --init --recursive

# Rebuild (binaries are platform-specific, not committed):
cd lonelybot && cargo build --release && cd ..
cd klondike-env && maturin develop --release && cd ..
```

---

## What Goes in Git vs What Stays Local

### IN GIT (synced between machines)
- All specs (`specs/`)
- All research docs (`*.md`)
- All source code (`klondike/`, `klondike-env/src/`, `scripts/`)
- Config files (`Cargo.toml`, `pyproject.toml`, `requirements.txt`)
- Test files (`tests/`)
- `CLAUDE.md`, `current_context.md`, `session_log.md`
- Lonelybot as submodule

### NOT IN GIT (stays local, rebuild per machine)
- Build artifacts (`target/`, `__pycache__/`, `.so`/`.dylib`/`.pyd`)
- Training data (`data/` — too large, regenerate on PC)
- Model checkpoints (`checkpoints/` — too large)
- Virtual environment (`.venv/` — platform-specific)
- Logs (`logs/`)

### Large files that need manual transfer (if you don't want to regenerate)
If you train a model on PC and want it on MacBook:
```bash
# Option A: Just copy the one file you need
scp user@pc:/path/to/checkpoints/best.pt ~/claude/solitaire/checkpoints/

# Option B: Use Git LFS (if checkpoints are <2GB)
git lfs install
git lfs track "*.pt"
git add .gitattributes checkpoints/best.pt
git commit -m "Add best model checkpoint"
git push

# Option C: Google Drive / Dropbox / USB drive for large files
```

---

## Per-Machine Setup (One-Time)

### MacBook Setup

```bash
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Python
python3 -m venv .venv
source .venv/bin/activate
pip install maturin numpy torch  # PyTorch with MPS support
pip install scipy  # for stats

# Build everything
cd lonelybot && cargo build --release && cd ..
cd klondike-env && maturin develop --release && cd ..
```

### PC Setup (Windows)

```powershell
# Rust
# Download from https://rustup.rs/ (rustup-init.exe)

# Python (use official installer or winget)
winget install Python.Python.3.12

# Virtual environment
python -m venv .venv
.venv\Scripts\activate

# PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install maturin numpy scipy

# Build everything
cd lonelybot; cargo build --release; cd ..
cd klondike-env; maturin develop --release; cd ..
```

### PC Setup (Linux)

```bash
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Python
sudo apt install python3.12 python3.12-venv  # or your distro's equivalent

python3 -m venv .venv
source .venv/bin/activate

# PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install maturin numpy scipy

# Build everything
cd lonelybot && cargo build --release && cd ..
cd klondike-env && maturin develop --release && cd ..
```

---

## Quick Reference: Platform Differences

| Thing | MacBook (M4) | PC (Windows/Linux) |
|-------|-------------|-------------------|
| PyTorch device | `"mps"` | `"cuda"` |
| Binary extension | `.dylib` / `.so` | `.pyd` / `.so` |
| Activate venv | `source .venv/bin/activate` | `.venv\Scripts\activate` (Win) or `source` (Linux) |
| Parallel MCTS | 8 processes | 16 processes |
| Full training | Short runs only | Full runs here |

### Handling device in code

The code should auto-detect. In `klondike/nn/trainer.py` and `pipeline.py`:

```python
import torch

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
```

---

## Transfer Checklist

### Before switching machines:
- [ ] All code changes committed and pushed
- [ ] `session_log.md` updated with what you did and where you stopped
- [ ] `current_context.md` updated with any new findings
- [ ] Note which phase/step you're on

### After arriving on new machine:
- [ ] `git pull && git submodule update --init --recursive`
- [ ] Activate virtual environment
- [ ] Rebuild Rust crates (`cargo build --release`, `maturin develop --release`)
- [ ] Run a quick smoke test (e.g., `python -c "import klondike_env"`)
- [ ] Read `session_log.md` to remember where you left off
