# Final Cleanup Patch List

This document captures the minimal cleanup actions needed to fully close the remaining spec gaps.

## 1. Wire deterministic PIMC seeds through all call sites

**Files**:
- `/Users/Daniel/claude/solitaire/specs/phase-2-search-evaluation.md`
- `/Users/Daniel/claude/solitaire/specs/phase-3-supervised-learning.md`

Change shared interface to pass context:

```python
SearchFn = Callable[[KlondikeEnv, int, int], SearchResult]  # env, game_seed, move_num
```

In benchmark loop, call:

```python
result = search_fn(env, seed, n_moves)
```

In ablations/data generation, pass seed/move count:

```python
search_fn = lambda env, s, m: pimc_search(env, pimc_config, game_seed=s, move_num=m)
result = pimc_search(env, SEARCH_CONFIG, game_seed=seed, move_num=n_moves)
```

## 2. Make uncertainty features real (not TODO zeros)

**File**:
- `/Users/Daniel/claude/solitaire/specs/phase-1-pyo3-bridge.md`

Replace placeholder uncertainty-feature comments with explicit baseline logic (even simple priors), e.g., compute unseen red fraction and unseen average rank, then fill per-pile features deterministically.

## 3. Fix final-save RAM spike in data generation

**File**:
- `/Users/Daniel/claude/solitaire/specs/phase-3-supervised-learning.md`

Replace `np.array(memmap_slice)` truncation with streaming copy into correctly sized memmaps:

```python
obs_final = np.lib.format.open_memmap(
    path / "observations_final.npy",
    mode="w+",
    dtype=np.float32,
    shape=(pos_idx, OBSERVATION_DIM),
)
obs_final[:] = obs_mm[:pos_idx]
del obs_final
# Repeat for policies/masks/values, then atomically rename *_final.npy
```

## 4. Align aux-loss claims with implementation

**File**:
- `/Users/Daniel/claude/solitaire/specs/phase-3-supervised-learning.md`

Pick one:
- Implement auxiliary targets end-to-end now, or
- Remove aux term from loss docs/config for Phase 3 and defer to Phase 6.

Minimal safe option: defer and remove from Phase 3 loss formula and acceptance criteria.

## 5. Fix Phase 3 runtime estimate consistency

**File**:
- `/Users/Daniel/claude/solitaire/specs/phase-3-supervised-learning.md`

Replace the fixed `2-4 sec/game` estimate with conditional estimates:
- Fast path (vectorized/all-moves-at-once HOP): `X-Y`
- Per-move HOP path: much slower (minutes/game)

## 6. Fix Python MCTS returned value metric

**File**:
- `/Users/Daniel/claude/solitaire/specs/phase-2-search-evaluation.md`

Do not return `root.value` unless root accumulates value.

Either backprop into root too, or compute final value from children:

```python
total = sum(c.visit_count for c in root.children.values())
value = sum(c.value * c.visit_count for c in root.children.values()) / max(total, 1)
```

## 7. Optional: Prevent Phase 5 "done" ambiguity

**File**:
- `/Users/Daniel/claude/solitaire/specs/phase-5-deployment.md`

Keep `NotImplementedError`/`pass`, but mark those sections as scaffolding and remove them from strict acceptance criteria until implemented.

