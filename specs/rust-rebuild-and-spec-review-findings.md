# Consolidated Findings: Rust Rebuild + Spec Review

Date: 2026-03-04
Scope: Combined findings from two requests:
1) Comprehensive Rust rebuild planning (including architecture changes)
2) Review of spec plans after Phase 4, with focus on Phase 5 and Phase 6

---

## A) Rust Rebuild Findings (Architecture + Migration)

### Current State Findings
1. The repo is already hybrid: Rust handles core game/search primitives, while Python orchestrates env/search/eval/training loops.
2. Existing Rust components are strong and reusable:
   - `lonelybot` engine/search internals
   - `klondike-env` PyO3 bridge exposing game operations
3. Current bottlenecks and complexity are mainly in orchestration/dataflow layers (Python-level loops, object crossing, script-based control plane), not in core move generation.

### Directional Findings for Speed
1. A Rust-first architecture is compatible with your speed goal and can remove cross-language overhead in hot paths.
2. Python can be fully removed from active runtime if desired, with current Python stack kept only as rollback/archive.
3. Largest performance gains are expected from:
   - Batched inference
   - Parallel self-play workers
   - Reduced serialization/object churn
   - Typed dataflow + memory-mapped storage
   Not from language switch alone.

### Proposed Architecture Findings
1. Promote Rust to full control plane:
   - Rust CLIs for benchmark/train/eval/ablations
   - Typed configs and structured logs
2. Keep `lonelybot` as domain core dependency rather than rewriting solved engine logic.
3. Split new Rust workspace by responsibility:
   - domain / encoding / search / eval / ml / pipeline / apps
4. Keep compatibility adapters only as optional boundary tools, not part of performance-critical runtime.

### Migration Findings (Execution Strategy)
1. Use parity-first migration gates (seed-locked correctness before optimization) to avoid silent regressions.
2. Port order should prioritize iteration speed:
   - actions + encoding + stats + benchmark first
   - search + data generation second
   - full training pipeline after correctness parity
3. Enforce objective cutover gates:
   - Behavior parity on fixtures
   - Win-rate CI overlap on fixed seed sets
   - Throughput target vs current pipeline
4. Preserve current Python path untouched during transition for immediate fallback.

---

## B) Spec Review Findings (Phase 5/6 + Alignment)

### P0 Findings
1. **Phase 5 core acceptance is currently unachievable as written.**
   - Core sign-off requires manual input + advisor output, but legal-mask path remains unimplemented/deferred.
   - References:
     - `specs/phase-5-deployment.md:211`
     - `specs/phase-5-deployment.md:217`
     - `specs/phase-5-deployment.md:334`
     - `specs/phase-5-deployment.md:343`

2. **Phase 5 advisor is specified as greedy policy inference, which can underperform search-backed play and conflict with >50% objective.**
   - References:
     - `specs/phase-5-deployment.md:154`
     - `specs/phase-5-deployment.md:190`
     - `specs/phase-6-iteration-optimization.md:69`
     - `specs/phase-6-iteration-optimization.md:79`
     - `session_log.md:31`

### P1 Findings
3. **Phase 6 auxiliary-target ablation is inconsistent with current implementation state.**
   - Spec says auxiliary targets are currently used; implementation defines aux heads but training loss does not use aux terms.
   - References:
     - `specs/phase-6-iteration-optimization.md:143`
     - `klondike/nn/model.py:72`
     - `klondike/alphazero/pipeline.py:190`

4. **Rule-fidelity verification sample size in Phase 5 is too small for deployment-critical confidence.**
   - Current threshold (10 games) is unlikely to catch rare rule/timer/disconnection edge cases.
   - References:
     - `specs/phase-5-deployment.md:327`
     - `specs/phase-5-deployment.md:338`

5. **Specs remain Python/PyO3-centric and are misaligned with strict Rust-only direction.**
   - Phase definitions and end-state tree still bind future work to Python components.
   - References:
     - `specs/00-overview.md:36`
     - `specs/00-overview.md:97`
     - `specs/phase-5-deployment.md:69`
     - `specs/phase-5-deployment.md:140`

### P2 Findings
6. **Repeated tuning on a single tuning set (test_set_C) risks tuning-set overfit.**
   - Good A/B discipline exists for final claims, but C-only repeated sweeps still bias selection.
   - References:
     - `specs/phase-6-iteration-optimization.md:61`
     - `specs/phase-6-iteration-optimization.md:247`

7. **Spec snippet defect: `np.mean` is used without a visible numpy import in advisor code block.**
   - Reference:
     - `specs/phase-5-deployment.md:241`

---

## C) Cross-Cutting Consolidated Findings

1. Phase 4 implementation exists and is functioning, but it is currently a smoke-tested stage rather than a final-performance stage.
   - Reference: `session_log.md:27`
2. The project is at the right point to re-baseline around a Rust-only roadmap before Phase 5/6 implementation.
3. If Phase 5 proceeds, deployment recommendation quality should be explicitly tied to search-backed policy (not greedy-only policy).
4. Phase 6 should be rewritten with Rust-native deliverables, acceptance criteria, and experiment harnesses to avoid architectural drift.

