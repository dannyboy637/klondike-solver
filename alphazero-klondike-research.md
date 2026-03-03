# Building an AlphaZero-style Klondike solver under imperfect information

**The most viable path to >50% win rate on draw-3 Klondike combines AlphaZe\*\*'s PC-PIMC framework with Lonelybot's Rust engine via PyO3 bindings, training a ~2M-parameter ResNet on your RTX 4060 in 1–4 weeks.** This approach works because Klondike's hidden information is non-adversarial — the cards don't adapt to your strategy — which eliminates PIMC's known theoretical weaknesses (strategy fusion, exploitability). The AlphaZe\*\* paper (Blüml, Czech & Kersting, 2023) proved this architecture competitive on Stratego using only **3 determinizations per move**, and Klondike's simpler information structure means even fewer samples may suffice. Your hardware is adequate: the network fits comfortably in <1 GB VRAM, leaving ample room for batched inference during MCTS self-play.

---

## PIMC determinization is unusually well-suited to solitaire

Perfect Information Monte Carlo works by sampling possible arrangements of hidden cards consistent with observations, solving each determinized world with a perfect-information solver, then aggregating results via policy or value averaging. For Klondike, this means taking the set of **unseen cards** (52 minus all face-up tableau cards, waste pile cards, and foundation cards), randomly assigning them to face-down tableau positions and unobserved stock positions, and running MCTS on each resulting "Thoughtful Solitaire" instance.

The canonical critique of PIMC is **strategy fusion**, formalized by Frank & Basin (1998) in "Search in Games with Incomplete Information." Strategy fusion occurs when PIMC believes it can use a different strategy in each determinized world, but the player must actually commit to a single action across all indistinguishable states. In adversarial games, this lets opponents exploit the resulting policy. Long, Sturtevant, Buro & Furtak (2010) identified three measurable properties that predict PIMC failure: **leaf correlation** (how similar payoffs are across worlds), **bias** (asymmetry in payoff structure), and **disambiguation factor** (how much information is revealed during play). Card games like Skat and Hearts land in the PIMC-friendly zone — correlation ~0.8–1.0, disambiguation ~0.6 — losing only **0.063 tournament points per deal** versus a Nash equilibrium solution.

**Klondike is even more favorable than trick-taking games for PIMC.** There is no adversary exploiting information asymmetry. The hidden cards are randomly placed by the shuffle, not strategically hidden by an opponent. Progressive card revelation (flipping face-down cards) provides high disambiguation. The single known failure mode in Klondike is **optimistic bias**: Bjarnason, Fern & Tadepalli (2009) showed that HOP (Hindsight Optimization) is "overly optimistic" because determinization makes future card flips correlated — the solver "knows" which card will appear next in each world, making all choices look equally viable when they're not. This bias doesn't prevent PIMC from working; it just means raw PIMC slightly overestimates win probability.

Three mitigations for this optimistic bias are established in the literature. **ISMCTS** (Cowling, Powley & Whitehouse, 2012) builds a single tree over information sets rather than game states, aggregating statistics across determinizations and preventing the algorithm from making different decisions in indistinguishable states. It was commercialized in AI Factory's Spades, winning **64%** against knowledge-based AI with 2,500 iterations in <0.25 seconds on mobile hardware. **RecPIMC** (Furtak & Buro, 2013) finishes games using PIMC as a recursive playout module rather than a perfect-info solver, with 1,600 top-level worlds and 10 second-level worlds breaking through PIMC's saturation ceiling in Skat. **EPIMC** (Arjonilla, Cazenave & Saffidine, 2024) postpones the perfect-info evaluator to depth d, provably reducing strategy fusion monotonically with depth — at depth 3 with 1,000 seconds, it achieved near-**100%** win rates in Dark Chess and Dark Hex.

For practical determinization counts, Bridge AI (GIB) uses **50 fixed determinizations**, Skat's Kermit saturates around **100–200 worlds**, and ISMCTS in Dou Di Zhu shows rapid improvement up to ρ < 20 with diminishing returns after. AlphaZe\*\* uses only **3–12 determinizations** per move when combined with neural network guidance. For Klondike, start with **50–100 determinizations** during evaluation and **3–5 during training** (where speed matters more than precision).

---

## AlphaZe\*\* provides the simplest lift from AlphaZero to imperfect information

The AlphaZe\*\* paper (Frontiers in AI 6, 2023; code at github.com/BluemlJ/AlphaZe, now part of CrazyAra) makes the minimal change needed: it replaces MCTS in AlphaZero with **Policy Combining PIMC (PC-PIMC)**. At each decision point, sample n determinized worlds consistent with current observations, run standard MCTS (with neural network guidance) on each as a perfect-info game, obtain per-world policies π₁...πₙ, then average them: **π_combined = (1/n) × Σπᵢ**. This averaged policy becomes both the action distribution for play and the training target for the network.

The critical architectural detail: **the neural network sees only the current observation (visible cards, pile structure, stock state), NOT the determinized state.** Determinized states are used internally by MCTS for planning only. This forces the network to generalize across the information set. AlphaZe\*\* used CrazyAra's **RISEv2 mobile architecture: 13 residual blocks × 256 channels** with pre-activation, depthwise separable convolutions, and Squeeze-and-Excitation layers, outputting dual policy/value heads.

**TrueSight Learning (TSL)** is a bootstrapping trick from the same paper: for the first N training epochs, the network receives the **true world state** (all hidden information revealed), effectively training on Thoughtful Solitaire. After N epochs, it switches to PC-PIMC with sampled determinizations. This mirrors how humans learn card games — cards face-up first. For Klondike, this means training Phase 1 on Thoughtful Solitaire (where Lonelybot already achieves ~82% solve rate), then Phase 2 with hidden cards.

For single-player adaptation, several changes from the standard two-player AlphaZero loop are needed. MCTS becomes pure expected-value maximization — no minimax, no sign-flipping at opponent nodes, just maximizing at every node. **Q-value initialization** is critical: initialize Q(s,a) to the network's value prediction V(s), not to 0 (from tmoer/alphazero_singleplayer). In two-player games with binary outcomes, 0 is a neutral value; in single-player stochastic games, initializing at 0 creates pathological exploration. The value target is simply binary win/loss — Klondike's ~50% winnability provides naturally balanced training signal without needing Ranked Reward (R2). Chance nodes (drawing unknown cards, revealing face-down cards) use expectimax: average over possible outcomes weighted by probability.

For comparison, **DeepNash** (Perolat et al., Science 2022) achieved 97% win rate against all existing Stratego bots and 84% against human experts, but used Regularized Nash Dynamics with **no search at all** — purely model-free, DeepMind-scale distributed training on hundreds of TPUs. **Player of Games** (Schmid et al., 2021) uses Growing-Tree CFR with a Counterfactual Value-and-Policy Network, handling both perfect and imperfect info but at lower strength than specialized algorithms. **ReBeL** (Brown & Bakhtin, NeurIPS 2020) transforms imperfect-info games into continuous-state perfect-info games via Public Belief States and CFR subgame solving. None of these are practical on consumer hardware. AlphaZe\*\*'s PC-PIMC is the only approach demonstrated to work at moderate scale (single GPU feasible).

---

## Why MuZero is theoretically elegant but practically inferior here

**Stochastic MuZero** (Antonoglou et al., ICLR 2022) is the most relevant MuZero variant for Klondike. It factors transitions into deterministic action-conditioned steps (afterstate dynamics φ) and stochastic chance transitions (conditioned on VQ-VAE chance codes). The MCTS tree alternates between decision nodes and chance nodes. It achieved state-of-the-art on **2048** (a stochastic single-player game, the closest analog to Klondike in the MuZero literature) without domain knowledge, and matched MuZero on Go.

However, MuZero's representation function maps observations to latent states via a **learned, deterministic compression** — it does not explicitly maintain belief distributions over hidden states. For Atari, observation stacking (32 frames) provides sufficient history. For Klondike's combinatorial hidden information (up to 21 unknown face-down cards from a remaining deck), the latent state must implicitly encode card probabilities, which is fundamentally harder. The muzero-general GitHub issue #130 acknowledges this limitation, recommending observation stacking as a practical but theoretically limited workaround.

The compute disparity is decisive. MuZero requires learning three functions (representation, dynamics, prediction) end-to-end versus AlphaZero's single policy-value network with an existing simulator. **LightZero's benchmarks** (NeurIPS 2023 Spotlight) confirm: "In board game environments, AlphaZero's sample efficiency greatly exceeds that of MuZero. Employing AlphaZero directly is advantageous when an environment simulator is available." DeepMind's original MuZero used 16 TPUv3 for training plus 1,000 TPUs for self-play. EfficientZero brought this down to 4×20GB GPUs for 7 hours per Atari game, but that's still image-based learning from scratch. A Stochastic MuZero for Klondike would need to learn 52-card deck dynamics from experience — estimated **3–7 days** for basic play, potentially weeks for strong play on an RTX 4060.

**MuZero Unplugged** (Schrittwieser et al., NeurIPS 2021) offers one useful capability: learning from offline data via the Reanalyse algorithm, which revisits past episodes with the current model to compute improved training targets. If you have game logs from Lonelybot's MCTS solver, MuZero Unplugged could bootstrap from them. But this advantage doesn't outweigh the complexity overhead.

Open-source implementations worth referencing: **muzero-general** (werner-duvaud, PyTorch + Ray, tested on GTX 1050Ti), **LightZero** (OpenDILab, C++ MCTS, unified AlphaZero/MuZero/EfficientZero), **DHDev0/Stochastic-muzero** (PyTorch, supports Gymnasium environments). **BetaZero** (Moss et al., Stanford, 2024) extends AlphaZero to POMDPs by performing MCTS in belief space with particle-filter-based belief representations — theoretically closest to what you need but implemented in Julia (POMDPs.jl ecosystem), not Python.

---

## State representation requires encoding both certainty and uncertainty

No published neural network encoding for Klondike exists. The closest precedent is **Hanabi's canonical encoding** from DeepMind's Hanabi Learning Environment: **658 binary values** for a 2-player game, where each unknown card gets a **25-bit possibility mask** indicating which identities remain consistent with observations, plus explicit bits for revealed color/rank knowledge. The key design principle — encode what each unknown card *could* be, not just that it's unknown — transfers directly to Klondike.

Four encoding proposals emerge from synthesizing the card game literature, ordered by complexity:

**Proposal A (card-centric flat vector, ~1,019 dims):** For each of 52 cards, encode location (foundation, tableau pile 1–7, waste, stock, unknown) and visibility (face-up/face-down/unseen) as binary features. Add global features: foundation top ranks (4 × 14 one-hot = 56 bits), tableau face-down counts (7 × 7 one-hot = 49 bits), stock remaining (unary, 24 bits), waste top cards (3 × 53 one-hot = 159 bits). This is the simplest approach and mirrors Skat's encoding (Rebstock thesis, U. Alberta, 2019).

**Proposal B (belief-augmented, ~3,432 dims):** For each face-down position and stock position, encode a probability vector over possible card identities: P(card c) = 1/|unseen cards| for each unseen card c, 0 for located cards. This is the maximum-entropy prior and provides the network with explicit uncertainty information. The network must learn correlations (if card X is in position A, it can't be in position B) but receives the full posterior. At **14 KB per state, batch size 8,192 uses only ~112 MB** — trivial for 8 GB VRAM.

**Proposal D (recommended hybrid, ~857 dims):** Binary card-centric encoding (~700 values) plus an uncertainty summary (~157 values): the 52-bit unseen card set mask, plus per-pile belief hints (fraction of unseen cards that are red/black for alternating-color constraints, rank distribution for foundation readiness). At **3.4 KB per state**, batch size 4,096 uses ~14 MB.

The choice between **MLP and CNN** architectures is clear: card games lack the spatial locality that makes CNNs powerful for Go/Chess. A 13×4 grid (ranks × suits) is too small for convolutions to exploit meaningful spatial patterns. **MLP with residual blocks** is recommended: Input(857) → Linear(512) + ReLU + BatchNorm → 4 Residual Blocks(512) → Policy Head(|actions|) + Value Head(1). Parameter count: **~2M parameters, ~8 MB in FP32**. With FP16 and batch size 1,024, total VRAM usage is approximately 100 MB — **1.2% of your 8 GB**.

An alternative **Transformer architecture** treats each card as a token (52 cards × 66 features → 128-dim embeddings, 4 layers, 4 heads) and naturally captures card-card interactions via attention ("this card blocks that card"). This is more expensive but may be worth testing if MLP performance plateaus. Research from ICML 2023 (Wang et al.) shows that RNNs trained on observation sequences in POMDPs naturally develop internal representations correlating with Bayesian beliefs — suggesting that an **LSTM history encoder** concatenated to the state encoding could implicitly learn belief states without explicit probability vectors.

---

## Training data generation exploits solitaire's natural curriculum

The self-play equivalent for single-player games generates random deals, plays through them using MCTS + neural network, collects (state, MCTS policy, game outcome) triples, and trains on these. The **Ranked Reward (R2)** method (Laterre et al., 2018) creates a virtual opponent by binarizing outcomes: if a game's result exceeds the α-th percentile (typically 75th) of the last L results (typically L=250), label it +1, otherwise -1. However, Klondike's natural binary win/loss signal with ~50% winnability makes R2 unnecessary — **direct win/loss targets provide balanced, informative training signal** without additional engineering.

The recommended training loop from the Morpion Solitaire AlphaZero paper (Wang et al., 2020) and Expert Iteration (Anthony, Tian & Barber, NeurIPS 2017) is:

1. Generate 200–500 random deals per iteration
2. For each deal, play using MCTS (100–400 simulations/move) with current network + PIMC (3–5 determinizations)
3. Record (state, MCTS visit distribution, game outcome) for every move
4. Add to replay buffer (100K–500K positions)
5. Sample 10–50 mini-batches (256–1,024 each), train with loss = MSE(v_pred, z) + CrossEntropy(p_pred, π) + L2 regularization
6. Evaluate on fixed test set of 1,000+ deals every 5 iterations

**Curriculum learning is strongly recommended.** Start with **Phase 1: Thoughtful Solitaire** (all cards visible, perfect information) to validate the pipeline and teach basic strategy. Lonelybot solves ~82% of Thoughtful games, providing a strong baseline. Then **Phase 2: gradually introduce hidden information** — reduce the number of visible face-down cards from all to none, or switch from Thoughtful to PIMC-based play. This mirrors AlphaZe\*\*'s TrueSight Learning. A more aggressive curriculum starts from near-won positions (few cards remaining) and works backward, similar to DeepCubeA's backward training for Rubik's Cube (Agostinelli et al., Nature Machine Intelligence 2019), which achieved 100% solve rate using approximate value iteration on states of increasing scramble distance.

From comparable domains, estimated training games needed:

- Morpion Solitaire (R2+AlphaZero): 5,000 games, ~1 week on GPU server
- Peg Solitaire (A2C): ~11,000 games, 53 minutes on CPU
- 9×9 Gomoku (AlphaZero): ~2,000–3,000 games, ~2 days single PC
- KataGo 9×9 Go: ~4.2 million games, 19 days on 28 V100s

**Estimate for Klondike: 50,000–500,000 training games**, with 100,000 as a reasonable middle target. Temperature annealing follows AlphaZero: τ = 1.0 for the first 15–20 moves (exploration), then τ → 0 (greedy). No network gating is needed — in single-player games, there's no risk of "forgetting how to beat yourself," so continuous updates are safe.

---

## Practical implementation on your hardware

**Framework: Custom PyTorch, based on alpha-zero-general.** Stable Baselines3 doesn't support MCTS. RLlib's AlphaZero contrib is buggy (multiple unresolved GitHub issues #13177, #14477, #32228). CleanRL lacks MCTS implementations. The best starting points are **alpha-zero-general** (suragnair, 4.4k stars, clean Game/NeuralNet abstractions), **kevaday/alphazero-general** (enhanced fork with Cython MCTS — **30× faster** than pure Python), and **LightZero** (OpenDILab, NeurIPS 2023 Spotlight, C++ MCTS, unified AlphaZero/MuZero under one API). Reference **KataGo** for optimization tricks:

- **Playout cap randomization**: Most moves use fast search (100 playouts), occasional full search (600 playouts). Only full-search moves train policy. Value net benefits from more games. This resolves the "tension between policy and value training."
- **Auxiliary targets**: Predict foundation card count, face-down cards remaining — not just win/loss. "Whenever a target can be expressed as a sum of sub-events, predicting those subevents provides sharper gradients."
- **Drop batch normalization**: KataGo uses Fixup initialization instead, yielding 1.6–1.8× training speedup and eliminating train/eval mode differences.
- **Stochastic weight averaging**: Snapshot every 250K samples, average every 4 snapshots.

**Rust-Python interop via PyO3 is the clear winner.** Function call overhead is **~60–70 ns** (measured benchmarks from PyO3 issue #1607), versus ~1–10 ms for subprocess IPC. Build with maturin (`pip install maturin; maturin develop`). The pattern for exposing Lonelybot:

```rust
#[pyclass]
struct KlondikeEnv { game: lonelybot::Game }

#[pymethods]
impl KlondikeEnv {
    fn step(&mut self, action: usize) -> (Vec<f32>, f32, bool) { /* obs, reward, done */ }
    fn get_legal_actions(&self) -> Vec<usize> { /* ... */ }
    fn reset(&mut self, seed: u64) { /* ... */ }
}
```

Use `py.allow_threads(|| ...)` to release the GIL during Rust computation, enabling true multi-threading across self-play workers. Build a **vectorized environment** with rayon's thread pool for batch stepping — essentially a mini-EnvPool in Rust. Lonelybot already has a `python/` directory suggesting some interop work has begun. Key modifications needed: PyO3 bindings, observation vector extraction (~857 features), action space definition, and PIMC sampling (enumerate/sample plausible hidden card configurations).

**Compute estimates for your RTX 4060 + Ryzen 5700X:**

| Parameter | Estimate |
|-----------|----------|
| Network size | 4–10 ResNet blocks × 128–256 channels, ~2M params |
| VRAM usage (FP16, batch 512) | <1 GB |
| Games per hour (400 MCTS sims/move) | 100–900 |
| Training games needed | 50K–500K |
| Optimistic timeline | ~6 days (50K games × 10s/game) |
| Realistic timeline | **2–4 weeks** (200K games × 15s/game) |
| Conservative timeline | ~16 weeks (500K games × 20s/game) |

Optimization priorities, ordered by impact: (1) PyO3 bindings with GIL release, (2) FP16 mixed precision via `torch.amp` (2× speedup, 50% VRAM savings), (3) batched NN inference during MCTS (batch 16–64 leaf evaluations), (4) async pipeline overlapping CPU self-play with GPU training, (5) `torch.compile(mode='max-autotune')` for 20–40% inference speedup, (6) CUDA graphs for fixed-shape inference during MCTS (2–4× latency reduction for small batches), and (7) KataGo's playout cap randomization for more games per compute unit.

---

## Conclusion: a concrete path to >50% win rate

The architecture that maximizes your probability of success combines **AlphaZe\*\*'s PC-PIMC** (3–5 determinizations during training, 50–100 during evaluation) with Lonelybot's Rust MCTS engine, a **~2M-parameter MLP-ResNet** on the ~857-dim hybrid state encoding, and a **two-phase curriculum** (Thoughtful Solitaire → imperfect information). This approach is theoretically justified: PIMC's failure modes are specific to adversarial settings, and Klondike's non-adversarial hidden information makes determinization nearly ideal. The current SOTA of **47.58%** from Lonelybot's HOP/MCTS uses no learned policy — adding a trained neural network to guide search should provide the marginal improvement needed to cross 50%.

Three novel insights emerge from this research. First, Klondike may be the *ideal* testbed for AlphaZe\*\*-style methods because it combines imperfect information (benefiting from PIMC) with single-player structure (eliminating strategy fusion concerns entirely). Second, the belief-augmented state encoding (Proposal C/D) combined with an LSTM history encoder could provide implicit Bayesian inference over hidden cards without explicit particle filtering — a capability no existing Klondike solver exploits. Third, the auxiliary prediction targets from KataGo (predicting foundation progress, face-down counts) could provide dramatically sharper gradients than binary win/loss alone, potentially reducing training games needed by an order of magnitude. The biggest engineering risk is not the algorithm but the **Rust-Python integration latency** in the MCTS inner loop; batching NN queries and releasing the GIL during Rust computation are essential to keeping throughput high enough for practical training timelines.