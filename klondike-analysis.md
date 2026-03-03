# Klondike Solitaire at 2× payout is a losing bet

**A skilled player without undo wins roughly 33–43% of Klondike draw-1 games with unlimited passes — well below the 50% threshold needed for a 2× payout to break even.** The gap between theoretical solvability (~82–91%) and what humans actually achieve (~33–43%) is enormous, and that gap is precisely what makes this game profitable for any casino offering it. The "no undo" rule is the single most important mechanism converting a potentially beatable game into a reliable house earner. Even the world's best Klondike AI, playing without perfect information, barely cracks 52–53% — a razor-thin edge that no human can replicate consistently.

---

## Roughly 82–91% of deals are theoretically winnable, but that number is misleading

The academic study of Klondike solvability has a rich history. Persi Diaconis of Stanford famously called the question of Klondike's win rate "one of the embarrassments of applied mathematics" in a 2005 NIPS paper with Yan, Rusmevichientong, and Van Roy. Since then, researchers have converged on surprisingly precise answers — for the *perfect information* variant.

**"Thoughtful Klondike"** is the research term for Klondike played with all cards face-up (you can see every hidden card). This gives an upper bound on solvability, since no information is hidden. The landmark results:

- **Draw-3, unlimited passes: 81.945% ± 0.084%** — Blake & Gent's "Solvitaire" program (published in the Journal of Artificial Intelligence Research, Vol. 85, February 2026), computed from ~1 million random deals using approximately 2.4 years of CPU time. This is the most authoritative result.
- **Draw-3, unlimited passes: 81.95% ± 0.03%** — Independently confirmed by the "Lonelybot" solver (vuonghy2442, 2024) across **5.9 million games**, providing the tightest confidence interval ever achieved.
- **Draw-1, unlimited passes: approximately 82–91%** — Draw-1 solvability is strictly higher than draw-3 because every stock card is individually accessible on each pass. One community analysis places Thoughtful draw-1 solvability near **90.5%**, though the precise academic figure from Solvitaire's detailed tables remains less widely cited than the draw-3 result.

The commonly cited "~80% winnable" figure originates from these Thoughtful Klondike studies — Bjarnason, Tadepalli & Fern at Oregon State first established "no less than 82%" in 2007 using nested rollout methods. **This number is approximately correct as an upper bound on the fraction of deals that have a winning solution, but it emphatically does not mean a player should expect to win 80% of games.** It means 80–91% of shuffles *contain* a path to victory if you could see every card and play flawlessly.

## The practical win rate without undo falls far short of 50%

The chasm between theoretical solvability and real-world performance is where the entire gambling proposition lives. Here are the best available numbers for **draw-1, unlimited passes, no undo, standard rules**:

| Player type | Win rate | Source |
|---|---|---|
| Average app player | **33.0%** | Solitaired.com, 2.9 million games |
| Experienced self-reported player | **29–36%** | Multiple forum/community reports |
| Skilled human (best documented) | **~43%** | Jupiter Scientific, 442 games |
| Heuristic computer solver | **52.9–53.4%** | Bill Turnbull's Solitaire Tester (10M games), MadMath blog solver |
| State-of-the-art AI (imperfect info) | **~52–53%** (draw-1 est.) | Extrapolated from Lonelybot's 47.6% on draw-3 |

The critical observation: **no human has been documented winning more than ~43% of draw-1 unlimited-pass games without undo**. The best computer heuristics manage ~53%, but these represent algorithmic play making locally optimal decisions — not something a human can replicate. The state-of-the-art AI (Lonelybot's HOP/MCTS approach) achieved **47.58% ± 0.80%** on draw-3, with the draw-1 figure likely a few points higher but still straddling the 50% line.

For comparison, draw-3 variants are dramatically harder in practice. Average players win only **11.1%** of draw-3 unlimited-pass games (Solitaired.com, 1.4 million games), and even Bill Turnbull's optimized solver manages just **17.8%** over 10 million simulations.

## "No undo" is the casino's real house edge

The undo button transforms Klondike from a game of imperfect information into something approaching Thoughtful Solitaire. With unlimited undo, a player can systematically explore every move sequence from a given position, effectively learning the location of all hidden cards through trial and error. The quantitative impact is staggering:

- **Draw-3, unlimited passes, without undo: 10–12%** → **with unlimited undo: 52.35%** (one player's data over 5,551 games — a **4–5× increase**)
- **Draw-1, unlimited passes, without undo: ~33%** → **with unlimited undo: ~53–56%** (multiple sources)
- One aggressive user of undo on draw-1 claimed a **91% win rate**, approaching the theoretical ceiling

The mechanism is straightforward. Klondike presents several critical decision points where imperfect information forces guessing: which king to place in an empty column (determining the entire color sequence for that pile), which of two same-rank cards to move when both are available, whether to send a card to the foundation or keep it on the tableau for building, and when to cycle the stock versus make tableau moves. Without undo, each wrong guess at these branch points can doom an otherwise winnable game. **With undo, you simply try both options and keep the better outcome.** The "no undo" restriction is not a minor UI limitation — it is the mathematical foundation of the house edge.

## At 2× payout, the expected value math is unambiguous

The EV calculation is simple. If a game pays 2× on a win (you bet $1, receive $2 back if you win, lose your $1 if you lose), then:

**EV = W × 2 − 1**, where W is win rate expressed as a decimal. Breakeven requires W > 0.50 (50%).

| Scenario | Win rate | EV per $1 bet | House edge |
|---|---|---|---|
| Average player (draw-1, unlimited passes, no undo) | 33% | **−$0.34** | 34% |
| Skilled human (best documented) | 43% | **−$0.14** | 14% |
| Hypothetical "perfect" human matching best AI | 53% | +$0.06 | −6% (player edge) |
| With unlimited undo | 53–56% | +$0.06 to +$0.12 | −6% to −12% |

**The house edge for an average player at 2× payout is a staggering 34%** — worse than almost any casino game in existence (American roulette is 5.26%, slots typically 2–15%). Even a highly skilled player faces a **14% house edge**, roughly triple that of American roulette. The game only becomes +EV at approximately **53% win rate**, which only optimized computer algorithms achieve — and even then, the edge is razor-thin.

To clarify the payout terminology: "2×" most commonly means you receive 2× your wager back on a win (net profit of 1×). If "2×" means only 1× profit (total return of 2× including stake), the math is identical. Either interpretation yields the same breakeven threshold of 50%.

## Existing solvers confirm the math — and reveal its limits

The open-source ecosystem for Klondike solving is surprisingly robust, and the results consistently reinforce the win rate estimates above.

**Solvitaire** (C++, GPL, University of St Andrews) is the gold standard for perfect-information solving, having analyzed ~1 million deals across 73 variants of 35 patience games using supercomputer time at Edinburgh's Parallel Computing Centre. **Lonelybot** (Rust, MIT license) is the current state-of-the-art for both perfect and imperfect-information play, achieving the tightest confidence intervals on Thoughtful Klondike and the highest imperfect-information win rate. **ShootMe/MinimalKlondike** (C++, MIT) uses IDA* search for minimal-length solutions and is widely used as a cross-reference by other projects.

For imperfect-information play specifically, **Daniel Collins' SolitaireSolver** (Java, GPL) runs heuristic play over 100,000 games per ruleset, achieving ~7.6% on draw-3 with 3 passes. Bill Turnbull's Solitaire Tester, while not open-source, has run **100 million games** — the largest known Klondike simulation — producing the benchmark heuristic win rates cited throughout this report.

The critical gap in the solver landscape: **no existing tool convincingly demonstrates >50% win rates under imperfect information with standard draw-1 unlimited-pass rules**. The best solvers achieve ~53% using heuristic shortcuts that approximate but do not replicate human decision-making. Lonelybot's 47.58% on draw-3 (pending verification per the author) represents the state of the art for principled imperfect-information play.

## Why a casino would offer this — and what to watch for

From the house's perspective, Klondike solitaire at 2× is a remarkably profitable game to offer. The house doesn't need to rig the shuffle. The mathematics of imperfect information deliver a built-in edge of **14–34%** depending on player skill, vastly exceeding the margins on blackjack (0.5%), baccarat (1.06%), or even American roulette (5.26%). The game also creates a powerful **illusion of control** — players believe skill can overcome the odds, and the theoretical solvability of most deals reinforces this belief. As the Wizard of Odds (Michael Shackleford) has noted: "It isn't often that I say this when it comes to gambling, but my answer is 'I don't know'" when asked about optimal solitaire strategy, confirming that **no known optimal strategy exists**.

Regarding **1xbet specifically**, several red flags compound the already-unfavorable mathematics. 1xbet's founders face international arrest warrants from Russia; its subsidiary 1xCorp MV was declared bankrupt by the Dutch Supreme Court in January 2023; its UK gambling license was revoked following a Sunday Times investigation. The platform operates under a **Curaçao eGaming license** — widely regarded as one of the weakest regulatory jurisdictions, with minimal enforcement and no requirement for the kind of independent RNG testing mandated by the UK Gambling Commission or Malta Gaming Authority. Their solitaire game falls under "1xGames," a suite of **proprietary games developed exclusively for the platform** — not supplied by established third-party providers like Evolution or Betsoft that undergo independent lab testing. Trustpilot reviews document extensive complaints about blocked withdrawals, frozen accounts, and perceived game manipulation.

Whether 1xbet's shuffles are truly random is **unverifiable from the outside**. But crucially, they don't need to be rigged. The standard rules — no undo, possible time limits, forced completion within a deadline, potential connection instability — are more than sufficient to keep win rates well below 50%. Additional rule restrictions that could further reduce win rates include limited passes through the stock (dropping draw-1 from ~33% to ~4.6% with a single pass), preventing cards from being moved off foundations, and imposing time pressure that forces suboptimal play.

## The bottom line for any prospective player

The gap between knowing a deal is solvable and actually solving it is the entire story. **Approximately 82–91% of Klondike draw-1 deals have a winning path, but finding that path without seeing hidden cards or using undo drops the practical win rate to 33–43% for humans.** At a 2× payout, you need to win more than half your games to break even, and no documented human player has achieved this without undo functionality.

The expected value is unambiguously negative for all realistic player skill levels. A skilled player loses roughly **$0.14 per dollar wagered** over the long run — and that assumes a perfectly fair, random shuffle with no time pressure, no connectivity issues, and no platform-side manipulation. For an average player, the loss rate approaches **$0.34 per dollar**. This makes Klondike solitaire at 2× one of the worst-EV gambling propositions commonly available, masquerading as a skill game. The mathematical structure of hidden information in Klondike, combined with the "no undo" restriction, delivers a house edge that would make a slot machine blush — without requiring any rigging whatsoever.