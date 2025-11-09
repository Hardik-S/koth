# Phase 1 Deliverable

0. **Group Members**
   - Hardik Shrestha (solo project lead)

1. **Project Title**
   - Interpreting KotH Evaluator Disagreements in the Middlegame

2. **Dataset Link**
   - Primary: [Lichess KotH PGN archive](https://database.lichess.org/variant/koth/)
   - Backup: [Hugging Face KotH PGN mirror](https://huggingface.co/datasets/hardikshrestha/koth-pgns)

3. **Project Description**

   The King of the Hill (KotH) chess variant rewards racing a king to the central squares while still respecting checkmate rules. This project concentrates on middlegame positions (moves 10–20) where central tension and king activity strongly influence outcomes.

   Two evaluators will be implemented and compared:
   - A hand-crafted Minimax/Alpha-Beta search enhanced with KotH-specific heuristics, such as king proximity to the center, central pawn structure, and safety considerations.
   - A neural network (NNUE) evaluator tailored for KotH that updates efficiently with incremental position changes.

   The analysis pipeline will extract positions from the datasets, evaluate them with both approaches, and flag disagreements of roughly 200 centipawns or more. For each flagged position, the goal is to interpret *why* the evaluators diverge—examining central king routes, tactical motifs, and strategic plans unique to KotH. Insights will be documented to guide stronger KotH play and inform future evaluator improvements.
