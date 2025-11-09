# Project Summary for Conversational Agents

This repository houses a solo King of the Hill (KotH) chess evaluation project led by Hardik Shrestha. The effort studies middlegame positions, especially moves 10–20, comparing two evaluators:

- A hand-tuned heuristic built on Minimax/Alpha-Beta search and KotH-specific tactical heuristics.
- A neural network (NNUE) evaluator adapted for KotH to provide efficiently updatable position scores.

The core objective is to surface and interpret major disagreements (≈200 centipawns or more) between these evaluators to better understand strategic biases around central king play.

## Datasets and References
- **Primary**: Lichess KotH archive of PGN games (bulk downloads via https://database.lichess.org/variant/koth/).
- **Backup**: Mirrored KotH PGN dataset hosted on Hugging Face (https://huggingface.co/datasets/hardikshrestha/koth-pgns).

Focus analysis on middlegame positions extracted from these sources, filtered to the move range where KotH-specific dynamics emerge.

## Guidance for AI Assistance
1. Prioritize interpretability: explain *why* the evaluators diverge, referencing KotH win conditions and king centralization tactics.
2. Use both centipawn and win-probability perspectives when assessing disagreement severity.
3. Document any recurring motifs (e.g., premature king marches, central pawn structures) that drive evaluator bias.
4. When suggesting experiments, emphasize reproducible pipelines for re-evaluating candidate positions and logging evaluator outputs.
