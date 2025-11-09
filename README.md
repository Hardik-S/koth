# King of the Hill Evaluation Study

## Overview
This repository supports a solo research project examining evaluation strategies for the King of the Hill (KotH) chess variant. The study contrasts a hand-crafted heuristic evaluator against a neural network architecture (NNUE) to illuminate how each approach interprets middlegame positions and central king activity.

## Objectives
- Build a reproducible pipeline that can ingest KotH positions between moves 10 and 20.
- Implement both the heuristic (Minimax/Alpha-Beta) evaluator and the NNUE evaluator for KotH-specific rules.
- Identify positions where the evaluators disagree by roughly 200 centipawns or more.
- Analyze disagreements to uncover strategic biases and actionable insights about KotH play.

## Data Sources
- **Primary dataset**: [Lichess KotH PGN archive](https://database.lichess.org/variant/koth/) containing bulk KotH games.
- **Backup dataset**: [Hugging Face KotH PGN mirror](https://huggingface.co/datasets/hardikshrestha/koth-pgns) for redundancy.

### Usage Notes
1. Download PGN archives and filter for KotH middlegames (moves 10–20).
2. Convert PGN positions into formats suitable for the heuristic and NNUE evaluators.
3. Store extracted positions and evaluator outputs for repeatable analyses.

## Methodology
1. **Position Extraction**: Parse PGNs, annotate KotH-specific metadata, and isolate middlegame snapshots.
2. **Heuristic Evaluation**: Run a Minimax/Alpha-Beta search augmented with KotH heuristics (king distance to center, central pawn structure, safety metrics).
3. **NNUE Evaluation**: Adapt an efficiently updatable neural network evaluator to KotH board representation and compute scores for the same positions.
4. **Disagreement Analysis**: Flag positions where evaluation scores diverge by ≈200 centipawns, then investigate tactical and strategic factors behind the gap.
5. **Interpretability Reporting**: Summarize patterns, biases, and takeaways to guide future KotH strategy development.

## Evaluation
- Quantitatively compare evaluator outputs using centipawn scores and estimated win probabilities.
- Qualitatively review flagged positions, emphasizing central king routes, piece coordination, and tactical motifs unique to KotH.
- Document insights and recommendations in follow-up reports (e.g., Phase deliverables).

## Repository Expectations
- Source code for data processing, evaluator implementations, and batch evaluation scripts.
- Configuration files or notebooks that reproduce extraction, evaluation, and analysis steps.
- Reporting artifacts (markdown, visualizations) capturing key findings from evaluator disagreements.
