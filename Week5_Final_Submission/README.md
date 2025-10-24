# Week 5 Final Submission - Recommendation Systems

## Required Files

### 1. Three Recommender Systems
- `recommender.py` - Collaborative filtering
- `advanced_recommender_week5.py` - Content-based filtering
- `heuristic_recommender.py` - Global heuristic (popularity)

### 2. Final Recommendations
- `final_recommendations.csv` - 20 users with 2 recommendations each

### 3. Report
- `writeup.md` - 2-page report answering all reflection questions

## How to Run
```bash
python recommender.py                    # Collaborative
python advanced_recommender_week5.py     # Content-based
python heuristic_recommender.py          # Heuristic
python evaluate_similarity_based.py      # Evaluation
```

## Results

| Method | Precision@2 |
|--------|-------------|
| Heuristic | 0.819 |
| Collaborative | 0.569 |
| Content-Based | 0.476 |

Final submission uses hybrid approach (60% collaborative + 40% content-based).

See `writeup.md` for full analysis.
