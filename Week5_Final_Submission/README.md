# Week 5: Recommendation Systems

Emanuel Gonzalez

---

## Final Submission

**Method:** Hybrid (60% Collaborative + 40% Content-Based)

**Performance:**
- 20/20 users successful
- 19 unique items (50% coverage)
- Similarity-based precision: 0.476

**Rationale:** Hybrid recommends 19 unique items vs collaborative's 10 items and heuristic's 2 items, providing better diversity with minimal precision tradeoff.

---

## How to Run
```bash
python generate_final_submission.py  # Creates final_recommendations.csv
python evaluate_similarity_based.py  # Evaluates all methods
python final_validation.py           # Validates submission
```

---

## Results Summary

### Evaluation Set (9 users)

| Method | Precision@2 | Unique Items | Coverage |
|--------|-------------|--------------|----------|
| Heuristic | 0.819 | 2 | 5% |
| Collaborative | 0.569 | 12 | 32% |
| Hybrid | 0.476 | 11 | 29% |

### Final Submission Set (20 users)

| Method | Unique Items | Coverage |
|--------|--------------|----------|
| Hybrid | 19 | 50% |
| Collaborative | 10 | 26% |
| Heuristic | 2 | 5% |

---

## Key Files

**Submission:**
- final_recommendations.csv
- writeup.md

**Code:**
- recommender.py (collaborative)
- advanced_recommender_week5.py (content-based + hybrid)
- heuristic_recommender.py (popularity)

**Visualizations:**
- method_comparison_real.png
- user_segmentation_real.png
- diversity_analysis_real.png
- evaluation_comparison.png