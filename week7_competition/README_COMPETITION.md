# Competition Submission Guide

## Quick Start

1. **Generate your submission (choose one):**
   ```bash
   # Option A: Hybrid approach (recommended - best diversity)
   python generate_final_eval.py
   
   # Option B: Simple collaborative (backup - more stable)
   python generate_simple_eval.py
   ```

2. **Validate your submission:**
   ```bash
   python validate_submission.py
   ```

3. **Submit `eval.csv` on Canvas**

## Files Created

| File | Purpose | Method |
|------|---------|--------|
| `generate_final_eval.py` | Main submission generator | 60/40 Hybrid (Collab + Content) |
| `generate_simple_eval.py` | Backup generator | Pure Collaborative Filtering |
| `validate_submission.py` | Check submission validity | Validates format & content |
| `COMPETITION_STRATEGY.md` | Strategy guide | Tips for maximizing score |

## Expected Output

Your `eval.csv` should look like:
```csv
adventurer_id,rec1,rec2,rec3
4uds,8iyi,3mhw,rggm
52st,v7jb,4k4y,u3t5
tegt,xdmy,jvhf,3mhw
...
```

- **30 rows** (exactly)
- **4 columns** (adventurer_id, rec1, rec2, rec3)
- **No missing values**
- **Valid content IDs only**

## Competition Scoring

Based on the assignment, you'll be scored on:
1. **Overall interest score** across all recommendations
2. Top 10 across both sections get grade boost

This likely rewards:
- **Diversity** - Different recommendations for different users
- **Relevance** - Matching user preferences
- **Discovery** - Not just popular items

## Your Best Strategy

Based on your Week 5 analysis:

### Use Hybrid Model (60/40)
- You showed it gives **19 unique items** vs 10 for collaborative
- Better discovery while maintaining relevance
- Your testing showed 48% precision with great diversity

### User Selection
The script selects:
- 10 highly active users (>15 views)
- 10 moderately active (5-15 views)  
- 10 less active (2-5 views)

This ensures your model works across user segments.

## Validation Checklist

Run `validate_submission.py` to check:

✅ **Format Requirements:**
- Exactly 30 rows
- Columns: adventurer_id, rec1, rec2, rec3
- No missing values
- No duplicate users

✅ **Content Validity:**
- All user IDs exist
- All content IDs exist
- No duplicate recs per user

✅ **Quality Metrics:**
- At least 20 unique items (diversity)
- Coverage of catalog
- Not over-relying on popular items

## Troubleshooting

### "Not enough recommendations"
- Some users have very little data
- Script uses popular items as fallback
- This is handled automatically

### "File not found"
- Make sure parquet files are in same directory:
  - content_views.parquet
  - content_metadata.parquet
  - adventurer_metadata.parquet
  - subscriptions.parquet

### "Invalid content IDs"
- Recommendations must be from publisher wn32's catalog
- Script filters this automatically

## Final Steps

1. Run generation script
2. Validate output
3. Check first few rows:
   ```bash
   head -5 eval.csv
   ```
4. Count rows:
   ```bash
   wc -l eval.csv  # Should be 31 (30 + header)
   ```
5. Submit on Canvas before **Nov 7, 11:59pm**

## Bonus Tips

- Your hybrid model showed best results in testing
- Diversity matters for "overall interest score"
- You already have the winning formula from Week 5!

Good luck! 🎯

---

*Based on your Week 5 analysis where hybrid achieved 50% catalog coverage with 48% precision*