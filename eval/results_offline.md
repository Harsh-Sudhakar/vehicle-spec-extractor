### Evaluation (offline rule-based)

| # | doc | query | kind | hit@k | values | correct | #specs |
|---|---|---|---|---|---|---|---|
| 1 | f150 | Torque for brake caliper bolts | table | ✓ | 3/3 | ✅ | 18 |
| 2 | f150 | front brake fluid capacity | table | ✓ | 1/1 | ✅ | 4 |
| 3 | f150 | minimum front brake disc thickness | table | ✓ | 1/1 | ✅ | 10 |
| 4 | f150 | brake pad minimum thickness | table | ✓ | 1/1 | ✅ | 8 |
| 5 | f150 | lower ball joint nut torque | table | ✓ | 1/1 | ✅ | 8 |
| 6 | f150 | wheel speed sensor bolt torque front suspension | table | ✓ | 1/1 | ✅ | 8 |
| 7 | f150 | tie-rod end nut torque | table | ✓ | 1/1 | ✅ | 6 |
| 8 | f150 | brake master cylinder nut torque | table | ✓ | 1/1 | ✅ | 8 |
| 9 | f150 | camber adjustment cam bolt nut tightening torque | prose | ✓ | 1/1 | ✅ | 5 |
| 10 | f150 | stabilizer bar link nut torque | table+prose | ✓ | 2/2 | ✅ | 8 |
| 11 | figo | engine oil capacity for the diesel including filter | table | ✓ | 0/1 | ❌ | 1 |
| 12 | figo | rear tyre pressure when laden | table | ✓ | 0/1 | ❌ | 0 |
| 13 | figo | brake fluid specification | table | ✓ | 0/1 | ❌ | 1 |
| 14 | figo | spark plug gap | table | ✓ | 1/1 | ✅ | 1 |
| 15 | figo | fuel tank capacity | table | ✓ | 0/1 | ❌ | 2 |
| 16 | figo | engine idle speed | table | ✓ | 1/1 | ✅ | 10 |
| 17 | figo | gross vehicle weight of the diesel | table | ✓ | 0/1 | ❌ | 0 |
| 18 | figo | exhaust valve clearance for the petrol engine | table | ✓ | 0/1 | ❌ | 1 |
| 19 | figo | oil filter part number 1.2L Duratec | table | ✓ | 0/1 | ❌ | 0 |
| 20 | figo | petrol engine cooling system capacity | table | ✓ | 0/1 | ❌ | 3 |
| 21 | figo | torque for brake caliper bolts | negative | – | – | ✅ | 0 |

- **Retrieval hit@k:** 20/20 (100%)
- **Item recall:** 14/22 (64%)
- **Query accuracy:** 13/21 (62%)
- **Negatives (correct empty answer):** 1/1
- **Median latency:** 2.2s
