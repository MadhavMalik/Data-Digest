# an_d32e2ca9b8dd — final answer

**Question:** What factors are associated with the amount passengers pay for NYC yellow taxi trips?
**Dataset:** nyc_tlc_yellow_2026_01 — 3,724,889 rows × 20 columns
**Stop reason:** `completed`

## Answer

The amount passengers pay for NYC yellow taxi trips is strongly associated with trip duration, trip distance, and whether the trip involves an airport pickup.

### Key findings

- trip_duration_minutes ~ total_amount: n=3,509,466, pearson=+0.699, spearman=+0.889, MI=0.250, stability=1.00, dir=positive, very strong
- trip_distance ~ total_amount: n=3,509,466, pearson=+0.867, spearman=+0.847, MI=0.229, stability=1.00, dir=positive, very strong
- is_airport_pickup ~ total_amount: n=2,515,206, pearson=+0.643, spearman=+0.443, MI=0.164, stability=1.00, dir=positive, strong

### Caveats

- Data accuracy is not guaranteed as records are submitted by technology providers.
- Trips with RatecodeID 2 (JFK) and 3 (Newark) use flat or administered fares, which may affect fare calculations.
- Negative fare amounts in the dataset likely indicate refunds or voided trips.
- A small number of records have timestamps outside the nominal month of the file.

### Unresolved

- The exact impact of passenger count on fare, given the mixed associations with trip distance.
- The role of average speed in determining fare, as the relationship appears nonlinear.
- The effect of tip fraction on fare, as the relationship is likely nonlinear.

## Degradations

- residual stage failed: NameError: name 'RelationshipResult' is not defined
