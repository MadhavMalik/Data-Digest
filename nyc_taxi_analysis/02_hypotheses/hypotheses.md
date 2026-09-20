# Hypotheses proposed by the planner

## Round 1 — source: `llm`

_This round focuses on exploring how trip characteristics, vendor differences, and temporal factors relate to fare amounts, prioritizing hypotheses that could reveal non-obvious patterns._

- **trip_distance → fare_amount** (high)
  - transformations: `fare_amount / trip_distance`
  - rationale: Longer trips are expected to have higher fares, but the relationship may be influenced by fixed fees at airports.
- **trip_duration_minutes → fare_amount** (medium)
  - transformations: `fare_amount / trip_duration_minutes`
  - rationale: Longer trip durations may lead to higher fares due to time-based charges.
- **payment_type → fare_amount** (medium)
  - transformations: `fare_amount`
  - rationale: Different payment types may be associated with different fare amounts due to varying tipping behaviors and fare adjustments.
- **is_rush_hour → fare_amount** (medium)
  - transformations: `fare_amount`
  - rationale: Rush hour trips may have higher fares due to increased demand and potential surcharges.
- **VendorID → fare_amount** (low)
  - transformations: `fare_amount`
  - rationale: Different vendors may have different fare structures or reporting practices.
- **pickup_hour → fare_amount** (low)
  - transformations: `fare_amount`
  - rationale: Fares may vary by time of day due to demand fluctuations and surcharges.

## Round 2 — source: `llm`

_This round focuses on exploring potential nonlinear effects and group differences that could reveal unexpected influences on fare amounts, particularly considering fixed fees and categorical variables._

- **trip_distance → total_per_mile** (high)
  - transformations: `log(trip_distance)`
  - rationale: Investigating the relationship between trip distance and cost per mile could reveal how distance affects pricing efficiency, especially given the known fixed fees for airport pickups.
- **passenger_count → total_amount** (medium)
  - transformations: `log1p(passenger_count)`
  - rationale: Exploring how the number of passengers affects total fare could uncover nonlinear effects, especially given the high proportion of null and zero values.
- **payment_type → total_amount** (medium)
  - rationale: Different payment types may correlate with different fare amounts due to varying tipping behaviors and potential disputes.
- **pickup_hour → total_amount** (medium)
  - rationale: Time of day may influence fare amounts due to demand fluctuations and potential surcharges.
- **is_airport_pickup → total_per_mile** (high)
  - rationale: Airport pickups are known to have fixed fees, which could significantly affect the cost per mile, especially for short trips.
- **VendorID → total_amount** (low)
  - rationale: Different vendors might have varying fare structures or service quality, potentially affecting total fare amounts.

## Round 3 — source: `llm`

_This round focuses on exploring fare efficiency per mile and group differences to uncover nuanced factors affecting total fare beyond simple distance and duration metrics._

- **trip_distance → total_amount** (high)
  - transformations: `total_per_mile`
  - rationale: Testing the relationship between total amount per mile and trip distance could reveal how distance affects the cost efficiency of trips, especially considering airport fees.
- **trip_duration_minutes → total_amount** (medium)
  - transformations: `total_per_mile`
  - rationale: Investigating total amount per mile against trip duration could highlight how time affects fare efficiency, especially during rush hours.
- **payment_type → total_amount** (medium)
  - transformations: `fare_per_mile`
  - rationale: Different payment types may influence fare per mile due to varying tipping behaviors and transaction fees.
- **is_airport_pickup → total_amount** (high)
  - transformations: `total_per_mile`
  - rationale: Airport pickups have fixed fees that could significantly affect the total cost per mile, revealing differences in fare structure.
- **pickup_day_of_week → total_amount** (medium)
  - transformations: `fare_per_mile`
  - rationale: Day of the week could influence fare per mile due to variations in demand and traffic patterns.
- **passenger_count → total_amount** (low)
  - transformations: `fare_per_mile`
  - rationale: Passenger count might affect fare per mile in nonlinear ways due to shared ride dynamics and negotiated fares.
