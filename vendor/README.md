# Vendor: `international_results` (up to FIFA World Cup 2026 'Round of 16')

This directory contains a vendored copy of the open-source GitHub repository:

- `martj42/international_results`

## What was included
From the upstream project, I copied the match data and related files needed for modeling international football results:

- `results.csv` (all international matches prior to the 2026 World Cup, plus World Cup 2026 matches up through the **Round of 16**)
- `shootouts.csv`
- `goalscorers.csv`

## How this is used in this project
This data is treated as **read-only input** for the tournament predictor. The code in this repo uses:
- historical international match results to estimate team strength / scoring rates
- World Cup 2026 results through the Round of 16 as the latest completed tournament data point
- subsequent tournament rounds (quarterfinals, semifinals, final) are predicted going forward

## Upstream updates
If upstream changes are pulled in the future (e.g., adding newer rounds), the project should update the submodule/re-vendored snapshot and then recompute any derived datasets/models as needed.
