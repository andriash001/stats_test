# Stats Test by Process Step

This module compares good vs bad batch yields per process step and returns a
summary table with p-values per step.

In a typical manufacturing setting, production data includes many process
steps, each with batches labeled as good or bad. This tool quickly checks, per
step, whether the good and bad populations are statistically different, and
adds a `rank` column so you can spot the most significant differences first.
Yield is just an example metric here; you can use any numeric measurement that
makes sense for your process.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```python
import pandas as pd
from stat_tests import compare_good_bad_by_step, make_dummy_data

# Use real data
# df = pd.read_csv("your_data.csv")
# result = compare_good_bad_by_step(df)

# Or use dummy data to validate setup
result = compare_good_bad_by_step(make_dummy_data())
print(result)
```

### Quick example (dummy data)

```python
from stat_tests import compare_good_bad_by_step, make_dummy_data

df = make_dummy_data()  # 150 batches, 3 steps, good/bad statuses by default
out = compare_good_bad_by_step(df)
print(out.head(1))
```

Example output (columns):

```
process_step | test_result | p_value | rank
```

## Input schema

Required columns:
- process_step
- batch_id
- batch_status (good or bad)
- batch_yield (0-100)

## Output schema

- process_step
- test_result (statistically different or not)
- p_value
- rank (1 = smallest p-value, most statistically different)

## Configuration

You can customize thresholds using `TestConfig`:

```python
from stat_tests import TestConfig, compare_good_bad_by_step

config = TestConfig(alpha=0.01, min_normality_n=10)
result = compare_good_bad_by_step(df, config=config)
```

## Notes

- The statistical test is auto-selected per step: Welch t-test for normal data
  (Shapiro, n >= 8), otherwise Mann-Whitney U.
- The default significance level is alpha = 0.05 (configurable via TestConfig).
