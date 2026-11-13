# Synthetic load disaggregation example

Companion code for *Load Disaggregation under Incomplete Inventory with Structured
Unclaimed Demand*, submitted to ICASSP 2027.
[Accompanying preprint](https://arxiv.org/abs/2610.15247).

The generator combines named loads, a varying background and measurement noise.
The standalone decoder estimates the loads and background from aggregate power.

## Usage

```sh
pip install -r requirements.txt
python synthetic_site.py
python decode.py --input days/day_001.csv
python decode.py --input days/day_001.csv --restricted --out restricted.csv
```

Tested with Python 3.12, NumPy 2.5.4 and PyYAML 6.0.3.

`synthetic_site.py` writes one labeled commissioning day and `days` evaluation days.
Each CSV includes aggregate power, background, channel power and state labels.
`decode.py` reads only `aggregate_w` and writes estimates to `decoded.csv`.
Use `--background constant` for a fixed background, or `--config path.yaml`
to use another configuration.

| File | Purpose |
|---|---|
| `synthetic_site.py` | Synthetic data generation |
| `decode.py` | Calibration, decoding and factorial Viterbi |
| `utils.py` | Config loading, validation and CSV input/output |

Run the checks with `python -m unittest -q`.

## Configuration and calibration

Edit `config.yaml` to add loads or change their power levels and timing.
`markov` loads use a mean dwell per state; `schedule` loads follow timed blocks
with random jitter. State numbers index `levels_w`, starting at zero.
An optional `support_order` limits the state order when `--restricted` is used;
`[1, 0, 1]` allows an ON, OFF, ON sequence with empty phases allowed.

Calibration regenerates the labeled commissioning day and estimates state levels,
transition probabilities, noise and background range. The decoder uses exact
factorial Viterbi; restrictions preserve the original costs of allowed paths.

The factorized Viterbi implementation is described in
[Mostafavi and Cox, Section 3.3](https://arxiv.org/abs/2610.15247).
Factorizing the transition step preserves exact joint MAP inference.

## Reference results

Reference values from the 50-day synthetic experiment with labeled calibration.
Intervals are 95% bootstrap intervals over days.

| Background | Support | Compressor-count agreement |
|---|---|---|
| Constant | Unrestricted | .699 [.683, .715] |
| Constant | Restricted | .771 [.736, .807] |
| Dynamic | Unrestricted | .967 [.956, .977] |
| Dynamic | Restricted | .993 [.985, .998] |

The paired restriction gains were +.073 with a constant background and +.026
with a dynamic background.

Code license: [MIT](LICENSE).
