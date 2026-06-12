# Pitcher Matchup Scout

Streamlit scouting app for pitcher-vs-current-roster matchup prep.

The app reads a local pitch-level Parquet dataset, lets you select any pitcher in that dataset, then lets you select a current MLB team. It filters the selected pitcher’s historical pitches to current roster hitters and shows matchup history pitch by pitch.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## AI Setup

The AI brief and attack-plan features use AWS Bedrock through `boto3`. To use
them, configure AWS credentials locally with access to Bedrock Runtime in
`us-west-2`.

If you use a named AWS profile, set it before starting the app:

```bash
export AWS_PROFILE=your-local-profile-name
```

Optional AI environment settings:

```bash
export AWS_REGION=us-west-2
export BEDROCK_MODEL_ID=google.gemma-3-4b-it
```

## Parquet Options

Large all-pitches Parquet dataset for the last two years:

```bash
python scripts/build_all_pitches_parquet.py --start 2024-06-03 --output data/statcast_all_pitches_last2years.parquet --overwrite
```

Large all-pitches Parquet dataset for the last five years:

```bash
python scripts/build_all_pitches_parquet.py --overwrite
```

The all-pitches builder writes each completed date chunk into `data/cache/` as Parquet, then assembles the final file. If a long Baseball Savant pull times out, rerun the same command and it will reuse cached chunks instead of starting over.

Generated local files, not committed to GitHub:

```text
data/statcast_all_pitches_last2years.parquet
data/statcast_all_pitches_last5years.parquet
```

The app is Parquet-only and filters pitcher/batter rows through Parquet reads. Full Statcast Parquet files and cache files are generated locally and ignored by Git because they are large. Keep `data/sample_statcast.parquet` in the repo as safe sample data.

## Run

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Workflow

1. Pick a pitch dataset source in the sidebar.
2. Pick any pitcher from that dataset.
3. Pick a current MLB team.
4. Review current roster hitters with or without matchup history.
5. Open each plate appearance and inspect every pitch in order.

The team filter uses current roster player IDs. If a current player faced the selected pitcher while playing for another team, that historical matchup still appears.

## What The App Shows

- Current roster table for the selected team.
- PA and pitch counts against the selected pitcher.
- Zero-PA rows for current players with no history against the selected pitcher.
- Every historical plate appearance in correct pitch sequence.
- Individual pitch detail with count, pitch type, velocity, movement, zone, result, xwOBA, and Savant link.
- Batter profile vs the selected pitcher: pitch mix seen by count, result summary, contact quality, pitch log, and zone heatmap.
- All-dataset batter profile: every pitch the selected current-roster batter has faced in the selected dataset, across all pitchers, plus graphical zone heatmaps for hits, whiffs, fouls, takes, and damage.
- AI attack plan generator: select a pitcher and current-roster batter, then generate a concise pitcher-batter game plan using AWS Bedrock. The prompt combines pitcher tendencies, batter weaknesses, direct matchup history when available, pitch-level samples, count/location patterns, whiff/chase/damage indicators, and contact quality.
- Team-level AI scouting brief: generate a fast scouting summary for the selected pitcher against the selected current roster.


## Tests

```bash
pytest
```

If your local Anaconda/Python 3.13 build crashes while pytest imports its debugger plugin:

```bash
pytest -p no:debugging
```
