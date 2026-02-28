# Sentinel Forge v1.0

AI Influence, Boundary & Alignment Intelligence Platform.  
Pure Python TF-IDF. No sklearn. No torch. Runs on Termux / 4GB Android.

---

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --no-access-log
```

Docs: `http://localhost:8000/docs`

## Termux

```bash
pkg install python -y && pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --no-access-log
```

## Test

```bash
pytest tests/ -v
```

---

## API

### GET /health
```json
{"status": "ok", "service": "sentinel-forge"}
```

### POST /analyze
```json
{
  "transcript": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "label": "optional"
}
```

### POST /analyze/batch
Up to 50 transcripts. Returns all results + mean_alignment_risk.

---

## Output Fields

| Field | Range | Description |
|---|---|---|
| boundary_pressure_score | 0–1 | User-side pressure on assistant limits |
| narrative_phase | string | Dominant escalation phase |
| identity_drift_score | 0–1 | Assistant self-description drift |
| policy_consistency_delta | -1–1 | Temporal refusal→compliance shift |
| escalation_index | 0–1 | Composite escalation metric |
| alignment_risk | 0–1 | Final risk score |
| summary_classification | string | Stable / Escalating / High Risk |

## RAM Profile
- Idle: ~80MB  
- Under load: ~120MB  
- Safe for 4GB Android
