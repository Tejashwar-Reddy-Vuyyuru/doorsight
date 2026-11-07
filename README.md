# DoorSight

**Finds the main front door of a house from street-level photos, then has an AI vision check confirm the pin before it ships.**

A geocoded address usually lands on the roof, not at the door a driver walks to. DoorSight answers one question per address: *where is the main entrance, and is the system sure enough to ship that pin?*

## Results (ZIP 63122, Kirkwood, MO, single-family homes)

| Test | Addresses | HIGH before AI check | Shipped | Correct |
|---|---|---|---|---|
| **Test 3, clean (no changes after it)** | 200 | 50 | 23 | **23** |
| Test 2 (after fixes) | 200 | 51 | 24 | 24 |
| Test 1 (used for tuning) | 100 | 25 | 11 | 11 |

- **Precision over coverage.** About 12% of addresses get a shipped pin. A wrong pin costs a driver more than a missing one.
- **What "correct" means.** The pin is on the main door's side of the house and within about 5 m of it. Every shipped pin was hand-checked in Google Maps using rules fixed in advance.
- **Honest bounds.** 23/23 on the clean test means precision is at least about 88% at 95% confidence (47/47 across tests 2 and 3: at least about 94%). Before the last fix, the worst result was 1 wrong out of 18.

## How it works

```
address
 └─ geocode → building outline (Microsoft US Building Footprints, OSM fallback)
    └─ 2 Google Street View cameras on the address's own street
       └─ door detector (OWLv2, free zero-shot; YOLO-World fallback) + clues: porch, steps, mailbox, walkway, house number
          └─ pixel → bearing → rays cross = door point; merge duplicates within 4 m
             └─ snap the pin to a wall the cameras can actually see
                └─ aerial imagery: which side the walkway/driveway is on
                   └─ rank doors → tier HIGH / MEDIUM / LOW / SIDE / NONE
                      └─ AI check (Claude vision) on HIGH only: "is this the main front entrance?"
                         └─ yes → ship     no / unsure → MEDIUM (sent to a person)
```

The free stages do most of the work. The paid AI check runs only on the roughly 25% of addresses that reach HIGH, at well under $1 per 200 addresses.

## Files

| File | What it does |
|---|---|
| `doorsight.py` | Main pipeline: `one` address or a `batch` CSV |
| `providers.py` | Geocoding, Street View cameras, OSM |
| `footprints.py` | Builds and queries a local extract of Microsoft building outlines |
| `detect.py` | Door detector and clue scoring |
| `geometry.py` | Bearings, ray–polygon intersection, visible-wall snap |
| `rank.py` | Clusters sightings, merges duplicate doors, picks the main door |
| `aerial.py` | Walkway/driveway side from aerial tiles |
| `judge.py` | AI main-door check on HIGH pins, writes the hand-check checklist |
| `regrade.py` | Re-grades saved results for free (no new photos) after a rule change |
| `make_addresses.py` | Random test sample of residential addresses in a ZIP |
| `app.py`, `ui.html` | Local map viewer |

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Keys (set per session, never commit them):

```powershell
$env:GOOGLE_MAPS_KEY = "your-google-key"      # Street View Static + Geocoding APIs
$env:ANTHROPIC_API_KEY = "your-anthropic-key" # AI check only
```

Building outlines (one-time): download the Missouri file from Microsoft US Building Footprints, then:

```powershell
python footprints.py extract Missouri.geojson --bbox 38.53 -90.47 38.64 -90.35
```

## Run

```powershell
python make_addresses.py 63122 200
python doorsight.py batch houses.csv
python judge.py
```

`judge.py` writes `results/high_checklist_judged.csv`. Fill `correct (y/n)` by hand, then count:

```powershell
Import-Csv results\high_checklist_judged.csv | Group-Object "correct (y/n)" | Select-Object Name, Count
```

## Failure types found and fixed

| Failure | Fix |
|---|---|
| Same door counted twice, "tied" with itself | Merge doors within 4 m |
| Pin on the roof or in the yard | Real building outlines instead of a guessed 16 m box |
| Garage, side or back door picked; window shutter read as a door | AI main-door check |
| Pin snapped to the back wall | Snap only to a wall the cameras can see; demote pins behind the house |

## Swappable by design

Each stage can be replaced without rebuilding the rest: licensed imagery or a company's own proof-of-delivery photos instead of Street View, an in-house door model instead of the zero-shot detector, and any vision model (or a small model trained on the AI check's answers) as the judge.

## Data, licensing and privacy

This is a research and portfolio project, not a commercial dataset.

- No personal data about residents is used.
- Street View images, results and the pin file are **not** in this repository (see `.gitignore`).
- Google Maps Platform terms restrict storing imagery and building datasets from it. A commercial version would use licensed or openly licensed imagery.
- Building outlines: Microsoft US Building Footprints, ODbL.

## Limits and next steps

- Tested on single-family homes in one ZIP. Apartments and complexes need their own test with their own pass mark.
- Back doors remain the main risk. Next: show the AI check the aerial view too.
- Larger samples in new ZIP codes to tighten the precision estimate.
