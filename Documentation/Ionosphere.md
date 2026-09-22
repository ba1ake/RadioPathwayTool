# `ionosphere.py` — Ionospheric Observation Module

## 1. Purpose

`ionosphere.py` is responsible for retrieving and processing **current ionospheric propagation information** from the Australian Bureau of Meteorology's Space Weather Services (SWS).

The module currently focuses on the SWS **Automated MUF Report**, which provides a near-real-time assessment of ionospheric conditions at monitoring stations throughout Australia, the South Pacific, and Antarctica.

The module converts the human-readable SWS report into structured Python objects that can be consumed by the propagation-analysis engine.

### Primary purpose

```text
SWS MUF Report
      ↓
HTTP request
      ↓
HTML extraction
      ↓
Report timestamp extraction
      ↓
Station condition extraction
      ↓
Structured observations
      ↓
Propagation analysis
```

The module is deliberately focused on **data collection and parsing**. It does not determine whether a particular amateur-radio band is good or bad. That responsibility belongs to `band_favorability.py`.

---

# 2. Current Capabilities

`ionosphere.py` currently provides the following capabilities:

### Live SWS data retrieval

The module connects directly to:

```text
https://www.sws.bom.gov.au/HF_Systems/1/2/4
```

This is the official SWS **Automated Report on Current Australian Ionospheric Conditions**.

The report is updated periodically and contains current conditions for multiple ionospheric monitoring stations.

---

### Report timestamp detection

The module extracts the timestamp associated with the SWS report.

For example:

```text
2026-09-22 23:31 UTC
```

This allows the application to distinguish between:

* when the SWS report was generated
* when the Python application retrieved it

This is important because propagation data is time-sensitive.

---

### Station condition extraction

The module identifies individual monitoring stations and their reported conditions.

Example:

```text
Norfolk Is      near normal
Niue Is         enhanced by 32%
Townsville      depressed by -17%
```

These are converted into structured observations.

---

### Percentage deviation extraction

Where SWS provides a numerical deviation from expected conditions, the module extracts the percentage.

For example:

```text
enhanced by 32%
```

becomes:

```text
condition = "enhanced"
percent_difference = 32
```

Similarly:

```text
depressed by -17%
```

can be represented as a negative deviation.

The SWS report defines conditions relative to monthly predicted values. Conditions within approximately ±15% of predicted values are described as normal.

---

### Missing-data handling

The module recognises that some stations may not have usable MUF observations.

For example:

```text
no vertical MUF data
```

or:

```text
no vertical MUF data or possible ionospheric absorption
```

These are represented as observations where data is unavailable rather than causing the entire collection process to fail.

This is important because individual ionosonde stations can temporarily have missing or unusable data.

---

# 3. `IonosphereObservation`

The primary data structure produced by the module is:

```python
@dataclass
class IonosphereObservation:
    station: str
    station_name: str
    timestamp_utc: datetime | None
    condition: str
    percent_difference: float | None
    data_available: bool
    source: str
```

Each instance represents the current SWS assessment for one station.

---

## 3.1 `station`

A short internal identifier for the station.

Example:

```python
"norfolk"
```

This is intended for programmatic use.

---

## 3.2 `station_name`

The human-readable station name.

Example:

```python
"Norfolk Is"
```

This is primarily intended for displaying results to the user.

---

## 3.3 `timestamp_utc`

The timestamp associated with the SWS report.

Example:

```python
datetime(2026, 9, 22, 23, 31, tzinfo=timezone.utc)
```

The timestamp is stored in UTC so that all propagation calculations can use a consistent time reference.

---

## 3.4 `condition`

The textual assessment supplied by SWS.

Typical values include:

```text
near normal
enhanced
depressed
no vertical MUF data
```

The field intentionally retains the original SWS meaning rather than immediately converting everything into a numerical score.

This allows the propagation engine to make its own interpretation later.

---

## 3.5 `percent_difference`

The numerical difference from predicted conditions when SWS supplies one.

Example:

```python
32
```

for:

```text
enhanced by 32%
```

If no percentage is available:

```python
None
```

Example:

```text
near normal
```

does not contain a specific percentage, so:

```python
percent_difference = None
```

---

## 3.6 `data_available`

Boolean indicating whether usable MUF information was reported.

Example:

```python
True
```

for:

```text
Norfolk Is: near normal
```

and:

```python
False
```

for:

```text
Brisbane: no vertical MUF data
```

This allows downstream code to distinguish between:

```text
bad propagation
```

and:

```text
no data
```

These are fundamentally different situations.

---

## 3.7 `source`

Identifies where the observation originated.

For example:

```python
"SWS MUF Report"
```

This becomes useful if additional ionospheric data sources are added later.

---

# 4. Station List

The module currently recognises the following SWS stations:

```python
STATIONS = {
    "brisbane": "Brisbane",
    "canberra": "Canberra",
    "cocos": "Cocos Is",
    "darwin": "Darwin",
    "hobart": "Hobart",
    "learmonth": "Learmonth",
    "niue": "Niue Is",
    "norfolk": "Norfolk Is",
    "perth": "Perth",
    "sydney": "Sydney",
    "townsville": "Townsville",
    "casey": "Casey",
    "davis": "Davis",
    "mawson": "Mawson",
}
```

These stations provide geographically distributed observations.

Of particular interest to the RadioPathwayTool are stations such as:

* **Norfolk Island** — useful for South Pacific propagation assessment
* **Hobart** — useful for southern Australian conditions
* **Canberra** — useful for eastern Australian conditions
* **Niue** — useful for Pacific conditions

The application should not treat any single station as representing the entire region.

---

# 5. Main Processing Functions

## `fetch_muf_report()`

Responsible for retrieving the live SWS MUF report.

Conceptually:

```text
Internet
   ↓
SWS website
   ↓
HTTP GET
   ↓
HTML document
```

The request uses a normal HTTP user-agent so that the request resembles a standard web client.

The function returns the retrieved HTML/report data.

---

# 6. Report Timestamp Parsing

The module searches the report for the SWS update timestamp.

Example source text:

```text
last updated 22 Sep 2026 23:31 UT
```

This is converted into a Python `datetime`.

The application therefore has two important times:

```text
Report time:
2026-09-22 23:31 UTC

Retrieval time:
2026-09-22 23:50 UTC
```

The difference between these values can be used later to determine the age of the propagation information.

---

# 7. Station Parsing

The parser searches the report for known station names followed by their reported condition.

For example:

```text
Norfolk Is : near normal
```

is converted into:

```python
IonosphereObservation(
    station="norfolk",
    station_name="Norfolk Is",
    condition="near normal",
    percent_difference=None,
    data_available=True,
    ...
)
```

And:

```text
Niue Is : enhanced by 32%
```

becomes approximately:

```python
IonosphereObservation(
    station="niue",
    station_name="Niue Is",
    condition="enhanced by 32%",
    percent_difference=32,
    data_available=True,
    ...
)
```

---

# 8. Handling Missing Data

The SWS system does not always provide usable MUF information.

For example:

```text
Brisbane : no vertical MUF data.
```

The module does not interpret this as poor propagation.

Instead:

```python
data_available = False
```

This distinction is important.

### Example

These two situations should not produce the same result:

```text
Hobart:
depressed by 20%
```

versus:

```text
Brisbane:
no vertical MUF data
```

The first contains an actual propagation observation.

The second simply means the system cannot currently provide that particular observation.

---

# 9. Current Example

A successful run currently produces output similar to:

```text
HF PROPAGATION IONOSPHERIC OBSERVATIONS

Retrieved:
2026-09-22 23:50:10 UTC

SWS MUF REPORT

Brisbane        no vertical MUF data
Canberra        near normal
Cocos Is        no vertical MUF data
Darwin          near normal
Hobart          near normal
Learmonth       near normal
Niue Is         enhanced by 32%
Norfolk Is      near normal
Perth           near normal
Sydney          no vertical MUF data
Townsville      near normal
Casey           near normal
Davis           no vertical MUF data or possible ionospheric absorption
Mawson          near normal
```

This demonstrates that the module is successfully retrieving and parsing live SWS information.

---

# 10. What `ionosphere.py` Does NOT Do

The module intentionally does not currently:

### Determine band quality

It does not decide:

```text
10m = good
20m = bad
80m = excellent
```

That is handled by:

```text
band_favorability.py
```

---

### Calculate propagation paths

It does not currently calculate:

* maximum usable frequency for a specific path
* take-off angle
* skip distance
* hop count
* expected signal strength
* exact propagation direction

Those are separate analysis functions.

---

### Determine actual amateur-radio activity

The SWS MUF report describes ionospheric conditions.

It does **not** tell us whether somebody is currently transmitting on:

```text
28.400 MHz
```

For actual activity detection, future integrations could include:

* PSK Reporter
* WSPR
* DX clusters
* SDR observations

These would complement the ionospheric data rather than replace it.

---

# 11. Data Flow Through RadioPathwayTool

The intended architecture is:

```text
                    ┌────────────────────┐
                    │  Space Weather     │
                    │  NOAA / SWS        │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ space_weather.py   │
                    │                    │
                    │ Solar / geomagnetic│
                    │ conditions         │
                    └─────────┬──────────┘
                              │
                              │
                              ▼
                    ┌────────────────────┐
                    │ ionosphere.py      │
                    │                    │
                    │ Live MUF /         │
                    │ ionospheric state  │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ band_favorability  │
                    │                    │
                    │ 80m / 40m / 20m / │
                    │ 15m / 10m analysis │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │     main.py        │
                    │                    │
                    │ Combine results    │
                    └─────────┬──────────┘
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
          Discord Bot                 Web/API
```

---

# 12. Why This Data Is Useful

The important advantage of the SWS MUF report is that it provides **observed ionospheric behaviour**, rather than relying exclusively on theoretical solar/geomagnetic indicators.

For example:

```text
Solar conditions:
good

Geomagnetic conditions:
quiet

BUT

Norfolk:
near normal

Niue:
enhanced by 32%
```

The propagation engine can use those observations as evidence that the ionosphere is actually behaving differently across the region.

This is particularly useful for detecting conditions that broad global indices may not capture well.

---

# 13. Future Extensions

The module is currently a **v1 ionospheric observation collector**.

Potential future additions include:

## Regional weighting

Instead of treating all stations equally, stations could be weighted according to the propagation path.

For example, a New Zealand → Australia path could give greater weight to:

```text
Hobart
Canberra
Norfolk Island
```

while a New Zealand → Pacific path could give greater weight to:

```text
Norfolk Island
Niue
```

---

## Report age

Add:

```python
report_age_minutes
```

so the propagation engine can determine whether the observation is:

```text
< 30 minutes old
30–60 minutes old
1–2 hours old
> 2 hours old
```

Older observations could automatically receive lower confidence.

---

## Numerical MUF data

Future versions could attempt to obtain actual:

```text
foF2
MUF
M(3000)F2
```

measurements from SWS ionogram/autoscaled data.

This would allow the system to move from:

```text
"Norfolk: near normal"
```

toward:

```text
"Norfolk MUF: 18.4 MHz"
"foF2: 6.2 MHz"
```

which would provide considerably more information to the band-analysis engine.

---

## Actual propagation observations

Future integrations could combine ionospheric conditions with real amateur-radio observations:

```text
SWS ionosphere
       +
NOAA space weather
       +
PSK Reporter
       +
WSPR
       +
DX cluster
       +
SDR observations
       ↓
Observed propagation state
```

This would eventually allow RadioPathwayTool to distinguish between:

```text
10m theoretically possible
```

and:

```text
10m is actually open right now
```

---

# 14. Design Principle

`ionosphere.py` should remain primarily a **data acquisition and normalization module**.

It should answer:

> "What is the ionosphere currently reporting?"

It should not answer:

> "Should I use 10 metres?"

That second question belongs to the analysis layer.

Keeping those responsibilities separate makes the system easier to test, expand, and eventually connect to Discord or another interface.

---

# 15. Current Status

**Status: v1 complete**

Current implementation successfully:

* connects to the official SWS MUF Report
* retrieves current report data
* identifies the report timestamp
* identifies monitoring stations
* extracts station conditions
* extracts percentage deviations
* handles missing MUF data
* converts results into structured Python observations
* provides data suitable for downstream propagation analysis

The next development stage should focus on **integrating these observations with `band_favorability.py`**, rather than substantially expanding the scraper itself.
