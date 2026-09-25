# 📡 RadioPathwayTool

**RadioPathwayTool** is a modular amateur-radio propagation and space-weather analysis system designed to help operators understand **when, where, and why HF radio bands are likely to be usable**.

The project combines space-weather observations, ionospheric conditions, propagation models, band-specific analysis, and real-time radio observations into a single system.

It is being developed primarily for **HF amateur-radio operators, DXers, and radio propagation enthusiasts**.

---

## 🎯 Goals

RadioPathwayTool aims to:

* Monitor current space-weather conditions.
* Analyse ionospheric and HF propagation conditions.
* Estimate propagation favourability across amateur-radio bands.
* Identify potentially useful band openings.
* Compare predicted conditions against real-world radio observations.
* Generate automated propagation reports.
* Provide alerts when significant propagation changes occur.
* Make propagation information accessible through multiple interfaces.
* Maintain historical data for future analysis and model improvement.

### Planned interfaces

The system is intended to support several ways of interacting with the same backend:

* 🌐 **Web dashboard** — forward-facing browser interface.
* 🤖 **Discord bot** — existing interface for reports and interaction.
* 📱 **Telegram bot** — planned mobile interface.
* 🔔 **Push notifications** — planned notification system using services such as ntfy.

All interfaces should use the same underlying propagation engine rather than implementing separate propagation logic.

---

# 🏗️ Architecture

The project is designed as a modular system where individual components can be developed and replaced independently.

```text
                         ┌─────────────────────┐
                         │   External Sources  │
                         │                     │
                         │ NOAA / NASA / SWS   │
                         │ RBN / DX data       │
                         │ Propagation data    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Data Collection   │
                         │                     │
                         │ Space Weather       │
                         │ Propagation Data    │
                         │ Radio Observations  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │       Analysis Engine        │
                    │                              │
                    │ Propagation Engine           │
                    │ Ionosphere Analysis          │
                    │ Space Weather Analysis       │
                    │ Band Favorability            │
                    └──────────────┬───────────────┘
                                   │
                 ┌─────────────────┼─────────────────┐
                 │                 │                 │
                 ▼                 ▼                 ▼
          🌐 Web Interface    🤖 Discord       📱 Telegram
                 │                 │                 │
                 └─────────────────┼─────────────────┘
                                   │
                                   ▼
                            🔔 Notifications
```

The central principle is:

> **Interfaces should consume the same analysis engine rather than duplicate its logic.**

---

# 📂 Current Project Structure

The project is currently being consolidated into a simpler Python-based architecture.

```text
RadioProgram/
│
├── propagation_engine.py
├── ionosphere.py
├── space_weather.py
├── band_favorability.py
├── ai_assistant.py
│
├── main.py
├── bot.py
├── discord_bot.py
│
├── venv/
│
└── README.md
```

### Core modules

| File                    | Purpose                                                |
| ----------------------- | ------------------------------------------------------ |
| `propagation_engine.py` | Main propagation calculations and route analysis       |
| `ionosphere.py`         | Ionospheric condition analysis                         |
| `space_weather.py`      | Space-weather data collection and processing           |
| `band_favorability.py`  | Determines how favourable individual amateur bands are |
| `ai_assistant.py`       | AI-assisted interpretation of propagation information  |
| `main.py`               | Main application / testing entry point                 |
| `discord_bot.py`        | Discord interface                                      |
| `bot.py`                | Earlier bot implementation / legacy functionality      |

The structure will evolve as the system grows.

---

# ☀️ Data Sources

RadioPathwayTool is intended to combine multiple sources rather than relying on a single propagation indicator.

## Space Weather

Potential sources include:

* **NOAA Space Weather Prediction Center**

  * Solar flux
  * Sunspot data
  * K-index
  * A-index
  * Geomagnetic conditions
  * Solar X-ray flux
  * Solar wind information

* **NASA**

  * Solar observations
  * CME information
  * Solar-wind observations
  * Space-weather events

* **Australian Bureau of Meteorology Space Weather Services**

  * HF propagation information
  * Regional propagation charts
  * Path-specific propagation predictions

## Amateur Radio / Real-World Observations

Potential sources include:

* Reverse Beacon Network (RBN)
* PSK Reporter
* DX cluster information
* DXMaps
* Amateur-radio propagation reports
* Other publicly available amateur-radio observations

These observations are particularly useful because they provide **real-world evidence of propagation**, rather than relying entirely on theoretical models.

---

# 📡 Propagation Analysis

The system analyses propagation on a band-by-band basis.

Initial focus includes:

* 80 m
* 40 m
* 20 m
* 15 m
* 10 m

Additional amateur bands can be added later.

The system considers factors such as:

* Frequency
* Solar activity
* Geomagnetic activity
* Time of day
* Solar illumination
* Ionospheric conditions
* Geographic path
* Distance
* MUF
* Expected propagation mode
* Real-world amateur-radio reports

---

# 🌎 Path Analysis

Propagation is not treated as simply:

```text
"20 m = good"
```

Instead, the system aims to answer questions such as:

```text
Nelson, New Zealand
        ↓
Sydney, Australia
        ↓
20 m
        ↓
What are the current conditions?
```

Different paths can have significantly different propagation conditions at the same time.

The system therefore supports path-specific analysis.

Example:

```text
FROM: Nelson, New Zealand
TO:   Sydney, Australia

Band: 20 m

Current conditions:
    MUF:        ...
    Ionosphere: ...
    Solar:      ...
    Geomagnetic: ...

Assessment:
    ...

Best window:
    ...
```

---

# 📊 Band Favorability

`band_favorability.py` provides a way of converting multiple propagation factors into a band-specific assessment.

The system should avoid treating simple thresholds as absolute rules.

For example:

```text
SFI > 150
K < 2
A < 10
```

may indicate favourable conditions for some propagation scenarios, but these values **do not guarantee that a band is open**.

The project therefore aims to combine:

```text
Space weather
      +
Ionospheric conditions
      +
Path geometry
      +
Propagation model
      +
Real-world observations
      =
Propagation assessment
```

---

# 🚀 Spontaneous Band Openings

One of the project's objectives is detecting unexpected or short-lived openings.

For example:

```text
10 m

Normally:
    Poor

Suddenly:
    Multiple stations detected
    across a region

        ↓

Potential opening detected
```

Real-world observations can be used to validate whether an apparent opening is actually occurring.

This is particularly useful for bands such as **10 m**, where propagation can change rapidly.

---

# 🔭 Propagation Modelling

The project can incorporate external propagation models and prediction services.

One important source being investigated is the **Australian Bureau of Meteorology Space Weather Services path-specific HF propagation system**.

Example path:

```text
Christchurch
     ↓
   Sydney
```

The system can use path-specific propagation information as an input to its own analysis rather than relying solely on generic band predictions.

---

# 🤖 AI Assistant

The project includes an AI-assisted interpretation layer.

The purpose is not to replace the propagation calculations.

Instead:

```text
Raw data
   ↓
Propagation engine
   ↓
Calculated conditions
   ↓
AI assistant
   ↓
Human-readable explanation
```

For example, a user could ask:

> What is the best band to Australia this afternoon?

The AI layer can interpret the calculated propagation information and explain the result in plain language.

---

# 💬 Interfaces

## Discord

Discord is currently the primary bot interface.

The Discord bot can be used to expose propagation information and reports through Discord.

Planned functionality includes:

```text
/propagation
/forecast
/spaceweather
/bands
/path
```

The exact command structure is still under development.

---

## 📱 Telegram

A Telegram interface is planned.

The goal is to provide a simple mobile interface without requiring the user to access a Discord server.

Example:

```text
User:
What's 20m looking like to Australia?

Bot:
20 m currently appears favourable for
the Australia path.

Best window:
...

Current conditions:
...
```

Telegram should use the same backend as the Discord bot.

---

## 🌐 Web Interface

A forward-facing web dashboard is planned.

The web interface will eventually provide a browser-based view of the system.

Possible dashboard:

```text
┌─────────────────────────────────────────┐
│         📡 RADIOPATHWAYTOOL             │
├─────────────────────────────────────────┤
│                                         │
│ CURRENT CONDITIONS                      │
│                                         │
│ 80 m     ████████░░                     │
│ 40 m     █████████░                     │
│ 20 m     ██████████                     │
│ 15 m     ███████░░░                     │
│ 10 m     ███░░░░░░░                     │
│                                         │
│ ☀ Solar conditions                      │
│ 🌎 Geomagnetic conditions                │
│ 📡 Active paths                         │
│                                         │
│ [ Propagation ] [ Space Weather ]       │
│ [ Forecast ]    [ Paths ]               │
│                                         │
└─────────────────────────────────────────┘
```

The web interface is intended to be built around the existing Python backend rather than creating a separate analysis system.

A likely implementation is:

* FastAPI
* Jinja templates
* JavaScript for interactive components
* REST/API endpoints
* Existing Python propagation modules

---

# 🔔 Alerts

The system can generate alerts when significant events occur.

Potential alerts include:

### Band opening

```text
📡 BAND ALERT

10 m propagation activity detected.

Multiple observations have been detected
on the path.

Band: 28 MHz
Region: ...
Time: ...
```

### Space weather

```text
☀️ SPACE WEATHER ALERT

Significant solar activity detected.

X-ray class: ...
K-index: ...
Solar flux: ...
```

### Propagation change

```text
📡 PROPAGATION UPDATE

20 m conditions have changed significantly
for the selected path.

Path:
Nelson → Australia

Change:
...
```

Alerts can eventually be delivered through:

* Discord
* Telegram
* Web notifications
* ntfy / push notifications

---

# 🗄️ Data Storage

Historical data will eventually be stored for analysis and trend detection.

Initial database options:

* SQLite
* PostgreSQL

SQLite is suitable for initial development.

PostgreSQL can be introduced if the project grows to require:

* Larger datasets
* Multiple concurrent users
* More complex queries
* Historical propagation analysis
* Multiple deployments

---

# ⏱️ Automation

The system is intended to operate continuously on a Linux server.

Potential collection intervals:

```text
Space weather:
    5–15 minutes

Propagation data:
    5–30 minutes

Band analysis:
    5–15 minutes

Alert evaluation:
    Continuous / scheduled
```

The exact intervals depend on the data source.

Possible scheduling systems include:

* systemd timers
* Cron
* APScheduler
* Celery

For the current single-server deployment, simple scheduling is preferred where practical.

---

# 🖥️ Deployment

The current target environment is:

```text
OS:
    Rocky Linux 8

Architecture:
    x86_64

Python:
    Python 3.12

Environment:
    Python virtual environment

Server:
    Linux VM
```

The system is designed to eventually run continuously as a collection of services.

Example:

```text
radio-pathway.service
radio-discord.service
radio-telegram.service
radio-web.service
```

---

# 🔐 Configuration & Security

API keys, bot tokens, and other credentials **must not be committed to Git**.

Use environment variables or a local `.env` file.

Example:

```text
NOAA_API_KEY=...
DISCORD_BOT_TOKEN=...
TELEGRAM_BOT_TOKEN=...
```

A `.gitignore` file should exclude:

```text
.env
*.key
*.token
__pycache__/
venv/
```

Bot credentials should be treated as secrets and regenerated if they are accidentally exposed.

---

# 🧪 Development Philosophy

RadioPathwayTool is being developed incrementally.

The intended development progression is:

```text
1. Collect reliable data
        ↓
2. Process and normalize data
        ↓
3. Build propagation calculations
        ↓
4. Validate calculations
        ↓
5. Add real-world observations
        ↓
6. Generate useful assessments
        ↓
7. Add alerts
        ↓
8. Add interfaces
```

The propagation engine should remain independent from the user interfaces wherever possible.

This allows:

```text
Discord
Telegram
Website
CLI
```

to all use the same underlying calculations.

---

# 🛣️ Roadmap

## Phase 1 — Core Propagation Engine

* [x] Space-weather data collection
* [x] Ionospheric analysis
* [x] Band favourability calculations
* [x] Initial propagation engine
* [x] Path-specific propagation work
* [ ] Further validation against observed contacts

## Phase 2 — Automated Analysis

* [ ] Automated scheduled collection
* [ ] Historical database
* [ ] Propagation trend analysis
* [ ] Real-time observation integration
* [ ] Band-opening detection
* [ ] Improved path scoring/analysis

## Phase 3 — Interfaces

* [x] Discord integration
* [ ] Telegram bot
* [ ] Web dashboard
* [ ] Mobile-friendly interface
* [ ] Push notifications

## Phase 4 — Advanced Features

* [ ] Historical propagation maps
* [ ] Interactive path maps
* [ ] Automatic opening detection
* [ ] PSK Reporter integration
* [ ] Reverse Beacon Network integration
* [ ] User-defined stations
* [ ] Saved propagation paths
* [ ] AI-assisted propagation explanations
* [ ] Long-term propagation statistics

---

# 📜 Project Status

**Active development**

RadioPathwayTool is currently a personal development project focused on understanding and predicting HF amateur-radio propagation from New Zealand.

The architecture and individual components are expected to change as the propagation models and data sources are validated.

---

# 📡 Intended Use

RadioPathwayTool is an **informational propagation-analysis tool**.

Predictions and assessments should be treated as estimates rather than guarantees. Actual HF propagation depends on many factors, including ionospheric variability, antenna systems, local noise, terrain, equipment, and changing space-weather conditions.

The ultimate goal is to combine multiple independent data sources with real-world amateur-radio observations to produce a more useful picture of current HF conditions.
