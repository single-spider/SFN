# SFN Showcase Service

This is a bounded, local presentation service for the Panda peg-in-hole
simulation.  A run starts the actual PyBullet Panda insertion environment,
uses the selected SFSS, SFMS, or MFMS controller, and streams measured
joint/peg/hole telemetry over a WebSocket.  It does not expose arbitrary robot
motion commands.

Install the optional service dependencies from the project root:

```powershell
..\.venv\Scripts\python.exe -m pip install -e '.[showcase]'
```

Run the service locally:

```powershell
..\.venv\Scripts\python.exe scripts\showcase_serve.py
```

The browser client uses `GET /v1/health`, `POST /v1/sessions`, `POST
/v1/sessions/{session_id}/commands`, and `WS /v1/sessions/{session_id}/stream`.
Only the allowlisted shapes and the three presentation methods are accepted.
There are at most two five-minute sessions, and the only commands are
`start`, `pause`, `reset`, and `close`.

For public hosting, the website defaults to recorded telemetry replay.  The
live service is intentionally bound to a local workstation and should only be
shared through a temporary authenticated tunnel for a supervised demo; it is
not a hardware-control endpoint.

Replay documents use the versioned `sfn.showcase.replay/v2` schema and retain
the measured telemetry fields used by the browser visualisation.
