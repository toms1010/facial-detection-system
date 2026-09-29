# Privacy

## What this application does

Everything runs on your machine. Face detection, expression classification,
tracking and hardware telemetry all execute locally. There is no server
component and no account required to use the vision features.

## The guarantees, and how they are enforced

| Guarantee | How it is enforced |
|---|---|
| No network egress | `tests/test_privacy.py` parses every module's AST and fails if an HTTP or socket client is imported. There is no HTTP client dependency at all. |
| No frames written to disk | All three storage flags default to `false`. `save_frame` returns `None` when they are off. |
| Deletable at any time | Settings has a "Delete stored camera data now" button; the app wipes on exit by default. |
| Camera can be switched off | The Settings toggle releases the device immediately. |
| Host name redacted | Hardware snapshots replace the hostname with `redacted` before storage, so telemetry is safe to paste into a bug report. |
| No biometric storage | The schema has no column for a face image or an embedding. A test asserts this over every table. |

## What is stored, by default

- Settings, in `~/.config/linux-ai-vision/settings.json`
- Rotating logs, in `~/.local/share/linux-ai-vision/logs`

That is all. Log messages never contain frame data, and IP addresses and home
directory paths are scrubbed before they are written.

## If you enable storage

With `privacy.store_frames`, `store_snapshots` or `store_face_crops` turned on,
images are written to `~/.local/share/linux-ai-vision/`. These are opt-in
per-flag and independent of one another. The privacy settings do **not** include
`allow_network_upload` as a master switch, because nothing in the application
can upload anything — the flag exists only so the intent is visible.

## With the optional database

Enabling MySQL adds metadata storage: dataset paths, labels, model versions,
training metrics, experiment notes and, if you record them, detection sessions.
`detection_sessions` stores **measurements only** — a tracking ID, a bounding
box, an expression label, a confidence, a latency. No image, embedding or
biometric template.

Detection data is not collected automatically. Sessions are created only by
explicit calls from application code that you trigger.

## Limits of what a camera can infer

This matters as much as the storage question. A classifier reads pixels. It
cannot see intent, thought, tone of voice, or the difference between a genuine
expression and a posed one. It performs worse on faces that are small, turned
away, partially occluded, or lit unevenly, and its error rates are not evenly
distributed across skin tones or across the cultures that associate particular
facial movements with particular expressions.

**Do not use expression output to make decisions about people** — hiring,
grading, access control, health, or surveillance. Doing so is both unreliable
and, in many jurisdictions, unlawful.

## Reporting a problem

If you find a route by which data leaves the machine, please report it. That
would be a serious defect and it is treated as one.
