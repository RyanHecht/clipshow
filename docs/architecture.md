# Clip Show architecture (target: distributable Docker Compose stack)

Two containers, deployable on one host or split across machines:

- **web**: the hub. Jellyfin sign-in, the title directory, every title's search page, clip rendering,
  the job queue, and the API workers talk to. Light: CPU only (query embeddings + ffmpeg renders).
- **worker** (one or more): claims indexing jobs from web over HTTP, does the heavy work (decode,
  shot cuts, voice activity, thumbnails, embeddings) on its own hardware (GPU optional), and uploads
  the results. No shared filesystem with web is required; both need read access to the media.

## Jellyfin

Jellyfin is the identity provider and the library catalogue; no plugin.

- The Jellyfin server is fixed by `JELLYFIN_URL` at startup (never settable from the website).
- Sign in with Jellyfin username/password (`POST /Users/AuthenticateByName`) or Quick Connect.
  The Jellyfin token is kept server-side in the session. Jellyfin admins can commission and manage
  titles; other users can search.
- Commissioning: browse the server's series (`/Items?IncludeItemTypes=Series`), pick one; episodes,
  file paths and subtitle streams come from `/Shows/{id}/Episodes`. Jellyfin's file paths are mapped
  to container paths with configured prefix rules (e.g. `/mnt/pool/media` -> `/media`).
- "Open in Jellyfin" opens the episode in a new tab (`/web/#/details?id=...`); the hub then looks for
  the user's newly active web session and starts the episode at the moment
  (`POST /Sessions/{id}/Playing?startPositionTicks=...`). If that fails, the user still has the page.

## Storage

```
$DATA/
  hub.sqlite               titles, settings, sessions, jobs
  titles/<slug>/
    index.sqlite           episodes, moments, FTS, vectors, speakers, cached cuts/VAD
    thumbs/ renders/ subs/ transcripts/
```

Per-title settings (skip patterns, theme detection, speaker aliases, transcript source) live in the
hub DB and are editable in the UI.

## Workers

Workers claim jobs and fetch what they need over HTTP (`/api/worker/*`, with a shared token), index on
their own hardware, and upload each finished episode: moments, embeddings, subtitles, and a tar of
thumbnails. The web app stores them, so it is the only writer of the SQLite databases and the titles'
folders, and nothing but the media is shared between machines. Workers need the media at the same path
as the web app (both containers mount it at `/media`).

## Status

- Done: multi-title hub; Jellyfin sign-in (password and Quick Connect), admin roles, commissioning from
  Jellyfin libraries, folder mapping, "Open in Jellyfin" with seek; per-title themes; favorites; the
  render cache; networked workers; container images (web; worker in cpu / cuda / rocm flavors) and
  compose files for one machine or a separate worker.
- Later: uploading transcripts in the admin page, subtitles from speech (Whisper) for episodes without
  any, reading speaker tags from WebVTT/ASS subtitles.
