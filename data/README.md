# Data

## letterboxd_export/

Personal Letterboxd export. Download the ZIP from
https://letterboxd.com/data/export/ and extract here.

### Files consumed by ingestion

| File | Signal |
|------|--------|
| `ratings.csv` | Core taste signal (star ratings) |
| `watched.csv` | Full watch history (includes unrated) |
| `watchlist.csv` | Intent / want-to-watch |
| `likes/films.csv` | Liked films (no star rating) |

Ingestion reads whatever is present; empty files are skipped (no-op).
Other files in the export are Letterboxd internals and ignored.

This folder's contents are gitignored. Do not commit personal export data.
