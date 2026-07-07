"""
github_store.py — uses a private GitHub repo as the persistent datastore.
Streamlit Cloud's disk is wiped on restarts; this keeps each athlete's
harvested Strava cache safe in a repo you own and can inspect.

Layout inside the data repo:
    athletes/{athlete_id}/gradepace_streams.parquet
    athletes/{athlete_id}/gradepace_activities.parquet
"""
import base64
import io
import posixpath

import pandas as pd
import requests

API = "https://api.github.com"


class GitHubStore:
    def __init__(self, repo, token):
        """repo: 'username/repo-name' | token: fine-grained PAT with Contents RW on that repo."""
        self.repo = repo
        self.headers = {
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _url(self, path):
        return f"{API}/repos/{self.repo}/contents/{path}"

    # ---------------- low-level bytes ----------------
    def read_bytes(self, path):
        """Raw file bytes, or None if the file doesn't exist. Works for files >1MB."""
        r = requests.get(
            self._url(path),
            headers={**self.headers, "Accept": "application/vnd.github.raw+json"},
            timeout=60,
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.content

    def _sha(self, path):
        """Current blob sha (needed to update an existing file). Uses the parent
        directory listing so it works even for files >1MB."""
        d, name = posixpath.split(path)
        url = self._url(d) if d else f"{API}/repos/{self.repo}/contents"
        r = requests.get(url, headers=self.headers, timeout=60)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        for entry in r.json():
            if entry.get("name") == name:
                return entry.get("sha")
        return None

    def write_bytes(self, path, data, message):
        body = {"message": message, "content": base64.b64encode(data).decode()}
        sha = self._sha(path)
        if sha:
            body["sha"] = sha
        r = requests.put(self._url(path), headers=self.headers, json=body, timeout=120)
        r.raise_for_status()
        return True

    # ---------------- parquet convenience ----------------
    def read_parquet(self, path):
        raw = self.read_bytes(path)
        if raw is None:
            return None
        return pd.read_parquet(io.BytesIO(raw))

    def write_parquet(self, path, df, message):
        buf = io.BytesIO()
        df.to_parquet(buf, index=False, compression="zstd")
        return self.write_bytes(path, buf.getvalue(), message)

    # ---------------- athlete cache ----------------
    def athlete_paths(self, athlete_id):
        base = f"athletes/{athlete_id}"
        return f"{base}/gradepace_streams.parquet", f"{base}/gradepace_activities.parquet"

    def load_athlete_cache(self, athlete_id, stream_cols, meta_cols):
        sp, mp = self.athlete_paths(athlete_id)
        streams = self.read_parquet(sp)
        meta = self.read_parquet(mp)
        if streams is None:
            streams = pd.DataFrame(columns=stream_cols)
        if meta is None:
            meta = pd.DataFrame(columns=meta_cols)
        return streams, meta

    def save_athlete_cache(self, athlete_id, streams, meta):
        sp, mp = self.athlete_paths(athlete_id)
        self.write_parquet(mp, meta, f"Update activity index for athlete {athlete_id}")
        self.write_parquet(sp, streams, f"Update stream cache for athlete {athlete_id}")
        return True
