"""Spotify metadata fetcher with real audio feature extraction.

Replaces the previous version that fabricated audio features. Now fetches
the actual danceability/energy/valence/tempo/etc. for each track via the
audio_features module (Spotify API → librosa preview fallback).
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials

from audio_features import FeatureUnavailable, get_audio_features

load_dotenv()


def _get_spotify_client() -> spotipy.Spotify:
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set in .env")
    return spotipy.Spotify(
        client_credentials_manager=SpotifyClientCredentials(
            client_id=client_id, client_secret=client_secret
        )
    )


def search_playlists(sp: spotipy.Spotify, query: str, limit: int = 5) -> list:
    print(f"Searching Spotify for playlists matching: {query!r}")
    try:
        results = sp.search(q=query, type="playlist", limit=limit)
    except Exception as exc:
        print(f"Search failed: {exc}")
        return []

    items = (results or {}).get("playlists", {}).get("items") or []
    valid = [p for p in items if p and p.get("id") and p.get("name")]
    for i, p in enumerate(valid, 1):
        owner = (p.get("owner") or {}).get("display_name", "Unknown")
        track_count = (p.get("tracks") or {}).get("total", 0)
        print(f"  {i}. {p['name']} ({p['id']}) by {owner} — {track_count} tracks")
    return valid


def _get_track_metadata(sp: spotipy.Spotify, track_id: str) -> dict | None:
    try:
        track = sp.track(track_id)
    except Exception as exc:
        print(f"  ! could not fetch metadata for {track_id}: {exc}")
        return None
    return {
        "track_id": track_id,
        "track_name": track["name"],
        "artist": track["artists"][0]["name"] if track["artists"] else "",
        "album_name": track["album"]["name"],
        "release_date": track["album"]["release_date"],
        "duration_ms": track["duration_ms"],
        "popularity": track["popularity"],
        "preview_url": track.get("preview_url") or "",
    }


def fetch_spotify_metadata(
    playlist_id: str, sp: spotipy.Spotify | None = None, track_limit: int = 50
) -> pd.DataFrame:
    """Fetch playlist metadata + real audio features for each track."""
    if sp is None:
        sp = _get_spotify_client()

    print(f"Fetching playlist {playlist_id}...")
    track_ids: list[str] = []
    offset = 0
    while True:
        try:
            page = sp.playlist_items(playlist_id, fields="items(track(id))", offset=offset)
        except Exception as exc:
            print(f"  ! playlist_items failed at offset {offset}: {exc}")
            break
        items = page.get("items") or []
        if not items:
            break
        for item in items:
            tid = (item.get("track") or {}).get("id")
            if tid:
                track_ids.append(tid)
        offset += len(items)
        print(f"  collected {len(track_ids)} track ids so far")

    if not track_ids:
        print("No tracks found in playlist.")
        return pd.DataFrame()

    if len(track_ids) > track_limit:
        print(f"Limiting to first {track_limit} tracks for tractable runtime")
        track_ids = track_ids[:track_limit]

    rows = []
    spotify_audio_features_works = True
    librosa_count = 0
    skipped = 0

    for i, tid in enumerate(track_ids, 1):
        print(f"[{i}/{len(track_ids)}] {tid}")
        meta = _get_track_metadata(sp, tid)
        if meta is None:
            skipped += 1
            continue

        feature_sp = sp if spotify_audio_features_works else None
        try:
            features = get_audio_features(tid, meta["preview_url"], sp=feature_sp)
        except FeatureUnavailable as exc:
            print(f"  ! {exc}")
            skipped += 1
            continue

        source = features.pop("_source", "unknown")
        if source != "spotify_api":
            spotify_audio_features_works = False
            if source == "librosa_preview":
                librosa_count += 1

        meta.update(features)
        rows.append(meta)

    if not rows:
        print("No tracks produced usable features.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    print(
        f"\nDone: {len(df)} tracks with features "
        f"(librosa fallback used for {librosa_count}, skipped {skipped})"
    )
    return df


def main(argv: list[str] | None = None) -> int:
    sp = _get_spotify_client()

    fallback_playlists = [
        ("37i9dQZF1DXcBWIGoYBM5M", "Today's Top Hits"),
        ("37i9dQZF1DXa2EiKmMLhFD", "Release Radar"),
        ("37i9dQZEVXbNG2KDcFcKOF", "Spotify Viral 50"),
    ]

    playlist_id, playlist_name = None, None
    for query in ("pop", "hits"):
        candidates = search_playlists(sp, query)
        if candidates:
            playlist_id = candidates[0]["id"]
            playlist_name = candidates[0]["name"]
            print(f"Using search result: {playlist_name} ({playlist_id})")
            break

    if not playlist_id:
        for fid, fname in fallback_playlists:
            try:
                sp.playlist(fid, fields="id,name")
                playlist_id, playlist_name = fid, fname
                print(f"Using fallback: {playlist_name} ({playlist_id})")
                break
            except Exception as exc:
                print(f"Fallback {fname} unavailable: {exc}")

    if not playlist_id:
        print("Could not find any accessible playlist. Aborting.")
        return 1

    df = fetch_spotify_metadata(playlist_id, sp=sp)
    if df.empty:
        return 1

    assert playlist_name is not None  # set together with playlist_id above
    safe_name = "".join(c if c.isalnum() or c == " " else "_" for c in playlist_name)
    output_file = f"spotify_metadata_{safe_name.replace(' ', '_')}.xlsx"
    df.to_excel(output_file, index=False)
    print(f"Wrote {output_file} with {list(df.columns)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
