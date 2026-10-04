#!/usr/bin/env python3
import sys
import os
import json
import hashlib
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, jsonify, request, send_file, render_template, make_response, abort

import pillow_heif
import numpy as np
from PIL import Image, ImageOps

pillow_heif.register_heif_opener()

app = Flask(__name__, template_folder='templates')


def _count_heics(dirpath, exts=None):
    """Count media files in a directory using os.scandir (fast). Bails after
    2000 entries. `exts` defaults to HEIC-only (used by tools, like the Map
    Generator, that only ever read HEIC); pass a broader set (e.g. PT_PHOTO_EXTS)
    for tools that also handle JPG/PNG."""
    exts = exts or {'.heic'}
    count = 0
    scanned = 0
    try:
        with os.scandir(dirpath) as it:
            for e in it:
                scanned += 1
                if scanned > 2000:
                    break
                try:
                    if e.is_file() and Path(e.name).suffix.lower() in exts:
                        count += 1
                except OSError:
                    pass
    except (PermissionError, OSError):
        pass
    return count


def _folder_tag_status(dirpath, exts=None):
    """Return {tagged, total} if _ratings.json exists, else None."""
    ratings_file = Path(dirpath) / '_ratings.json'
    if not ratings_file.exists():
        return None
    try:
        with open(ratings_file) as f:
            ratings = json.load(f)
        total = _count_heics(dirpath, exts)
        if total == 0:
            return None
        tagged = sum(
            1 for v in ratings.values()
            if isinstance(v, dict) and any(v.get(k) for k in ('me', 'blog', 'hidden'))
        )
        return {'tagged': tagged, 'total': total}
    except Exception:
        return None


def _listdirs_json(path_str, default_to_home=True, exts=None):
    raw = path_str or (str(Path.home()) if default_to_home else '')
    p = Path(raw).expanduser().resolve()
    if not p.is_dir():
        return None
    items = []
    permission_error = False
    try:
        with os.scandir(p) as it:
            for e in it:
                try:
                    if e.is_dir() and not e.name.startswith('.'):
                        items.append((e.name, e.path))
                except OSError:
                    pass
    except PermissionError:
        permission_error = True
    except OSError:
        pass
    items.sort(key=lambda x: x[0].lower())
    entries = [
        {'name': name, 'path': dpath,
         'heic_count': _count_heics(dpath, exts),
         'tag_status': _folder_tag_status(dpath, exts)}
        for name, dpath in items
    ]
    parent = str(p.parent) if str(p.parent) != str(p) else None
    return {'path': str(p), 'parent': parent, 'entries': entries,
            'permission_error': permission_error}


# ── Photo Tagger state ────────────────────────────────────────────────────────
# PT_FOLDER lives in memory, so a server restart (e.g. relaunching to pick up a
# code change) used to silently wipe it while an already-open browser tab kept
# working from its stale in-memory photo list — every save/star/note then 500'd
# and every image 404'd, with nothing surfaced to the user. Persisting the last
# folder to disk and restoring it on startup means a restart just quietly
# resumes where it left off instead of breaking an in-progress tagging session.
_PT_STATE_FILE = Path.home() / '.travel_blog_pt_state.json'


def _pt_load_last_folder():
    try:
        folder = json.loads(_PT_STATE_FILE.read_text()).get('folder')
        if folder and Path(folder).is_dir():
            return Path(folder)
    except Exception:
        pass
    return None


def _pt_persist_folder(folder):
    try:
        _PT_STATE_FILE.write_text(json.dumps({'folder': str(folder)}))
    except Exception:
        pass


# Formats the Photo Tagger can list/serve/crop. Keys double as the mimetype map
# for serving originals; HEIC is listed first only for readability. MP4 clips
# are listed and played alongside the stills (they can be tagged, noted, moved
# and deleted like any photo; crop/straighten/copy-as-PNG don't apply).
PT_MIME = {'.heic': 'image/heic', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
           '.png': 'image/png', '.mp4': 'video/mp4'}
PT_VIDEO_EXTS = {'.mp4'}
PT_PHOTO_EXTS = set(PT_MIME)   # everything the tagger lists, stills and clips alike

PT_FOLDER = _pt_load_last_folder()
_geocache: dict = {}   # (rounded lat, lon) → {place, landmark}
_geocache_cc: dict = {}  # (rounded lat, lon) → {country, city}, for the General table
PT_RATINGS_FILE = (PT_FOLDER / '_ratings.json') if PT_FOLDER else None
_pt_meta_cache = {}
_pt_cache_lock = threading.Lock()


def _dms_to_decimal(dms, ref):
    d, m, s = float(dms[0]), float(dms[1]), float(dms[2])
    dec = d + m / 60 + s / 3600
    if ref in ('S', 'W'):
        dec = -dec
    return round(dec, 6)


def _utc_sort_key(date_raw, exif_ifd, gps):
    """True UTC instant ('YYYY-MM-DD HH:MM:SS') for chronological sorting.

    The EXIF DateTime tag is naive local wall-clock time: photos taken while
    the phone was on a different timezone (flight days) would otherwise
    interleave with same-wall-clock photos hours apart in real time. Prefer
    local time minus OffsetTimeOriginal; fall back to the GPS timestamp
    (always UTC); fall back to the naive local time itself.
    """
    local = None
    if date_raw:
        try:
            local = datetime.strptime(date_raw, '%Y:%m:%d %H:%M:%S')
        except ValueError:
            pass
    offset = exif_ifd.get(0x9011) or exif_ifd.get(0x9010)  # OffsetTimeOriginal / OffsetTime
    if local and offset:
        m = re.fullmatch(r'([+-])(\d{2}):(\d{2})', offset.strip())
        if m:
            sign = 1 if m.group(1) == '+' else -1
            delta = timedelta(hours=int(m.group(2)), minutes=int(m.group(3)))
            return (local - sign * delta).strftime('%Y-%m-%d %H:%M:%S')
    gps_date, gps_time = gps.get(29), gps.get(7)  # GPSDateStamp, GPSTimeStamp (UTC)
    if gps_date and gps_time:
        try:
            h, mi, s = (int(float(v)) for v in gps_time)
            base = datetime.strptime(gps_date, '%Y:%m:%d')
            return (base + timedelta(hours=h, minutes=mi, seconds=s)).strftime('%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError):
            pass
    return local.strftime('%Y-%m-%d %H:%M:%S') if local else None


# MP4/QuickTime timestamps count seconds from 1904-01-01.
_MP4_EPOCH = datetime(1904, 1, 1)


def _mp4_creation_time(filepath):
    """Recording time from the MP4 'mvhd' header, or None. Both ends of the
    file are scanned: a recorder that doesn't rewrite for fast-start leaves
    'moov' at the tail (all of this user's clips do). The value is naive local
    wall-clock with no zone stored, so it's used as-is."""
    try:
        size = filepath.stat().st_size
        with open(filepath, 'rb') as f:
            head = f.read(min(size, 1 << 20))
            tail = b''
            if size > (1 << 20):
                f.seek(max(0, size - (4 << 20)))
                tail = f.read()
    except OSError:
        return None
    for chunk in (head, tail):
        i = chunk.find(b'mvhd')
        if i < 0:
            continue
        # box: [size][b'mvhd'][version:1][flags:3][creation_time:4 or 8]...
        version = chunk[i + 4] if i + 4 < len(chunk) else None
        try:
            if version == 0:
                raw = struct.unpack('>I', chunk[i + 8:i + 12])[0]
            elif version == 1:
                raw = struct.unpack('>Q', chunk[i + 8:i + 16])[0]
            else:
                continue
            stamp = _MP4_EPOCH + timedelta(seconds=raw)
        except (struct.error, OverflowError, ValueError):
            continue
        if datetime(1990, 1, 1) < stamp < datetime(2100, 1, 1):   # reject junk/zero headers
            return stamp
    return None


def _video_metadata(filepath):
    """Date + sort key for a clip. Unlike a photo — whose EXIF offset or GPS
    stamp pins a true UTC instant — an MP4 only carries naive local time, so a
    clip sorts by wall-clock; within a day's folder that's the same order.
    Falls back to the file's mtime when the header can't be read. MP4s carry no
    GPS here, so the coordinates stay empty and the map row just stays hidden."""
    stamp = _mp4_creation_time(filepath)
    if stamp is None:
        try:
            stamp = datetime.fromtimestamp(filepath.stat().st_mtime)
        except OSError:
            return {'date': None, 'sort_key': None, 'lat': None, 'lon': None, 'gmaps_url': None}
    text = stamp.strftime('%Y-%m-%d %H:%M:%S')
    return {'date': text, 'sort_key': text, 'lat': None, 'lon': None, 'gmaps_url': None}


def _extract_metadata(filepath):
    filepath = Path(filepath)
    if filepath.suffix.lower() in PT_VIDEO_EXTS:
        return _video_metadata(filepath)
    try:
        img = Image.open(filepath)
        exif = img.getexif()
        date_raw = exif.get(306)
        date_display = date_raw.replace(':', '-', 2) if date_raw else None
        gps = exif.get_ifd(0x8825)
        sort_key = _utc_sort_key(date_raw, exif.get_ifd(0x8769), gps)
        lat = lon = gmaps_url = None
        if gps:
            lat_dms, lat_ref = gps.get(2), gps.get(1)
            lon_dms, lon_ref = gps.get(4), gps.get(3)
            if lat_dms and lat_ref and lon_dms and lon_ref:
                lat = _dms_to_decimal(lat_dms, lat_ref)
                lon = _dms_to_decimal(lon_dms, lon_ref)
                gmaps_url = f"https://www.google.com/maps?q={lat},{lon}"
        return {'date': date_display, 'sort_key': sort_key, 'lat': lat, 'lon': lon, 'gmaps_url': gmaps_url}
    except Exception:
        return {'date': None, 'sort_key': None, 'lat': None, 'lon': None, 'gmaps_url': None}


def _heic_to_png(src, dst):
    """Losslessly convert a HEIC to PNG, baking in EXIF rotation. Skips if dst is already up to date."""
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return
    img = Image.open(src)
    img = ImageOps.exif_transpose(img)
    img.convert('RGB').save(dst, format='PNG')


# ── Display cache ───────────────────────────────────────────────────────────
# Safari re-decodes the full 24MP HEVC of every HEIC on each navigation
# (~300-900ms). A full-resolution JPEG re-encode (same pixels, no downscale)
# decodes much faster and serves instantly, so the everyday view uses this
# cache while zoom/crop/copy still read the untouched original HEIC.
PT_DISPLAY_CACHE = Path(tempfile.gettempdir()) / 'pt_display_cache'
_pt_warm_executor = ThreadPoolExecutor(max_workers=4)
_pt_warm_token = 0


def _display_jpg_path(src):
    key = hashlib.md5(str(Path(src).resolve()).encode()).hexdigest()
    return PT_DISPLAY_CACHE / f'{key}.jpg'


def _heic_to_display_jpg(src, dst):
    """Full-resolution JPEG re-encode of a HEIC for fast browser display — same
    pixel dimensions (no downscale), EXIF rotation baked in. Skips if up to date."""
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = ImageOps.exif_transpose(Image.open(src))
    img.convert('RGB').save(dst, format='JPEG', quality=92)


# ── Live Photo frame cache ──────────────────────────────────────────────────
# The scrubber used to seek a <video> element, but an iPhone Live Photo MOV is
# HEVC with only 2-4 sync samples in its whole ~2s, and browsers snap a seek to
# the nearest keyframe — so the "continuous" slider could only ever reach a
# handful of distinct stills. Instead every frame is decoded server-side once
# and cached as a preview JPEG, which makes the picker genuinely frame-by-frame
# (and lets Save grab that exact frame at full resolution).
PT_FRAME_CACHE = Path(tempfile.gettempdir()) / 'pt_frame_cache'
PT_FRAME_PREVIEW_PX = 1100   # long edge of a scrub preview — sharp on screen, quick to ship
PT_FRAME_MAX = 900           # sanity cap; a Live Photo is ~50-90 frames
PT_FRAME_CACHE_KEEP = 40     # videos kept extracted before the oldest are pruned


def _frame_cache_dir(mov):
    mov = Path(mov)
    key = hashlib.md5(f'{mov.resolve()}:{mov.stat().st_mtime_ns}'.encode()).hexdigest()
    return PT_FRAME_CACHE / key


def _prune_frame_cache(keep=PT_FRAME_CACHE_KEEP):
    """Drop the least recently extracted videos so /tmp can't grow without
    bound (a Live Photo's frames are ~8MB)."""
    try:
        dirs = sorted((d for d in PT_FRAME_CACHE.iterdir() if d.is_dir()),
                      key=lambda d: d.stat().st_mtime, reverse=True)
    except OSError:
        return
    for d in dirs[keep:]:
        shutil.rmtree(d, ignore_errors=True)


def _extract_frames(mov):
    """Decode every frame of `mov` into cached preview JPEGs, returning
    {count, fps, duration}. Re-extraction is skipped when the cache is already
    complete for this file+mtime. OpenCV honours the rotation matrix, so frames
    come out upright — the same orientation the still is shown in."""
    cache = _frame_cache_dir(mov)
    meta_file = cache / 'meta.json'
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text())
            try:
                os.utime(cache, None)  # keep recently used videos out of the prune tail
            except OSError:
                pass
            return meta
        except Exception:
            shutil.rmtree(cache, ignore_errors=True)
    import cv2  # imported lazily: only the frame picker needs it, and it's slow to load
    cap = cv2.VideoCapture(str(mov))
    if not cap.isOpened():
        raise RuntimeError('could not open the Live Photo video')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cache.mkdir(parents=True, exist_ok=True)
    i = 0
    try:
        while i < PT_FRAME_MAX:
            ok, frame = cap.read()
            if not ok:
                break
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img.thumbnail((PT_FRAME_PREVIEW_PX, PT_FRAME_PREVIEW_PX))
            img.save(cache / f'{i:04d}.jpg', format='JPEG', quality=82)
            i += 1
    finally:
        cap.release()
    if i == 0:
        shutil.rmtree(cache, ignore_errors=True)
        raise RuntimeError('no frames could be decoded')
    meta = {'count': i, 'fps': round(fps, 3), 'duration': round(i / fps, 3)}
    meta_file.write_text(json.dumps(meta))
    _prune_frame_cache()
    return meta


def _decode_frame_full(mov, index):
    """The one frame at `index` at the video's native resolution, as a PIL
    image. Reads forward rather than seeking: frame-accurate seeking is exactly
    what this codec doesn't support, and a whole Live Photo decodes in ~0.2s."""
    import cv2
    cap = cv2.VideoCapture(str(mov))
    if not cap.isOpened():
        raise RuntimeError('could not open the Live Photo video')
    try:
        for _ in range(index):
            if not cap.grab():
                raise RuntimeError('frame is past the end of the video')
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError('frame is past the end of the video')
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()


def _video_poster_path(src):
    key = hashlib.md5(str(Path(src).resolve()).encode()).hexdigest()
    return PT_DISPLAY_CACHE / f'{key}_poster.jpg'


def _make_video_poster(src, dst):
    """Still for a clip's <video poster>, so navigating onto it shows the scene
    instead of a black box while the browser fetches enough of the file to
    decode. Grabs ~1s in — the first frame is often a blurred half-exposure."""
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return
    import cv2  # lazily imported: only video thumbnails and the frame picker need it
    cap = cv2.VideoCapture(str(src))
    try:
        if not cap.isOpened():
            raise RuntimeError('could not open the video')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        grabbed = 0
        for _ in range(int(min(max(fps, 1), 30))):
            if not cap.grab():
                break
            grabbed += 1
        ok, frame = cap.retrieve() if grabbed else (False, None)
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok:
            raise RuntimeError('no frames could be decoded')
    finally:
        cap.release()
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img.thumbnail((1400, 1400))
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, format='JPEG', quality=85)


def _warm_one(src, token):
    if token != _pt_warm_token:
        return
    try:
        _heic_to_display_jpg(src, _display_jpg_path(src))
    except Exception:
        pass


def _kick_warm(folder):
    """Fire background jobs to pre-build the display JPEGs for a folder. A new
    call supersedes any in-flight warm via the token (stale jobs no-op)."""
    global _pt_warm_token
    _pt_warm_token += 1
    token = _pt_warm_token
    p = Path(folder)
    for f in sorted(p.glob('*.HEIC')) + sorted(p.glob('*.heic')):
        _pt_warm_executor.submit(_warm_one, f, token)


def _load_ratings():
    if PT_RATINGS_FILE and Path(PT_RATINGS_FILE).exists():
        with open(PT_RATINGS_FILE) as f:
            return json.load(f)
    return {}


def _save_ratings(ratings):
    with open(PT_RATINGS_FILE, 'w') as f:
        json.dump(ratings, f, indent=2)


def _load_ratings_at(folder):
    """Ratings for an arbitrary folder (not just the one being tagged), empty
    when it has no _ratings.json yet. Used when a photo is moved and its tags
    have to follow it into the target folder's tracker."""
    ratings_file = Path(folder) / '_ratings.json'
    if ratings_file.exists():
        try:
            with open(ratings_file) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_ratings_at(folder, ratings):
    with open(Path(folder) / '_ratings.json', 'w') as f:
        json.dump(ratings, f, indent=2)


def _matching_mov(heic_name):
    """Sidecar Live Photo video for a HEIC ('IMG_1.HEIC' -> its 'IMG_1.MOV'/'.mov'),
    or None. Matches by stem, case-insensitive extension."""
    if not PT_FOLDER:
        return None
    stem = Path(heic_name).stem
    for ext in ('.MOV', '.mov'):
        cand = Path(PT_FOLDER) / f'{stem}{ext}'
        if cand.exists():
            return cand
    return None


def _pt_photo_files():
    """All taggable photos in PT_FOLDER (HEIC, JPG/JPEG, PNG)."""
    if not PT_FOLDER:
        return []
    result = []
    try:
        with os.scandir(PT_FOLDER) as it:
            for e in it:
                if e.is_file() and Path(e.name).suffix.lower() in PT_PHOTO_EXTS:
                    result.append(e.name)
    except OSError:
        pass
    return sorted(result)


# ── Home ──────────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    resp = make_response(render_template('home.html'))
    resp.headers['Cache-Control'] = 'no-store'
    return resp


# ── Photo Tagger ──────────────────────────────────────────────────────────────

@app.route('/photo-tagger')
def photo_tagger():
    resp = make_response(render_template('photo_tagger.html'))
    resp.headers['Cache-Control'] = 'no-store'
    return resp


@app.route('/photo-tagger/api/status')
def pt_status():
    return jsonify({'folder': str(PT_FOLDER) if PT_FOLDER else None})


@app.route('/photo-tagger/api/folder', methods=['POST'])
def pt_set_folder():
    global PT_FOLDER, PT_RATINGS_FILE
    folder = (request.json or {}).get('folder', '').strip()
    p = Path(folder).expanduser().resolve()
    if not p.is_dir():
        return jsonify({'error': 'Not a valid directory'}), 400
    PT_FOLDER = p
    PT_RATINGS_FILE = p / '_ratings.json'
    _pt_persist_folder(PT_FOLDER)
    with _pt_cache_lock:
        _pt_meta_cache.clear()
    _kick_warm(PT_FOLDER)  # pre-build display JPEGs in the background
    return jsonify({'ok': True, 'folder': str(PT_FOLDER)})


@app.route('/photo-tagger/api/photos')
def pt_photos():
    files = _pt_photo_files()
    ratings = _load_ratings()
    photos = []
    for filename in files:
        with _pt_cache_lock:
            meta = _pt_meta_cache.get(filename)
        if meta is None:
            meta = _extract_metadata(Path(PT_FOLDER) / filename)
            with _pt_cache_lock:
                _pt_meta_cache[filename] = meta
        entry = ratings.get(filename, {'me': False, 'hidden': False, 'blog': False, 'note': ''})
        is_video = Path(filename).suffix.lower() in PT_VIDEO_EXTS
        photos.append({
            'filename': filename,
            'date': meta['date'],
            'sort_key': meta.get('sort_key') or meta['date'],
            'lat': meta['lat'],
            'lon': meta['lon'],
            'gmaps_url': meta['gmaps_url'],
            'tags': {'me': entry.get('me', False), 'blog': entry.get('blog', False), 'hidden': entry.get('hidden', False)},
            'note': entry.get('note', ''),
            'kind': 'video' if is_video else 'photo',
            # Live Photo sidecars belong to HEIC stills; an MP4 is the clip itself.
            'has_mov': (not is_video) and _matching_mov(filename) is not None,
        })
    photos.sort(key=lambda ph: (ph['sort_key'] is None, ph['sort_key'], ph['filename']))
    return jsonify({'folder': str(PT_FOLDER), 'photos': photos})


@app.route('/photo-tagger/api/image/<path:filename>')
def pt_image(filename):
    if not PT_FOLDER:
        abort(404)
    filepath = Path(PT_FOLDER) / filename
    mimetype = PT_MIME.get(filepath.suffix.lower())
    if not filepath.exists() or not mimetype:
        abort(404)
    return send_file(filepath, mimetype=mimetype)


@app.route('/photo-tagger/api/display/<path:filename>')
def pt_display(filename):
    """Fast display view. HEIC gets a full-resolution JPEG re-encode, cached
    (Safari decodes raw HEIC slowly); JPG/PNG already decode fast natively, so
    they're served as-is. Falls back to the original HEIC if re-encoding fails."""
    if not PT_FOLDER:
        abort(404)
    folder = Path(PT_FOLDER).resolve()
    filepath = (folder / filename).resolve()
    ext = filepath.suffix.lower()
    mimetype = PT_MIME.get(ext)
    if not str(filepath).startswith(str(folder)) or not mimetype or not filepath.exists():
        abort(404)
    if ext != '.heic':
        return send_file(filepath, mimetype=mimetype, conditional=True)
    dst = _display_jpg_path(filepath)
    try:
        _heic_to_display_jpg(filepath, dst)
    except Exception:
        return send_file(filepath, mimetype='image/heic')
    return send_file(dst, mimetype='image/jpeg', conditional=True)


@app.route('/photo-tagger/api/poster/<path:filename>')
def pt_poster(filename):
    """Poster still for an MP4 in the current folder, cached next to the
    display JPEGs."""
    if not PT_FOLDER:
        abort(404)
    folder = Path(PT_FOLDER).resolve()
    filepath = (folder / filename).resolve()
    if filepath.parent != folder or filepath.suffix.lower() not in PT_VIDEO_EXTS \
            or not filepath.exists():
        abort(404)
    dst = _video_poster_path(filepath)
    try:
        _make_video_poster(filepath, dst)
    except Exception:
        abort(404)
    return send_file(dst, mimetype='image/jpeg', conditional=True)


@app.route('/photo-tagger/api/video/<path:filename>')
def pt_video(filename):
    """Serve the Live Photo sidecar MOV for a HEIC so the browser can scrub it.
    conditional=True enables HTTP range requests, which <video> seeking needs."""
    if not PT_FOLDER:
        abort(404)
    mov = _matching_mov(filename)
    if not mov:
        abort(404)
    return send_file(mov, mimetype='video/quicktime', conditional=True)


@app.route('/photo-tagger/api/frames/<path:filename>')
def pt_frames(filename):
    """Extract (or reuse) every frame of a photo's Live Photo sidecar and report
    how many there are, so the picker's slider can be one notch per frame."""
    if not PT_FOLDER:
        abort(404)
    mov = _matching_mov(filename)
    if not mov:
        return jsonify({'error': 'No Live Photo video for this photo'}), 404
    try:
        meta = _extract_frames(mov)
    except Exception as e:
        return jsonify({'error': f'Frame extraction failed: {e}'}), 500
    return jsonify({'ok': True, **meta})


@app.route('/photo-tagger/api/frame-image/<int:index>/<path:filename>')
def pt_frame_image(index, filename):
    """One cached preview frame. Immutable for this file+mtime, so the browser
    can hold every frame of the video in cache while scrubbing."""
    if not PT_FOLDER:
        abort(404)
    mov = _matching_mov(filename)
    if not mov:
        abort(404)
    jpg = _frame_cache_dir(mov) / f'{index:04d}.jpg'
    if not jpg.exists():
        abort(404)
    resp = make_response(send_file(jpg, mimetype='image/jpeg', conditional=True))
    resp.headers['Cache-Control'] = 'private, max-age=86400, immutable'
    return resp


@app.route('/photo-tagger/api/star', methods=['POST'])
def pt_star():
    data = request.json
    filename, category, value = data.get('filename'), data.get('category'), bool(data.get('value'))
    if category not in ('me', 'hidden', 'blog'):
        return jsonify({'error': 'invalid category'}), 400
    ratings = _load_ratings()
    if filename not in ratings:
        ratings[filename] = {'me': False, 'hidden': False, 'blog': False, 'note': ''}
    ratings[filename][category] = value
    _save_ratings(ratings)
    return jsonify({'ok': True})


@app.route('/photo-tagger/api/note', methods=['POST'])
def pt_note():
    data = request.json or {}
    filename, note = data.get('filename'), data.get('note', '')
    if not filename:
        return jsonify({'error': 'Missing filename'}), 400
    ratings = _load_ratings()
    if filename not in ratings:
        ratings[filename] = {'me': False, 'hidden': False, 'blog': False, 'note': ''}
    ratings[filename]['note'] = note
    _save_ratings(ratings)
    return jsonify({'ok': True})


def _move_to_trash(filepath):
    """Move a file to ~/.Trash, renaming on collision. Returns the trash path."""
    filepath = Path(filepath)
    trash = Path.home() / '.Trash' / filepath.name
    if trash.exists():
        i = 1
        while (Path.home() / '.Trash' / f'{filepath.stem}_{i}{filepath.suffix}').exists():
            i += 1
        trash = Path.home() / '.Trash' / f'{filepath.stem}_{i}{filepath.suffix}'
    shutil.move(str(filepath), str(trash))
    return trash


@app.route('/photo-tagger/api/delete', methods=['POST'])
def pt_delete():
    filename = (request.json or {}).get('filename', '')
    if not PT_FOLDER or not filename:
        return jsonify({'error': 'Missing data'}), 400
    folder = Path(PT_FOLDER).resolve()
    filepath = (folder / filename).resolve()
    if not str(filepath).startswith(str(folder)):
        return jsonify({'error': 'Invalid path'}), 403
    if not filepath.exists():
        return jsonify({'error': 'File not found'}), 404
    _move_to_trash(filepath)
    ratings = _load_ratings()
    ratings.pop(filename, None)
    _save_ratings(ratings)
    with _pt_cache_lock:
        _pt_meta_cache.pop(filename, None)
    return jsonify({'ok': True})


@app.route('/photo-tagger/api/move', methods=['POST'])
def pt_move():
    """Move one photo (plus its Live Photo sidecar MOV) into another folder,
    carrying its me/blog/hidden tags and note over to that folder's
    _ratings.json. A photo that turns out to be filed under the wrong day can
    be re-filed mid-session without losing the tagging already done on it."""
    data = request.json or {}
    filename = (data.get('filename') or '').strip()
    target = (data.get('target') or '').strip()
    if not PT_FOLDER or not filename or not target:
        return jsonify({'error': 'Missing data'}), 400
    folder = Path(PT_FOLDER).resolve()
    filepath = (folder / filename).resolve()
    if filepath.parent != folder:
        return jsonify({'error': 'Invalid path'}), 403
    if not filepath.exists():
        return jsonify({'error': 'File not found'}), 404
    dest_dir = Path(target).expanduser().resolve()
    if not dest_dir.is_dir():
        return jsonify({'error': 'Target is not a folder'}), 400
    if dest_dir == folder:
        return jsonify({'error': 'Photo is already in that folder'}), 400

    mov = _matching_mov(filename)  # resolve while the photo is still here
    dest = _pi_unique_dest(dest_dir, filepath.name)
    try:
        shutil.move(str(filepath), str(dest))
    except Exception as e:
        return jsonify({'error': f'Move failed: {e}'}), 500
    # Keep the Live Photo pair together, under the new stem if the photo had to
    # be renamed to avoid a collision in the target folder.
    if mov:
        try:
            shutil.move(str(mov), str(_pi_unique_dest(dest_dir, f'{dest.stem}{mov.suffix}')))
        except Exception:
            pass

    warnings = []
    entry = _load_ratings_at(folder).get(filename)
    tagged = isinstance(entry, dict) and (
        any(entry.get(k) for k in ('me', 'blog', 'hidden')) or entry.get('note'))
    # Only create/touch the target's tracker when there is something to carry.
    if tagged:
        try:
            dest_ratings = _load_ratings_at(dest_dir)
            dest_ratings[dest.name] = entry
            _save_ratings_at(dest_dir, dest_ratings)
        except Exception as e:
            warnings.append(f'tags could not be written to {dest_dir}: {e}')
    if entry is not None and not warnings:
        try:
            ratings = _load_ratings_at(folder)
            ratings.pop(filename, None)
            _save_ratings_at(folder, ratings)
        except Exception as e:
            warnings.append(f'old tag entry could not be cleared: {e}')
    with _pt_cache_lock:
        _pt_meta_cache.pop(filename, None)
    return jsonify({'ok': True, 'filename': dest.name, 'target': str(dest_dir),
                    'renamed': dest.name != filename, 'tags_moved': bool(tagged),
                    'warning': '; '.join(warnings) or None})


@app.route('/photo-tagger/api/copy-image', methods=['POST'])
def pt_copy_image():
    """Convert the photo to PNG and put it on the macOS system clipboard."""
    filename = (request.json or {}).get('filename', '')
    if not PT_FOLDER or not filename:
        return jsonify({'error': 'Missing data'}), 400
    folder = Path(PT_FOLDER).resolve()
    filepath = (folder / filename).resolve()
    if not str(filepath).startswith(str(folder)):
        return jsonify({'error': 'Invalid path'}), 403
    if not filepath.exists():
        return jsonify({'error': 'File not found'}), 404
    png = Path(tempfile.gettempdir()) / f'pt_clipboard_{filepath.stem}.png'
    try:
        _heic_to_png(filepath, png)
    except Exception as e:
        return jsonify({'error': f'PNG conversion failed: {e}'}), 500
    # Pass the path as an argument instead of splicing it into the script,
    # so paths with quotes/special characters can't break the AppleScript.
    script = ('on run argv\n'
              'set the clipboard to (read (POSIX file (item 1 of argv)) as «class PNGf»)\n'
              'end run')
    result = subprocess.run(['osascript', '-e', script, str(png)],
                            capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return jsonify({'error': result.stderr.strip() or 'Clipboard copy failed'}), 500
    return jsonify({'ok': True})


_PT_CROP_FORMAT = {'.heic': 'HEIF', '.jpg': 'JPEG', '.jpeg': 'JPEG', '.png': 'PNG'}


@app.route('/photo-tagger/api/crop', methods=['POST'])
def pt_crop():
    """Crop the original photo at full native resolution and write a NEW
    adjacent file in the SAME format (never overwrites). The crop rectangle
    arrives in native image pixels (x, y, w, h) as measured against the
    exif-transposed, upright image — the same orientation the browser
    displays. EXIF (date/time, GPS) is copied so the crop sorts to the same
    instant and lands right after its source by filename
    ('IMG_0001.HEIC' -> 'IMG_0001_crop.HEIC')."""
    data = request.json or {}
    filename = data.get('filename', '')
    if not PT_FOLDER or not filename:
        return jsonify({'error': 'Missing data'}), 400
    folder = Path(PT_FOLDER).resolve()
    filepath = (folder / filename).resolve()
    if not str(filepath).startswith(str(folder)):
        return jsonify({'error': 'Invalid path'}), 403
    fmt = _PT_CROP_FORMAT.get(filepath.suffix.lower())
    if not filepath.exists() or not fmt:
        return jsonify({'error': 'File not found'}), 404
    try:
        x = int(round(float(data['x']))); y = int(round(float(data['y'])))
        w = int(round(float(data['w']))); h = int(round(float(data['h'])))
    except (KeyError, TypeError, ValueError):
        return jsonify({'error': 'Invalid crop rectangle'}), 400
    try:
        img = ImageOps.exif_transpose(Image.open(filepath))  # bake in rotation
        W, H = img.size
        x0, y0 = max(0, min(x, W)), max(0, min(y, H))
        x1, y1 = max(0, min(x + w, W)), max(0, min(y + h, H))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return jsonify({'error': 'Crop too small'}), 400
        cropped = img.crop((x0, y0, x1, y1))
        save_kwargs = {'format': fmt}
        if fmt == 'HEIF':
            save_kwargs['quality'] = 100
        elif fmt == 'JPEG':
            save_kwargs['quality'] = 95
            if cropped.mode not in ('RGB', 'L'):  # JPEG has no alpha channel
                cropped = cropped.convert('RGB')
        exif = img.info.get('exif')  # datetime + GPS; orientation already neutralised
        if exif:
            save_kwargs['exif'] = exif
        dest = _pi_unique_dest(folder, f'{filepath.stem}_crop{filepath.suffix}')
        cropped.save(dest, **save_kwargs)
    except Exception as e:
        return jsonify({'error': f'Crop failed: {e}'}), 500
    with _pt_cache_lock:
        _pt_meta_cache.pop(dest.name, None)
    return jsonify({'ok': True, 'filename': dest.name})


def _pt_order_quad(points):
    """Sort 4 arbitrary (x, y) points into (top-left, top-right,
    bottom-right, bottom-left) order, regardless of the order the user
    clicked them in. Standard corner-sort: TL/BR are the extremes of
    x+y, TR/BL are the extremes of x-y."""
    pts = np.array(points, dtype=np.float64)
    s = pts.sum(axis=1)
    diff = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(diff)]
    bl = pts[np.argmin(diff)]
    return tl, tr, br, bl


def _pt_perspective_coeffs(dst_pts, src_pts):
    """Solve the 8 coefficients PIL's Image.transform(..., Image.PERSPECTIVE)
    needs: for each destination pixel (x, y) it samples the source at
    ((a*x + b*y + c) / (g*x + h*y + 1), (d*x + e*y + f) / (g*x + h*y + 1)).
    Exact solve (4 point correspondences -> 8 equations, 8 unknowns) is the
    same homography-fitting math used by any perspective/document-flatten
    tool, just spelled out instead of hidden behind a CV library."""
    A, B = [], []
    for (x, y), (u, v) in zip(dst_pts, src_pts):
        A.append([x, y, 1, 0, 0, 0, -x * u, -y * u]); B.append(u)
        A.append([0, 0, 0, x, y, 1, -x * v, -y * v]); B.append(v)
    coeffs = np.linalg.solve(np.array(A, dtype=np.float64), np.array(B, dtype=np.float64))
    return coeffs.tolist()


@app.route('/photo-tagger/api/perspective-crop', methods=['POST'])
def pt_perspective_crop():
    """Straighten an arbitrary quadrilateral (4 corners the user placed over
    a tilted/skewed subject - a sign, a document, a building facade) into an
    upright rectangle, using a true projective transform (not just a crop).
    Points arrive in native image pixels against the exif-transposed image,
    same convention as /crop. Output size is picked from the quad's own
    longest top/bottom and left/right edges so the result isn't stretched
    beyond the resolution actually present. Saved as a NEW adjacent file,
    same format, EXIF copied ('IMG_0001.HEIC' -> 'IMG_0001_persp.HEIC')."""
    data = request.json or {}
    filename = data.get('filename', '')
    points = data.get('points')
    if not PT_FOLDER or not filename:
        return jsonify({'error': 'Missing data'}), 400
    folder = Path(PT_FOLDER).resolve()
    filepath = (folder / filename).resolve()
    if not str(filepath).startswith(str(folder)):
        return jsonify({'error': 'Invalid path'}), 403
    fmt = _PT_CROP_FORMAT.get(filepath.suffix.lower())
    if not filepath.exists() or not fmt:
        return jsonify({'error': 'File not found'}), 404
    try:
        if not isinstance(points, list) or len(points) != 4:
            raise ValueError
        pts = [(float(p[0]), float(p[1])) for p in points]
    except (TypeError, ValueError, IndexError):
        return jsonify({'error': 'Need exactly 4 points'}), 400
    try:
        img = ImageOps.exif_transpose(Image.open(filepath))  # bake in rotation
        W, H = img.size
        tl, tr, br, bl = _pt_order_quad(pts)
        for px, py in (tl, tr, br, bl):
            if not (0 <= px <= W and 0 <= py <= H):
                return jsonify({'error': 'Points must be within the photo'}), 400
        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)
        out_w = int(round(max(width_top, width_bottom)))
        out_h = int(round(max(height_left, height_right)))
        if out_w < 2 or out_h < 2:
            return jsonify({'error': 'Selection too small'}), 400
        dst_pts = [(0, 0), (out_w - 1, 0), (out_w - 1, out_h - 1), (0, out_h - 1)]
        src_pts = [tuple(tl), tuple(tr), tuple(br), tuple(bl)]
        coeffs = _pt_perspective_coeffs(dst_pts, src_pts)
        warped = img.transform((out_w, out_h), Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC)
        save_kwargs = {'format': fmt}
        if fmt == 'HEIF':
            save_kwargs['quality'] = 100
        elif fmt == 'JPEG':
            save_kwargs['quality'] = 95
            if warped.mode not in ('RGB', 'L'):  # JPEG has no alpha channel
                warped = warped.convert('RGB')
        exif = img.info.get('exif')  # datetime + GPS; orientation already neutralised
        if exif:
            save_kwargs['exif'] = exif
        dest = _pi_unique_dest(folder, f'{filepath.stem}_persp{filepath.suffix}')
        warped.save(dest, **save_kwargs)
    except Exception as e:
        return jsonify({'error': f'Perspective crop failed: {e}'}), 500
    with _pt_cache_lock:
        _pt_meta_cache.pop(dest.name, None)
    return jsonify({'ok': True, 'filename': dest.name})


# Clockwise degrees -> the transpose that performs it. Transposes reorder
# pixels instead of resampling them, so the rotation itself is lossless.
_PT_ROTATE = {90: Image.Transpose.ROTATE_270,
              180: Image.Transpose.ROTATE_180,
              270: Image.Transpose.ROTATE_90}


@app.route('/photo-tagger/api/rotate', methods=['POST'])
def pt_rotate():
    """Rotate a photo 90/180/270 degrees clockwise and write a NEW adjacent
    file in the SAME format (never overwrites), the way /crop does. The source
    is read upright (EXIF rotation baked in) so the result matches what was on
    screen, and EXIF is copied with Orientation neutralised so viewers don't
    re-rotate it ('IMG_0001.HEIC' -> 'IMG_0001_rot90.HEIC')."""
    data = request.json or {}
    filename = (data.get('filename') or '').strip()
    try:
        degrees = int(data.get('degrees'))
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid rotation'}), 400
    if not PT_FOLDER or not filename:
        return jsonify({'error': 'Missing data'}), 400
    if degrees not in _PT_ROTATE:
        return jsonify({'error': 'Rotation must be 90, 180 or 270'}), 400
    folder = Path(PT_FOLDER).resolve()
    filepath = (folder / filename).resolve()
    if filepath.parent != folder:
        return jsonify({'error': 'Invalid path'}), 403
    fmt = _PT_CROP_FORMAT.get(filepath.suffix.lower())
    if not filepath.exists() or not fmt:
        return jsonify({'error': 'File not found'}), 404
    try:
        img = ImageOps.exif_transpose(Image.open(filepath))   # bake in rotation
        rotated = img.transpose(_PT_ROTATE[degrees])
        save_kwargs = {'format': fmt}
        if fmt == 'HEIF':
            save_kwargs['quality'] = 100
        elif fmt == 'JPEG':
            save_kwargs['quality'] = 95
            if rotated.mode not in ('RGB', 'L'):   # JPEG has no alpha channel
                rotated = rotated.convert('RGB')
        exif = img.info.get('exif')   # datetime + GPS; orientation already neutralised
        if exif:
            save_kwargs['exif'] = exif
        dest = _pi_unique_dest(folder, f'{filepath.stem}_rot{degrees}{filepath.suffix}')
        rotated.save(dest, **save_kwargs)
    except Exception as e:
        return jsonify({'error': f'Rotate failed: {e}'}), 500
    with _pt_cache_lock:
        _pt_meta_cache.pop(dest.name, None)
    return jsonify({'ok': True, 'filename': dest.name})


@app.route('/photo-tagger/api/frame', methods=['POST'])
def pt_frame():
    """Save one frame of the Live Photo MOV as a NEW adjacent HEIC, decoded
    server-side at the video's native resolution (~1920x1440 — inherently lower
    than the still). The browser sends the frame index it is showing, so what
    gets saved is exactly the frame that was picked. Date/time + GPS are copied
    from the source HEIC so it sorts to the same instant and lands right after
    it ('IMG_0001.HEIC' -> 'IMG_0001_frame.HEIC')."""
    data = request.json or {}
    filename = (data.get('filename') or '').strip()
    try:
        index = int(data.get('index'))
    except (TypeError, ValueError):
        return jsonify({'error': 'Missing frame index'}), 400
    if not PT_FOLDER or not filename or index < 0:
        return jsonify({'error': 'Missing data'}), 400
    folder = Path(PT_FOLDER).resolve()
    src = (folder / filename).resolve()
    if src.parent != folder:
        return jsonify({'error': 'Invalid path'}), 403
    if not src.exists() or src.suffix.lower() != '.heic':
        return jsonify({'error': 'File not found'}), 404
    mov = _matching_mov(filename)
    if not mov:
        return jsonify({'error': 'No Live Photo video for this photo'}), 404
    try:
        frame = _decode_frame_full(mov, index).convert('RGB')
        # Copy EXIF from the still, with Orientation neutralised (the decoded
        # frame is already upright) so viewers don't re-rotate it.
        exif = ImageOps.exif_transpose(Image.open(src)).info.get('exif')
        save_kwargs = {'format': 'HEIF', 'quality': 100}
        if exif:
            save_kwargs['exif'] = exif
        dest = _pi_unique_dest(folder, f'{src.stem}_frame.HEIC')
        frame.save(dest, **save_kwargs)
    except Exception as e:
        return jsonify({'error': f'Frame save failed: {e}'}), 500
    with _pt_cache_lock:
        _pt_meta_cache.pop(dest.name, None)
    return jsonify({'ok': True, 'filename': dest.name})


@app.route('/photo-tagger/api/geocode', methods=['POST'])
def pt_geocode():
    import urllib.parse
    lat = (request.json or {}).get('lat')
    lon = (request.json or {}).get('lon')
    if lat is None or lon is None:
        return jsonify({'error': 'No coordinates'}), 400

    key = (round(lat, 4), round(lon, 4))
    if key in _geocache:
        return jsonify(_geocache[key])

    hdrs = {'User-Agent': 'TravelBlogTools/1.0 (jeremiaslcc@gmail.com)'}
    result = {'place': '', 'landmark': ''}

    # 1. Nominatim — city / neighbourhood
    try:
        url = (f'https://nominatim.openstreetmap.org/reverse'
               f'?lat={lat}&lon={lon}&format=json&zoom=14&addressdetails=1')
        req = urllib.request.Request(url, headers=hdrs)
        with urllib.request.urlopen(req, timeout=6) as r:
            ndata = json.loads(r.read())
        addr = ndata.get('address', {})
        parts = []
        for k in ('suburb', 'quarter', 'neighbourhood', 'city_district'):
            if addr.get(k):
                parts.append(addr[k]); break
        for k in ('city', 'town', 'village', 'municipality'):
            if addr.get(k):
                parts.append(addr[k]); break
        if addr.get('country'):
            parts.append(addr['country'])
        result['place'] = ', '.join(parts)
        result['city']  = addr.get('city') or addr.get('town') or addr.get('village', '')
    except Exception:
        pass

    # 2. Overpass — nearest named tourism/historic/amenity within 300 m
    try:
        query = (f'[out:json][timeout:6];'
                 f'(node["name"]["tourism"](around:300,{lat},{lon});'
                 f'node["name"]["historic"](around:300,{lat},{lon});'
                 f'node["name"]["amenity"~"^(museum|theatre|place_of_worship|arts_centre)$"]'
                 f'(around:300,{lat},{lon});'
                 f'way["name"]["tourism"](around:300,{lat},{lon});'
                 f'way["name"]["historic"](around:300,{lat},{lon}););'
                 f'out center;')
        body = urllib.parse.urlencode({'data': query}).encode()
        req  = urllib.request.Request(
            'https://overpass-api.de/api/interpreter', data=body,
            headers={**hdrs, 'Content-Type': 'application/x-www-form-urlencoded'})
        with urllib.request.urlopen(req, timeout=8) as r:
            odata = json.loads(r.read())
        els = odata.get('elements', [])
        if els:
            def _d(el):
                la = el.get('lat') or el.get('center', {}).get('lat', lat)
                lo = el.get('lon') or el.get('center', {}).get('lon', lon)
                return (la - lat)**2 + (lo - lon)**2
            closest = min(els, key=_d)
            result['landmark'] = closest.get('tags', {}).get('name', '')
    except Exception:
        pass

    _geocache[key] = result
    return jsonify(result)


@app.route('/open-privacy-settings')
def open_privacy_settings():
    subprocess.Popen(['open', 'x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles'])
    return jsonify({'ok': True})


@app.route('/photo-tagger/api/listdirs')
def pt_listdirs():
    data = _listdirs_json(request.args.get('path', ''), exts=PT_PHOTO_EXTS)
    if data is None:
        return jsonify({'error': 'not a directory'}), 400
    return jsonify(data)


# ── Day Splitter ──────────────────────────────────────────────────────────────

# Files this tool created/manages elsewhere — never treated as loose camera-roll output.
_DAY_SPLITTER_SKIP = {'_ratings.json', 'blogpost_skeleton.md', 'map.html'}

# Directories the Day Splitter must never sort in place. Running it on $HOME (or /,
# or /Users) sweeps loose dotfiles — .zshrc, .zsh_history, .gitconfig, .claude.json —
# into YYYY-MM-DD folders by mtime, silently wrecking the user's shell and logins.
_DAY_SPLITTER_PROTECTED = {Path.home().resolve(), Path.home().resolve().parent, Path('/')}


def _reject_day_splitter_dir(p):
    """Return an error string if p is unsafe to sort in place, else None."""
    if p in _DAY_SPLITTER_PROTECTED:
        return (f"Refusing to run on {p} — this is a protected directory. "
                "Point the Day Splitter at a photo folder, not your home directory.")
    return None


def _media_date(filepath):
    """Best-effort capture date. EXIF DateTimeOriginal (works for HEIC/JPG/PNG via
    Pillow) first, falling back to the file's mtime for videos and anything else
    Pillow can't open or that lacks EXIF."""
    try:
        img = Image.open(filepath)
        date_raw = img.getexif().get(306)
        if date_raw:
            return date_raw[:10].replace(':', '-')  # "YYYY-MM-DD"
    except Exception:
        pass
    try:
        return datetime.fromtimestamp(filepath.stat().st_mtime).strftime('%Y-%m-%d')
    except Exception:
        return 'unknown'


def _day_splitter_files(p):
    """Top-level files in p, split into (movable, aae). AAE sidecars are edit
    metadata with no value once orphaned from their photo — they get deleted
    rather than moved; everything else is a real photo/video to sort by day."""
    movable, aae = [], []
    for f in sorted(p.glob('*')):
        # Skip dotfiles — pathlib's glob('*') matches them (unlike the shell), and
        # they are never camera output; moving them (.zshrc, .gitconfig, …) is what
        # broke the user's shell/logins repeatedly.
        if not f.is_file() or f.name.startswith('.') or f.name in _DAY_SPLITTER_SKIP:
            continue
        if f.suffix.lower() == '.aae':
            aae.append(f)
        else:
            movable.append(f)
    return movable, aae


@app.route('/day-splitter')
def day_splitter():
    resp = make_response(render_template('day_splitter.html'))
    resp.headers['Cache-Control'] = 'no-store'
    return resp


@app.route('/day-splitter/api/listdirs')
def ds_listdirs():
    data = _listdirs_json(request.args.get('path', ''))
    if data is None:
        return jsonify({'error': 'not a directory'}), 400
    return jsonify(data)


@app.route('/day-splitter/api/preview', methods=['POST'])
def ds_preview():
    folder = (request.json or {}).get('folder', '').strip()
    p = Path(folder).expanduser().resolve()
    if not p.is_dir():
        return jsonify({'error': 'Not a valid directory'}), 400
    err = _reject_day_splitter_dir(p)
    if err:
        return jsonify({'error': err}), 400

    movable, aae = _day_splitter_files(p)
    groups = defaultdict(list)
    for f in movable:
        groups[_media_date(f)].append(f.name)

    days = sorted(
        [{'date': d, 'count': len(names), 'files': names} for d, names in groups.items()],
        key=lambda x: x['date']
    )
    return jsonify({
        'folder': str(p), 'days': days, 'total': sum(d['count'] for d in days),
        'aae_count': len(aae),
    })


@app.route('/day-splitter/api/run', methods=['POST'])
def ds_run():
    folder = (request.json or {}).get('folder', '').strip()
    p = Path(folder).expanduser().resolve()
    if not p.is_dir():
        return jsonify({'error': 'Not a valid directory'}), 400
    err = _reject_day_splitter_dir(p)
    if err:
        return jsonify({'error': err}), 400

    movable, aae = _day_splitter_files(p)
    moved = defaultdict(int)
    errors = []

    deleted_aae = 0
    for f in aae:
        try:
            _move_to_trash(f)
            deleted_aae += 1
        except Exception as e:
            errors.append({'file': f.name, 'error': str(e)})

    for f in movable:
        date = _media_date(f)
        dest_dir = p / date
        dest_dir.mkdir(exist_ok=True)
        try:
            shutil.move(str(f), str(dest_dir / f.name))
            moved[date] += 1
        except Exception as e:
            errors.append({'file': f.name, 'error': str(e)})

    return jsonify({
        'ok': True,
        'deleted_aae': deleted_aae,
        'moved': dict(moved),
        'errors': errors,
    })


# ── Map Generator ─────────────────────────────────────────────────────────────

_DOT_HEX = ['#E53935','#1976D2','#388E3C','#F57C00','#7B1FA2','#0097A7','#C2185B','#F9A825']

_mg_groups: dict | None = None   # last generated GPS groups (for HTML export)
_mg_colors: dict | None = None   # last generated group → color mapping


def _gps_from_heic(filepath):
    try:
        img = Image.open(filepath)
        gps = img.getexif().get_ifd(0x8825)
        ld, lr = gps.get(2), gps.get(1)
        nd, nr = gps.get(4), gps.get(3)
        if ld and lr and nd and nr:
            return _dms_to_decimal(ld, lr), _dms_to_decimal(nd, nr)
    except Exception:
        pass
    return None, None


# CARTO's clean "Voyager" basemap now needs a free key (5M tiles/month, no
# account needed — get one at https://carto.com/basemaps/apikey and set the
# CARTO_API_KEY env var, e.g. in launch.sh). Without a key we fall back to
# keyless OpenStreetMap tiles so maps always render.
CARTO_API_KEY = os.environ.get('CARTO_API_KEY', '').strip()


def _map_tile_js():
    """Leaflet tile-layer JS for the trip maps — a single Python source of truth
    shared by the exported map and the live preview. Returns raw JS."""
    if CARTO_API_KEY:
        return (
            "L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png?key="
            + CARTO_API_KEY + "', { subdomains: 'abcd', maxZoom: 20, "
            "attribution: '&copy; <a href=\"https://openstreetmap.org\">OpenStreetMap</a> "
            "&copy; <a href=\"https://carto.com\">CARTO</a>' }).addTo(map);"
        )
    return (
        "L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', "
        "{ subdomains: 'abc', maxZoom: 19, "
        "attribution: '&copy; <a href=\"https://openstreetmap.org\">OpenStreetMap</a> contributors' }).addTo(map);"
    )


def _build_map_html(groups, colors):
    """groups: {label: [[lat, lon], ...]}, colors: {label: '#hex'}
    → standalone, self-contained Leaflet page (light OSM-style tiles, pannable/zoomable)."""
    groups_json = json.dumps(groups)
    colors_json = json.dumps(colors)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trip Map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html, body {{ margin: 0; padding: 0; height: 100%; font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif; }}
  #map {{ width: 100%; height: 100%; }}
  #legend {{
    position: absolute; bottom: 16px; right: 16px; z-index: 1000;
    background: rgba(255,255,255,0.92); border-radius: 10px; box-shadow: 0 4px 16px rgba(0,0,0,0.18);
    padding: 10px 14px; font-size: 12px; color: #333; max-width: 240px;
  }}
  .legend-row {{ display: flex; align-items: center; gap: 7px; padding: 2px 0; }}
  .legend-dot {{ width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; border: 1px solid rgba(0,0,0,0.15); }}
  .leaflet-popup-content-wrapper {{ border-radius: 8px !important; }}
</style>
</head>
<body>
<div id="map"></div>
<div id="legend"></div>
<script>
  const groups = {groups_json};
  const colors = {colors_json};

  const map = L.map('map', {{ preferCanvas: true }});
  {_map_tile_js()}

  const bounds = L.latLngBounds();
  const legend = document.getElementById('legend');

  Object.entries(groups).forEach(([label, pts]) => {{
    const color = colors[label] || '#888';
    pts.forEach(([lat, lon]) => {{
      L.circleMarker([lat, lon], {{
        radius: 6, fillColor: color, color: 'rgba(255,255,255,0.95)',
        weight: 1.5, opacity: 1, fillOpacity: 0.9,
      }}).bindPopup(`<b>${{label}}</b>`).addTo(map);
      bounds.extend([lat, lon]);
    }});
    const row = document.createElement('div');
    row.className = 'legend-row';
    row.innerHTML = `<div class="legend-dot" style="background:${{color}}"></div><span>${{label}} (${{pts.length}})</span>`;
    legend.appendChild(row);
  }});

  if (bounds.isValid()) map.fitBounds(bounds, {{ padding: [40, 40], maxZoom: 16 }});
</script>
</body>
</html>"""


@app.route('/map-generator')
def map_generator():
    resp = make_response(render_template('map_generator.html', tile_js=_map_tile_js()))
    resp.headers['Cache-Control'] = 'no-store'
    return resp


@app.route('/map-generator/api/listdirs')
def mg_listdirs():
    data = _listdirs_json(request.args.get('path', ''))
    if data is None:
        return jsonify({'error': 'not a directory'}), 400
    return jsonify(data)


@app.route('/map-generator/api/subfolders', methods=['POST'])
def mg_subfolders():
    folder = (request.json or {}).get('folder', '').strip()
    p = Path(folder).expanduser().resolve()
    if not p.is_dir():
        return jsonify({'error': 'Not a valid directory'}), 400
    result = []
    root_count = _count_heics(str(p))
    if root_count:
        result.append({'name': f'{p.name} (root)', 'path': str(p), 'count': root_count})
    items = []
    try:
        with os.scandir(p) as it:
            for e in it:
                try:
                    if e.is_dir() and not e.name.startswith('.'):
                        items.append((e.name, e.path))
                except OSError:
                    pass
    except (PermissionError, OSError):
        pass
    for name, dpath in sorted(items, key=lambda x: x[0].lower()):
        count = _count_heics(dpath)
        if count:
            result.append({'name': name, 'path': dpath, 'count': count})
    return jsonify({'parent': str(p), 'folders': result})


@app.route('/map-generator/api/generate', methods=['POST'])
def mg_generate():
    global _mg_groups, _mg_colors
    selections = (request.json or {}).get('folders', [])
    if not selections:
        return jsonify({'error': 'No folders selected'}), 400

    def get_gps(f): return (f, *_gps_from_heic(f))

    groups: dict[str, list] = {}
    for sel in selections:
        p = Path(sel['path']).expanduser().resolve()
        label = sel['label']
        files = sorted(p.glob('*.HEIC')) + sorted(p.glob('*.heic'))
        if not files:
            continue
        with ThreadPoolExecutor(max_workers=10) as ex:
            results = list(ex.map(get_gps, files))
        pts = [[la, lo] for _, la, lo in results if la is not None]
        if pts:
            groups[label] = pts

    if not groups:
        return jsonify({'error': 'No GPS data found in selected folders'}), 400

    sorted_labels = sorted(groups.keys())
    colors = {lbl: _DOT_HEX[i % len(_DOT_HEX)] for i, lbl in enumerate(sorted_labels)}

    _mg_groups = groups
    _mg_colors = colors

    return jsonify({
        'ok': True,
        'groups': groups,
        'colors': colors,
        'total': sum(len(v) for v in groups.values()),
    })


@app.route('/map-generator/api/export', methods=['POST'])
def mg_export():
    if not _mg_groups:
        return jsonify({'error': 'Generate a map first'}), 400
    html = _build_map_html(_mg_groups, _mg_colors)
    resp = make_response(html)
    resp.headers['Content-Type'] = 'text/html; charset=utf-8'
    resp.headers['Content-Disposition'] = 'attachment; filename="trip_map.html"'
    resp.headers['Cache-Control'] = 'no-store'
    return resp


# ── Blogpost Skeleton ───────────────────────────────────────────────────────

TEMPLATE_PATH = Path(__file__).resolve().parent / 'Obsidian Vault' / 'Journal Template.md'
_DAY_HEADING_RE = re.compile(r'^Day (\d+|n) \+ pics$')


def _fmt_day(date_str):
    """'YYYY-MM-DD[ HH:MM:SS]' -> 'YY/MM/DD', matching the template's date format."""
    if not date_str or len(date_str) < 10:
        return ''
    y, m, d = date_str[:10].split('-')
    return f'{y[2:]}/{m}/{d}'


def _load_template_sections():
    """Parse Journal Template.md into an ordered [(title, body_lines)] list.
    Every heading in the template is a single '# ', so this is flat — no nesting."""
    text = TEMPLATE_PATH.read_text()
    sections = []
    title, body = None, []
    for line in text.split('\n'):
        if line.startswith('# '):
            if title is not None:
                sections.append((title, body))
            title, body = line[2:].strip(), []
        else:
            body.append(line)
    if title is not None:
        sections.append((title, body))
    return sections


def _sample_gps_points(groups, max_points=6):
    """Dedup GPS points to ~1km clusters and cap the count, so the Nominatim
    lookups below stay cheap — this is a best-effort fill-in, not a hard requirement."""
    seen, samples = set(), []
    for pts in groups.values():
        for lat, lon in pts:
            key = (round(lat, 2), round(lon, 2))
            if key in seen:
                continue
            seen.add(key)
            samples.append((lat, lon))
            if len(samples) >= max_points:
                return samples
    return samples


def _reverse_geocode_country_city(lat, lon):
    key = (round(lat, 4), round(lon, 4))
    if key in _geocache_cc:
        return _geocache_cc[key]
    result = {'country': '', 'city': ''}
    try:
        url = (f'https://nominatim.openstreetmap.org/reverse'
               f'?lat={lat}&lon={lon}&format=json&zoom=10&addressdetails=1')
        req = urllib.request.Request(url, headers={'User-Agent': 'TravelBlogTools/1.0 (jeremiaslcc@gmail.com)'})
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read())
        addr = data.get('address', {})
        result['country'] = addr.get('country', '')
        result['city'] = addr.get('city') or addr.get('town') or addr.get('village') or ''
    except Exception:
        pass
    _geocache_cc[key] = result
    return result


@app.route('/blogpost-skeleton')
def blogpost_skeleton():
    resp = make_response(render_template('blogpost_skeleton.html'))
    resp.headers['Cache-Control'] = 'no-store'
    return resp


@app.route('/blogpost-skeleton/api/listdirs')
def bs_listdirs():
    data = _listdirs_json(request.args.get('path', ''))
    if data is None:
        return jsonify({'error': 'not a directory'}), 400
    return jsonify(data)


@app.route('/blogpost-skeleton/api/subfolders', methods=['POST'])
def bs_subfolders():
    folder = (request.json or {}).get('folder', '').strip()
    p = Path(folder).expanduser().resolve()
    if not p.is_dir():
        return jsonify({'error': 'Not a valid directory'}), 400
    result = []
    root_count = _count_heics(str(p))
    if root_count:
        result.append({
            'name': f'{p.name} (root)', 'path': str(p), 'count': root_count,
            'tag_status': _folder_tag_status(str(p)),
        })
    items = []
    try:
        with os.scandir(p) as it:
            for e in it:
                try:
                    if e.is_dir() and not e.name.startswith('.'):
                        items.append((e.name, e.path))
                except OSError:
                    pass
    except (PermissionError, OSError):
        pass
    for name, dpath in sorted(items, key=lambda x: x[0].lower()):
        count = _count_heics(dpath)
        if count:
            result.append({
                'name': name, 'path': dpath, 'count': count,
                'tag_status': _folder_tag_status(dpath),
            })
    return jsonify({'parent': str(p), 'folders': result})


@app.route('/blogpost-skeleton/api/generate', methods=['POST'])
def bs_generate():
    data = request.json or {}
    parent = (data.get('parent') or '').strip()
    selections = data.get('folders', [])
    if not parent or not selections:
        return jsonify({'error': 'Missing parent folder or folder selection'}), 400
    parent_p = Path(parent).expanduser().resolve()
    if not parent_p.is_dir():
        return jsonify({'error': 'Not a valid parent directory'}), 400
    if not TEMPLATE_PATH.exists():
        return jsonify({'error': f'Journal Template.md not found at {TEMPLATE_PATH}'}), 500

    blog_photos_dir = parent_p / 'blog_photos'
    me_photos_dir = parent_p / 'me_photos'
    hidden_photos_dir = parent_p / 'hidden_photos'

    groups: dict[str, list] = {}            # label -> [[lat, lon], ...], for map.html
    all_dates = []                          # every photo's 'YYYY-MM-DD', for date range / unique days
    day_notes = defaultdict(list)           # 'YYYY-MM-DD' -> [(timestamp, note), ...]
    blog_photos = []                        # [(date_str, label, relative_path)]
    to_convert_blog = []                    # [(src_path, dst_path)]
    to_convert_me = []                      # [(src_path, dst_path)]
    to_move = []                            # [(src_path, dst_path)] — hard move, ✕ Hidden
    ratings_cleanup = []                    # [(ratings_file, ratings_dict, [filenames_moved])]

    for sel in sorted(selections, key=lambda s: s['label'].lower()):
        p = Path(sel['path']).expanduser().resolve()
        label = sel['label']
        files = sorted(p.glob('*.HEIC')) + sorted(p.glob('*.heic'))
        if not files:
            continue

        with ThreadPoolExecutor(max_workers=10) as ex:
            metas = list(ex.map(_extract_metadata, files))

        pts = [[m['lat'], m['lon']] for m in metas if m['lat'] is not None]
        if pts:
            groups[label] = pts
        all_dates += [m['date'][:10] for m in metas if m['date']]

        ratings_file = p / '_ratings.json'
        ratings = {}
        if ratings_file.exists():
            try:
                with open(ratings_file) as rf:
                    ratings = json.load(rf)
            except Exception:
                ratings = {}

        seen_notes_per_day = defaultdict(set)
        hidden_filenames_this_folder = []
        for filename, entry in ratings.items():
            if not isinstance(entry, dict):
                continue
            filepath = p / filename
            if not filepath.exists():
                continue

            note = entry.get('note', '').strip()
            if note:
                meta = _extract_metadata(filepath)
                ts = meta['date'] or ''
                day = ts[:10]
                if note not in seen_notes_per_day[day]:
                    seen_notes_per_day[day].add(note)
                    day_notes[day].append((ts, note))

            if entry.get('blog'):
                meta = _extract_metadata(filepath)
                out_path = blog_photos_dir / f'{label}_{Path(filename).stem}.png'
                to_convert_blog.append((filepath, out_path))
                rel = str(out_path.relative_to(parent_p))
                blog_photos.append((meta['date'] or '', label, rel))

            if entry.get('me'):
                out_path = me_photos_dir / f'{label}_{Path(filename).stem}.png'
                to_convert_me.append((filepath, out_path))

            if entry.get('hidden'):
                # Hard move — deferred until after the PNG-conversion batches below,
                # so a photo tagged both me/blog *and* hidden still gets converted
                # before its source file is relocated out from under it.
                out_path = hidden_photos_dir / f'{label}_{filename}'
                to_move.append((filepath, out_path))
                hidden_filenames_this_folder.append(filename)

        if hidden_filenames_this_folder:
            ratings_cleanup.append((ratings_file, ratings, hidden_filenames_this_folder))

    if to_convert_blog:
        blog_photos_dir.mkdir(exist_ok=True)
        with ThreadPoolExecutor(max_workers=6) as ex:
            list(ex.map(lambda pair: _heic_to_png(*pair), to_convert_blog))

    if to_convert_me:
        me_photos_dir.mkdir(exist_ok=True)
        with ThreadPoolExecutor(max_workers=6) as ex:
            list(ex.map(lambda pair: _heic_to_png(*pair), to_convert_me))

    if to_move:
        hidden_photos_dir.mkdir(exist_ok=True)
        for src, dst in to_move:
            shutil.move(str(src), str(dst))
        for ratings_file, ratings, filenames in ratings_cleanup:
            for fn in filenames:
                ratings.pop(fn, None)
            with open(ratings_file, 'w') as rf:
                json.dump(ratings, rf, indent=2)

    # ── Map (best-effort — only if any selected photo carried GPS) ─────────
    map_path = None
    if groups:
        sorted_labels = sorted(groups.keys())
        colors = {lbl: _DOT_HEX[i % len(_DOT_HEX)] for i, lbl in enumerate(sorted_labels)}
        html = _build_map_html(groups, colors)
        map_path = parent_p / 'map.html'
        map_path.write_text(html)

    blog_photos.sort(key=lambda t: t[0])

    # ── Trip date range + unique days ───────────────────────────────────────
    unique_days = sorted(set(d for d in all_dates if d))
    day_list = unique_days or ['']   # always emit at least one Day block
    date_range = ''
    if unique_days:
        start, end = _fmt_day(unique_days[0]), _fmt_day(unique_days[-1])
        date_range = start if start == end else f'{start} to {end}'

    # ── Countries / cities, best-effort from a sample of GPS points ─────────
    countries, cities = set(), set()
    try:
        for i, (lat, lon) in enumerate(_sample_gps_points(groups)):
            if i > 0:
                time.sleep(1.1)  # Nominatim usage policy: max 1 req/sec
            r = _reverse_geocode_country_city(lat, lon)
            if r['country']:
                countries.add(r['country'])
            if r['city']:
                cities.add(r['city'])
    except Exception:
        pass

    # ── Tagging notes: sorted by photo timestamp within each day ────────────
    notes_lines = []
    for day in sorted(day_notes.keys()):
        entries = sorted(day_notes[day], key=lambda t: t[0])
        if day in unique_days:
            day_label = f'Day {unique_days.index(day) + 1} — {_fmt_day(day)}'
        else:
            day_label = _fmt_day(day) if day else 'Day ? — unknown date'
        notes_lines.append(f'* {day_label}')
        for _, note in entries:
            notes_lines.append(f'    * {note}')

    # ── Build the journal file from Journal Template.md ─────────────────────
    lines = []
    for title, body in _load_template_sections():
        day_match = _DAY_HEADING_RE.match(title)
        if title == '(remove) tagging notes':
            lines.append(f'# {title}')
            lines.append('')
            lines.extend(notes_lines)
            lines.append('')
        elif title == 'General':
            lines.append(f'# {title}')
            lines.append('')
            lines.append(f'| Date    | {date_range} |')
            lines.append('| ------- | ------------------------ |')
            lines.append(f'| Country | {", ".join(sorted(countries))} |')
            lines.append(f'| Cities  | {", ".join(sorted(cities))} |')
            lines.append('| Type    | Solo / Friends / Partner |')
            lines.append('')
        elif title == 'Itemized itinerary':
            lines.append(f'# {title}')
            lines.append('')
            for n, day in enumerate(day_list, start=1):
                lines.append(f'{_fmt_day(day)} day {n}: Where')
                lines.append('* ')
            lines.append('')
        elif day_match:
            if day_match.group(1) != '1':
                continue  # emitted in one shot below, alongside "Day 1 + pics"
            for n, day in enumerate(day_list, start=1):
                lines.append(f'# Day {n} ({_fmt_day(day)}) + pics')
                if n == 1:
                    lines.append('if you revisited a place much maybe deep dive jus 1st')
                lines.append('')
        else:
            lines.append(f'# {title}')
            lines.extend(body)

    md_path = parent_p / 'blogpost_skeleton.md'
    md_path.write_text('\n'.join(lines))

    return jsonify({
        'ok': True,
        'parent': str(parent_p),
        'map_path': str(map_path) if map_path else None,
        'md_path': str(md_path),
        'blog_photos_dir': str(blog_photos_dir) if to_convert_blog else None,
        'me_photos_dir': str(me_photos_dir) if to_convert_me else None,
        'hidden_photos_dir': str(hidden_photos_dir) if to_move else None,
        'gps_points': sum(len(v) for v in groups.values()),
        'unique_days': len(day_list),
        'blog_photo_count': len(blog_photos),
        'me_photo_count': len(to_convert_me),
        'hidden_photo_count': len(to_move),
        'countries': sorted(countries),
        'cities': sorted(cities),
    })


@app.route('/blogpost-skeleton/api/preview-map')
def bs_preview_map():
    p = Path((request.args.get('parent') or '')).expanduser().resolve() / 'map.html'
    if not p.exists():
        abort(404)
    resp = make_response(send_file(p, mimetype='text/html'))
    resp.headers['Cache-Control'] = 'no-store'
    return resp


@app.route('/blogpost-skeleton/api/preview-md')
def bs_preview_md():
    p = Path((request.args.get('parent') or '')).expanduser().resolve() / 'blogpost_skeleton.md'
    if not p.exists():
        abort(404)
    return p.read_text(), 200, {'Content-Type': 'text/plain; charset=utf-8'}


@app.route('/blogpost-skeleton/api/reveal', methods=['POST'])
def bs_reveal():
    folder = (request.json or {}).get('parent', '').strip()
    p = Path(folder).expanduser().resolve()
    if not p.is_dir():
        return jsonify({'error': 'Not a valid directory'}), 400
    subprocess.Popen(['open', str(p)])
    return jsonify({'ok': True})


# ── Phone Import (iPhone via afcclient / libimobiledevice) ───────────────────

PHONE_EXPORT_DIR = Path.home() / 'photo_tool_export'
PHONE_MEDIA_EXTS = {'.heic', '.jpg', '.jpeg', '.png', '.mov', '.mp4', '.dng', '.gif', '.webp'}

_pi_state = {
    'running': False,
    'message': 'idle',
    'copied': 0,
    'skipped': 0,
    'error': None,
    'done': False,
}
_pi_lock = threading.Lock()


def _afc(*args, timeout=600):
    # afcclient's global option parser eats subcommand flags, hence the '--'
    return subprocess.run(
        ['afcclient', '--'] + list(args),
        capture_output=True, text=True, timeout=timeout,
    )


def _afc_ls(path):
    r = _afc('ls', path, timeout=60)
    if r.returncode != 0 or 'No device found' in r.stdout + r.stderr:
        raise RuntimeError('No iPhone found. Is it plugged in, unlocked and trusted?')
    return [line.strip() for line in r.stdout.splitlines() if line.strip()]


def _afc_mtime(path):
    """File mtime as epoch seconds, or None if unreadable."""
    r = _afc('info', path, timeout=60)
    for line in r.stdout.splitlines():
        if 'st_mtime' in line:
            digits = ''.join(c for c in line if c.isdigit())
            if len(digits) >= 10:
                return int(digits[:10])
    return None


def _pi_unique_dest(dest_dir, filename):
    """Never overwrite: IMG_0001.HEIC -> IMG_0001_1.HEIC, _2, ..."""
    dest = dest_dir / filename
    if not dest.exists():
        return dest
    stem, suffix = Path(filename).stem, Path(filename).suffix
    n = 1
    while True:
        dest = dest_dir / f'{stem}_{n}{suffix}'
        if not dest.exists():
            return dest
        n += 1


def _pi_set(**kw):
    with _pi_lock:
        _pi_state.update(kw)


def _pi_worker(start_ts, end_ts):
    try:
        _pi_set(message='Scanning phone folders…')
        folders = sorted(f for f in _afc_ls('/DCIM')
                         if f.endswith('APPLE') and not f.startswith('.'))
        copied = skipped = 0
        for i, folder in enumerate(folders):
            _pi_set(message=f'Scanning {folder} ({i + 1}/{len(folders)})…')
            files = sorted(f for f in _afc_ls(f'/DCIM/{folder}')
                           if Path(f).suffix.lower() in PHONE_MEDIA_EXTS)
            if not files:
                continue
            # DCIM folders fill chronologically: skip folders entirely
            # outside the range using only their first/last file.
            first_mt = _afc_mtime(f'/DCIM/{folder}/{files[0]}')
            last_mt = _afc_mtime(f'/DCIM/{folder}/{files[-1]}')
            if last_mt is not None and last_mt < start_ts:
                continue
            if first_mt is not None and first_mt > end_ts:
                continue
            for f in files:
                src = f'/DCIM/{folder}/{f}'
                mt = _afc_mtime(src)
                if mt is None or not (start_ts <= mt <= end_ts):
                    skipped += 1
                    continue
                day = datetime.fromtimestamp(mt).strftime('%Y-%m-%d')
                dest_dir = PHONE_EXPORT_DIR / day
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = _pi_unique_dest(dest_dir, f)
                r = _afc('get', src, str(dest))
                if r.returncode == 0 and dest.exists():
                    copied += 1
                else:
                    dest.unlink(missing_ok=True)
                    skipped += 1
                _pi_set(copied=copied, skipped=skipped,
                        message=f'{folder}: copying… ({copied} copied)')
        _pi_set(running=False, done=True, copied=copied, skipped=skipped,
                message=f'Done. {copied} files copied to {PHONE_EXPORT_DIR}')
    except Exception as e:
        _pi_set(running=False, done=True, error=str(e), message=f'Error: {e}')


@app.route('/phone-import')
def phone_import():
    resp = make_response(render_template('phone_import.html'))
    resp.headers['Cache-Control'] = 'no-store'
    return resp


@app.route('/phone-import/api/start', methods=['POST'])
def pi_start():
    data = request.json or {}
    try:
        start = datetime.strptime(data.get('start', ''), '%Y-%m-%d')
        end = datetime.strptime(data.get('end', ''), '%Y-%m-%d')
    except ValueError:
        return jsonify({'error': 'Invalid dates (expected YYYY-MM-DD)'}), 400
    if end < start:
        return jsonify({'error': 'End date is before start date'}), 400
    with _pi_lock:
        if _pi_state['running']:
            return jsonify({'error': 'An import is already running'}), 409
        _pi_state.update(running=True, done=False, error=None,
                         copied=0, skipped=0, message='Starting…')
    start_ts = start.timestamp()
    end_ts = (end + timedelta(days=1)).timestamp() - 1
    threading.Thread(target=_pi_worker, args=(start_ts, end_ts), daemon=True).start()
    return jsonify({'ok': True})


@app.route('/phone-import/api/status')
def pi_status():
    with _pi_lock:
        return jsonify(dict(_pi_state))


# ── Prune Photos ──────────────────────────────────────────────────────────────

_PRUNE_IMG_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff', '.tif', '.heic', '.svg'}
_PRUNE_WIKI_RE = re.compile(r'\[\[([^\]|#]+)')            # [[t]] / ![[t|300]] / ![[t#heading]]
_PRUNE_MDLINK_RE = re.compile(r'!\[[^\]]*\]\(([^)]+)\)')  # ![alt](target "title")


def _prune_scan(root):
    """Walk root (skipping dot-dirs like .obsidian) and return (referenced,
    images, md_count): the lowercased basenames of every image referenced from
    any .md in the tree, every image file found, and how many notes were read.
    References are matched vault-wide by filename because that is how Obsidian
    resolves ![[embeds]] — a photo used by a note in another folder is NOT
    redundant, even though its own folder's notes never mention it."""
    referenced, images, md_count = set(), [], 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith('.'))
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            ext = p.suffix.lower()
            if ext == '.md':
                try:
                    text = p.read_text(errors='ignore')
                except OSError:
                    continue
                md_count += 1
                targets = [m.group(1) for m in _PRUNE_WIKI_RE.finditer(text)]
                for m in _PRUNE_MDLINK_RE.finditer(text):
                    t = re.sub(r'\s+"[^"]*"$', '', m.group(1).strip())  # drop optional "title"
                    targets.append(urllib.parse.unquote(t))
                for t in targets:
                    name = Path(t.strip()).name
                    if Path(name).suffix.lower() in _PRUNE_IMG_EXTS:
                        referenced.add(name.lower())
            elif ext in _PRUNE_IMG_EXTS:
                images.append(p)
    return referenced, images, md_count


@app.route('/prune-photos')
def prune_photos():
    resp = make_response(render_template('prune_photos.html'))
    resp.headers['Cache-Control'] = 'no-store'
    return resp


@app.route('/prune-photos/api/listdirs')
def pp_listdirs():
    data = _listdirs_json(request.args.get('path', ''))
    if data is None:
        return jsonify({'error': 'not a directory'}), 400
    return jsonify(data)


@app.route('/prune-photos/api/preview', methods=['POST'])
def pp_preview():
    folder = (request.json or {}).get('folder', '').strip()
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        return jsonify({'error': 'Not a valid directory'}), 400
    referenced, images, md_count = _prune_scan(root)
    folders = {}
    for img in images:
        rel = img.relative_to(root)
        fkey = str(rel.parent)
        f = folders.setdefault(fkey, {'folder': fkey, 'total': 0, 'unreferenced': []})
        f['total'] += 1
        if img.name.lower() not in referenced:
            f['unreferenced'].append(str(rel))
    result = sorted(folders.values(), key=lambda f: f['folder'])
    return jsonify({
        'root': str(root),
        'folders': result,
        'md_count': md_count,
        'total_images': len(images),
        'total_unreferenced': sum(len(f['unreferenced']) for f in result),
    })


@app.route('/prune-photos/api/image')
def pp_image():
    """Serve one image inside the scanned root, so the preview list can show
    what would be deleted before the user confirms."""
    root = Path(request.args.get('root', '')).expanduser().resolve()
    rel = request.args.get('rel', '')
    if not root.is_dir():
        abort(404)
    filepath = (root / rel).resolve()
    if not str(filepath).startswith(str(root) + os.sep):
        abort(403)
    if not filepath.is_file() or filepath.suffix.lower() not in _PRUNE_IMG_EXTS:
        abort(404)
    return send_file(filepath)


@app.route('/prune-photos/api/run', methods=['POST'])
def pp_run():
    data = request.json or {}
    root = Path(data.get('root', '')).expanduser().resolve()
    files = data.get('files') or []
    if not root.is_dir() or not isinstance(files, list):
        return jsonify({'error': 'Invalid request'}), 400
    deleted, errors = 0, []
    for rel in files:
        filepath = (root / str(rel)).resolve()
        if not str(filepath).startswith(str(root) + os.sep) or filepath.suffix.lower() not in _PRUNE_IMG_EXTS:
            errors.append({'file': str(rel), 'error': 'Invalid path'})
            continue
        if not filepath.is_file():
            errors.append({'file': str(rel), 'error': 'File not found'})
            continue
        try:
            _move_to_trash(filepath)
            deleted += 1
        except Exception as e:
            errors.append({'file': str(rel), 'error': str(e)})
    return jsonify({'ok': True, 'deleted': deleted, 'errors': errors})


# ── Launch ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get('TRAVEL_BLOG_PORT', 57001))
    url = f'http://localhost:{port}'
    print(f"  Travel Blog Tools → {url}")
    if not os.environ.get('TRAVEL_BLOG_LAUNCHER'):
        subprocess.Popen(['open', '-a', 'Safari', url])
    if PT_FOLDER:
        _kick_warm(PT_FOLDER)  # warm the display cache for the last-used folder
    app.run(port=port, debug=False, use_reloader=False, threaded=True)
