#!/usr/bin/env python3
import sys
import os
import json
import subprocess
import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_file, render_template, abort

import pillow_heif
from PIL import Image

pillow_heif.register_heif_opener()

app = Flask(__name__)

FOLDER = None
RATINGS_FILE = None
_meta_cache = {}
_cache_lock = threading.Lock()


def dms_to_decimal(dms, ref):
    d, m, s = float(dms[0]), float(dms[1]), float(dms[2])
    dec = d + m / 60 + s / 3600
    if ref in ('S', 'W'):
        dec = -dec
    return round(dec, 6)


def extract_metadata(filepath):
    try:
        img = Image.open(filepath)
        exif = img.getexif()

        date_raw = exif.get(306)
        date_display = None
        if date_raw:
            date_display = date_raw.replace(':', '-', 2)

        gps = exif.get_ifd(0x8825)
        lat = lon = gmaps_url = None
        if gps:
            lat_dms = gps.get(2)
            lat_ref = gps.get(1)
            lon_dms = gps.get(4)
            lon_ref = gps.get(3)
            if lat_dms and lat_ref and lon_dms and lon_ref:
                lat = dms_to_decimal(lat_dms, lat_ref)
                lon = dms_to_decimal(lon_dms, lon_ref)
                gmaps_url = f"https://www.google.com/maps?q={lat},{lon}"

        return {'date': date_display, 'lat': lat, 'lon': lon, 'gmaps_url': gmaps_url}
    except Exception:
        return {'date': None, 'lat': None, 'lon': None, 'gmaps_url': None}


def load_ratings():
    if RATINGS_FILE and Path(RATINGS_FILE).exists():
        with open(RATINGS_FILE) as f:
            return json.load(f)
    return {}


def save_ratings(ratings):
    with open(RATINGS_FILE, 'w') as f:
        json.dump(ratings, f, indent=2)


def heic_files():
    if not FOLDER:
        return []
    p = Path(FOLDER)
    files = sorted(p.glob('*.HEIC')) + sorted(p.glob('*.heic'))
    seen = set()
    result = []
    for f in files:
        if f.name not in seen:
            seen.add(f.name)
            result.append(f.name)
    return result


@app.route('/')
def index():
    from flask import make_response
    resp = make_response(render_template('index.html'))
    resp.headers['Cache-Control'] = 'no-store'
    return resp


@app.route('/api/photos')
def api_photos():
    files = heic_files()
    ratings = load_ratings()
    photos = []
    for filename in files:
        filepath = Path(FOLDER) / filename
        with _cache_lock:
            meta = _meta_cache.get(filename)
        if meta is None:
            meta = extract_metadata(filepath)
            with _cache_lock:
                _meta_cache[filename] = meta
        tags = ratings.get(filename, {'me': False, 'hidden': False, 'blog': False})
        photos.append({
            'filename': filename,
            'date': meta['date'],
            'lat': meta['lat'],
            'lon': meta['lon'],
            'gmaps_url': meta['gmaps_url'],
            'tags': tags,
        })
    return jsonify({'folder': str(FOLDER), 'photos': photos})


@app.route('/api/image/<path:filename>')
def api_image(filename):
    if not FOLDER:
        abort(404)
    filepath = Path(FOLDER) / filename
    if not filepath.exists() or filepath.suffix.lower() != '.heic':
        abort(404)
    return send_file(filepath, mimetype='image/heic')


@app.route('/api/star', methods=['POST'])
def api_star():
    data = request.json
    filename = data.get('filename')
    category = data.get('category')
    value = bool(data.get('value'))
    if category not in ('me', 'hidden', 'blog'):
        return jsonify({'error': 'invalid category'}), 400
    ratings = load_ratings()
    if filename not in ratings:
        ratings[filename] = {'me': False, 'hidden': False, 'blog': False}
    ratings[filename][category] = value
    save_ratings(ratings)
    return jsonify({'ok': True})


@app.route('/api/folder', methods=['POST'])
def api_set_folder():
    global FOLDER, RATINGS_FILE
    folder = (request.json or {}).get('folder', '').strip()
    p = Path(folder).expanduser().resolve()
    if not p.is_dir():
        return jsonify({'error': 'Not a valid directory'}), 400
    FOLDER = p
    RATINGS_FILE = p / '_ratings.json'
    with _cache_lock:
        _meta_cache.clear()
    return jsonify({'ok': True, 'folder': str(FOLDER)})


@app.route('/api/listdirs')
def api_listdirs():
    raw = request.args.get('path', str(Path.home()))
    p = Path(raw).expanduser().resolve()
    if not p.is_dir():
        return jsonify({'error': 'not a directory'}), 400
    entries = []
    try:
        for item in sorted(p.iterdir(), key=lambda x: x.name.lower()):
            if item.is_dir() and not item.name.startswith('.'):
                heic_count = sum(1 for _ in item.glob('*.HEIC')) + sum(1 for _ in item.glob('*.heic'))
                entries.append({'name': item.name, 'path': str(item), 'heic_count': heic_count})
    except PermissionError:
        pass
    return jsonify({
        'path': str(p),
        'parent': str(p.parent) if str(p.parent) != str(p) else None,
        'entries': entries,
    })


@app.route('/api/status')
def api_status():
    return jsonify({'folder': str(FOLDER) if FOLDER else None})


if __name__ == '__main__':
    if len(sys.argv) > 1:
        folder_arg = Path(sys.argv[1]).expanduser().resolve()
        if folder_arg.is_dir():
            FOLDER = folder_arg
            RATINGS_FILE = FOLDER / '_ratings.json'
            print(f"  Folder: {FOLDER}")
        else:
            print(f"  Warning: '{sys.argv[1]}' is not a valid directory")

    url = 'http://localhost:5001'
    print(f"  Open {url} in Safari")
    subprocess.Popen(['open', '-a', 'Safari', url])
    app.run(port=5001, debug=False, use_reloader=False)
