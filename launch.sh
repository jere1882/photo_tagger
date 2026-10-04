#!/bin/bash
PORT=57001
APP_DIR="/Users/jere/travelBLog"
LOGFILE="/tmp/travelblog_57001.log"
PYTHON="/opt/homebrew/Caskroom/miniforge/base/bin/python3"
# Free CARTO Voyager basemap key (5M tiles/month, no account needed): get one at
# https://carto.com/basemaps/apikey and put it in .env next to this script, as
# CARTO_API_KEY="...". .env is gitignored, so the key stays on this machine.
# Without one, the maps fall back to plain OpenStreetMap tiles.
CARTO_API_KEY=""
[ -f "$APP_DIR/.env" ] && . "$APP_DIR/.env"

# Kill any existing server on this port
lsof -ti :$PORT 2>/dev/null | xargs kill -9 2>/dev/null
sleep 0.3

# Run the server through Terminal — Terminal has Full Disk Access so
# python3 inherits it and can read ~/Desktop without TCC prompts.
osascript << APPLES
tell application "Terminal"
    set w to do script "TRAVEL_BLOG_PORT=$PORT TRAVEL_BLOG_LAUNCHER=1 CARTO_API_KEY='$CARTO_API_KEY' '$PYTHON' '$APP_DIR/app.py' > '$LOGFILE' 2>&1"
    delay 0.4
    set miniaturized of w to true
end tell
APPLES

# Poll until server answers (up to 12 s)
for i in $(seq 1 24); do
    sleep 0.5
    if curl -sf "http://localhost:$PORT/" > /dev/null 2>&1; then
        open -a Safari "http://localhost:$PORT/"
        exit 0
    fi
done

LOG_TAIL=$(tail -15 "$LOGFILE" 2>/dev/null || echo "(no log)")
osascript -e "display alert \"Travel Blog failed to start\" message \"$LOG_TAIL\""
exit 1
