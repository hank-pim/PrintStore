# PrintStore

Simple local/network 3D print file viewer.

- Browse folders (including UNC/network shares on Windows)
- Thumbnail grid view for `.3mf`/`.stl` (cached locally)
- Click a file to preview it in a PyVista 3D viewport

## Requirements

- Python 3.10+ recommended
- Works best with a virtual environment (`.venv`)

## Install

```bash
python -m venv .venv
```

Windows:

```bat
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux/macOS:

```bash
. .venv/bin/activate
pip install -r requirements.txt
```

## Run

Windows (double-click):
- `run.bat`

Windows (from terminal):

```bat
.venv\Scripts\python.exe app.py
```

Linux/macOS:

```bash
chmod +x run.sh
./run.sh
```

### Optional arguments

- Open a specific folder at startup (works with UNC paths):

```bat
run.bat "\\server\share\3dprints"
```

- Enable debug logs:

```bat
run.bat --debug
```

Or set `PRINTSTORE_DEBUG=1`.

## Thumbnail cache

Thumbnails are cached under your local profile:

- Windows: `%LOCALAPPDATA%\PrintStore\thumbs`

Cache keys include file modification time; delete the folder to regenerate.

## Notes

- Many `.3mf` files store slicer-specific color/material info; this viewer currently renders geometry and basic mesh colors when available.
- If you hit a rendering/import error, re-run with `--debug` and paste the console output.
