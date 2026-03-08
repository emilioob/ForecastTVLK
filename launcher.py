from __future__ import annotations

import sys
from pathlib import Path

from streamlit.web.cli import main as streamlit_main


def resource_path(*parts: str) -> Path:
    base_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base_path.joinpath(*parts)


def main() -> int:
    app_path = resource_path("app.py")
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--global.developmentMode=false",
    ]
    return streamlit_main()


if __name__ == "__main__":
    raise SystemExit(main())
