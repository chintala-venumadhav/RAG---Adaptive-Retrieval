"""
run.py – Project Entry Point
==============================
Launches the Streamlit application.  Simply run:

    py run.py

This is equivalent to:

    py -m streamlit run ui/streamlit_app.py

Author  : B.Tech CSE-AI Student Project
"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Launch the Streamlit UI."""
    app_path: Path = Path(__file__).resolve().parent / "ui" / "streamlit_app.py"

    if not app_path.exists():
        print(f"❌ Error: Streamlit app not found at {app_path}")
        sys.exit(1)

    print(f"🚀 Starting Adaptive RAG UI …")
    print(f"   App: {app_path}")
    print(f"   Open: http://localhost:8501")
    print()

    try:
        subprocess.run(
            ["py", "-m", "streamlit", "run", str(app_path)],
            check=True,
        )
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user.")
    except FileNotFoundError:
        print(
            "❌ Streamlit is not installed.\n"
            "   Run:  pip install -r requirements.txt"
        )
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"❌ Streamlit exited with error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

