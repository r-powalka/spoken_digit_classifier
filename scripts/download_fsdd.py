"""Download the Free Spoken Digit Dataset (FSDD)."""

import argparse
import subprocess
from pathlib import Path

FSDD_URL = (
    "https://github.com/Jakobovski/free-spoken-digit-dataset.git"
)


def download_fsdd(data_dir, update=False):
    """Clone FSDD into data_dir and return the repository path."""
    data_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = data_dir / "free-spoken-digit-dataset"

    if repo_dir.exists():
        if not (repo_dir / ".git").is_dir():
            raise FileExistsError(
                f"{repo_dir} exists but is not a Git repository."
            )

        if not update:
            print(f"FSDD already exists at {repo_dir}")
            return repo_dir

        subprocess.run(
            ["git", "-C", str(repo_dir), "pull", "--ff-only"],
            check=True,
        )
    else:
        subprocess.run(
            ["git", "clone", FSDD_URL, str(repo_dir)],
            check=True,
        )

    print(f"FSDD is available at {repo_dir}")
    return repo_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download the Free Spoken Digit Dataset."
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update an existing clone.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    download_fsdd(data_dir, update=args.update)


if __name__ == "__main__":
    main()
