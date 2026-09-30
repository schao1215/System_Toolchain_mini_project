"""Tennessee Eastman fault detectors trained on normal data only.

Run the full pipeline with `uv run miniproject`.
This package does not call the course evidence script.
"""


def main() -> None:
    """Entry point registered in pyproject.toml as `miniproject`."""
    from miniproject.pipeline import run

    run()
