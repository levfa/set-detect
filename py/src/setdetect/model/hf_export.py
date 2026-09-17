import pathlib as pl
import shutil


def render_model_card(*, frontmatter: dict[str, str | list[str]], body: str) -> str:
    """Hand-format a README.md string: a YAML frontmatter block followed by a
    Markdown body. No YAML library needed for content this simple (plain strings
    and lists of strings)."""
    lines = ["---"]
    for key, value in frontmatter.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            lines.extend(f"  - {item}" for item in value)
        else:
            lines.append(f"{key}: {value}")
    lines.append("---")
    lines.append("")
    return "\n".join(lines) + "\n" + body


def stage_files(out_dir: pl.Path, files: list[tuple[pl.Path, str]]) -> None:
    """Copy each (local_path, path_in_repo) pair into out_dir, creating parent dirs
    as needed. Only ever adds/overwrites the given files -- never deletes or
    otherwise touches anything else already in out_dir."""
    for local_path, path_in_repo in files:
        dest = out_dir / path_in_repo
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)


def write_text(out_dir: pl.Path, path_in_repo: str, content: str) -> None:
    """Write a generated text file (README.md, config.json) into out_dir/path_in_repo."""
    dest = out_dir / path_in_repo
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content)
