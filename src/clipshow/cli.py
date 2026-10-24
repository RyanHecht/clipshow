"""Admin/dev CLI: manage titles, index, search, render. The web app (and worker) are the product's interface."""

import shutil
from pathlib import Path
from typing import Annotated

import typer

from . import titles, store

app = typer.Typer(no_args_is_help=True, add_completion=False)
_state: dict = {}


@app.callback()
def main(title: Annotated[str | None, typer.Option("--title", "-t", envvar="CLIPSHOW_TITLE",
                                                  help="Which title (show) to work on; optional when there's only one")] = None):
    _state["title"] = title


def current_title() -> titles.Title:
    slug = _state.get("title")
    if slug:
        title = titles.get_title(slug)
        if title is None:
            raise typer.BadParameter(f"no title {slug!r}; see `clipshow titles list`")
        return title
    all_titles = titles.list_titles()
    if len(all_titles) == 1:
        return all_titles[0]
    raise typer.BadParameter("choose a title with --title (see `clipshow titles list`)" if all_titles
                             else "no titles yet; create one with `clipshow titles add`")


def fmt_time(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


# --- titles -------------------------------------------------------------------------------------

titles_app = typer.Typer(no_args_is_help=True, help="Create and manage titles (one per show).")
app.add_typer(titles_app, name="titles")


@titles_app.command("list")
def titles_list():
    """List titles."""
    for s in titles.list_titles():
        typer.echo(f"{s.slug:24s} {s.name:28s} {s.media or ''}")


@titles_app.command("add")
def titles_add(
    media: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="Folder with the show's videos and subtitles")],
    name: Annotated[str, typer.Option(help="Show name shown on the title")],
    slug: Annotated[str | None, typer.Option(help="URL name (default: from the name)")] = None,
):
    """Create a title for a show folder."""
    title = titles.create_title(name, str(media.resolve()), slug=slug)
    typer.echo(f"created title {title.slug!r}; index it with: clipshow -t {title.slug} index")


@titles_app.command("settings")
def titles_settings():
    """Show a title's settings."""
    import json

    title = current_title()
    typer.echo(json.dumps({"name": title.name, "media": title.media, **title.settings.to_dict()}, indent=2))


# --- indexing ----------------------------------------------------------------------------------

@app.command()
def index(
    path: Annotated[Path | None, typer.Argument(help="Show folder or a single episode file (default: the title's media folder)")] = None,
    describe: Annotated[bool, typer.Option(help="Also describe new moments with Gemini (needs an OpenRouter key)")] = False,
    shots: Annotated[bool, typer.Option(help="Detect shot cuts (better boundaries)")] = True,
    force: Annotated[bool, typer.Option(help="Re-index episodes that are already indexed")] = False,
    workers: Annotated[int, typer.Option(help="Episodes prepared in parallel")] = 6,
):
    """Index episodes: subtitles -> moments -> thumbnails -> embeddings."""
    from .indexer import index_path

    title = current_title()
    path = path or (Path(title.media) if title.media else None)
    if path is None or not path.exists():
        raise typer.BadParameter("give a media path, or create the title with one (`clipshow titles add`)")
    touched = index_path(title, path, shots=shots, force=force, workers=workers, log=typer.echo)
    if describe and touched:
        from .describe import describe_pending

        describe_pending(store.connect(title.db_path), episode_ids=touched, log=typer.echo)


@app.command()
def describe(
    limit: Annotated[int | None, typer.Option(help="Max moments to describe this run")] = None,
    workers: Annotated[int, typer.Option(help="Concurrent Gemini requests")] = 8,
    season: Annotated[list[int] | None, typer.Option(help="Only these season(s); repeatable")] = None,
):
    """Describe all not-yet-described moments with Gemini (resumable)."""
    from .describe import describe_pending

    conn = store.connect(current_title().db_path)
    episode_ids = None
    if season:
        q = f"SELECT id FROM episodes WHERE indexed_at IS NOT NULL AND season IN ({','.join('?' * len(season))})"
        episode_ids = [r[0] for r in conn.execute(q, season)]
        typer.echo(f"{len(episode_ids)} indexed episode(s) in season(s) {season}")
        if not episode_ids:
            return
    describe_pending(conn, episode_ids=episode_ids, limit=limit, workers=workers, log=typer.echo)


# --- search / render ---------------------------------------------------------------------------

@app.command()
def search(
    query: str,
    k: Annotated[int, typer.Option("-k", help="Number of results")] = 10,
    fmt: Annotated[str | None, typer.Option("--format", "-f", help="Render results as gif | clip | still")] = None,
    out: Annotated[Path, typer.Option(help="Where to copy rendered files")] = Path("out"),
):
    """Search moments; optionally render each result."""
    from .search import search as run_search

    title = current_title()
    conn = store.connect(title.db_path)
    results = run_search(conn, query, k=k, settings=title.settings)
    if not results:
        typer.echo("no results")
        raise typer.Exit(1)
    if fmt:
        out.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(results, 1):
        hits = " ".join(f"{s}#{r}" for s, r in sorted(m.hits.items(), key=lambda x: x[1]))
        typer.echo(f"{i:2d}. [{m.id}] {m.label} {fmt_time(m.start)}-{fmt_time(m.end)}  ({hits})")
        typer.echo(f"     {(m.speaker + ': ') if m.speaker else ''}{m.text or '[no dialogue]'}")
        if m.notes:
            typer.echo(f"     [{m.notes}]")
        if fmt:
            from .render import render

            f = render(m, fmt, title.renders, m.ds, m.de)
            dest = out / f"{i:02d}_{m.label}_{int(m.start)}s{f.suffix}"
            shutil.copy(f, dest)
            typer.echo(f"     -> {dest}")


# --- transcripts -------------------------------------------------------------------------------

transcripts_app = typer.Typer(no_args_is_help=True, help="Speaker-labelled transcripts (<video>.transcript.txt files).")
app.add_typer(transcripts_app, name="transcripts")


@transcripts_app.command("apply")
def transcripts_apply():
    """Attach speakers and stage directions from transcript files to already-indexed moments (no re-index)."""
    from .indexer import apply_annotations

    matched, total = apply_annotations(current_title(), log=typer.echo)
    typer.echo(f"annotated {matched}/{total} moments")


@app.command()
def stats():
    """Show what's indexed."""
    conn = store.connect(current_title().db_path)
    q = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
    typer.echo(f"episodes: {q('SELECT count(*) FROM episodes WHERE indexed_at IS NOT NULL')}")
    silent = q("SELECT count(*) FROM moments WHERE kind = 'silent'")
    typer.echo(f"moments:  {q('SELECT count(*) FROM moments')} ({silent} silent)")
    typer.echo(f"described: {q('SELECT count(*) FROM moments WHERE description IS NOT NULL')}")
