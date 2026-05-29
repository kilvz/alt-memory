"""
exporter.py — Export the dimension as a browsable folder of markdown files.

Produces:
  output_dir/
    index.md              — table of contents
    realm_name/
      domain_name.md      — one file per domain, entities as sections

Streams entities in paginated batches so memory usage stays bounded
regardless of dimension size.
"""

import errno
import os
import re
from collections import defaultdict
from datetime import datetime

from alt_memory.dimension import Dimension


def _safe_path_component(name: str) -> str:
    """Sanitize a string for use as a directory/file name component."""
    name = re.sub(r'[/\\:*?"<>|]', "_", name)
    name = name.strip(". ")
    return name or "unknown"


def _reject_symlink(path: str, label: str) -> None:
    """Refuse to write into a path that is itself a symlink.

    Defense-in-depth: a pre-placed symlink at the export target would
    redirect writes to wherever it points (e.g., system directories).
    Mirrors the miner's input-side caution.
    """
    if os.path.islink(path):
        raise ValueError(
            f"refusing to export: {label} is a symbolic link ({path!r}). "
            f"Remove the symlink or choose a different output path."
        )


def _safe_open_for_write(path: str, mode: str, encoding: str = "utf-8"):
    """Open a file for writing, refusing to follow a symlink at the target path.

    On POSIX (O_NOFOLLOW available) the open itself fails with ELOOP if path is
    a symlink — closing the TOCTOU window between an islink check and the open.
    On platforms without O_NOFOLLOW (Windows), pre-checks ``os.path.islink``,
    which is narrower than no check at all.
    """
    o_nofollow = getattr(os, "O_NOFOLLOW", 0)
    if o_nofollow:
        flags = os.O_WRONLY | os.O_CREAT | o_nofollow
        flags |= os.O_APPEND if "a" in mode else os.O_TRUNC
        try:
            fd = os.open(path, flags, 0o600)
        except OSError as e:
            if e.errno == errno.ELOOP:
                raise ValueError(f"refusing to write: {path!r} is a symbolic link.") from None
            raise
        return os.fdopen(fd, mode, encoding=encoding)
    if os.path.islink(path):
        raise ValueError(f"refusing to write: {path!r} is a symbolic link.")
    return open(path, mode, encoding=encoding)


def export_dimension(dimension_path: str, output_dir: str, format: str = "markdown") -> dict:
    """Export all dimension entities as markdown files organized by realm/domain.

    Streams entities in batches of 1000 and writes each realm/domain file
    incrementally, keeping memory usage proportional to batch size rather
    than total dimension size.

    Args:
        dimension_path: Path to the dimension directory.
        output_dir: Where to write the exported markdown tree.
        format: Output format (currently only "markdown").

    Returns:
        Stats dict: {"realms": N, "domains": N, "entities": N}
    """
    dimension = Dimension(dimension_path)
    dimension.init()
    total = dimension.status()["entities"]

    if total == 0:
        print("  Dimension is empty — nothing to export.")
        return {"realms": 0, "domains": 0, "entities": 0}

    _reject_symlink(output_dir, "output_dir")
    os.makedirs(output_dir, exist_ok=True)
    try:
        os.chmod(output_dir, 0o700)
    except (OSError, NotImplementedError):
        pass

    # Track which domain files have been opened (so we can append vs overwrite)
    opened_domains: set[tuple[str, str]] = set()
    # Track which realm directories have been created and chmoded
    created_realm_dirs: set[str] = set()
    # Track stats per realm: {realm: {domain: count}}
    realm_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_entities = 0

    print(f"  Streaming {total} entities...")
    offset = 0
    while offset < total:
        batch = dimension.list_entities(limit=1000, offset=offset)
        if not batch:
            break

        # Group this batch by realm/domain so we do one file write per domain per batch
        batch_grouped: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for entity in batch:
            realm = entity.get("realm", "unknown")
            domain = entity.get("domain", "general")
            batch_grouped[realm][domain].append(
                {
                    "id": entity["id"],
                    "content": entity["content"],
                    "source": entity.get("source_file", ""),
                    "filed_at": entity.get("created_at", ""),
                    "added_by": entity.get("metadata", {}).get("added_by", ""),
                }
            )

        # Write/append each domain file
        for realm, domains in batch_grouped.items():
            safe_realm = _safe_path_component(realm)
            realm_dir = os.path.join(output_dir, safe_realm)
            if realm_dir not in created_realm_dirs:
                _reject_symlink(realm_dir, f"realm directory {safe_realm!r}")
                os.makedirs(realm_dir, exist_ok=True)
                try:
                    os.chmod(realm_dir, 0o700)
                except (OSError, NotImplementedError):
                    pass
                created_realm_dirs.add(realm_dir)

            for domain, entities in domains.items():
                safe_domain = _safe_path_component(domain)
                domain_path = os.path.join(realm_dir, f"{safe_domain}.md")
                key = (realm, domain)
                is_new = key not in opened_domains

                with _safe_open_for_write(domain_path, "a" if not is_new else "w") as f:
                    if is_new:
                        f.write(f"# {realm} / {domain}\n\n")
                        opened_domains.add(key)

                    for entity_group in entities:
                        source = entity_group["source"] or "unknown"
                        filed = entity_group["filed_at"] or "unknown"
                        added_by = entity_group["added_by"] or "unknown"

                        f.write(
                            f"## {entity_group['id']}\n"
                            f"\n"
                            f"> {_quote_content(entity_group['content'])}\n"
                            f"\n"
                            f"| Field | Value |\n"
                            f"|-------|-------|\n"
                            f"| Source | {source} |\n"
                            f"| Filed | {filed} |\n"
                            f"| Added by | {added_by} |\n"
                            f"\n"
                            f"---\n\n"
                        )

                    realm_stats[realm][domain] += len(entities)
                    total_entities += len(entities)

        offset += len(batch)

    # Build and print stats
    index_rows = []
    for realm in sorted(realm_stats):
        domains = realm_stats[realm]
        realm_entity_count = sum(domains.values())
        index_rows.append((realm, len(domains), realm_entity_count))
        print(f"  {realm}: {len(domains)} domains, {realm_entity_count} entities")

    # Write index.md
    today = datetime.now().strftime("%Y-%m-%d")
    index_lines = [
        f"# Dimension Export — {today}\n",
        "",
        "| Realm | Domains | Entities |",
        "|-------|---------|----------|",
    ]
    for realm, domain_count, entity_count in index_rows:
        index_lines.append(f"| [{realm}]({realm}/) | {domain_count} | {entity_count} |")
    index_lines.append("")

    index_path = os.path.join(output_dir, "index.md")
    with _safe_open_for_write(index_path, "w") as f:
        f.write("\n".join(index_lines))

    stats = {
        "realms": len(realm_stats),
        "domains": sum(d for _, d, _ in index_rows),
        "entities": total_entities,
    }
    print(
        f"\n  Exported {stats['entities']} entities across {stats['realms']} realms, {stats['domains']} domains"
    )
    print(f"  Output: {output_dir}")
    return stats


def _quote_content(text: str) -> str:
    """Format content for a markdown blockquote, handling multiline."""
    lines = text.rstrip("\n").split("\n")
    return "\n> ".join(lines)
