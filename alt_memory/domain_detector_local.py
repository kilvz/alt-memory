"""domain_detector_local.py — Local setup, no API required.

Two ways to define domains without calling any AI:
  1. Auto-detect from folder structure (zero config)
  2. Define manually in .alt-memory.yaml
"""

import logging
import os
from collections import defaultdict
from pathlib import Path

import yaml

from alt_memory.config import normalize_realm_name
from alt_memory.dimension import SKIP_DIRS

logger = logging.getLogger(__name__)

FOLDER_DOMAIN_MAP = {
    "frontend": "frontend", "front-end": "frontend", "front_end": "frontend",
    "client": "frontend", "ui": "frontend", "views": "frontend", "components": "frontend",
    "pages": "frontend",
    "backend": "backend", "back-end": "backend", "back_end": "backend",
    "server": "backend", "api": "backend", "routes": "backend", "services": "backend",
    "controllers": "backend", "models": "backend", "database": "backend", "db": "backend",
    "docs": "documentation", "doc": "documentation", "documentation": "documentation",
    "wiki": "documentation", "readme": "documentation", "notes": "documentation",
    "design": "design", "designs": "design", "mockups": "design", "wireframes": "design",
    "assets": "design", "storyboard": "design",
    "costs": "costs", "cost": "costs", "budget": "costs", "finance": "costs",
    "financial": "costs", "pricing": "costs", "invoices": "costs", "accounting": "costs",
    "meetings": "meetings", "meeting": "meetings", "calls": "meetings",
    "meeting_notes": "meetings", "standup": "meetings", "minutes": "meetings",
    "team": "team", "staff": "team", "hr": "team", "hiring": "team",
    "employees": "team", "people": "team",
    "research": "research", "references": "research", "reading": "research", "papers": "research",
    "planning": "planning", "roadmap": "planning", "strategy": "planning",
    "specs": "planning", "requirements": "planning",
    "tests": "testing", "test": "testing", "testing": "testing", "qa": "testing",
    "scripts": "scripts", "tools": "scripts", "utils": "scripts",
    "config": "configuration", "configs": "configuration", "settings": "configuration",
    "infrastructure": "configuration", "infra": "configuration", "deploy": "configuration",
}

def _walk_files(project_dir: str) -> list:
    project_path = Path(project_dir).expanduser().resolve()
    files = []
    for root, dirs, filenames in os.walk(project_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for filename in filenames:
            filepath = Path(root) / filename
            if filepath.is_symlink():
                continue
            try:
                if filepath.stat().st_size > 50 * 1024 * 1024:
                    continue
            except OSError:
                continue
            files.append(filepath)
    return sorted(files)


def detect_domains_from_folders(project_dir: str) -> list:
    project_path = Path(project_dir).expanduser().resolve()
    found_domains = {}

    for item in project_path.iterdir():
        try:
            is_dir = item.is_dir()
        except OSError:
            continue
        if is_dir and item.name not in SKIP_DIRS:
            name_lower = item.name.lower().replace("-", "_")
            if name_lower in FOLDER_DOMAIN_MAP:
                domain_name = FOLDER_DOMAIN_MAP[name_lower]
                if domain_name not in found_domains:
                    found_domains[domain_name] = item.name
            elif len(item.name) > 2 and item.name[0].isalpha():
                clean = item.name.lower().replace("-", "_").replace(" ", "_")
                if clean not in found_domains:
                    found_domains[clean] = item.name

    for item in project_path.iterdir():
        try:
            item_is_dir = item.is_dir()
        except OSError:
            continue
        if item_is_dir and item.name not in SKIP_DIRS:
            try:
                subitems = list(item.iterdir())
            except OSError:
                continue
            for subitem in subitems:
                try:
                    subitem_is_dir = subitem.is_dir()
                except OSError:
                    continue
                if subitem_is_dir and subitem.name not in SKIP_DIRS:
                    name_lower = subitem.name.lower().replace("-", "_")
                    if name_lower in FOLDER_DOMAIN_MAP:
                        domain_name = FOLDER_DOMAIN_MAP[name_lower]
                        if domain_name not in found_domains:
                            found_domains[domain_name] = subitem.name

    domains = []
    for domain_name, original in found_domains.items():
        domains.append({"name": domain_name, "description": f"Files from {original}/", "keywords": [domain_name, original.lower()]})

    if not any(d["name"] == "general" for d in domains):
        domains.append({"name": "general", "description": "Files that don't fit other domains", "keywords": []})

    return domains


def detect_domains_from_files(project_dir: str) -> list:
    project_path = Path(project_dir).expanduser().resolve()
    keyword_counts = defaultdict(int)
    for root, dirs, filenames in os.walk(project_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for filename in filenames:
            name_lower = filename.lower().replace("-", "_").replace(" ", "_")
            for keyword, domain in FOLDER_DOMAIN_MAP.items():
                if keyword in name_lower:
                    keyword_counts[domain] += 1

    domains = []
    for domain, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True):
        if count >= 2:
            domains.append({"name": domain, "description": f"Files related to {domain}", "keywords": [domain]})
        if len(domains) >= 6:
            break

    if not domains:
        domains = [{"name": "general", "description": "All project files", "keywords": []}]

    return domains


def print_proposed_structure(project_name: str, domains: list, total_files: int, source: str):
    print(f"\n{'=' * 55}")
    print("  Alt Memory Init — Local setup")
    print(f"{'=' * 55}")
    print(f"\n  REALM: {project_name}")
    print(f"  ({total_files} files found, domains detected from {source})\n")
    for domain in domains:
        print(f"    DOMAIN: {domain['name']}")
        print(f"          {domain['description']}")
    print(f"\n{'─' * 55}")


def get_user_approval(domains: list) -> list:
    print("  Review the proposed domains above.")
    print("  Options:")
    print("    [enter]  Accept all domains")
    print("    [edit]   Remove or rename domains")
    print("    [add]    Add a domain manually")
    print()

    choice = input("  Your choice [enter/edit/add]: ").strip().lower()

    if choice in ("", "y", "yes"):
        return domains

    if choice == "edit":
        print("\n  Current domains:")
        for i, domain in enumerate(domains):
            print(f"    {i + 1}. {domain['name']} — {domain['description']}")
        remove = input("\n  Domain numbers to REMOVE (comma-separated, or enter to skip): ").strip()
        if remove:
            to_remove = {int(x.strip()) - 1 for x in remove.split(",") if x.strip().isdigit()}
            domains = [d for i, d in enumerate(domains) if i not in to_remove]

    if choice == "add" or input("\n  Add any missing domains? [y/N]: ").strip().lower() == "y":
        while True:
            new_name = input("  New domain name (or enter to stop): ").strip().lower().replace(" ", "_")
            if not new_name:
                break
            new_desc = input(f"  Description for '{new_name}': ").strip()
            domains.append({"name": new_name, "description": new_desc, "keywords": [new_name]})
            print(f"  Added: {new_name}")

    return domains


def save_config(project_dir: str, project_name: str, domains: list):
    config = {
        "realm": project_name,
        "domains": [{"name": d["name"], "description": d["description"], "keywords": d.get("keywords", [d["name"]])} for d in domains],
    }
    config_path = Path(project_dir).expanduser().resolve() / ".alt-memory.yaml"
    tmp_path = config_path.with_suffix(".yaml.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    os.replace(str(tmp_path), str(config_path))

    print(f"\n  Config saved: {config_path}")
    print("\n  Next step:")
    print(f"    alt-memory mine {project_dir}")
    print(f"\n{'=' * 55}\n")


def detect_domains_local(project_dir: str, yes: bool = False):
    project_path = Path(project_dir).expanduser().resolve()
    project_name = normalize_realm_name(project_path.name)

    if not project_path.exists():
        raise FileNotFoundError(f"Directory not found: {project_dir}")

    files = _walk_files(project_dir)

    domains = detect_domains_from_folders(project_dir)
    source = "folder structure"

    if len(domains) <= 1:
        domains = detect_domains_from_files(project_dir)
        source = "filename patterns"

    if not domains:
        domains = [{"name": "general", "description": "All project files", "keywords": []}]
        source = "fallback (flat project)"

    print_proposed_structure(project_name, domains, len(files), source)
    if yes:
        approved_domains = domains
    else:
        approved_domains = get_user_approval(domains)
    save_config(project_dir, project_name, approved_domains)
