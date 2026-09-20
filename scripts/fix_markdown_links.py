"""
Convert absolute file:/// URLs across all repository Markdown files to clean repository-relative links.
Also sanitizes host paths in notebook outputs to portable [DATA_ROOT] placeholders.
"""

import json
import os
import re
from pathlib import Path

PREFIX_PATTERN = re.compile(r'file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/([^)\s"]+)')

def fix_markdown_file(md_path: Path, repo_root: Path):
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    rel_dir = md_path.parent.relative_to(repo_root)

    def replace_match(match):
        target_path_str = match.group(1)
        # Compute relative path from md file directory to target path
        if str(rel_dir) == '.':
            target_rel = target_path_str
        else:
            target_rel = os.path.relpath(target_path_str, str(rel_dir)).replace('\\', '/')
        return target_rel

    new_content = PREFIX_PATTERN.sub(replace_match, content)

    # Also catch any generic file:/// links or Windows absolute paths if present
    new_content = re.sub(r'file:///[a-zA-Z]:/[^)\s"]+/ADHD200_GitHub_Staging/([^)\s"]+)', replace_match, new_content)

    if new_content != content:
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Fixed links in: {md_path.relative_to(repo_root)}")


def sanitize_notebook_outputs(nb_path: Path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    modified = False
    for cell in nb.get('cells', []):
        for out in cell.get('outputs', []):
            if 'text' in out:
                new_text = []
                for line in out['text']:
                    cleaned_line = line.replace('/home/nvidia/23BRS1236/adhd_data', '[DATA_ROOT]')
                    cleaned_line = cleaned_line.replace('/home/nvidia/23BRS1236/mnt/ADHD200', '[DATA_ROOT]')
                    cleaned_line = cleaned_line.replace('/home/nvidia/23BRS1236', '[PROJECT_ROOT]')
                    cleaned_line = cleaned_line.replace('/lp-dev/23BRS1236/mnt/ADHD200', '[DATA_ROOT]')
                    if cleaned_line != line:
                        modified = True
                    new_text.append(cleaned_line)
                out['text'] = new_text

    if modified:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Sanitized outputs in: {nb_path.name}")


def main():
    repo_root = Path(__file__).resolve().parent.parent
    md_files = list(repo_root.rglob('*.md'))
    print(f"Scanning {len(md_files)} markdown files for file:/// links...")
    for md in md_files:
        fix_markdown_file(md, repo_root)

    nb_files = list((repo_root / 'notebooks').rglob('*.ipynb'))
    print(f"Sanitizing outputs in {len(nb_files)} notebooks...")
    for nb in nb_files:
        sanitize_notebook_outputs(nb)

    print("Fix complete.")


if __name__ == '__main__':
    main()
