#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "beautifulsoup4",
#     "html2text",
#     "requests",
# ]
# ///
"""
Scrape NVIDIA PTX ISA documentation and convert to markdown.
Preserves tables, math notation, code blocks, and other formatting.
"""

import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import html2text
import requests
from bs4 import BeautifulSoup, NavigableString, Tag


class PTXDocScraper:
    """Scrapes and converts PTX ISA documentation to markdown."""

    BASE_URL = "https://docs.nvidia.com/cuda/parallel-thread-execution/"

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configure html2text for better markdown conversion
        self.h2t = html2text.HTML2Text()
        self.h2t.body_width = 0  # Don't wrap text
        self.h2t.ignore_links = False
        self.h2t.ignore_images = False
        self.h2t.ignore_emphasis = False
        self.h2t.skip_internal_links = False
        self.h2t.protect_links = True
        self.h2t.unicode_snob = True
        self.h2t.use_automatic_links = False

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

    def fetch_page(self, url: str) -> BeautifulSoup | None:
        """Fetch a page and return BeautifulSoup object."""
        try:
            print(f"Fetching: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            print(f"Error fetching {url}: {e}", file=sys.stderr)
            return None

    def convert_to_markdown(self, element: Tag | NavigableString) -> str:
        """Convert an HTML element to markdown."""
        markdown = self.h2t.handle(str(element))
        return self._clean_markdown(markdown)

    def _clean_markdown(self, markdown: str) -> str:
        """Clean up converted markdown."""
        # Remove excessive newlines (more than 3)
        markdown = re.sub(r'\n{4,}', '\n\n\n', markdown)

        # Fix common conversion issues
        markdown = re.sub(r'\\\[', '[', markdown)
        markdown = re.sub(r'\\\]', ']', markdown)

        # Ensure code blocks are properly formatted
        markdown = re.sub(r'```\s*\n\s*```', '', markdown)

        # Convert relative image paths to absolute URLs
        # Pattern: ![alt text](_images/filename.png) -> ![alt text](https://docs.nvidia.com/cuda/parallel-thread-execution/_images/filename.png)
        markdown = re.sub(
            r'!\[(.*?)\]\(_images/(.*?)\)',
            r'![\1](https://docs.nvidia.com/cuda/parallel-thread-execution/_images/\2)',
            markdown
        )

        return markdown.strip()

    def sanitize_filename(self, title: str, section_num: str = "") -> str:
        """Convert title to safe filename."""
        # Remove section number from title if present
        title = re.sub(r'^\d+(\.\d+)*\.?\s*', '', title)

        # Remove special characters, replace spaces with hyphens
        filename = re.sub(r'[^\w\s-]', '', title.lower())
        filename = re.sub(r'[-\s]+', '-', filename)
        filename = filename.strip('-')

        # Add section number prefix
        if section_num:
            filename = f"{section_num}-{filename}"

        return filename if filename else "index"

    def extract_sections_from_page(self, soup: BeautifulSoup) -> list[dict[str, Any]]:
        """Extract all sections from the single-page documentation."""
        sections = []

        # Find main content area
        content = None
        for selector in [
            {'role': 'main'},
            {'class': 'document'},
            {'class': 'body'},
            {'itemprop': 'articleBody'}
        ]:
            content = soup.find('div', selector) or soup.find('section', selector)
            if content:
                break

        if not content:
            print("Warning: Could not find main content area", file=sys.stderr)
            return sections

        # Find all heading tags that represent sections
        headings = content.find_all(['h1', 'h2', 'h3', 'h4'])

        for i, heading in enumerate(headings):
            # Extract section number and title
            heading_text = heading.get_text(strip=True)

            # Skip if empty
            if not heading_text:
                continue

            # Extract section number if present
            section_match = re.match(r'^(\d+(?:\.\d+)*)\.\s*(.+)$', heading_text)
            if section_match:
                section_num = section_match.group(1)
                title = section_match.group(2)
            else:
                section_num = ""
                title = heading_text

            # Get the anchor ID
            anchor_id = heading.get('id', '')
            if not anchor_id:
                # Try to find it in a child element
                anchor = heading.find('a')
                if anchor:
                    anchor_id = anchor.get('id', '')

            # Determine level from heading tag
            level = int(heading.name[1]) - 1  # h1 -> 0, h2 -> 1, etc.

            # Collect content until next heading of same or higher level
            content_elements = []
            current = heading.next_sibling

            while current:
                # Stop if we hit another heading of equal or higher importance
                if isinstance(current, Tag) and current.name in ['h1', 'h2', 'h3', 'h4']:
                    current_level = int(current.name[1]) - 1
                    if current_level <= level:
                        break

                if isinstance(current, Tag):
                    content_elements.append(current)

                current = current.next_sibling

            sections.append({
                'title': title,
                'section_num': section_num,
                'level': level,
                'anchor': anchor_id,
                'heading': heading,
                'content': content_elements
            })

        return sections

    def save_section(self, section: dict[str, Any], parent_dir: Path) -> None:
        """Save a section as a markdown file."""
        # Create filename
        filename = self.sanitize_filename(section['title'], section['section_num'])

        # Convert heading and content to markdown
        markdown_parts = []

        # Add the heading
        level_prefix = '#' * (section['level'] + 1)
        title_with_num = f"{section['section_num']}. {section['title']}" if section['section_num'] else section['title']
        markdown_parts.append(f"{level_prefix} {title_with_num}\n")

        # Add content
        for element in section['content']:
            # Remove unwanted UI elements
            for class_name in ['headerlink', 'viewcode-link', 'navigation', 'related']:
                for unwanted in element.find_all(class_=class_name):
                    unwanted.decompose()

            md = self.convert_to_markdown(element)
            if md:
                markdown_parts.append(md)

        markdown = '\n\n'.join(markdown_parts)

        # Add frontmatter
        url = f"{self.BASE_URL}index.html#{section['anchor']}" if section['anchor'] else self.BASE_URL
        frontmatter = f"""---
title: "{title_with_num}"
section: {section['section_num']}
url: {url}
---

"""

        # Determine output file
        output_file = parent_dir / f"{filename}.md"

        # Write file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(frontmatter + markdown)

        print(f"  Saved: {output_file.name}")

    def scrape_all(self) -> None:
        """Scrape the entire documentation."""
        # Fetch the main page
        soup = self.fetch_page(self.BASE_URL + "index.html")
        if not soup:
            print("Failed to fetch documentation", file=sys.stderr)
            return

        print("Extracting sections from page...")
        sections = self.extract_sections_from_page(soup)

        if not sections:
            print("No sections found!", file=sys.stderr)
            return

        print(f"Found {len(sections)} sections")

        # Build a set of section numbers that have children
        # A section has children if another section's number starts with this section's number + "."
        parent_sections = set()
        for section in sections:
            if section['section_num']:
                for other in sections:
                    if other['section_num'] and other['section_num'] != section['section_num']:
                        if other['section_num'].startswith(section['section_num'] + '.'):
                            parent_sections.add(section['section_num'])
                            break

        print(f"Identified {len(parent_sections)} parent sections to skip (have subsections)")

        # Group sections by top-level chapter
        current_chapter = None
        chapter_dir = self.output_dir

        for section in sections:
            # Skip "Notices" section
            if 'notice' in section['title'].lower() and section['level'] == 0:
                print(f"Skipping: {section['title']}")
                continue

            # Create new directory for top-level sections (chapters)
            # Do this even if we're skipping the parent file itself
            if section['level'] == 0:
                chapter_name = self.sanitize_filename(section['title'], section['section_num'])
                chapter_dir = self.output_dir / chapter_name
                chapter_dir.mkdir(exist_ok=True)
                current_chapter = chapter_name

                # If this is a parent section with children, skip saving the file but keep the directory
                if section['section_num'] in parent_sections:
                    print(f"\nChapter: {section['title']} (skipping parent file, keeping directory)")
                    continue
                else:
                    print(f"\nChapter: {section['title']}")

            # Skip parent sections that have subsections (but not chapter-level, we handled that above)
            elif section['section_num'] in parent_sections:
                print(f"  Skipping parent: {section['section_num']} {section['title']}")
                continue

            # Save the section
            self.save_section(section, chapter_dir)

        print(f"\n✓ Scraping complete!")
        print(f"Documentation saved to: {self.output_dir}")

        # Create an index file
        self.create_index(sections)

    def create_index(self, sections: list[dict[str, Any]]) -> None:
        """Create an index file listing all sections."""
        index_path = self.output_dir / "INDEX.md"

        lines = ["# PTX ISA 9.1 Documentation Index\n"]

        current_level = -1
        for section in sections:
            # Skip notices
            if 'notice' in section['title'].lower() and section['level'] == 0:
                continue

            indent = "  " * section['level']
            title_with_num = f"{section['section_num']}. {section['title']}" if section['section_num'] else section['title']

            # Create relative link
            if section['level'] == 0:
                chapter_name = self.sanitize_filename(section['title'], section['section_num'])
                filename = self.sanitize_filename(section['title'], section['section_num'])
                link = f"{chapter_name}/{filename}.md"
            else:
                # Find parent chapter
                parent_section = section['section_num'].split('.')[0] if section['section_num'] else ""
                # This is approximate - we'd need to track the chapter context
                link = f"section-{section['section_num']}.md" if section['section_num'] else ""

            lines.append(f"{indent}- [{title_with_num}]({link})")

        with open(index_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"\n✓ Created index: {index_path}")


def main() -> None:
    """Main entry point."""
    # Output to cuda_skill/references/ptx-docs/
    output_dir = Path(__file__).parent / "cuda_skill" / "references" / "ptx-docs"

    scraper = PTXDocScraper(output_dir)
    scraper.scrape_all()


if __name__ == "__main__":
    main()
