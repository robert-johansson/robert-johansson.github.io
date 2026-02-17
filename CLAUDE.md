# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

"The Learning Machine" — Jekyll-based research blog hosted on GitHub Pages at robert-johansson.se. Topics: probabilistic programming, machine psychology, AGI. Uses the "So Simple" remote theme.

## Build & Development Commands

```bash
bundle install          # Install dependencies
bundle exec jekyll serve  # Local dev server with live reload
bundle exec jekyll build  # Build static site to _site/
```

Jekyll version is managed by the `github-pages` gem — no need to pin a version manually.

## Architecture

- **Static site generator**: Jekyll with kramdown markdown processor
- **Theme**: `mmistakes/so-simple-theme` (remote theme via `_config.yml`)
- **Hosting**: GitHub Pages (custom domain via `CNAME`)
- **Plugins**: jekyll-seo-tag, jekyll-sitemap, jekyll-feed, jekyll-paginate
- **Interactive pages**: Scittle (ClojureScript in browser via script tags, no build step)

### Key directories

- `_posts/` — Blog posts in Markdown with YAML front matter
- `_data/` — Site data files (`navigation.yml` for nav menu, `text.yml` for theme labels)
- `_bibliography/` — BibTeX references (`references.bib`)
- `_site/` — Generated output (do not edit directly)
- `images/`, `sounds/`, `files/` — Static assets

### Content conventions

- Posts use the naming convention `YYYY-MM-DD-title.md`
- Permalink structure: `/:categories/:title/`
- Front matter typically includes `title`, `categories`, `tags`, `date`
- Scittle pages use `.html` extension with Jekyll front matter and `<script type="application/x-scittle">` tags
