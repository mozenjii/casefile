# Review screen design

The review screen is a working tool for one person checking one case at a desk. It should feel calm,
dense and trustworthy. Nothing on it is decoration.

## Layout

Three columns on a wide screen, stacked below 1100px:

1. **Dossier**: the recommendation, the five criteria with their result and reason, what was requested
   from the provider, the drafted claims, and anything the grounding check rejected.
2. **Source documents**: the documents as received, with every evidence span marked.
3. **Decision and timeline**: reviewer name, approve / request information / reject, release, and every
   event in the case's log.

Below 700px the criteria table becomes one card per criterion.

## Rules

- **Every claim is one click from its source.** Evidence ids are plain links to the marked span, so
  they work without JavaScript; the CSS `:target` state highlights the span.
- **Status is never colour alone.** Met, not met, unknown and conflict each carry their word.
- **The version is always visible.** The page shows the dossier hash being decided on, and the form
  posts it back.
- **Keyboard first.** A, I and R decide; focus outlines are always visible. The reviewer name is
  remembered in the browser.
- **No web fonts, no client framework, no build step.** System fonts, server-rendered HTML, a few lines
  of script.

## Tokens

Colours are CSS custom properties on `:root`, with a dark set under `prefers-color-scheme: dark`:
surface layers (`--bg`, `--surface`, `--line`), text (`--text`, `--muted`), one accent, and a
foreground/background pair per status. Radius is 6px. Text is 15px with 1.5 line height; documents are
monospace at 13px so spans line up with what was received.

## Motion

Only the highlight of a cited span fades in, over 250ms, and only when the reader has not asked for
reduced motion.
