# Alt Memory Mine

When the user invokes this skill, follow these steps:

## 1. Ask what to mine

Ask the user what they want to mine and where the source data is located.
Clarify:
- Is it a project directory (code, docs, notes)?
- Is it conversation exports (Claude, ChatGPT, Slack)?
- Do they want auto-classification (decisions, milestones, problems)?

## 2. Choose the mining mode

There are three mining modes:

### Project mining

    alt-memory mine <dir>

Mines code files, documentation, and notes from a project directory.

### Conversation mining

    alt-memory mine <dir> --mode convos

Mines conversation exports from Claude, ChatGPT, or Slack into the dimension.

### General extraction (auto-classify)

    alt-memory mine <dir> --mode convos --extract general

Auto-classifies mined content into decisions, milestones, and problems.

## 3. Optionally split mega-files first

If the source directory contains very large files, suggest splitting them
before mining:

    alt-memory split <dir> [--dry-run]

Use --dry-run first to preview what will be split without making changes.

## 4. Optionally tag with a wing

If the user wants to organize mined content under a specific realm, add the
--realm flag:

    alt-memory mine <dir> --realm <name>

## 5. Show progress and results

Run the selected mining command and display progress as it executes. After
completion, summarize the results including:
- Number of items mined
- Categories or classifications applied
- Any warnings or skipped files

## 6. Suggest next steps

After mining completes, suggest the user try:
- /alt-memory:search -- search the newly mined content
- /alt-memory:status -- check the current state of their dimension
- Mine more data from additional sources
