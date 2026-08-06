# Knowledge packs

Anything dropped in this folder is chunked, embedded and indexed when the server
starts. Supported: `.txt`, `.md`, `.pdf`, `.docx`, `.html`.

This is the seam that makes one platform serve many domains. The code does not
change between an electricity board and a university admissions desk — the
knowledge pack, the agent profile and the tool set do.

Load without restarting:

```bash
vaani ingest ./knowledge --agent default
```

or via the API:

```bash
curl -F "file=@citizen-charter.pdf" \
     "http://localhost:8080/api/knowledge/upload?agent_key=default"
```

Check retrieval quality before trusting it on a call — this is the fastest way
to diagnose a wrong answer:

```bash
curl "http://localhost:8080/api/knowledge/search?q=last+date+to+pay+the+bill"
```

## Writing content that works on a phone call

Retrieved text is read aloud, so the source matters more than usual.

- One fact per paragraph. Chunks are split on paragraph boundaries.
- Write numbers as words: "fifteenth of March", not "15/03".
- Avoid tables. A table read aloud is unintelligible; write the rows as
  sentences instead.
- Put the answer first. Callers hang up during preamble.
- Scanned PDFs index as empty — the ingest log warns when it sees one. Run OCR
  first.
