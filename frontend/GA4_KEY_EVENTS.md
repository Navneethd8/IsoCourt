# GA4 key events

Property: `G-TET6JN36Q4`.

Hits are sent only from `www.isocourt.fit` and `isocourt.fit`, and only when the server has set the `ic_vid` cookie. That cookie is the GA4 client id. Preview URLs, the Hugging Face Space, and requests flagged as bots do not send hits and do not get `ic_vid`.

After a deploy has sent each event at least once, open Admin → Data display → Events and mark only these two as key events:

- `analyze_completed` — total clips analyzed. One event when a clip finishes.
- `live_session_started` — total times gone live. One event when the live socket is open.

Read each total as that event’s own count in Reports → Events. Pick the date range you want; the full property history is the all-time number. The combined Key events card adds every key event together, so it is not either total.

Leave these as ordinary events. They are the outage signal and must not be added into the Key events total:

- `analyze_failed`
- `live_session_failed`
- `live_coaching_delivered`

Unmark older click or “started” events if any were marked (`analyze_click`, `Stream Started`, `Session Started`, and the rest of the pre-revamp names). Those names are no longer sent.

What to watch: `analyze_started` with `analyze_completed` near zero, or `live_session_started` with no `live_coaching_delivered`.
