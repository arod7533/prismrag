# API authentication and rate limits

> Fictional Northstar Analytics demonstration document.

Harbor’s read API uses workspace-scoped bearer tokens created by an Admin. Each token must have an expiry of no more than ninety days. Tokens can be revoked immediately. Never place tokens in query parameters or public repositories.

Starter allows sixty requests per minute per workspace, Growth allows three hundred and Enterprise allows one thousand. Exceeding the limit returns HTTP 429 with a Retry-After header in seconds. Clients should wait that duration and retry with jitter. Limits aggregate across tokens in one workspace.

The API returns paginated results with a maximum page size of five hundred rows. API requests do not trigger an immediate connector refresh and cannot bypass plan retention limits.
