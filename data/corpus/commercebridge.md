# CommerceBridge connector

> Fictional Northstar Analytics demonstration document.

CommerceBridge imports orders, refunds and inventory counts from supported retail stores. An Admin creates a read-only integration token in the source system and enters it in Harbor. The connector does not write inventory changes back to the store.

The first synchronization imports the preceding thirty days, subject to plan retention limits. Subsequent refresh frequency follows the workspace plan. A token must allow order and inventory read scopes. Error CB-401 indicates an expired or revoked token; replace the token and retry once. Never send a token to support.

The optional multi-currency add-on costs $15 per workspace per month. It uses the daily exchange rate saved on import and does not retroactively recalculate historical orders when rates change.
