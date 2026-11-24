# Warehouse Sync connector

> Fictional Northstar Analytics demonstration document.

Warehouse Sync imports tables from a customer-managed PostgreSQL warehouse using a read-only database user. The connector needs TLS, a reachable host and explicit access to the selected tables. Store the password only in the connector secret field.

The connector uses an updated_at timestamp as its incremental cursor. Rows without a valid cursor are skipped and reported in the import log. Deletes are not inferred from missing rows; customers must supply an is_deleted boolean for deletion propagation. Large initial loads should be scheduled outside store reporting peaks.

Warehouse Sync is included on Growth and Enterprise and unavailable on Starter. A schema change pauses the affected table until an Admin reviews and approves the new mapping.
