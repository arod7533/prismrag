# Data retention policy

> Fictional Northstar Analytics demonstration document.

Status: CURRENT. Harbor retains imported source rows for ninety days on Starter, one hundred eighty days on Growth and three hundred sixty-five days on Enterprise. Aggregated dashboard results are kept for the same retention period as source rows.

After workspace deletion is verified, active workspace data is removed within seven calendar days. Encrypted backups expire on a rolling thirty-day schedule. Backup expiration is distinct from active-data deletion; deletion does not instantly erase every backup copy.

An Owner must submit a deletion request with the workspace ID. Platform Engineering performs the deletion and Customer Operations confirms completion. Retention settings do not authorize customers to upload prohibited sensitive data.
