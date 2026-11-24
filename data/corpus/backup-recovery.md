# Backup and disaster recovery

> Fictional Northstar Analytics demonstration document.

Platform Engineering takes encrypted database backups every six hours and retains them for thirty days. The recovery point objective is six hours; the recovery time objective is eight hours after a disaster is declared. These are recovery targets, not a guarantee that every outage resolves in eight hours.

A recovery drill is performed quarterly in an isolated environment. The on-call incident commander approves restoration, and a second engineer verifies restored row counts and application health. Customer-specific point-in-time restores are not offered through the product interface.

Workspace deletion removes active data within seven days, while backup copies expire according to the thirty-day backup schedule. Restored systems must reapply pending deletion requests before serving traffic.
