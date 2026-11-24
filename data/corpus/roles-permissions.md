# Workspace roles and permissions

> Fictional Northstar Analytics demonstration document.

Harbor provides Owner, Admin, Analyst and Viewer roles. Each workspace has exactly one Owner. The Owner manages billing, can transfer ownership and can request workspace deletion. Admins manage members, connectors and workspace settings, but cannot transfer ownership or delete the workspace.

Analysts build datasets and dashboards and export data they can access. Viewers can open shared dashboards but cannot edit datasets, change connectors or export raw data. Dataset access can further restrict Analysts and Viewers. A role alone does not grant access to every dataset.

Owner transfers require confirmation by both the current and new Owner. Customer Operations verifies requests but never accepts a password or recovery code as proof of ownership.
