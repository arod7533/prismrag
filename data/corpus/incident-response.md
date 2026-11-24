# Incident response playbook

> Fictional Northstar Analytics demonstration document.

The on-call Platform Engineering lead becomes incident commander for severity-one incidents. The commander assigns an investigator and a communications owner. Customer Operations publishes the first status update within thirty minutes of confirmed impact and subsequent updates every sixty minutes until recovery.

For suspected data exposure, immediately restrict the affected integration token, preserve audit logs and page the security lead. Do not delete evidence or paste secrets into an incident chat. Customer notification content must be reviewed by the security lead.

After service recovery, a blameless incident review is due within five business days. It records impact, timeline, root cause, corrective actions and owners. Review incident-2026-10.md for an example involving delayed imports.
