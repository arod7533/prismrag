"""Human-authored document topics, independent of the embedding model."""
TOPIC_COLORS = {
    "Product & plans": "#60a5fa", "Customer operations": "#fbbf24",
    "Security & access": "#a78bfa", "Integrations & data": "#2dd4bf",
    "Incidents": "#fb7185", "Company & people": "#a3e635", "Uncategorized": "#94a3b8"}
GROUPS = {
    "Product & plans": ["harbor-product", "pricing-current", "pricing-archive", "release-notes", "dashboard-sharing"],
    "Customer operations": ["billing-refunds", "onboarding", "support", "sla"],
    "Security & access": ["roles-permissions", "sso", "data-retention", "security", "backup-recovery"],
    "Integrations & data": ["commercebridge", "warehouse-sync", "csv-import", "api-limits", "metrics-definitions"],
    "Incidents": ["incident-response", "incident-2026-10"],
    "Company & people": ["company-overview", "employee-onboarding", "expense-policy"]}
TOPICS = {name + ".md": topic for topic, names in GROUPS.items() for name in names}

def topic_for(doc):
    return doc.metadata.get("topic", TOPICS.get(doc.metadata["source"], "Uncategorized"))
