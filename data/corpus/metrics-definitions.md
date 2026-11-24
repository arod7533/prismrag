# Retail metric definitions

> Fictional Northstar Analytics demonstration document.

Gross sales is the sum of gross_amount before refunds. Net sales equals gross sales minus recorded refund amounts. Orders counts distinct order_id values per store, excluding rows marked deleted. Average order value equals net sales divided by orders and is undefined when there are no orders.

Harbor reports sales by the workspace time zone, not the browser time zone. Currency conversion uses the exchange rate stored at import time when the multi-currency add-on is enabled. Comparing a source dashboard that uses UTC or a different refund cutoff can produce apparent mismatches.

A delayed connector refresh affects freshness, not the metric formulas. Always inspect the most recent successful refresh before investigating an unexpected daily total.
