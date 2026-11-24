# CSV import requirements

> Fictional Northstar Analytics demonstration document.

Harbor accepts UTF-8 CSV files of at most fifty megabytes and one million rows each. Required sales columns are order_id, ordered_at, store_id, currency and gross_amount. Currency uses a three-letter code. ordered_at must include a time zone or use the workspace default time zone.

Duplicate order_id values within the same store update the existing row rather than creating another sale. Blank gross_amount values are rejected. Currency symbols such as $ must be removed from numeric fields. The import preview shows rejected rows before submission.

CSV files are not scheduled connectors. To automate imports or handle larger datasets, use Warehouse Sync on Growth or Enterprise. The CSV limit does not increase on Enterprise.
