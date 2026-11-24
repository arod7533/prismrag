# Northstar Analytics demo corpus

24 fictional company documents spanning product, billing, support, security, data integrations, incidents and employee operations. The app loads these automatically from `data/corpus/`.

Includes a historical pricing distractor, unreleased versus released features, overlapping policies, exact numerical facts and questions requiring multiple sources. Nothing here is a claim about a real company. The `.invalid` address cannot receive mail.

`data/eval.json` contains 20 curated demonstration questions, reference answers and document-level relevance labels. Labels are authored alongside the synthetic documents and are development data, not an independent or held-out benchmark. Precision measures selected labeled evidence, which may omit other useful passages. Add independent human-reviewed and unanswerable cases before claiming general quality.
