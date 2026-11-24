# Security architecture

> Fictional Northstar Analytics demonstration document.

Harbor encrypts network traffic with TLS and stored datasets with AES-256. Platform Engineering controls production access using individual accounts, MFA and least-privilege roles. Production access is reviewed quarterly. These are fictional design statements, not an independent certification.

Harbor’s demo company has no SOC 2 report, ISO 27001 certificate or HIPAA compliance offering. Do not claim those certifications. Customers must not upload full payment-card numbers, patient medical records or government identity documents.

Suspected vulnerabilities should be reported to security@example.invalid with a description and reproduction steps, without customer secrets. The .invalid domain is intentionally non-routable for this fictional corpus.
