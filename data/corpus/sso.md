# Single sign-on setup

> Fictional Northstar Analytics demonstration document.

SAML single sign-on is available only on Enterprise. Growth and Starter use email sign-in with optional time-based one-time-password MFA. Enterprise can enforce SAML for ordinary members while keeping one emergency Owner account protected with MFA.

Setup requires an identity-provider metadata URL, the Harbor service-provider entity ID, and matching email addresses. Test with one Admin before enforcing SAML. If a user receives an audience mismatch error, compare the configured entity ID to the value shown in Harbor’s SSO settings.

SSO does not automatically provision users. Invite users explicitly; SCIM provisioning is not supported. Customer Operations assists with setup and Platform Engineering investigates persistent authentication failures.
