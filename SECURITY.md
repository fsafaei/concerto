# Security policy

## Reporting a vulnerability

We take the security of CONCERTO and CHAMBER seriously. If you believe you have
found a security vulnerability, please report it privately to:

**safaei.farhad@gmail.com**

Please do **not** open a public GitHub issue for security reports.

When reporting, please include:

- A description of the vulnerability and its potential impact.
- Steps to reproduce.
- Affected versions or commit SHAs.
- Any proof-of-concept code, logs, or screenshots.

## Disclosure timeline

- We acknowledge receipt within **5 business days**.
- We aim to validate, fix, and release a patched version within **90 days**.
- We will credit you in the release notes unless you ask to remain anonymous.

For critical vulnerabilities (remote code execution, supply-chain compromise,
secrets exfiltration), we will work with you to coordinate a faster timeline
and an embargoed advisory if appropriate.

## Scope

In scope:

- The CONCERTO method packages (`src/concerto/*`).
- The CHAMBER benchmark packages (`src/chamber/*`).
- The repository's CI/CD configuration (`.github/workflows/*`,
  `scripts/check_*.{sh,py}`).
- Any signed release artifact published from this repo.

Out of scope (please report upstream):

- Vulnerabilities in third-party dependencies. We will bump our pinned
  versions promptly once an upstream fix lands.
- Issues in code that has not yet shipped to a tagged release.
- Issues that require physical access to a developer's machine.

## Supported versions

The latest release on `main` is supported. Older Phase-0 dev tags
(`v0.0.x.dev*`) are not supported.
