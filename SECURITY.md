# Security Policy

XENO Corporation takes the security of xeno-lib seriously. We appreciate responsible disclosure and will work to validate, reproduce, and fix legitimate vulnerabilities affecting supported releases.

## Supported Versions

| Version | Supported |
|---|---|
| 0.1.x (latest) | Yes |
| < 0.1.0 | No |

Security fixes are applied to the latest supported release line. We do not guarantee backports for older branches.

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, report vulnerabilities through one of these channels:

### Preferred: GitHub Security Advisories

Use the repository Security Advisories feature to report the issue privately.

### Alternative: Email

Send a report to **security@bnkrsys.com** with:

- description of the issue
- affected command, API, or module
- reproduction steps or proof of concept
- expected impact
- any suggested mitigation

If needed, request an encrypted channel for sensitive follow-up.

## Response Timeline

| Stage | Target |
|---|---|
| Initial acknowledgment | Within 48 hours |
| Preliminary assessment | Within 7 days |
| Fix or mitigation plan | Within 30 days for critical or high severity |
| Public disclosure | After a fix is available, or after a coordinated disclosure window |

## Scope

### In Scope

- Memory safety issues or out-of-bounds access in image, audio, video, subtitle, or container parsing
- Denial of service via malformed media files or crafted model inputs
- Path traversal or unsafe file handling in CLI workflows
- Unsafe ONNX model loading or unexpected code execution paths
- GPU backend issues that can corrupt memory or crash the host process
- Supply-chain vulnerabilities in project dependencies

### Out of Scope

- Model output quality, bias, or hallucinations
- Performance issues under normal supported workloads
- Missing codecs or unsupported formats
- Crashes caused only by intentionally invalid command-line usage already rejected by argument parsing
- Vulnerabilities in third-party tools or drivers that are outside the xeno-lib codebase, unless xeno-lib is invoking them unsafely

## Threat Model

xeno-lib processes untrusted local media and optional model assets. The main security concerns are:

1. Malicious media inputs: crafted image, audio, video, subtitle, or container files that attempt to trigger parser or decoder vulnerabilities.
2. Optional AI/runtime assets: malformed model files or invalid tensor metadata causing memory or resource abuse.
3. Native backend interactions: GPU, codec, or system-library integrations that can fail unsafely if assumptions are wrong.
4. CLI file operations: commands that read, write, transform, or batch-process user files.

## Disclosure Policy

- We follow coordinated disclosure.
- Reporters may be credited publicly unless they prefer anonymity.
- Critical issues may be fixed and shipped outside the normal release cadence.
