This directory vendors `pastey` `0.1.1` from `as1100k/pastey`.

XENO-specific changes:
- patched the dependency graph so crates requesting `paste` resolve to this
  vendored maintained fork instead of the archived `paste` crate
- kept the proc-macro source and upstream license files intact

Reason:
- `rav1e` currently depends on `paste`
- `paste` is archived and flagged by RustSec as unmaintained
- `pastey` is the maintained successor intended to be source-compatible for
  token-pasting macro use cases
