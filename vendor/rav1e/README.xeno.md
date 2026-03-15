This directory vendors `rav1e` `0.8.1` from `xiph/rav1e`.

XENO-specific changes:
- patched the crate to resolve `paste` to the vendored maintained `pastey`
  fork under `vendor/pastey`
- kept the encoder source and upstream license files intact

Reason:
- upstream `rav1e` still depends on the archived `paste` crate
- RustSec flags `paste` as unmaintained
- `pastey` is the maintained successor and is intended to be source-compatible
  for the token-pasting macro usage in `rav1e`
