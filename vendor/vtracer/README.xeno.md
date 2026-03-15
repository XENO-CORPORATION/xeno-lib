This directory vendors `vtracer` `0.6.5` from `visioncortex/vtracer`.

XENO-specific changes:
- switched `xeno-lib` to a path dependency so the shipped dependency graph is explicit
- removed the upstream CLI target and its `clap 2` dependency
- moved the vendored library onto `image 0.25` so it shares the maintained image stack used by `xeno-lib`
- kept the library code and upstream license files intact

Reason:
- `xeno-lib` uses `vtracer` as an internal SVG tracing library
- the upstream CLI dependency chain pulled in `atty`, which is unmaintained and flagged by Dependabot
