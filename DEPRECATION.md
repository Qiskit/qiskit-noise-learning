# Deprecation policy

This library is in the `0.x` stage of development, where breaking changes are permitted between
releases. It is alpha software: no part of the public interface is yet considered stable, and we
do not commit any interface to a deprecation policy at this time.

In practice this means:

- We reserve the right to change or remove any interface between any two releases, including
  between minor versions, without a deprecation period.
- We do not currently promise to issue deprecation warnings before making a breaking change.
- You should pin your dependency on this package (for example, `qiskit-noise-learning==0.1.*`) and
  review the [changelog](CHANGELOG.md) before upgrading.

As the public interface stabilizes, we expect to adopt a stricter deprecation policy and document it
here.
