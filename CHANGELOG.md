# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2019-06-06
- Remove `new` constructors for zero-sized types
- Add Pert distribution
- Fix undefined behavior in `Poisson`
- Make all distributions return `Result`s instead of panicking
- Implement `f32` support for most distributions
- Rename `UnitSphereSurface` to `UnitSphere`
- Implement `UnitBall` and `UnitDisc`

## [0.1.0] - 2019-06-06
Initial release. This is equivalent to the code in `rand` 0.6.5.
