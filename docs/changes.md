---
layout: default
title: Change Log
nav_order: 2
---

# Change Log

## v0.2.2
- Added ChainedCalc helper class to performed chaining of PropCalcs more easily.
- Added matcalc.load_up alias for easier loading of universal MLIPs.

## v0.2.0
- Major new feature: Most PropCalc now supports chaining. This allows someone to obtain multiple properties at once.
- SofteningBenchmark added. (@bowen-bd)
- Major expansion in the number of supported calculators, including Orb and GRACE models.

## v0.1.2
- Emergency bug fix for bad default perturb_distance parameter in Relaxation.

## v0.1.1
- Provide model aliases "PBE" and "R2SCAN" which defaults to the TensorNet MatPES models.
- New CLI tools to quickly use prop calculators to compute properties from structure files.

## v0.1.0
- Added support for ORB and GRACE universal calculators (@atulcthakur)
- Option to perturb structure before relaxation (@rul048)
- Improved handling of stress units (@rul048)
- Option to relax strained structures in ElasticityCalc (@lbluque)

## v0.0.6
- Checkpointing and better handling of benchmarking.
- Most PropCalc can now be imported from the root level, e.g., `from matcalc import ElasticityCalc` instead of the more
  verbose `from matcalc.elasticity import ElasticityCalc`.

## v0.0.5

- Initial release of benchmarking tools with Elasticity and Phonon benchmark data.

## v0.0.2

- Minor updates to returned dicts.

## v0.0.1

- First release with all major components.
