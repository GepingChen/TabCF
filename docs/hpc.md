# HPC Notes

This repository does not include SLURM job scripts, since the originals contain machine-specific paths, mail settings, and other lab-local assumptions that would not transfer to your environment.

If you need scheduler support:

- start from the Python/R CLI entrypoints kept in this repo;
- wrap them in your own site-local scheduler templates;
- keep output directories repo-relative;
- provide credentials such as Hugging Face tokens through your scheduler environment rather than hardcoding them.
