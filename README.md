# Plotting
This branch is only for running stand-along python scripts for fixing aberrant copy number calls by the main branch pipeline. 

#Usage 
```
Plotting.py <intersect_cnr> <sorted_cns> <cleanname> <output> <correction_regions.json> [highlight_json]
```

where 'correction_regions.json' should something like 
```
{
  "5q": {
    "chromosome": "chr5",
    "start": 108800000,
    "end": 181538259,
    "copy_number": 3
  },
"5p": {
    "chromosome": "chr5",
    "start": 0,
    "end": 108800000,
    "copy_number": 2
  },
"8p": {
    "chromosome": "chr8",
    "start": 0,
    "end": 85200000,
    "copy_number": 2
  },
"8q": {
    "chromosome": "chr8",
    "start": 85200000,
    "end": 145138636,
    "copy_number":3
  }
}
```
# Citations

1. Talevich, E., Shain, A.H., Botton, T., & Bastian, B.C. (2014). CNVkit: Genome-wide copy number detection and visualization from targeted sequencing. PLOS Computational Biology 12(4):e1004873
2. i) Olshen, A.B., Bengtsson, H., Neuvial, P., Spellman, P.T., Olshen, R.A., & Seshan, V.E. (2011). Parent-specific copy number in paired tumor-normal studies using circular binary segmentation. Bioinformatics 27(15):2038–46.
  ii) Venkatraman, E.S., & Olshen, A.B. (2007). A faster circular binary segmentation algorithm for the analysis of array CGH data. Bioinformatics 23(6):657–63

# License and Attribution

This project is primarily licensed under the MIT License. However, it includes code originally developed by Eric Talevich, University of California under the Apache License, Version 2.0. See the `LICENSE` file for the MIT License and the `NOTICE` file for Apache License details.

See the `NOTICE` file for further attribution requirements.
