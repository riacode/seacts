# Data

This directory is for local notes only. Raw and processed DepMap data should not be committed.

The project is designed to download real DepMap files into the `seacts-data` Modal Volume:

```bash
modal run modal_data.py
```

The minimal raw file subset is:

- `CRISPRGeneEffect.csv`
- `Model.csv` or `sample_info.csv`
- `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`
- `PortalOmicsCNGeneLog2.csv`
- `OmicsSomaticMutationsMatrixDamaging.csv`

If running locally, place those files under `data/raw/`. That path is ignored by git.
