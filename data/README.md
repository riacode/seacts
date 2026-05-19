# Data

This directory is for local notes only. Raw and processed DepMap data should not be committed.

The project is designed to download real DepMap files into the `seacts-data` Modal Volume:

```bash
modal run modal_data.py
```

The Modal download set is:

- `CRISPRGeneEffect.csv`
- `Model.csv` or `sample_info.csv`
- `Gene.csv`
- `SubtypeMatrix.csv`
- `SubtypeTree.csv`
- `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`
- `PortalOmicsCNGeneLog2.csv`
- `OmicsSomaticMutationsMatrixDamaging.csv`
- `OmicsSomaticMutationsMatrixHotspot.csv`
- `OmicsGlobalSignatures.csv`
- `OmicsInferredMolecularSubtypes.csv`

The current baseline config uses the gene-aligned matrices: CRISPR gene effect, expression, copy number, damaging mutation, and hotspot mutation. The subtype and global-signature files are downloaded for later context features and analysis. If running locally, place those files under `data/raw/`. That path is ignored by git.
