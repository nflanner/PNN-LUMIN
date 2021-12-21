# Parameterized Neural Network using LUMIN

LUMIN is a deep learning and data-analysis ecosystem for High Energy Particle Physics written by Giles Strong, see https://github.com/GilesStrong/lumin.

Utilizing the PNN in LUMIN requires three distinct steps, importing, model training, and validation/testing, which I will walk through using the various notebooks within this repository.

## Importing

notebook:
https://github.com/nflanner/PNN-LUMIN/blob/main/res_import_filtered_Rev4.ipynb

First, we create a list of strings associated to all the resonant mass ROOT files used for training (I left out 900 GeV to allow for testing on a mass that was not used in the training).

```bash
res_mass = [260, 270, 300, 350, 400, 450, 500, 550, 600, 650, 750, 800, 1000]
res_mass2 = [1250, 1500, 1750, 2000, 2500, 3000]
sig_str = ['GluGluToRadionToHHTo2B2ZTo2L2J_M-'+str(i)+'_narrow_13TeV-madgraph-v2' for i in res_mass]
for val in res_mass2:
    sig_str.append('GluGluToRadionToHHTo2B2ZTo2L2J_M-'+str(val)+'_narrow_TuneCUETP8M1_PSWeights_13TeV-madgraph-pythia8')
```
