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

Create a dictionary with keys associated to each resonant mass file and values corresponding to the uproot3 object (to be placed in a pandas DataFrame). This was all done locally, so PATH was set to the location of the ROOT files on my local machine.

```bash
PATH = Path('../../../MC')
signal = {sig: uproot3.open(PATH/(sig+'.root'))["Events"] for sig in sig_str}
background = {i: uproot3.open(PATH/(i+'.root'))["Events"] for i in\
              ['DYJetsToLL_M-10to50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8', 'DYToLL_0J_13TeV-amcatnloFXFX-pythia8',\
               'DYToLL_1J_13TeV-amcatnloFXFX-pythia8','DYToLL_2J_13TeV-amcatnloFXFX-pythia8']}
```

Create two seperate DataFrames (one for signal events and one for background) and add each individual signal resonant mass (signal) or NaN (background). Background masses will eventually be reandomly sampled from our list of candidate masses.

```bash
res_mass += res_mass2
# create signal DataFrame
for i, tree in enumerate(signal.values()):
    if i == 0:
        pre_sig_df = tree.pandas.df(feats+pre_feats)
        pre_sig_df['res_mass'] = res_mass[i]
    else:
        add_df = tree.pandas.df(feats+pre_feats)
        add_df['res_mass'] = res_mass[i]
        pre_sig_df = pre_sig_df.append(add_df, ignore_index=True)
print('shape of signal df before preselection cuts: {}'.format(pre_sig_df.shape))
        
# create background DataFrame
for i, tree in enumerate(background.values()):
    if i == 0:
        pre_bkg_df = tree.pandas.df(feats+pre_feats)
        pre_bkg_df['res_mass'] = np.nan
    else:
        add_df = tree.pandas.df(feats+pre_feats)
        add_df['res_mass'] = np.nan
        pre_bkg_df = pre_bkg_df.append(add_df, ignore_index=True)
print('shape of background df before preselection cuts: {}'.format(pre_bkg_df.shape))
```

Apply preselection cuts and drop preselection features that are not used in the training of the PNN.

```bash
sig_df = pd.DataFrame(pre_sig_df.loc[(pre_sig_df.ngood_bjets > 0) & (pre_sig_df.lep_category == 2) & (pre_sig_df.Zlep_cand_mass > 15) &
                                     (pre_sig_df.leading_lep_pt > 20) & (pre_sig_df.trailing_lep_pt > 10) & (pre_sig_df.leading_Hbb_pt > 20) &
                                     (pre_sig_df.trailing_Hbb_pt > 20) & (pre_sig_df.leading_jet_pt > 20) & (pre_sig_df.trailing_jet_pt > 20)])
bkg_df = pd.DataFrame(pre_bkg_df.loc[(pre_bkg_df.ngood_bjets > 0) & (pre_bkg_df.lep_category == 2) & (pre_bkg_df.Zlep_cand_mass > 15) & 
                                     (pre_bkg_df.leading_lep_pt > 20) & (pre_bkg_df.trailing_lep_pt > 10) & (pre_bkg_df.leading_Hbb_pt > 20) &
                                     (pre_bkg_df.trailing_Hbb_pt > 20) & (pre_bkg_df.leading_jet_pt > 20) & (pre_bkg_df.trailing_jet_pt > 20)])

# drop preselection features that are not in our desired features
sig_df.drop(columns=[f for f in pre_feats], inplace=True)
bkg_df.drop(columns=[f for f in pre_feats], inplace=True)
```

Add signal and background flags for training and validation (1 and 0 respectively).

```bash
sig_df['Label'] = 1
bkg_df['Label'] = 0
```

Create two final DataFrames, one using even event id and one using odd event id. These two DataFrames will be used to build two ensembles of neural networks which will then be validated via inference on the opposite half of the events. This will allow for inference using the full dataset rather than using only a portion for training and the remaining portion for validation. Weights were also set to 1.

```bash
df_0 = sig_df[::2].append(bkg_df[::2], ignore_index=True)
df_1 = sig_df[1::2].append(bkg_df[1::2], ignore_index=True)
df_0.rename(index=str, columns={'Label':'gen_target', 'weight_central':'gen_weight'}, inplace=True)
df_1.rename(index=str, columns={'Label':'gen_target', 'weight_central':'gen_weight'}, inplace=True)

# add weights=1 for now
df_0['gen_weight'] = 1
df_1['gen_weight'] = 1
```

Resonant masses were then randomly smapled from our training masses and applied to all background events in both DataFrames.

```bash
add_bkg_res_mass(df_0)
add_bkg_res_mass(df_1)
```

Input transformation pipelines were then created for both DataFrames, and each DataFrame was copied (for validation) and transformed using the corresponding ensemble that they are associated to. Set_0_train and set_1_train will be used to train ensemble 0 and 1 respectively, so they are transformed via their respective transformation pipelines. To allow for full inference, each ensemble will validate on the opposite half of the events. Therefore, set_0_test and set_1_test must be transformed using the pipeline that corresponds to the opposite half of the data (input_pipe_1 and input_pipe0 respectively).

```bash
input_pipe_0 = fit_input_pipe(df_0, cont_feats, PATH/f'input_pipe_0')
input_pipe_1 = fit_input_pipe(df_1, cont_feats, PATH/f'input_pipe_1')

set_0_train = df_0.copy()
set_1_train = df_1.copy()
set_0_test  = df_0.copy()
set_1_test  = df_1.copy()

set_0_train[cont_feats] = input_pipe_0.transform(set_0_train[cont_feats].values.astype('float32'))
set_1_train[cont_feats] = input_pipe_1.transform(set_1_train[cont_feats].values.astype('float32'))

set_0_test[cont_feats] = input_pipe_1.transform(set_0_test[cont_feats].values.astype('float32'))
set_1_test[cont_feats] = input_pipe_0.transform(set_1_test[cont_feats].values.astype('float32'))
```

All weights are then normalized for training.

```bash
new_balance_weights_resonant(set_0_train)
new_balance_weights_resonant(set_1_train)
```

Finally, all test and training sets were saved as HDF5 files in the file location saved previously as PATH.

```bash
df2foldfile(df=set_0_train, n_folds=10,
            cont_feats=cont_feats, cat_feats=cat_feats, targ_feats='gen_target', wgt_feat='gen_weight',
            misc_feats=['gen_orig_weight', 'gen_strat_key', 'res_mass_orig'],
            savename=PATH/'train_0', targ_type='int', strat_key='gen_strat_key')

df2foldfile(df=set_1_train, n_folds=10,
            cont_feats=cont_feats, cat_feats=cat_feats, targ_feats='gen_target', wgt_feat='gen_weight',
            misc_feats=['gen_orig_weight', 'gen_strat_key', 'res_mass_orig'],
            savename=PATH/'train_1', targ_type='int', strat_key='gen_strat_key')
            
df2foldfile(df=set_0_test, n_folds=10,
            cont_feats=cont_feats, cat_feats=cat_feats, targ_feats='gen_target', wgt_feat='gen_weight',
            misc_feats=['gen_strat_key', 'res_mass_orig'],
            savename=PATH/'test_0', targ_type='int', strat_key='gen_strat_key')
            
df2foldfile(df=set_1_test, n_folds=10,
            cont_feats=cont_feats, cat_feats=cat_feats, targ_feats='gen_target', wgt_feat='gen_weight',
            misc_feats=['gen_strat_key', 'res_mass_orig'],
            savename=PATH/'test_1', targ_type='int', strat_key='gen_strat_key')
```
