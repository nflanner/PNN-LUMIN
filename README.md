# Parameterized Neural Network using LUMIN

LUMIN is a deep learning and data-analysis ecosystem for High Energy Particle Physics written by Giles Strong, see https://github.com/GilesStrong/lumin.

Utilizing the PNN in LUMIN requires three distinct steps, importing, model training, and validation/testing, which I will walk through using the various notebooks within this repository. Additionally, I will show two methods of predicting on events that were not used for training - one using the same preprocessing methods shown in the Importing section, and one using a transformed Numpy array. Not every cell will be mentioned in this walk-through, please see the notes in each notebook for more detail on every cell.

## Importing

notebook:
https://github.com/nflanner/PNN-LUMIN/blob/main/res_import_filtered_Rev4.ipynb

All events need to be uploaded to a pandas DataFrame, preprocessed for training, and saved as HDF5 files to train the PNN.

A list of strings was created associated to all the resonant mass ROOT files used for training (I left out 900 GeV to allow for testing on a mass that was not used in the training).

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

Input transformation pipelines were then created for both DataFrames (saved as PKL files), and each DataFrame was copied (for validation) and transformed using the corresponding ensemble that they are associated to. Set_0_train and set_1_train will be used to train ensemble 0 and 1 respectively, so they are transformed via their respective transformation pipelines. To allow for full inference, each ensemble will validate on the opposite half of the events. Therefore, set_0_test and set_1_test must be transformed using the pipeline that corresponds to the opposite half of the data (input_pipe_1 and input_pipe0 respectively).

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



## Training

notebooks: 
https://github.com/nflanner/PNN-LUMIN/blob/main/selected_set_0_Rev4.ipynb
https://github.com/nflanner/PNN-LUMIN/blob/main/selected_set_1_Rev4.ipynb

After all training and test sets are saved, two ensembles of neural networks were trained using each half of the full set of events.

Create a folder yielder object using the training set and associated input transformation pipeline.

```bash
train_fy = FoldYielder(PATH/'train_0.hdf5', input_pipe=PATH/'input_pipe_0.pkl')
```

The model can then be built by specifiying all desired hyperparameters and architecure. In this case, I left batch size as given in the example from Giles Strong and the objective was kept as classification. Cat_embedder is associated to categorical events such as channel, year, etc. and, therefore, was not utilized in this example since only one year and channel were used. The ideal body architecture was previosuly determined to be a dense fully-connected network, with a swish activation function, width of 25, depth of 3, and 0 dropout. Finally, the optimizer was kept to be Adam with an epsilon of 1e-8.

```bash
bs = 1024
objective = 'classification'
cat_embedder = CatEmbedder.from_fy(train_fy)

body = partial(FullyConnected, act='swish',width=25,depth=3,dense=True, do=0)
opt_args = {'opt':'adam', 'eps':1e-08}

n_out = 1
model_builder = ModelBuilder(objective, cont_feats=train_fy.cont_feats, n_out=n_out, cat_embedder=cat_embedder,
                             body=body, opt_args=opt_args)
Model(model_builder)
```

To determine the range of learning rate to be used for training, we take the bounds of learning rate to be such that the curve of loss versus learning rate is always negative. The lr_finder method will produce a plot of loss versus learning rate (see below) which was then used to manually determine the bounds of the training learning rate range. For exmaple, the range selected from the plot shown below was chosen to be [1e-3, 1e-2].

```bash
lr_finder = fold_lr_find(train_fy, model_builder, bs, lr_bounds=[1e-7,1e1])
```

![image](https://user-images.githubusercontent.com/76540759/146947985-56156ab5-ca3f-4fab-b53c-7329cfce9729.png)

Number of models, patience, and number of epochs were all left at the default values from the examples given by Giles Strong. callback partials and metric partials were also kept the same as given, but lr_range was taken from the previously mentioned lr_finder associated with our events. Train_models then trains an ensemble of 10 models using the specified hyperparameters and architecture, and plots each models' loss versus sub-epoch. The best performing model is then kept and results are saved as a PKL file in the train_weights file folder.

```bash
n_models = 10
patience = 5
n_epochs = 15

cb_partials = [partial(OneCycle, lengths=(45, 90), lr_range=[1e-3, 1e-2], mom_range=(0.85, 0.95), interp='cosine')]
metric_partials = [partial(AMS, n_total=250000, br=10, wgt_name='gen_orig_weight', main_metric=False)]

from lumin.nn.training.train import train_models

results, histories, cycle_losses = train_models(train_fy,  # Training data
                                                n_models,  # Number of models to train
                                                model_builder=model_builder,  # How to build models, losses, and optimisers
                                                bs=bs,  # Batch size
                                                cb_partials=cb_partials,  # List of uninitialised callbacks
                                                metric_partials=metric_partials,  # Pass our evaluation metrics
                                                n_epochs=n_epochs,  # Maximum number of epochs to train
                                                patience=patience)
```

The ensemble is then loaded and the associated transformation pipeline is saved along with the ensemble in the weights file folder.

```bash
with open('train_weights/results_file.pkl', 'rb') as fin:   
    results = pickle.load(fin)
    
ensemble = Ensemble.from_results(results, n_models, model_builder, metric='loss')
ensemble.add_input_pipe(train_fy.input_pipe)

name = f'weights/selected_set_0_{run_name}'

ensemble.save(name, feats=train_fy.cont_feats+train_fy.cat_feats, overwrite=True)
```

The partial dependence plots the trained models' response at any given mass point by keeping all input features constant and only varying input resonant mass. Therefore, for any given mass point model, we expect a low flat response for background and a peak at the given mass for signal events. This was checked for both background and signal (see plots below).

```bash
plot_1d_partial_dependence(m, train_df[train_df.gen_target==0], 'res_mass', train_feats=train_feats, y_lim=[0,1],
                           input_pipe=train_fy.input_pipe, wgt_name='gen_weight', sample_sz=int(len(train_df[train_df.gen_target==0])/2),
                           pdp_isolate_kargs={'cust_grid_points':masses}, n_clusters=5)
                           
for mass in sorted(train_df.res_mass.unique()):
    df = train_df[(train_df.gen_target==1)&(train_df.res_mass==mass)]
    print(f'Mass {lookup_mass(mass)}, N events {len(df)}, weight_sum {df.gen_weight.sum()}')
    plot_1d_partial_dependence(m, df, 'res_mass', train_feats=train_feats, n_clusters=5, y_lim=[0,1],
                               input_pipe=train_fy.input_pipe, wgt_name='gen_weight',
                               pdp_isolate_kargs={'cust_grid_points':masses})
```

Background

![image](https://user-images.githubusercontent.com/76540759/146951088-641e38c3-7ff2-4a76-9fe8-f6a6843fc87d.png)

Signal (400 GeV)

![image](https://user-images.githubusercontent.com/76540759/146951151-396b1272-bf18-47ef-b97e-25a81015a388.png)

The same procedure was then repeated using train_0.hdf5 and input_pipe_0.pkl.



## Validation

notebook:
https://github.com/nflanner/PNN-LUMIN/blob/main/application_all_mass_Rev4.ipynb

Finally, both trained ensembles are loaded into an Ensemble object, and both test sets are loaded as folder yielder objects.

```bash
ensemble_0 = Ensemble.from_save(f'weights/selected_set_0_{run_name}')
ensemble_1 = Ensemble.from_save(f'weights/selected_set_1_{run_name}')

set_0_fy = FoldYielder(PATH/'test_0.hdf5', input_pipe=PATH/'input_pipe_0.pkl')
set_1_fy = FoldYielder(PATH/'test_1.hdf5', input_pipe=PATH/'input_pipe_1.pkl')
```

For all masses, we then took a tuple of the ensemble, associated test set (opposite half), and transformation pipe, and madepredictions using the predict() method. To tell LUMIN that this is indeed a parameterized netowkr, we use the ParameterizedPrediction callback function which specifies the training features (train_feats), feature(s) to be parameterized (['res_mass']), and the transformed prediction point. For example, to predict on 400 GeV, the final kwarg in ParameterisedPrediction would be [preprocess_mass(400,pipe)] where the method preprocess_mass() transforms the 400 mass via the transformation pipeline, pipe. The predict() method then saves the predictions in the original folder yielder given as a kwarg. Note that once the cell below is has been run and the folder yielder has prediction scores, it cannot be run again unless a new folder yielder is created via the importing/preprocessing procedure.

```bash
mb = master_bar(masses)
for m in mb:
    for e, fy_s, pipe in ((ensemble_0, set_1_fy, set_0_fy.input_pipe),
                            (ensemble_1, set_0_fy, set_1_fy.input_pipe)):
        mass_param = ParametrisedPrediction(train_feats, ['res_mass'], [preprocess_mass(m,pipe)])
        e.predict(fy_s,cbs=[mass_param],verbose=False,
                    pred_name=f'pred_mass_{m}')
```

Predictions from each half of the event dataset were then appended into one final DataFrame that includes all class predictions.

```bash
df = load_df(set_0_fy, 800).append(load_df(set_1_fy, 800), ignore_index=True); df.head()
```

All masses were then looped through, histograms were plotted to visualize class prediction for all events, and ROC curves produced. We expected that all signal events should be pushed towards a class prediction of 1 while all background events should be pushed towards a class prediction of 0. The figure below shows one such mass point class prediction and ROC curve and boasts a clear separation of signal and background. The AUC curve shows true versus false positive rate while varying the threshold of what is considered to be signal or baackground from 1 (no signal) to 0 (everything is considered signal). Therefore we want the area under the curve of the ROC plot to be close to 1.

```bash
roc_scores = []
for mass in masses:
    df = load_df(set_0_fy, mass).append(load_df(set_1_fy, mass), ignore_index=True)
    print('mass: ', mass)
    make_cp(df)
    fpr, tpr, threshold = metrics.roc_curve(df.gen_target, df.pred)
    roc_auc = metrics.auc(fpr, tpr)
    roc_scores.append(roc_auc)
    make_roc(fpr, tpr, roc_auc)
```

Class prediction ( 400 GeV)

![image](https://user-images.githubusercontent.com/76540759/146955338-a6bc5377-fe46-4cbc-b3d2-cf4dd0a4383b.png)

ROC Curve (400 GeV)

![image](https://user-images.githubusercontent.com/76540759/146955391-c7d5ef68-6979-4c77-b086-e328895f56e5.png)

Finally, the PNN's AUC scores were compared at all mass points to that of a singley-trained DNN (previously recorded), plot seen below.

```bash
mass = [260, 270, 300, 350, 400, 450, 500, 550, 600, 650, 750, 800, 1000]
SNN_AUC = [0.93, 0.89, 0.88, 0.90, 0.91, 0.92, 0.93, 0.95, 0.96, 0.97, 0.97, 0.98, 0.98]

single, = plt.plot(mass, SNN_AUC, 'b', Label='Single DNN')
parameterized, = plt.plot(masses, roc_scores, 'r', Label='Parameterized DNN')

plt.legend(handles=[single, parameterized])
plt.xlabel('Resonant Mass')
plt.ylabel('AUC Score')
plt.axis([250, 1010, 0.85, 1])
```

PNN AUC score versus DNN

![image](https://user-images.githubusercontent.com/76540759/146955838-021106fb-0aba-47d8-8c5a-c36a5e6580c4.png)



## Predictions on mass points not used in training (HDF5 method)

notebook:
https://github.com/nflanner/PNN-LUMIN/blob/main/res_import_filtered_untrained_Rev4.ipynb
https://github.com/nflanner/PNN-LUMIN/blob/main/Import_Predict_FY.ipynb

Predictions on mass points that were not used for training can be done in one of two ways: following the importing procedure with the unused mass point and making predictions on the imported folder yielder, or using a transformed Numpy array; this section focuses on the former.

The res_import_filtered_untrained_Rev4.ipynb notebook follows the exact same steps as the previous import notebook but instead uses 900 GeV (which was not used during training) and saves the HDF5 files with a different name. 

The Import_Predict_FY.ipynb notebook follows the exact same steps as the previous prediction notebook but only uses 900 GeV (...) and, again, produces a class prediction histogram and ROC plot at 900 GeV (see figures below).

Class Prediction (900 GeV - not used in training)

![image](https://user-images.githubusercontent.com/76540759/146959035-b4c2c257-5642-4f47-b8b4-7051321678e2.png)

ROC plot (900 GeV - not used in training)

![image](https://user-images.githubusercontent.com/76540759/146959096-eae77418-5529-4d5e-b843-4a2d75047458.png)

Observing the plots above, the clear separation of signal and background coupled with the high AUC score shows that the PNN indeed discriminates between signal and background even for mass points not used in training.



## Predictions on mass points not used in training (Numpy method)

notebook:
https://github.com/nflanner/PNN-LUMIN/blob/main/Import_Predict_npArray.ipynb

The second way to predict on events not used in training is via a Numpy array. Similar to the importing method, all signal events for 900 GeV (not used in training) were imported into a pandas DataFrame. Note that no background events were included, only signal. Once the DataFrame was finished, we then converted it into two Numpy arrays based off even or odd event id (to allow for full inference) and the arrays were transformed using the transofrmation pipelines previously mentioned.

```bash
np_arr_0 = sig_df[::2].to_numpy()
np_arr_1 = sig_df[1::2].to_numpy()

with open(PATH/"input_pipe_0_untrained.pkl", 'rb') as pickle_file:
    input_pipe_0 = pickle.load(pickle_file)

with open(PATH/"input_pipe_1_untrained.pkl", 'rb') as pickle_file:
    input_pipe_1 = pickle.load(pickle_file)
    
transformed_input_0 = np.hstack((input_pipe_0.transform(np_arr_0[:,:len(cont_feats)]), np_arr_0[:,len(cont_feats):]))
transformed_input_1 = np.hstack((input_pipe_1.transform(np_arr_1[:,:len(cont_feats)]), np_arr_1[:,len(cont_feats):]))
```

Predictions were then made and saved into separate arrays using the ParameterisedPrediction() and predict() methods, and the two arrays were finally concatenated into ne final prediction array.

```bash
mass_param_0 = ParametrisedPrediction(train_feats, ['res_mass'], [preprocess_mass(900,ensemble_0.input_pipe)])
mass_param_1 = ParametrisedPrediction(train_feats, ['res_mass'], [preprocess_mass(900,ensemble_1.input_pipe)])

pred_0 = ensemble_0.predict(transformed_input_1, cbs=[mass_param_0])
pred_1 = ensemble_1.predict(transformed_input_0, cbs=[mass_param_1])
pred = np.concatenate((pred_0, pred_1), axis=0)
```

The prediction array was then visualized via a Numpy histogram as seen below.

```bash
bins = np.arange(0,1.1,0.05)

hist,bins = np.histogram(pred,bins=bins)  
plt.hist(pred, bins = bins, density=1) 
plt.show()
```

Numpy array predictions on 900 GeV (not used for training) signal events
![image](https://user-images.githubusercontent.com/76540759/146960750-9633dd6d-8a2e-4673-9b86-d4c491b32d23.png)

Again, the class predictions come out as expected. Since all the events used were signal, we indeed expect all events to be pushed towards a class prediction of 1. Therefore, if the transformation pipeline is saved, then we can eliminate the preprocessing and conversion to HDF5 files steps by instead making predictions directly on a transformed a Numpy array.
