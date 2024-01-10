# Amide coupling reaction dataset

- Publicly available information: optimized molecule 3D structures in SDF format, molecular descriptors and reaction descriptors;
- Available only when you have access to [Reaxys Database](https://www.reaxys.com/): the reaction CSV file (Reactioin information and yields). **Your organization is very likely to have already subscribed to Reaxys.**


All public information can be downloaded [here](https://drive.google.com/drive/folders/1IIUVKZJahufrSspAz5_nDCH7D29m7b1J?usp=share_link). List of files:
```
.
|____molecules
| |____sdf.tar.gz  # optimized 3D structures of molecules (need to be unzipped)
| |____morgan1024.csv  # Morgan descriptors of molecules
| |____mordred_clean.csv  # Mordred descriptors of molecules
| |____AEV00001.zip  # AEV descriptors of molecules (need to be unzipped)
| |____qm_descriptors.tar.gz  # QM descriptors of molecules (need to be unzipped)
| |____sdf_es.pkl  # a dictionary mapping from molecule inchikey to the electronic energy of the optimized 3D structure
| |____word_idx.txt  # mapping from reaction context to one-hot encoding index
| |____v_desps.csv  # steric descriptors of reactions
|____reactions_example.csv  # ID and links to the reactions from Reaxys
|____reaction_fp_desps.csv  # fingerprint descriptors of reactions
|____reaction_mordred_desps.csv  # Mordred descriptors of reactions 
|____reaction_aev_desps.csv # AEV descriptors of reactions
|____reaction_qm_desps.csv  # QM descriptors of reactions
|____qm_train_test_splits  # train/test splits for QM descriptors
|____data.py  # an interface for loading the dataset
```

## data splits
The whole dataset is split into 5 mutually exclusive parts:
- `normal_ids.pkl`: 31,622 amide coupling reactions. The yield may or may not contain outliers. This should be the main dataset for training. It mimics the real-world scenario where the yield is not always reliable.
- `train_uncertain_ids.pkl`: 2,292 amide coupling reactions. The yield for each reaction are known to be uncertain. This dataset may be used to validate the model performance on identifying uncertain reactions.
- `test_uncertain_ids.pkl`: 3,000 amide coupling reactions. The yield for each reaction are known to be uncertain. This dataset may be used to test the model performance on identifying uncertain reactions.
- `test_clean_ids.pkl`: 187 amide coupling reactions. The yield for each reaction are known to be reliable. This dataset may be used to test the model performance on predicting the yield of reliable reactions.

## dataset interface
If you have access to Reaxys, you can easily get full information as in our paper. **The reaction IDs and links are available in the [`reactions_example.csv` file](https://drive.google.com/file/d/1ka5l256TAc4p-FhPh1ZMnNva8VsrCJeY/view?usp=drive_link)**. After filling the columns of the reaction CSV file, you will be abale to use all the functions in `data.py`. If you don't have access to Reaxys, **You can still easily load our reaction descriptors**.
Here are the information you can get via Reaxys database:
```
    # columns of reactions_example.csv:
    ID_2: our unique ID for each reaction
    ID (Reaxys ID, reactions conducted at different conditions may have the same ID)
    Links
    Reaction (Reaction SMILES)
    Time
    Temperature
    pH
    Conditions
    Reaction Type
    Yield
    Reagent
    Catalyst
    Solvent
    acids (SMILES)
    amines (SMILES)
    products (SMILES)
    acylisoureas (SMILES)
    condition_bd: a list of 5 numbers. Each number represents a condition/reagent/solvent/catalysts
    reagent_bd: a list of 5 numbers. Each number represents a condition/reagent/solvent/catalysts
    solvent_bd
    catalyst_bd
    acid_key (InchiKey)
    amine_key (InchiKey)
    product_key (InchiKey)
    acylisourea_key (InchiKey)
    acid_centers2  (reaction center index)
    amine_centers2 (reaction center index)
    p_centers2 (reaction center indexes)
```


`data.py` is a script to load the dataset with different descriptors. The dataset can be accessed using the `rxn` class in `data.py`. Here is the interface for the `rxn` class:
```
class rxn(object):
    def __init__(self, folder: str):
        """
        A class for getting reaction information.
        folder: a directory containing all relevant files
        """
    
    def all_idx2(self):
        """Return the list of reaction indexes as we used in the paper.
        These IDs are named as ID_2 in the reaction dataframe."""
    
    def id2id2(self, id):
        """transforming Reaxys ID into ID_2
        id: Reaxys ID
        return: a list of ID_2"""

    def get_fp_fast(self, idx) -> List[int]:
        """"Given a reaction ID_2, return the fingerprint embedding of the reaction"""
    
    def get_mordred_fast(self, idx) -> List[float]:
        """"Given a reaction ID_2, return the Mordred embedding of the reaction"""
    
    def get_aev_fast(self, idx) -> List[float]:
        """"Given a reaction ID_2, return an AEV embedding of the reaction"""

    def get_qm_fast(self, idx) -> List[float]:
        """Given a reaction ID_2, return a list of QM descriptors for the reaction"""
    
    def get_steric_desps(self, idx) -> List[float]:
        """Given a reaction ID_2, return the steric descriptors of the reaction"""
```

Example usage:
```
folder = path_to_your_folder_with_all_reaction_files
rxns = rxn(folder)

# get all reaction id
ids = rxns.all_idx2()

# get the descriptors for the 1st reaction
id = ids[0]
fp = rxns.get_fp_fast(id)
mordred = rxns.get_mordred_fast(id)
aev = rxns.get_aev_fast(id)
qm = rxns.get_qm_fast(id)
steric = rxns.get_steric_desps(id)
```
