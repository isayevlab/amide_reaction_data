# Amide coupling reaction dataset

All public information can be downloaded at [here](https://drive.google.com/drive/folders/1IIUVKZJahufrSspAz5_nDCH7D29m7b1J?usp=share_link). List of files:
- sdf.tar.gz contains the optimized 3D structures of the molecules involved in the reactions;
- v_desps.csv contains the steric descriptors of reaction centers;
- AEV00001.tar.gz contains the AEV descriptors for each molecule;
- qm_descriptors.tar.gz contains the QM descriptors for each molecule;
- modred_clean.csv contains the Mordred descriptor for each molecule;
- morgan1024.csv contains the Morgan descriptor for each molecule;
- word_idx.txt contains the mapping from the reaction context to the one-hot encoding index;


**Besides the above molecule files, you also need to have access to reaction files to use the following code. The reaction files are under Reaxys patent.**
**Our local cluster server is down for accessing the data. Maintainance in progress...**

`data.py` is a script to load the dataset with different descriptors. The dataset can be accessed using the `rxn` class in `data.py`. Here is the interface for the `rxn` class:
```
class rxn(object):
    def __init__(self, folder: str):
        """
        A class for getting reaction information.
        folder: a directory containing all relevant files
        """
        self.folder = folder
    
    def all_idx2(self):
        """Return the list of reaction indexes"""

    def get_rxn(self, idx):
        """Given a reaction idx, return all reaction info as a tuple."""

    def get_rxn_3d(self, idx):
        """Given a reaction idx, return the keys for 3D structures of acid, amine, product"""

    def get_rxn_centers(self, idx):
        """Given a reaction idx, return reaction centers"""

    def get_context_one_hot(self, idx):
        """Given a reaction idx, return a one-hot embedding of the context"""

    def get_aev(self, idx, aggregation="sum"):
        """"Given a reaction idx, return the AEV embedding of the reaction"""

    def get_fp(self, idx):
        """Given a reaction idx, return reaction fingerprint"""

    def get_mordred(self, idx):
        """Given a reaction idx, return reaction Mordred descriptor"""

    def get_qm_descriptors(self, idx):
        """Given a reaction idx, return QM descriptor"""

    def get_steric_desps(self, idx):
        """Given a reaction idx, return the standardized steric descriptor"""

```

Example usage:
```
folder = path_to_your_folder_with_all_reaction_files
rxns = rxn(folder)

# get all reaction id
ids = rxns.all_idx2()

# get reaction information
# return a list containing the following information: Reaxys_ID, link, acid, amine, product, O-acylisourea, time, Temperature, condition, reagent, solvent, catalyst, yield
info = rxns.get_rxn(ids[0])

# get QM descriptors
des = rxns.get_qm_descriptors(ids[0])
```
