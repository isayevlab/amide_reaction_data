import os
import pickle
import json, ujson
import pandas as pd
import numpy as np
from rdkit import Chem
from typing import List


class rxn(object):
    """A class for get reaction information.

    Below are the columns of the reaction dataframe:
    ID (contains duplicates i.e. reactions conducted at different temperatures)
    Links
    Reaction
    Time: continuous value
    Temperature: continuous value
    pH: continuous value
    Conditions
    Reaction Type
    Yield
    Reagent
    Catalyst
    Solvent
    acids
    amines
    products
    acylisoureas
    condition_bd: a list of 5 numbers. Each number represents a condition/reagent/solvent/catalysts
    reagent_bd: a list of 5 numbers. Each number represents a condition/reagent/solvent/catalysts
    solvent_bd
    catalyst_bd
    ID_2: the unique ID for each reactions
    Label
    acid_key
    amine_key
    product_key
    acylisourea_key
    Yield_std
    acid_centers2
    amine_centers2
    p_centers2
    """
    def __init__(self, folder):
        self.folder = folder
        self.mols_folder = os.path.join(folder, "molecules")
        self.df = pd.read_csv(os.path.join(folder, "reactions6_6.csv"), index_col="ID_2", low_memory=False)
        context_path = os.path.join(folder, "molecules", "word_idx.txt")
        self.context_bd = json.load(open(context_path, "r"))
        self.context_dim = len(self.context_bd)
        # self.h5 = h5py.File(os.path.join(folder, "molecules", "AEV00001.h5"), "r")
        self.h5 = os.path.join(folder, "molecules", "AEV00001")
        self.aimnet = os.path.join(folder, "molecules", "aimnet_descriptors")
        self.fp = pd.read_csv(os.path.join(folder, "molecules", "morgan1024.csv"), index_col="ID", low_memory=False)  #ID is molecular InChiKey
        self.mordred0 = pd.read_csv(os.path.join(folder, "molecules", "mordred_clean.csv"), index_col="ID")
        mordred_scaler = pickle.load(open(os.path.join(folder, "molecules", "mordred_MaxAbsScaler.pkl"), "rb"))
        self.mordred = pd.DataFrame(mordred_scaler.transform(self.mordred0))
        self.mordred["ID"] = self.mordred0.index
        self.mordred.set_index("ID", inplace=True)
        self.sdf_es = pickle.load(open(os.path.join(folder, "molecules", "sdf_es.pkl"), "rb"))
        self.aev_df = pd.read_csv(os.path.join(folder, "reaction_aev_desps.csv"), index_col=0)
        #stetric descriptors
        self.steric_df = pd.read_csv(os.path.join(folder, 'molecules', 'v_desps.csv'), index_col='ids')
        steric_stats = self.steric_df.describe()
        self.steric_mean = steric_stats.loc['mean', :].values  # np array shape(24, )
        self.steric_std = steric_stats.loc['std', :].values  #np array shape(24, )

        aimnet_dct = {0: "E rxn", 1: "Ea rxn", 2: "Fukui acid", 3: "Fukui amine", 4: "Fukui rxn",
        5: "IP acid", 6: "IP amine", 7: "IP product", 8: "IP reactants", 9: "IP: rxn",
        10: "EA acid", 11: "EA amine", 12: "EA product", 13: "EA reactnats", 14: "EA rxn",
        15: "EN acid", 16: "EN amine", 17: "EN product", 18: "EN reactnats", 19: "EN rxn",
        20: "Hardness acid", 21: "Hardness amine", 22: "Hardness product", 23: "Hardness reactnats", 24: "Hardness rxn",
        25: "EP acid", 26: "EP amine", 27: "EP product", 28: "EP reactnats", 29: "EP rxn",
        30: "Q_C", 31: "Q_N", 32: "Q_pC", 33: "Q_pN", 34: "Q_pCC", 35: "Q_pNN", 36: "Q_CN",
        37: "Fukui1", 38: "Fukui2", 39: "Fukui3", 40: "Fukui4", 41: "Fukui5", 42: "Fukui6", 43: "Fukui7",
        44: "EP1", 45: "EP2", 46: "EP3", 47: "EP4", 48: "EP5", 49: "EP6", 50: "EP7"}
        # self.aimnet_descriptor_names = [val for _, val in aimnet_dct.items()]
        aimnet_descriptor_df0 = pd.read_csv(os.path.join(folder, "reaction_aimnet_desps.csv"), index_col=0)
        qm_scaler = pickle.load(open(os.path.join(folder, "qm_MaxAbsScaler.pkl"), "rb"))
        self.aimnet_descriptor_df = pd.DataFrame(qm_scaler.transform(aimnet_descriptor_df0))
        self.aimnet_descriptor_df["ID"] = aimnet_descriptor_df0.index
        self.aimnet_descriptor_df.set_index("ID", inplace=True)

    def all_idx2(self):
        return self.df.index

    def get_yield(self, idx):
        """Given reaction ID2, return the yield"""
        return float(self.df.loc[idx, "Yield"])
    

    def get_amine_type(self, idx):
        """Given reaction ID2, return the amine"""
        return self.df.loc[idx, "amine_type"]


    def get_rxn(self, idx):
        """Given a reaction idx (NOT REAXYS ID, but ID_2 in the reaction dataset), return all rxn info in the reaction.csv file
        idx: ID_2 in the reaction dataframe
        
        Return a tuple of 13 elements"""
        id = self.df.loc[idx, 'ID']  #Reaxys ID
        link = self.df.loc[idx, 'Links']

        smi1 = self.df.loc[idx, 'acids']
        smi2 = self.df.loc[idx, 'amines']
        smi3 = self.df.loc[idx, 'products']
        smi4 = self.df.loc[idx, 'acylisoureas'] 
        # context
        t = self.df.loc[idx, 'Time']
        T = self.df.loc[idx, 'Temperature']
        con = self.df.loc[idx, 'Conditions']
        rea = self.df.loc[idx, 'Reagent']
        sol = self.df.loc[idx, 'Solvent']
        cat = self.df.loc[idx, 'Catalyst']
        
        yld = self.df.loc[idx, 'Yield']      
        return [id, link, smi1, smi2, smi3, smi4, t, T, con, rea, sol, cat, yld]

    def get_rxn_size(self, idx):
        """Given a reaction ID2, return the number of atoms (H not included) in acid, amine, product"""
        id, link, smi1, smi2, smi3, smi4, t, T, con, rea, sol, cat, yld = self.get_rxn(idx)
        size1 = Chem.MolFromSmiles(smi1).GetNumAtoms()
        size2 = Chem.MolFromSmiles(smi2).GetNumAtoms()
        size3 = Chem.MolFromSmiles(smi3).GetNumAtoms()
        return [size1, size2, size3]

    def get_rxn_elements(self, idx):
        """Given a reaction ID2, return the elements (H not included) in acid, amine, product"""
        id, link, smi1, smi2, smi3, smi4, t, T, con, rea, sol, cat, yld = self.get_rxn(idx)
        atoms1 = [a.GetSymbol() for a in Chem.MolFromSmiles(smi1).GetAtoms()]
        atoms2 = [a.GetSymbol() for a in Chem.MolFromSmiles(smi2).GetAtoms()]
        atoms3 = [a.GetSymbol() for a in Chem.MolFromSmiles(smi3).GetAtoms()]
        atoms = atoms1 + atoms2 + atoms3
        return set(atoms)


    def get_rxn_3d(self, idx):
        """ID2"""
        acid_key = self.df.loc[idx, "acid_key"]
        amine_key = self.df.loc[idx, "amine_key"]
        p_key = self.df.loc[idx, "product_key"]
        return (acid_key, amine_key, p_key)

    
    def get_steric_desps(self, idx):
        """ID2"""
        desps = self.steric_df.loc[idx, :].values  #np array shape (24, )
        standardized_desps = np.nan_to_num((desps - self.steric_mean)/self.steric_std, nan=0)
        return list(standardized_desps)


    @staticmethod
    def get_rxn_centerr_helper(s):
        s2 = s[1:]
        s3 = s2[:-1]
        vals = []
        for val in s3.split(","):
            vals.append(int(val))
        return vals


    def get_rxn_centers(self, idx):
        """ID2, rxn centers using SDF index (starting from 1)"""
        acid_center = self.get_rxn_centerr_helper(self.df.loc[idx, "acid_centers2"])
        amine_center = self.get_rxn_centerr_helper(self.df.loc[idx, "amine_centers2"])
        p_center = self.get_rxn_centerr_helper(self.df.loc[idx, "p_centers2"])
        return (acid_center, amine_center, p_center)

    def id2id2(self, id):
        """transforming Reaxys id into ID2
        id: Reaxys ID
        
        return:
        ID2: a list of Reayxs ids"""
        df2 = self.df[self.df["ID"] == id]
        ID2 = list(df2.index)
        return ID2

    def string2vec(self, string):
        '''
        for string looks like [0, 0, 0, 0]
        '''
        l = len(string)
        string = string.strip()[1:l-1]
        ss = string.split(',')
        r = [int(val) for val in ss]
        return r


    def list2onehot(self, l):
        """
        given a list of integers, get the one-hot embedding
        """
        l = [val for val in l if val > 0]
        one_hots = np.zeros((len(l), self.context_dim))
        one_hots[np.arange(len(l)), np.array(l)-1] = 1
        one_hots = list(np.sum(one_hots, axis=0).astype(int))
        return one_hots


    def get_tT(self, idx):
        """given ID2, return the reactio time (h) and Temperature (C)
        idx: integer
        
        return
        an one-hot embedding as a list of integers
        
        time: mean=14.20, std=14.38
        Temperature: mean=20.93, std=11.13"""
        t = float(self.df.loc[idx, 'Time'])
        T = float(self.df.loc[idx, 'Temperature'])
 
        if t == t:
            time = t
        else:
            time = 14
        if T == T:
            temperature = T
        else:
            temperature = 20
        return (time, temperature)


    def get_context_one_hot(self, idx):
        """given ID2, return a one-hot embedding of the context
        idx: integer
        
        return
        an one-hot embedding as a list of integers"""
        con = self.string2vec(self.df.loc[idx, 'condition_bd'])
        rea = self.string2vec(self.df.loc[idx, 'reagent_bd'])
        sol = self.string2vec(self.df.loc[idx, 'solvent_bd'])
        cat = self.string2vec(self.df.loc[idx, 'catalyst_bd'])
        context = con + rea + sol + cat
        context_onehot = self.list2onehot(context)
        return context_onehot

    def get_aev(self, idx, aggregation="sum"):
        """"given ID2, return a AEV embedding of the reaction"""
        acid_key = self.df.loc[idx, "acid_key"]
        amine_key = self.df.loc[idx, "amine_key"]
        p_key = self.df.loc[idx, "product_key"]
        # with h5py.File(self.h5, "r") as f:

        # group = self.h5[acid_key]
        group = pickle.load(open(os.path.join(self.h5, f"{acid_key}.pkl"), "rb"))
        # species = group["species"][:]
        aevs = group["aevs"][:]
        if aggregation == "mean":
            descriptors1 = np.mean(aevs, axis=0)
        elif aggregation == "sum":
            descriptors1 = np.sum(aevs, axis=0)
        else:
            raise ValueError("aggregation method is invalid.")
        
        # group = self.h5[amine_key]
        # species = group["species"][:]
        group = pickle.load(open(os.path.join(self.h5, f"{amine_key}.pkl"), "rb"))
        aevs = group["aevs"][:]
        if aggregation == "mean":
            descriptors2 = np.mean(aevs, axis=0)
        elif aggregation == "sum":
            descriptors2 = np.sum(aevs, axis=0)
        else:
            raise ValueError("aggregation method is invalid.")

        # group = self.h5[p_key]
        # species = group["species"][:]
        group = pickle.load(open(os.path.join(self.h5, f"{p_key}.pkl"), "rb"))
        aevs = group["aevs"][:]
        if aggregation == "mean":
            descriptors3 = np.mean(aevs, axis=0)
        elif aggregation == "sum":
            descriptors3 = np.sum(aevs, axis=0)
        else:
            raise ValueError("aggregation method is invalid.")
        
        descriptors = list(descriptors1) + list(descriptors2) + list(descriptors3)
        return descriptors

    def get_aev_(self, idx):
        """"given ID2, return a AEV embedding of the reaction, using the dataset prepared by the get_aev method"""
        return list(self.aev_df.loc[idx, :])


    def get_fp(self, idx: int) -> List[int]:
        """given ID2, return rxn fingerprint"""
        acid_key = self.df.loc[idx, "acid_key"]
        amine_key = self.df.loc[idx, "amine_key"]
        p_key = self.df.loc[idx, "product_key"]


        acid_val = list(self.fp.loc[acid_key, :])
        amine_val = list(self.fp.loc[amine_key, :])
        p_val = list(self.fp.loc[p_key, :])

        fp = acid_val + amine_val + p_val
        return fp

    def get_mordred(self, idx: int) -> List[int]:
        """given ID2, return rxn fingerprint"""
        acid_key = self.df.loc[idx, "acid_key"]
        amine_key = self.df.loc[idx, "amine_key"]
        p_key = self.df.loc[idx, "product_key"]


        acid_val = list(self.mordred.loc[acid_key, :])
        amine_val = list(self.mordred.loc[amine_key, :])
        p_val = list(self.mordred.loc[p_key, :])

        mordred = acid_val + amine_val + p_val
        return mordred   

    @staticmethod
    def get_E(path):
        mol = next(Chem.SDMolSupplier(path))
        e = float(mol.GetProp("E_tot"))
        return e

    def get_qm_descriptors(self, idx) -> List[float]:
        """given ID2, return a list of QM descriptors for the reaction
        """
        acid_key, amine_key, p_key = self.get_rxn_3d(idx)
        acid_center, amine_center, p_centers = self.get_rxn_centers(idx)
        assert(len(acid_center) == 1)
        assert(len(amine_center) == 1)
        assert(len(p_centers) == 2)
        acid_center = acid_center[0]
        amine_center = amine_center[0]
        pc = sorted(p_centers)[0]  #asssuming C is first in SDF, N comes later
        pn = sorted(p_centers)[1]

        acid = ujson.load(open(os.path.join(self.aimnet, f"{acid_key}.json"), "r"))
        amine = ujson.load(open(os.path.join(self.aimnet, f"{amine_key}.json"), "r"))
        p = ujson.load(open(os.path.join(self.aimnet, f"{p_key}.json"), "r"))

        ## REACTION LEVEL (4 floats)
        # reaction energy (E_prod - E_amine - E_acid), float
        e_water = -76.47738479880026  #hartree
        e_acid = self.sdf_es[acid_key]  #10 tims faster than the above
        e_amine = self.sdf_es[amine_key]
        e_p = self.sdf_es[p_key]
        rxn_e2 = (e_p + e_water - e_acid - e_amine) * 627.5095
        
        #reaction activateion energy
        e_edc = -479.7027169296599
        e_dcc = -618.6981692019508
        e_dic = -384.99380721218193
        e_urea = self.sdf_es[idx]
        catalyst_type = self.df.loc[idx, "catalyst_type"]
        if catalyst_type == "EDC":
            e_cat = e_edc
        elif catalyst_type == "DCC":
            e_cat = e_dcc
        elif catalyst_type == "DIC":
            e_cat = e_dic
        else:
            raise ValueError(idx)
        rxn_ae = (e_urea - e_acid - e_cat) * 627.5095

        # acid (fukui_minus - fukui_0)
        acid_e = acid["f_el"]
        acid_n = acid["f_nuc"]
        rxn_acid_fukui = acid_e[acid_center-1] - acid_n[acid_center-1]

        # amine (fukui_0 - fukui_plus)
        amine_n = amine["f_nuc"]
        amine_rad = amine["f_rad"]
        rxn_amine_fukui = amine_n[amine_center-1] - amine_rad[amine_center-1]

        rxn_fukui = rxn_amine_fukui - rxn_acid_fukui

        ## MOLECULAR LEVEL
        # ionization potential, 5 numbers (acid, amine, product, amine - acid, p-acid-amine)
        acid_ip = acid["ip"]
        amine_ip = amine["ip"]
        p_ip = p["ip"]
        aa_ip = amine_ip - acid_ip
        rxn_ip = p_ip - acid_ip - amine_ip

        # electron affinity, 5 numbers (acid, amine, product, amine - acid, p-acid-amine)
        acid_ea = acid["ea"]
        amine_ea = amine["ea"]
        p_ea = p["ea"]
        aa_ea = amine_ea - acid_ea
        rxn_ea = p_ea - acid_ea - amine_ea

        # electronegativity, 5 numbers (acid, amine, product, amine - acid, p-acid-amine)
        acid_chi = acid["chi"]
        amine_chi = amine["chi"]
        p_chi = p["chi"]
        aa_chi = amine_chi - acid_chi
        rxn_chi = p_chi - acid_chi - amine_chi

        # hardness, 5 numbers (acid, amine, product, amine - acid, p-acid-amine)
        acid_eta = acid["eta"]
        amine_eta = amine["eta"]
        p_eta = p["eta"]
        aa_eta = amine_eta - acid_eta
        rxn_eta = p_eta - acid_eta - amine_eta

        # electrophilicity index, 5 numbers (acid, amine, product, amine - acid, p-acid-amine)
        acid_omega = acid["omega"]
        amine_omega = amine["omega"]
        p_omega = p["omega"]
        aa_omega = amine_omega - acid_omega
        rxn_omega = p_omega - acid_omega - amine_omega

        ## ATOMIC LEVEL (C in acid, N in the amine, C&N in the amide)
        # charges (C, N, C&N, C&N -C, C&N - N, N-C)
        C_charge = acid["charges"][1][acid_center-1]
        N_charge = amine["charges"][1][amine_center-1]
        pC_charge = p["charges"][1][pc-1]
        pN_charge = p["charges"][1][pn-1]
        pCC_charge = pC_charge - C_charge
        pNN_charge = pN_charge - N_charge
        CN_charge = N_charge - C_charge
        # fukui (C, N, C&N, C&N -C, C&N - N, N-C)
        C_fukui = acid["f_nuc"][acid_center-1]
        N_fukui = amine["f_nuc"][amine_center-1]
        pC_fukui = p["f_nuc"][pc-1]
        pN_fukui = p["f_nuc"][pn-1]
        pCC_fukui = pC_fukui - C_fukui
        pNN_fukui = pN_fukui - N_fukui
        CN_fukui = N_fukui - C_fukui
        # omega (C, N, C&N, C&N -C, C&N - N, N-C)
        C_omega = acid["omega_nuc"][acid_center-1]
        N_omega = amine["omega_nuc"][amine_center-1]
        pC_omega = p["omega_nuc"][pc-1]
        pN_omega = p["omega_nuc"][pn-1]
        pCC_omega = pC_omega - C_omega
        pNN_omega = pN_omega - N_omega
        CN_omega = N_omega - C_omega

        return [rxn_e2, rxn_ae, rxn_acid_fukui, rxn_amine_fukui, rxn_fukui,
                acid_ip, amine_ip, p_ip, aa_ip, rxn_ip,
                acid_ea, amine_ea, p_ea, aa_ea, rxn_ea,
                acid_chi, amine_chi, p_chi, aa_chi, rxn_chi,
                acid_eta, amine_eta, p_eta, aa_eta, rxn_eta,
                acid_omega, amine_omega, p_omega, aa_omega, rxn_omega,
                C_charge, N_charge, pC_charge, pN_charge, pCC_charge, pNN_charge, CN_charge,
                C_fukui, N_fukui, pC_fukui, pN_fukui, pCC_fukui, pNN_fukui, CN_fukui,
                C_omega, N_omega, pC_omega, pN_omega, pCC_omega, pNN_omega, CN_omega]

    def get_aimnet_descriptors_(self, idx) -> List[float]:
        """given ID2, return a list of AIMNET descriptors (50) for the reaction. The AIMNET descriptors were pre-saved as a DataFrame using get_aimnet_descriptors method
        """
        return list(self.aimnet_descriptor_df.loc[idx, :])


if __name__ == "__main__":
    import os, sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(root)
    from utils import folder, mols_folder

    rxns = rxn(folder)
    ids = rxns.all_idx2()
    for i, id in enumerate(ids):
        des = rxns.get_aimnet_descriptors(id)
        t, T = rxns.get_tT(id)
        y = rxns.get_yield(id)
        print(len(des), des[0], des[1], t, T, y)
        # print(des[0], des[1])
        if i == 100:
            break
    # example = rxns.get_rxn(16336)
    # acid, amine, p, acy = example[2], example[3], example[4], example[5]
    # print(example)

    # example_3d = rxns.get_rxn_3d(16336)
    # print(example_3d)

    # rxns = rxn(folder)
    # example = rxns.get_rxn(0)
    # print(example)
    # context = rxns.get_context_one_hot(0)
    # print(context)
    # print(len(context))
    # print(rxns.context_dim)