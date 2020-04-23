import torch.nn as nn


class DeepBiding(nn.module):
    def __init__(self, protein_pdb_file, parameter_file, args):
        """
        Should take as paramter a .pdb (python data bank), a parameter file for the weight and other paramters needed for
        the biding network and initialize the network with parameter file. You can also inherit from an other network,
        as you wish
        :param protein_pdb_file:
        :param parameter_file
        :param args:
        """
        pass

    def score_list(self, smiles):
        """
        Take a list of smiles for exemple [""CCCCCCC(C)C(C)C", "OC(NC(Br)=CF)=C(O)C(F)NC=CC=CCl", "Cc1c(C2=Cc3ccccc3C2=O)ccc(Nc2ccc(F)cc2F)c1F"...]
        and returns a list of biding scores with the initialized 3D protein
            :param smiles: list of smiles
            :return: list of float scores reprsenting biding affinities
        """
        scores = [0.3, 0.6, 0.8, 0.9]

        return scores
