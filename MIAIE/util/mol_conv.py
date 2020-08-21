import pandas
import numpy
import torch
import rdkit.Chem.Descriptors as dsc
from mendeleev import get_table
from rdkit import Chem
from sklearn.preprocessing import scale
from torch_geometric.data import Data


list_atom_feats = ['atomic_number', 'atomic_radius', 'atomic_volume', 'boiling_point', 'density',
                  'dipole_polarizability', 'electron_affinity', 'evaporation_heat', 'fusion_heat',
                  'lattice_constant', 'melting_point', 'period', 'specific_heat', 'thermal_conductivity',
                  'vdw_radius', 'covalent_radius_cordero', 'covalent_radius_pyykko', 'en_pauling',
                  'en_allen', 'heat_of_formation', 'vdw_radius_uff', 'vdw_radius_mm3', 'abundance_crust',
                   'abundance_sea', 'en_ghosh', 'vdw_radius_alvarez', 'c6_gb', 'atomic_weight',
                   'atomic_weight_uncertainty', 'atomic_radius_rahm']
list_bond_feats = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
list_mol_feats = ['mol_weight', 'num_rings']
num_atom_feats = len(list_atom_feats)
num_bond_feats = len(list_bond_feats)
num_mol_feats = len(list_mol_feats)


def get_mat_atom_props():
    tb_atm_props = get_table('elements')
    mat_atom_props = numpy.nan_to_num(numpy.array(tb_atm_props[list_atom_feats], dtype=numpy.float))
    mat_atom_props = scale(mat_atom_props)

    return mat_atom_props


def read_dataset(path_dataset):
    data = numpy.array(pandas.read_csv(path_dataset))
    mat_atom_props = get_mat_atom_props()
    dataset = list()

    for i in range(0, data.shape[0]):
        mol = Chem.MolFromSmiles(data[i, 0])
        atoms = mol.GetAtoms()
        atom_nums = [atom.GetAtomicNum() for atom in atoms]
        atom_feat_mat = numpy.empty([mol.GetNumAtoms(), num_atom_feats])
        bonds = list()
        bond_feats = list()
        mol_feats = numpy.empty(num_mol_feats)

        for j in range(0, len(atoms)):
            atom_feat_mat[j, :] = mat_atom_props[atoms[j].GetAtomicNum()-1, :]

        for bond in mol.GetBonds():
            bond_feat = numpy.zeros(4)
            bond_type = bond.GetBondType()

            if bond_type == 'SINGLE':
                bond_feat[0] = 1
            elif bond_type == 'DOUBLE':
                bond_feat[1] = 1
            elif bond_type == 'TRIPLE':
                bond_feat[2] = 1
            elif bond_type == 'AROMATIC':
                bond_feat[3] = 1

            bonds.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            bonds.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
            bond_feats.append(bond_feat)

        if len(bonds) > 0:
            bonds = torch.tensor(bonds, dtype=torch.long).cuda()
            mol_feats[0] = dsc.ExactMolWt(mol)
            mol_feats[1] = mol.GetRingInfo().NumRings()

            mol_graph = Data(x=torch.tensor(atom_feat_mat, dtype=torch.float).cuda(),
                             edge_index=bonds.t().contiguous(),
                             edge_feats=torch.tensor(bond_feats, dtype=torch.float).cuda(),
                             mol_feats=torch.tensor(mol_feats, dtype=torch.float).view(-1, num_mol_feats).cuda(),
                             y=torch.tensor(data[i, 1], dtype=torch.float).view(-1, 1).cuda(),
                             atom_nums=torch.tensor(atom_nums, dtype=torch.long),
                             id=i)

            dataset.append(mol_graph)

        if i % 100 == 0:
            print('Data loading: {:.2f}%'.format(((i + 1) / data.shape[0]) * 100))

    return dataset
