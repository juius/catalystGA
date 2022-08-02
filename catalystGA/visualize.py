import matplotlib.pyplot as plt
import py3Dmol
from ppqm import chembridge, gaussian
from rdkit import Chem
from rdkit.Chem import Draw


def plot_scores(results, figsize=(8, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    gennums = [x[0] for x in results]
    if isinstance(results[0][1], list):
        # detailed
        # best scores
        scores = [x[1][0].score for x in results]
        ax.plot(gennums, scores)
    else:
        # only best scores
        scores = [x[1].score for x in results]
        ax.plot(gennums, scores)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Score")


def plot_population(population, molsPerRow=5, subImgSize=(100, 100), **kwargs):
    return Draw.MolsToGridImage(
        [ind.mol for ind in population],
        legends=[f"{ind.score:.2f}" for ind in population],
        **kwargs,
    )


def draw3d(
    mols,
    multipleConfs=False,
    confId=-1,
    atomlabel=False,
    width=600,
    height=400,
):
    p = py3Dmol.view(width=width, height=height)
    if type(mols) is not list:
        mols = [mols]
    for mol in mols:
        if isinstance(mol, str):
            if mol.endswith(".xyz"):
                xyz_f = open(mol)
                line = xyz_f.read()
                xyz_f.close()
                p.addModel(line, "xyz")
            elif mol.endswith(".log"):
                with open(mol) as f:
                    lines = f.readlines()
                geom = gaussian.get_opt_structure(lines)
                mol = chembridge.axyzc_to_molobj(
                    [chembridge.get_atom_str(a) for a in geom["atoms"]],
                    geom["coord"],
                    0,
                )
                mb = Chem.MolToMolBlock(mol)
                p.addModel(mb, "sdf")
        else:
            if multipleConfs:
                for conf in mol.GetConformers():
                    mb = Chem.MolToMolBlock(mol, confId=conf.GetId())
                    p.addModel(mb, "sdf")
            else:
                mb = Chem.MolToMolBlock(mol, confId=confId)
                p.addModel(mb, "sdf")
    p.setStyle({"sphere": {"radius": 0.4}, "stick": {}})
    if atomlabel:
        p.addPropertyLabels("index")
    else:
        p.setClickable(
            {},
            True,
            """function(atom,viewer,event,container) {
                   if(!atom.label) {
                    atom.label = viewer.addLabel(atom.index,{position: atom, backgroundColor: 'white', fontColor:'black'});
                   }}""",
        )
    p.zoomTo()
    p.show()
