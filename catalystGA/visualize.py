import matplotlib.pyplot as plt
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
