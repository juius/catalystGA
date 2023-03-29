import logging
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

import hide_warnings
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

XTB_CMD = "xtb"

_logger = logging.getLogger("xtb")


def xtb_calculate(
    atoms: List[str],
    coords: List[list],
    options: dict,
    scr: str = ".",
    n_cores: int = 1,
    detailed_input: dict = None,
) -> tuple:
    """Runs xTB calculation.

    Args:
        atoms (List[str]): List of atom symbols.
        coords (List[list]): 3xN list of atom coordinates.
        options (dict): xTB calculation options.
        scr (str, optional): Path to scratch directory. Defaults to '.'.
        n_cores (int, optional): Number of cores used in calculation. Defaults to 1.
        detailed_input (dict, optional): Detailed input for xTB calculation. Defaults to None.

    Returns:
        tuple: (atoms, coords, energy)
    """
    # Set Threads
    set_threads(n_cores)
    # Creat TMP directory
    tempdir = tempfile.TemporaryDirectory(dir=scr, prefix="XTB_")
    tmp_scr = Path(tempdir.name)
    xyz_file = write_xyz(atoms, coords, tmp_scr)
    # clean xtb method option
    for k, value in options.items():
        if "gfn" in k.lower():
            if value is not None and value is not True:
                options[k + str(value)] = None
                del options[k]
                break

    # Options to xTB command
    cmd = f"{XTB_CMD} --norestart --verbose --parallel {n_cores} "
    for key, value in options.items():
        if value is None or value is True:
            cmd += f"--{key} "
        else:
            cmd += f"--{key} {str(value)} "
    if detailed_input is not None:
        fpath = write_detailed_input(detailed_input, tmp_scr)
        cmd += f"--input {fpath.name} "

    result = run_xtb((cmd, xyz_file))
    if not normal_termination(result):
        _logger.warning("xTB did not terminate normally")
        _logger.info("".join(result))
        return atoms, coords, math.nan
    if "opt" in options:
        atoms, coords = read_opt_structure(result)
    energy = read_energy(result)

    return atoms, coords, energy


def set_threads(n_cores: int):
    """Set threads and procs environment variables."""
    _ = list(stream("ulimit -s unlimited"))
    os.environ["OMP_STACKSIZE"] = "4G"
    os.environ["OMP_NUM_THREADS"] = f"{n_cores},1"
    os.environ["MKL_NUM_THREADS"] = str(n_cores)
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"

    _logger.debug("Set OMP_STACKSIZE to 4G")
    _logger.debug(f"Set OMP_NUM_THREADS to {n_cores},1")
    _logger.debug(f"Set MKL_NUM_THREADS to {n_cores}")
    _logger.debug("Set OMP_MAX_ACTIVE_LEVELS to 1")


def write_xyz(atoms: List[str], coords: List[list], scr: Path):
    """Write xyz coordinate file."""
    natoms = len(atoms)
    xyz = f"{natoms} \n \n"
    for atomtype, coord in zip(atoms, coords):
        xyz += f"{atomtype}  {' '.join(list(map(str, coord)))} \n"
    with open(scr / "mol.xyz", "w") as inp:
        inp.write(xyz)
    _logger.debug(f"Written xyz-file to {scr / 'mol.xyz'}")
    return scr / "mol.xyz"


def write_detailed_input(details_dict: dict, scr: Path):
    """Write detailed input file for xTB calculation."""

    detailed_input_str = ""
    for key, value in details_dict.items():
        detailed_input_str += f"${key}\n"
        for subkey, subvalue in value.items():
            detailed_input_str += f'\t{subkey}: {", ".join([str(i) for i in subvalue])}\n'
    detailed_input_str += "$end\n"

    fpath = scr / "details.inp"
    _logger.debug(f"Writing detailed input to {fpath}")
    _logger.debug(detailed_input_str)
    with open(fpath, "w") as inp:
        inp.write(detailed_input_str)

    return fpath


def parse_coordline(line: str):
    """Parse coordinate line from xyz-file."""
    line = line.split()
    atom = line[0]
    coord = [float(x) for x in line[-3:]]
    return atom, coord


def run_xtb(args: Tuple[str]):
    """Runs xTB command for xyz-file in parent directory, logs and returns
    output."""
    cmd, xyz_file = args
    generator = stream(f"{cmd}-- {xyz_file.name} | tee xtb.out", cwd=xyz_file.parent)
    lines = []
    for line in generator:
        lines.append(line)
        _logger.debug(line.rstrip("\n"))
    return lines


def normal_termination(lines: List[str]):
    """Checks if xTB terminated normally."""
    for line in reversed(lines):
        if line.strip().startswith("normal termination"):
            return True
    return False


def read_opt_structure(lines: List[str]):
    """Reads optimized structure from xTB output."""
    for i, l in reversed(list(enumerate(lines))):
        if "final structure" in l:
            break
    n_atoms = int(lines[i + 2].rstrip())
    start = i + 4
    end = start + n_atoms
    atoms = []
    coords = []
    for line in lines[start:end]:
        atom, coord = parse_coordline(line)
        atoms.append(atom)
        coords.append(coord)

    return atoms, coords


def read_energy(lines: List[str]):
    """Reads energy from xTB output."""
    for line in reversed(list(lines)):
        if "TOTAL ENERGY" in line:
            energy = float(line.split()[-3])
            return energy
    return math.nan


def stream(cmd, cwd=None, shell=True):
    """Execute command in directory, and stream stdout."""
    popen = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        shell=shell,
        cwd=cwd,
    )
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line

    # Yield errors
    stderr = popen.stderr.read()
    popen.stdout.close()
    yield stderr

    return


#######################
# General RDKit Utils #
#######################


@hide_warnings
def _determineConnectivity(mol, **kwargs):
    """Determine bonds in molecule."""
    try:
        rdDetermineBonds.DetermineConnectivity(mol, **kwargs)
    finally:
        # cleanup extended hueckel files
        try:
            os.remove("nul")
            os.remove("run.out")
        except FileNotFoundError:
            pass
    return mol


def ac2xyz(atoms: List[str], coords: List[list]):
    """Converts atom symbols and coordinates to xyz string."""
    xyz = f"{len(atoms)}\n\n"
    for atom, coord in zip(atoms, coords):
        xyz += f"{atom} {coord[0]:.8f} {coord[1]:.8f} {coord[2]:.8f}\n"
    return xyz


def ac2mol(atoms: List[str], coords: List[list], useHueckel=True, **kwargs):
    """Converts atom symbols and coordinates to RDKit molecule."""
    xyz = ac2xyz(atoms, coords)
    rdkit_mol = Chem.MolFromXYZBlock(xyz)
    Chem.SanitizeMol(rdkit_mol)
    _determineConnectivity(rdkit_mol, useHueckel=useHueckel, **kwargs)
    return rdkit_mol
