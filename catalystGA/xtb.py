import logging
import os
import random
import re
import shutil
import string
import subprocess
import warnings
from pathlib import Path
from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

alphabet = string.ascii_lowercase + string.digits


STANDARD_PROPERTIES = {"xtb": {"total energy": "electronic_energy"}, "orca": {}}

_logger = logging.getLogger("xtb")


def ac2mol(
    atoms: List[str],
    coords: List[list],
    charge: int = 0,
    perceive_connectivity: bool = True,
    sanitize: bool = True,
):
    """Converts atom symbols and coordinates to RDKit molecule."""
    xyz = ac2xyz(atoms, coords)
    rdkit_mol = Chem.MolFromXYZBlock(xyz)
    if sanitize:
        Chem.SanitizeMol(rdkit_mol)
    if charge != 0:
        rdkit_mol.GetAtomWithIdx(0).SetFormalCharge(charge)
    if perceive_connectivity:
        determineConnectivity(rdkit_mol)
    return rdkit_mol


def adjacency_check(adj1, atoms, coords):
    "For detecting broken bonds during optimizations"
    mol = ac2mol(atoms, coords)
    determineConnectivity(mol, useHueckel=True)
    adj2 = Chem.GetAdjacencyMatrix(mol, force=True)
    return np.array_equal(adj1, adj2)


def determineConnectivity(mol, **kwargs):
    """Determine connectivity in molecule.

    Use to check if bonds are broken in xTB optimizations
    """
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


def xyz2ac(xyzblock: str):
    """Converts from xyz to atoms,coords."""
    lines = xyzblock.split("\n")
    atoms = []
    coords = []
    for line in lines[2:]:
        line = line.strip()
        if len(line) > 0:
            atom, x, y, z = line.split()
            atoms.append(atom)
            coords.append([float(x), float(y), float(z)])
        else:
            break
    return atoms, coords


def ac2xyz(atoms: List[str], coords: List[list]):
    """Converts atom symbols and coordinates to xyz string."""
    xyz = f"{len(atoms)}\n\n"
    for atom, coord in zip(atoms, coords):
        xyz += f"{atom} {coord[0]:.8f} {coord[1]:.8f} {coord[2]:.8f}\n"
    return xyz


def xtb_calculate(
    atoms: list[str],
    coords: list[list],
    charge: int = 0,
    multiplicity: int = 1,
    options: dict = {},
    scr: str = ".",
    n_cores: int = 1,
    detailed_input: None | dict = None,
    detailed_input_str: None | str = None,
    calc_dir: None | str = None,
    xtb_cmd: str = "xtb",
    force: bool = False,
) -> dict:
    """Run xTB calculation.

    Args:
        atoms (list[str]): list of atom symbols.
        coords (list[list]): 3xN list of atom coordinates.
        charge (int): Formal charge of molecule.
        multiplicity (int): Spin multiplicity of molecule.
        options (dict): xTB calculation options.
        scr (str, optional): Path to scratch directory. Defaults to '.'.
        n_cores (int, optional): Number of cores used in calculation. Defaults to 1.
        detailed_input (dict, optional): Detailed input for xTB calculation. Defaults to None.
        calc_dir (str, optional): Name of calculation directory, will be removed after calculation is None. Defaults to None.
        xtb_cmd (str): Path to xTB executable.

    Returns:
        dict: {'atoms': ..., 'coords': ..., ...}
    """
    check_executable(xtb_cmd)
    set_threads(n_cores)

    # create TMP directory
    work_dir = WorkingDir(root=scr, name=calc_dir)
    xyz_file = write_xyz(atoms, coords, work_dir)

    # clean xtb method option
    for k, value in options.items():
        if "gfn" in k.lower():
            if value is not None and value is not True:
                options[k + str(value)] = None
                del options[k]
                break

    # options to xTB command
    cmd = f"{xtb_cmd} --chrg {charge} --uhf {multiplicity-1} --norestart --verbose --parallel {n_cores} "
    for key, value in options.items():
        if value is None or value is True:
            cmd += f"--{key} "
        else:
            cmd += f"--{key} {str(value)} "
    if detailed_input is not None:
        fpath = write_detailed_input(detailed_input, work_dir)
        cmd += f"--input {fpath.name} "
    if detailed_input_str is not None:
        fpath = work_dir / "details.inp"
        with open(fpath, "w") as inp:
            inp.write(detailed_input_str)
        cmd += f"--input {fpath.name} "

    lines = run_xtb((cmd, xyz_file))
    if not normal_termination(lines) and not force:
        _logger.warning("xTB did not terminate normally")
        _logger.info("".join(lines))
        results = {"normal_termination": False, "log": "".join(lines)}
        if calc_dir:
            results["calc_dir"] = str(work_dir)
        else:
            work_dir.cleanup()
        return results

    # read results
    results = read_xtb_results(lines)
    if "hess" in options:
        results.update(read_thermodynamics(lines))
    if "grad" in options:
        with open(work_dir / "mol.engrad", "r") as f:
            grad_lines = f.readlines()
        results["grad"] = read_gradients(grad_lines)
    if "wbo" in options:
        results["wbo"] = read_wbo(work_dir / "wbo", len(atoms))
    if "pop" in options:
        results["mulliken"] = read_mulliken(work_dir / "charges")
    results["atoms"] = atoms
    results["coords"] = coords
    if "opt" in options:
        results["opt_coords"] = read_opt_structure(lines)[-1]
    if "ohess" in options:
        results.update(read_thermodynamics(lines))
        results["opt_coords"] = read_opt_structure(lines)[-1]
    if detailed_input or detailed_input_str:
        if any("scan" in x for x in (detailed_input, detailed_input_str) if x is not None):
            results["scan"] = read_scan(work_dir / "xtbscan.log")
    if calc_dir:
        results["calc_dir"] = str(work_dir)
    else:
        work_dir.cleanup()

    return results


def set_threads(n_cores: int) -> None:
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


def write_xyz(atoms: list[str], coords: list[list], scr: Path) -> Path:
    """Write xyz coordinate file."""
    natoms = len(atoms)
    xyz = f"{natoms} \n \n"
    for atomtype, coord in zip(atoms, coords):
        xyz += f"{atomtype}  {' '.join(list(map(str, coord)))} \n"
    with open(scr / "mol.xyz", "w") as inp:
        inp.write(xyz)
    _logger.debug(f"Written xyz-file to {scr / 'mol.xyz'}")
    return scr / "mol.xyz"


def write_detailed_input(details_dict: dict, scr: Path) -> Path:
    """Write detailed input file for xTB calculation."""
    detailed_input_str = ""
    for key, value in details_dict.items():
        detailed_input_str += f"${key}\n"
        for subkey, subvalue in value.items():
            detailed_input_str += f' {subkey}: {", ".join([str(i) for i in subvalue])}\n'
    detailed_input_str += "$end\n"

    fpath = scr / "details.inp"
    _logger.debug(f"Writing detailed input to {fpath}")
    _logger.debug(detailed_input_str)
    with open(fpath, "w") as inp:
        inp.write(detailed_input_str)

    return fpath


def run_xtb(args: tuple[str]) -> list[str]:
    """Run xTB command for xyz-file in parent directory, logs and returns
    output."""
    cmd, xyz_file = args
    generator = stream(f"{cmd}-- {xyz_file.name} | tee xtb.out", cwd=xyz_file.parent)
    lines = []
    for line in generator:
        lines.append(line)
        _logger.debug(line.rstrip("\n"))
    return lines


def normal_termination(lines: list[str], strict: bool = False) -> bool:
    """Check if xTB terminated normally."""
    first_check = 0
    for line in reversed(lines):
        if line.strip().startswith("normal termination"):
            first_check = 1
            if not strict:
                return first_check
        if "FAILED TO" in line:
            if strict:
                return max([0, first_check - 0.5])
    return first_check


def read_opt_structure(lines: list[str]) -> tuple[list[str], list[list[float]]]:
    """Read optimized structure from xTB output."""

    def _parse_coordline(line: str) -> tuple[str, list[list[float]]]:
        """Parse coordinate line from xyz-file."""
        line = line.split()
        atom = line[0]
        coord = [float(x) for x in line[-3:]]
        return atom, coord

    for i, l in reversed(list(enumerate(lines))):
        if "final structure" in l:
            break
    n_atoms = int(lines[i + 2].rstrip())
    start = i + 4
    end = start + n_atoms
    atoms = []
    coords = []
    for line in lines[start:end]:
        atom, coord = _parse_coordline(line)
        atoms.append(atom)
        coords.append(coord)

    return atoms, coords


def read_thermodynamics(lines: list[str]) -> dict:
    """Read thermodynamics output of frequency calculation."""
    thermo_idx = np.nan
    thermo_properties = {}
    for i, line in enumerate(lines):
        line = line.strip()
        if "THERMODYNAMIC" in line:
            thermo_idx = i
        if i > (thermo_idx + 2):
            if 20 * ":" in line:
                thermo_idx = np.nan
            elif 20 * "." in line:
                continue
            else:
                tmp = line.strip(":").strip().strip("->").split()[:-1]
                thermo_properties[" ".join(tmp[:-1])] = float(tmp[-1])
    return thermo_properties


def read_gradients(lines: list[str]) -> np.ndarray:
    """Read gradients from engrad file."""
    gradients = []
    gradient_idx = np.nan
    for i, line in enumerate(lines):
        line = line.strip()
        if "gradient" in line:
            gradient_idx = i
        if i > (gradient_idx + 1):
            if line.startswith("#"):
                gradient_idx = np.nan
                break
            gradients.append(float(line))
    gradients = np.asarray(gradients)
    return gradients.reshape(-1, 3)


def read_wbo(wbo_file, n_atoms) -> np.ndarray:
    """Read Wiberg bond order from wbo file."""
    data = np.loadtxt(wbo_file)
    wbo = np.zeros((n_atoms, n_atoms))
    for i, j, o in data:
        i = i.astype(int) - 1
        j = j.astype(int) - 1
        wbo[i, j] = wbo[j, i] = o
    return wbo


def read_mulliken(charges_file) -> np.ndarray:
    """Read Mulliken charges from charges file."""
    return np.loadtxt(charges_file)


def read_scan(scan_file) -> dict:
    """Read scan results from xTB output."""
    with open(scan_file, "r") as f:
        lines = f.readlines()
    nAtoms = int(lines[0])
    nFrames = int(len(lines) / (nAtoms + 2))
    pes = []
    traj = []
    for n in range(nFrames):
        pes.append(float(lines[n * (nAtoms + 2) + 1].split(":")[1].strip("xtb")))
        traj.append(xyz2ac("".join(lines[n * (nAtoms + 2) : n * (nAtoms + 2) + (nAtoms + 2)]))[1])
    return {"pes": pes, "traj": traj}


def read_xtb_results(lines: list[str]) -> dict:
    """Read basic results from xTB log."""

    def _get_runtime(lines: list[str]) -> float:
        """Reads xTB runtime in seconds."""

        pattern = r":\s+(\d+)\s+d,\s+(\d+)\s+h,\s+(\d+)\s+min,\s+([\d.]+)\s+sec"
        match = re.search(pattern, line)

        total_seconds = None
        if match:
            days_str, hours_str, mins_str, secs_str = match.groups()

            days = int(days_str)
            hours = int(hours_str)
            mins = int(mins_str)
            secs = float(secs_str)

            # If you want the total time in seconds:
            total_seconds = days * 86400 + hours * 3600 + mins * 60 + secs
        return total_seconds

    property_start_idx, dipole_idx, quadrupole_idx, runtime_idx, polarizability_idx = (
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    )
    properties = {}
    wall_time, cpu_time = np.nan, np.nan
    for i, line in enumerate(lines):
        line = line.strip()
        if "xtb version" in line:
            xtb_version = line.split()[3]
        elif "program call" in line:
            programm_call = line.split(":")[-1]
        elif "SUMMARY" in line:
            property_start_idx = i
        elif "molecular dipole" in line:
            dipole_idx = i
        elif "molecular quadrupole" in line:
            quadrupole_idx = i
        elif "total:" in line:
            runtime_idx = i
        elif "Mol. α(0) /au" in line:
            polarizability_idx = i

        # read property table
        if i > (property_start_idx + 1):
            if 20 * ":" in line:
                property_start_idx = np.nan
            elif 20 * "." in line:
                continue
            else:
                tmp = line.strip(":").strip().strip("->").split()[:-1]
                properties[" ".join(tmp[:-1])] = float(tmp[-1])

        # read polarizability
        if i == polarizability_idx:
            polarizability = float(line.split()[-1])

        # read dipole moment
        if i > (dipole_idx + 2):
            dip_x, dip_y, dip_z, dip_norm = [
                float(x) for x in line.split()[1:]
            ]  # norm is in Debye
            dipole_vec = np.array([dip_x, dip_y, dip_z])  # in a.u. (*2.5412 ot convert to Debye)
            dipole_idx = np.nan

        # read quadrupole moment
        if i > (quadrupole_idx + 3):
            quad_vec = np.array([float(x) for x in line.split()[1:]])  # in a.u.
            quad_vec[2], quad_vec[3] = quad_vec[3], quad_vec[2]
            quadrupole_mat = np.zeros((3, 3))
            indices = np.triu_indices(3)
            quadrupole_mat[indices] = quad_vec
            quadrupole_mat[indices[::-1]] = quad_vec
            quadrupole_idx = np.nan

        # read runtimes
        if i > runtime_idx:
            if i == (runtime_idx + 1):
                wall_time = _get_runtime(line)
            else:
                cpu_time = _get_runtime(line)
                runtime_idx = np.nan

    results = {
        "normal_termination": True,
        "programm_call": programm_call,
        "programm_version": xtb_version,
        "wall_time": wall_time,
        "cpu_time": cpu_time,
    }

    if not np.isnan(polarizability_idx):
        results["polarizability"] = polarizability
    if not np.isnan(dipole_idx):
        results["dipole_vec"] = dipole_vec
        results["dipole_norm"] = dip_norm
    if not np.isnan(quadrupole_idx):
        results["quadrupole_mat"] = quadrupole_mat

    results.update(properties)
    # add standardized property names
    for key, value in STANDARD_PROPERTIES["xtb"].items():
        if key in results:
            results[value] = results[key]
    return results


# def stream(cmd: str, cwd: None | Path = None, shell: bool = True) -> Generator[str, None, None]:
#     """Execute a command and stream stdout and stderr concurrently."""
#     with subprocess.Popen(
#         cmd,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#         text=True,  # Use text mode for string-based reading
#         shell=shell,
#         cwd=cwd,
#         bufsize=1,  # Line-buffered output for immediate feedback
#     ) as process:
#         try:
#             for line in iter(process.stdout.readline, ""):
#                 yield line.strip()
#             for line in iter(process.stderr.readline, ""):
#                 yield line.strip()
#         except KeyboardInterrupt:
#             print("\nCtrl+C pressed. Terminating the process...")
#             os.killpg(os.getpgid(process.pid), signal.SIGTERM)
#             process.wait()
#             print("Process terminated.")
#         finally:
#             process.stdout.close()
#             process.stderr.close()
#             process.wait()


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
        if "SKIPPING Reordering" in stdout_line:
            popen.kill()
        yield stdout_line

    # Yield errors
    stderr = popen.stderr.read()
    popen.stdout.close()
    yield stderr

    return


def check_executable(executable: str):
    """Check if executable is in PATH."""
    results = stream(f"which {executable}")
    result = next(results)
    if result.startswith("which: no"):
        warnings.warn(f"Executable {executable} not found in PATH")


class WorkingDir:
    def __init__(self, root: str = ".", name: str = None) -> None:
        self.root = Path(root)
        self.name = name if name else self._random_str()
        self.dir = self.root / self.name
        self.create()

    def __str__(self) -> str:
        return str(self.dir.resolve())

    def __repr__(self) -> str:
        return self.__str__()

    def __truediv__(self, name: str) -> str:
        return self.dir / name

    def _random_str(self) -> str:
        name = "_" + "".join(random.choices(alphabet, k=6))
        while (self.root / name).exists():
            name = "".join(random.choices(alphabet, k=6))
        return name

    def create(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)

    def cleanup(self) -> None:
        try:
            # print("removing ", self.dir.absolute())
            shutil.rmtree(self.dir.absolute())
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    atoms = ["C", "C", "C", "N", "C", "C", "N", "H", "H", "H", "H", "H", "H", "H"]
    coords = [
        [-0.150572945378, 1.06902383757551, 0.13369717980808],
        [-1.53340374624082, 0.97192911929508, 0.16014351260219],
        [-2.11046032356157, -0.28452881345742, 0.02365204850958],
        [-1.40326713049944, -1.39105857658305, -0.1287228595083],
        [-0.08701882046636, -1.30624454138783, -0.15461643853251],
        [0.58404976357399, -0.09554934782587, -0.02805090001267],
        [2.04601334595326, -0.0521981720584, -0.06393160516877],
        [0.3301095007743, 2.02983678857648, 0.23606713936603],
        [-2.14949770349705, 1.84686355411139, 0.28440826742643],
        [-3.18339157653865, -0.41415380049138, 0.03929295998759],
        [0.43604958186272, -2.24346585185004, -0.28257385567194],
        [2.37469662935334, 0.91726637325056, 0.04696293388512],
        [2.40041378960073, -0.42116004539483, -0.95829570995538],
        [2.44627963506354, -0.62656052376019, 0.69196732726456],
    ]

    options = {"opt": True, "alpb": "methanol", "wbo": True, "pop": True}
    results = xtb_calculate(atoms=atoms, coords=coords, charge=1, options=options)

    for key, value in results.items():
        print(key)
        print(value)
        print(80 * "*")
