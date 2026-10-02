"""Published M_D0 bounds from Cavero et al. (2026), Appendix B Tables 10-12."""

from __future__ import annotations

from pathlib import Path


def _assign(keys: str, value: int) -> dict[str, int]:
    return {key: value for key in keys.split()}


STANDARD_MDO_K = {
    **_assign("cycle10 cycle15 cycle20 cyclePow12-2 dStar5-5 dStar6-3 dStar6-6 dStar8-4 mobiousLadder12 p3xp3 p4xp4 p5xp5 p6xp6 path10 path15 path20 st3xst3 star5 star7 tree2x2 tree2x3 tree2x4 tree3x2 wheel5 wheel7 triangle6 triangle10", 1),
    **_assign("bipartite3x3 bipartite3x4 bipartite4x4 bipartite5x5 c3xc3 c3xc4 c3xk4 c4xc4 c4xc5 c4xk5 c5xc5 cyclePow10-2 cyclePow11-2 dStar8-8 k3xk3 k4xk4 k5 k6 k7 mobiousLadder6 mobiousLadder8 mobiousLadder10 mobiousLadder16 p3xk4 p4xc3 p4xc4 p4xk5 p5xc4 p5xc5 p6xc5 petersen st4xst4 st5xst5 star10 star15 star20 wheel10 wheel15 wheel20 triangle15 triangle21", 2),
    **_assign("bipartite7x8 k10", 3),
}

# A None value is deliberate: Table 12 reports that M_D0 did not solve it.
HB_MDO_K: dict[str, int | None] = {
    "arc130": 6, "ash85": 2, "bcspwr01": 1, "bcspwr02": 2, "bcspwr03": 2,
    "bcsstk01": 3, "bcsstk04": 7, "bcsstk05": 5, "bcsstk22": 2,
    "can__144": 3, "can__161": 3, "curtis54": 2, "dwt__234": 3,
    "fs_183_1": 6, "gent113": 5, "gre__115": 3, "gre__185": 3, "ibm32": 3,
    "impcol_b": 4, "impcol_c": 4, "lns__131": 3, "lund_a": 4, "lund_b": 4,
    "mcca": 6, "nos1": 2, "nos4": 3, "pores_1": 2, "steam3": 3,
    "west0132": 5, "west0156": 5, "west0167": 5, "will199": 14, "will57": 2,
    "494_bus": 3, "662_bus": 4, "685_bus": 3, "ash292": 3, "bcspwr04": 3,
    "bcspwr05": 3, "bcsstk06": None, "bcsstk19": 4, "bcsstk20": 2,
    "bcsstm07": None, "bp_____0": None, "bp__1000": None, "bp__1200": None,
    "bp__1400": None, "bp__1600": None, "bp___200": None, "bp___400": None,
    "bp___600": None, "bp___800": None, "can__292": None, "can__445": None,
    "can__715": 6, "can__838": None, "dwt__209": 4, "dwt__221": 3,
    "dwt__245": 3, "dwt__310": 3, "dwt__361": 3, "dwt__419": None,
    "dwt__503": None, "dwt__592": None, "dwt__878": 4, "dwt__918": 4,
    "dwt__992": None, "fs_541_1": None, "fs_680_1": None, "fs_760_1": None,
    "gr_30_30": 1, "gre_216a": None, "gre__343": None, "gre__512": None,
    "hor__131": None, "impcol_a": None, "impcol_d": None, "impcol_e": None,
    "jagmesh1": None, "jpwh_991": None, "lns__511": None, "mbeacxc": None,
    "mbeaflw": None, "mbeause": None, "mcfe": None, "nnc261": None,
    "nnc666": None, "nos2": 2, "nos3": None, "nos5": None, "nos6": 3,
    "nos7": None, "orsirr_2": None, "plat362": None, "plskz362": None,
    "pores_3": None, "saylr1": None, "saylr3": None, "sherman1": None,
    "sherman4": None, "shl____0": None, "shl__200": None, "shl__400": None,
    "steam1": None, "steam2": None, "str____0": None, "str__200": None,
    "str__600": None, "west0381": None, "west0479": None, "west0497": None,
    "west0655": None, "west0989": None,
}


def normalize_instance_name(instance: str | Path) -> str:
    name = Path(instance).name
    for suffix in (".mtx.rnd.mtx", ".mtx.rnd", ".mtx"):
        if name.endswith(suffix):
            return name.removesuffix(suffix)
    return name


def mdo_best_k(instance: str | Path) -> int | None:
    key = normalize_instance_name(instance)
    if key in STANDARD_MDO_K:
        return STANDARD_MDO_K[key]
    return HB_MDO_K.get(key)
