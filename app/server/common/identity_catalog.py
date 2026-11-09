# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

"""Curated identities for species that sources commonly abbreviate.

A formula appears here only when it names one compound in adsorption work;
isomeric formulas such as C4H10 or C3H6 must stay out. Formulas match
case-sensitively so that CO is never read as cobalt (Co).
"""

from __future__ import annotations

from typing import NamedTuple


###############################################################################
class CuratedAdsorbate(NamedTuple):
    inchi_key: str
    name: str
    formula: str
    aliases: tuple[str, ...] = ()


CURATED_ADSORBATES: tuple[CuratedAdsorbate, ...] = (
    CuratedAdsorbate("CURLTUGMZLYLDI-UHFFFAOYSA-N", "Carbon dioxide", "CO2"),
    CuratedAdsorbate("IJGRMHOSHXDMSA-UHFFFAOYSA-N", "Nitrogen", "N2"),
    CuratedAdsorbate("VNWKTOKETHGBQD-UHFFFAOYSA-N", "Methane", "CH4"),
    CuratedAdsorbate("UFHFLCQGNIYNRP-UHFFFAOYSA-N", "Hydrogen", "H2"),
    CuratedAdsorbate("MYMOFIZGZYHOMD-UHFFFAOYSA-N", "Oxygen", "O2"),
    CuratedAdsorbate("XKRFYHLGVUSROY-UHFFFAOYSA-N", "Argon", "Ar"),
    CuratedAdsorbate("SWQJXJOGLNCZEY-UHFFFAOYSA-N", "Helium", "He"),
    CuratedAdsorbate("DNNSSWSSYDEUBZ-UHFFFAOYSA-N", "Krypton", "Kr"),
    CuratedAdsorbate("FHNFHKCVQCLJFQ-UHFFFAOYSA-N", "Xenon", "Xe"),
    CuratedAdsorbate("UGFAIRIUMAVXCW-UHFFFAOYSA-N", "Carbon monoxide", "CO"),
    CuratedAdsorbate("XLYOFNOQVPJJNP-UHFFFAOYSA-N", "Water", "H2O"),
    CuratedAdsorbate("QGZKDVFQNNGYKY-UHFFFAOYSA-N", "Ammonia", "NH3"),
    CuratedAdsorbate("RWSOTUBLDIXVET-UHFFFAOYSA-N", "Hydrogen sulfide", "H2S"),
    CuratedAdsorbate("RAHZWNYVWXNFOC-UHFFFAOYSA-N", "Sulfur dioxide", "SO2"),
    CuratedAdsorbate("GQPLMRYTRLFLPF-UHFFFAOYSA-N", "Nitrous oxide", "N2O"),
    CuratedAdsorbate("MWUXSHHQAYIFBG-UHFFFAOYSA-N", "Nitric oxide", "NO"),
    CuratedAdsorbate("JCXJVPUVTGWSNB-UHFFFAOYSA-N", "Nitrogen dioxide", "NO2"),
    CuratedAdsorbate("SFZCNBIFKDRMGX-UHFFFAOYSA-N", "Sulfur hexafluoride", "SF6"),
    CuratedAdsorbate("OTMSDBZUPAUEDD-UHFFFAOYSA-N", "Ethane", "C2H6"),
    CuratedAdsorbate("VGGSQFUCUMXWEO-UHFFFAOYSA-N", "Ethylene", "C2H4", ("Ethene",)),
    CuratedAdsorbate("HSFWRNGVRCDJHI-UHFFFAOYSA-N", "Acetylene", "C2H2", ("Ethyne",)),
    CuratedAdsorbate("ATUOYWHBWRKTHZ-UHFFFAOYSA-N", "Propane", "C3H8"),
)

# Each group names one material. A protonated or metal-exchanged form is a
# different material and needs its own group.
CURATED_ADSORBENT_GROUPS: tuple[tuple[str, ...], ...] = (
    ("HKUST-1", "Cu-BTC", "CuBTC", "MOF-199", "Cu3(BTC)2", "Basolite C300"),
    ("MOF-5", "IRMOF-1"),
    ("ZIF-8", "MAF-4", "Basolite Z1200"),
    ("MIL-53(Al)", "Basolite A100"),
    ("Zeolite 13X", "13X", "NaX", "Zeolite NaX", "Molecular sieve 13X"),
    ("Zeolite 5A", "5A", "CaA", "Zeolite CaA", "Molecular sieve 5A"),
    ("Zeolite 4A", "4A", "NaA", "Zeolite NaA", "Molecular sieve 4A"),
    ("Zeolite 3A", "3A", "KA", "Zeolite KA", "Molecular sieve 3A"),
)

GENERIC_ADSORBENT_NAMES: frozenset[str] = frozenset(
    {
        "activated alumina",
        "activated carbon",
        "activated charcoal",
        "alginate beads",
        "alumina",
        "biochar",
        "carbon",
        "carbon nanotubes",
        "cellulose",
        "charcoal",
        "chitosan",
        "clay",
        "coal",
        "cof",
        "graphene",
        "graphene oxide",
        "mof",
        "nanoporous carbon",
        "porous carbon",
        "resin",
        "silica",
        "silica gel",
        "zeolite",
    }
)
