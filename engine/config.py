"""Constantes e valores padrao para o modelo de rompimento de barragens.

Normativas de referencia:
- CNRH 241/2024 (substitui 143/2012) — criterios de DPA
- Resolucao ANA 265/2025 — criterios complementares
- Lei 12.334/2010 (alterada pela 14.066/2020) — PNSB
"""

DATUM_OPTIONS = {
    "SIRGAS 2000 / UTM 23S (31983)": 31983,
    "SIRGAS 2000 / UTM 22S (31982)": 31982,
    "SIRGAS 2000 / UTM 24S (31984)": 31984,
    "SIRGAS 2000 / UTM 25S (31985)": 31985,
    "SIRGAS 2000 / UTM 21S (31981)": 31981,
    "SIRGAS 2000 / UTM 20S (31980)": 31980,
    "SIRGAS 2000 / UTM 19S (31979)": 31979,
    "SIRGAS 2000 / UTM 18S (31978)": 31978,
    "SIRGAS 2000 / UTM 17S (5396)": 5396,
}

DEFAULT_DATUM = 31982

# Mapa longitude -> EPSG para SIRGAS 2000 UTM Sul
# Fuso N cobre longitudes de (N-1)*6 - 180 ate N*6 - 180
_UTM_ZONES_SUL = {
    # fuso: (lon_min, lon_max, epsg)
    17: (-84, -78, 5396),
    18: (-78, -72, 31978),
    19: (-72, -66, 31979),
    20: (-66, -60, 31980),
    21: (-60, -54, 31981),
    22: (-54, -48, 31982),
    23: (-48, -42, 31983),
    24: (-42, -36, 31984),
    25: (-36, -30, 31985),
}


def auto_datum(lon):
    """Retorna o EPSG SIRGAS 2000 UTM Sul correto para a longitude dada."""
    for fuso, (lon_min, lon_max, epsg) in _UTM_ZONES_SUL.items():
        if lon_min <= lon < lon_max:
            return epsg
    # Fallback zona 23 (cobre RJ, ES, BA litoral)
    return 31983


def datum_label_from_epsg(epsg):
    """Retorna o label legivel de um EPSG."""
    for label, code in DATUM_OPTIONS.items():
        if code == epsg:
            return label
    return f"EPSG:{epsg}"
DEFAULT_N_SECOES = 21
DEFAULT_N_SIMPLIFICACAO = 8
DEFAULT_COMPRIMENTO_SECAO = 4000
DEFAULT_MANNING_K = 15
DEFAULT_FC = 1
DEFAULT_N_PONTOS_PERFIL = 81
DEFAULT_N_ALTURAS = 11
DEFAULT_PISO_ADAPTAVEL = True

# Metodos disponiveis
METODOS_QMAX_LABELS = {
    'ana_lnec': 'ANA-LNEC (conservador)',
    'froehlich_1995': 'Froehlich 1995',
    'froehlich_2008': 'Froehlich 2008',
    'mmc': 'MMC (USACE)',
    'webby_1996': 'Webby 1996',
    'vtg_1990': 'VTG 1990',
}

MODOS_RUPTURA = {
    'overtopping': 'Galgamento (overtopping)',
    'piping': 'Erosao interna (piping)',
}

METODOS_ATENUACAO = {
    'ana': 'ANA simplificado (empirico)',
    'muskingum_cunge': 'Muskingum-Cunge (hidrograma)',
}

# DPA — CNRH 241/2024
# Limites de pontuacao para classificacao
DPA_LIMITE_ALTO = 13
DPA_LIMITE_MEDIO = 7
