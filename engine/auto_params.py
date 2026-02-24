"""Calculo automatico de parametros otimos para modelagem de rompimento.

Todas as formulas sao baseadas na literatura:
- ANA/LNEC simplified methodology (Rolo et al.)
- Froehlich (1995, 2008) — breach parameters
- Wahl (2004), USBR DSO-98-04 — peak flow uncertainty analysis
- HEC-RAS dam break guidelines — cross-section spacing
- Manning's n for Brazilian rivers (Doce River study, SciELO)
"""

import math


def calcular_parametros_otimos(h_barr, volume_m3, rio_len_km, dem_resolution=30.0):
    """Calcula todos os parametros otimos a partir dos 4 inputs do usuario.

    Args:
        h_barr: altura da barragem (m)
        volume_m3: volume armazenado (m3)
        rio_len_km: comprimento do rio a modelar (km) — de comprimento_rio()
        dem_resolution: resolucao media do DEM em metros (default 30m Copernicus)

    Returns:
        dict com todos os parametros otimos
    """
    vol_hm3 = volume_m3 * 1e-6

    # ── 1. Comprimento da secao transversal (m) ─────────────────────────
    # Proporcional a energia (altura) + extensao lateral (sqrt volume).
    # HEC-RAS: secoes devem capturar toda a extensao da cheia.
    # Piso 500m (barragens pequenas), teto 8000m (grandes).
    comprimento_secao = max(500, min(8000, int(50 * h_barr + 20 * math.sqrt(vol_hm3))))

    # ── 2. Numero de secoes ──────────────────────────────────────────────
    # ANA/LNEC usa 21 para ~14km. ~1.5 secoes/km da boa resolucao espacial.
    # Minimo 5 (atenuacao precisa de pontos), maximo 30 (runtime).
    n_secoes = max(5, min(30, int(rio_len_km * 1.5) + 1))

    # ── 3. Segmentos de simplificacao ────────────────────────────────────
    # Metade das secoes preserva curvatura sem sobre-simplificar.
    n_simplificacao = max(5, n_secoes // 2)

    # ── 4. Pontos por perfil ─────────────────────────────────────────────
    # Nao ha beneficio em amostrar mais fino que o DEM (Pilotti 2016).
    # Impar para ter ponto central (talvegue).
    n_pontos = max(51, min(201, int(comprimento_secao / dem_resolution) + 1))
    if n_pontos % 2 == 0:
        n_pontos += 1

    # ── 5. Manning n ───────────────────────────────────────────────────────
    # Rios brasileiros: n = 0.028-0.085 (estudo Rio Doce, SciELO).
    # n=0.05 e o valor medio/balanceado para uso rural misto.
    # n=0.067 era o default legado (conservador demais).
    manning_n = 0.05

    # ── 6. Fator de correcao ─────────────────────────────────────────────
    # fc=1 e o mais conservador (sem reducao de altura). Padrao ANA.
    fc = 1

    # ── 7. Metodos ───────────────────────────────────────────────────────
    # ANA-LNEC = max(Froehlich, MMC) — menor incerteza global (Wahl 2004).
    # Overtopping = pior caso conservador para analise automatica.
    # Muskingum-Cunge: propagacao fisica do hidrograma (Ponce 1989).
    metodo_qmax = "ana_lnec"
    modo_ruptura = "overtopping"
    metodo_atenuacao = "muskingum_cunge"
    piso_adaptavel = True

    # ── 8. Buffer do poligono ────────────────────────────────────────────
    # 1.5x resolucao do DEM fecha gaps entre pixels. Minimo 30m.
    buffer_m = max(30.0, dem_resolution * 1.5)

    return {
        "comprimento_secao": comprimento_secao,
        "n_secoes": n_secoes,
        "n_simplificacao": n_simplificacao,
        "n_pontos_perfil": n_pontos,
        "manning_n": manning_n,
        "manning_k": 1.0 / manning_n,  # k = 1/n para calculo interno
        "fc": fc,
        "metodo_qmax": metodo_qmax,
        "modo_ruptura": modo_ruptura,
        "metodo_atenuacao": metodo_atenuacao,
        "piso_adaptavel": piso_adaptavel,
        "buffer_m": buffer_m,
    }


def resumo_parametros(params):
    """Retorna texto legivel com resumo dos parametros calculados."""
    from .config import METODOS_QMAX_LABELS, MODOS_RUPTURA, METODOS_ATENUACAO

    lines = [
        f"Secoes: {params['n_secoes']} x {params['comprimento_secao']}m",
        f"Simplificacao: {params['n_simplificacao']} segmentos",
        f"Pontos por perfil: {params['n_pontos_perfil']}",
        f"Manning n: {params['manning_n']}",
        f"Fator de correcao: {params['fc']}",
        f"Metodo Qmax: {METODOS_QMAX_LABELS.get(params['metodo_qmax'], params['metodo_qmax'])}",
        f"Ruptura: {MODOS_RUPTURA.get(params['modo_ruptura'], params['modo_ruptura'])}",
        f"Atenuacao: {METODOS_ATENUACAO.get(params['metodo_atenuacao'], params['metodo_atenuacao'])}",
        f"Buffer poligono: {params['buffer_m']:.0f}m",
    ]
    return "\n".join(lines)
