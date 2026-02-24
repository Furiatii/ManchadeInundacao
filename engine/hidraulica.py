"""Calculos hidraulicos para modelagem de rompimento de barragens.

Todas as funcoes sao puras (sem estado, sem side effects).
Baseado na metodologia simplificada da ANA, com extensoes para
multiplos metodos de ruptura e suporte a barragens pequenas.

Metodos de vazao de pico:
- Froehlich 1995, Froehlich 2008
- MMC (U.S. Army Corps of Engineers)
- ANA-LNEC (max entre Froehlich e MMC)
- Webby 1996
- VTG 1990 (Von Thun & Gillette)

Referencias:
- Rolo et al. (2022), RBRH v27 e8
- Oliveira et al. (2021), XXIV SBRH
- Froehlich (1995, 2008)
- Wahl (2004), USBR DSO-98-04
"""

import math
import numpy as np
import numpy.polynomial.polynomial as poly

G = 9.81  # aceleracao gravitacional (m/s2)

# ── Metodos de vazao de pico ────────────────────────────────────────────────

METODOS_QMAX = [
    'ana_lnec', 'froehlich_1995', 'froehlich_2008',
    'mmc', 'webby_1996', 'vtg_1990',
]


def _qmax_froehlich(altura, volume):
    """Froehlich (1995/2008): Qp = 0.607 * V^0.295 * H^1.24"""
    return 0.607 * (volume ** 0.295 * altura ** 1.24)


def _qmax_mmc(volume):
    """MMC (USACE): Qmax = 0.0039 * V^0.8122"""
    return 0.0039 * volume ** 0.8122


def _qmax_webby(altura, volume):
    """Webby (1996): Qp = 0.0443 * sqrt(g) * V^0.365 * H^1.405"""
    return 0.0443 * math.sqrt(G) * volume ** 0.365 * altura ** 1.405


def _qmax_vtg(altura, volume):
    """Von Thun & Gillette (1990): usa Froehlich para Qp."""
    return _qmax_froehlich(altura, volume)


def qmax_barragem(altura, volume, metodo='ana_lnec'):
    """Calcula a vazao maxima na barragem (m3/s).

    Args:
        altura: altura da barragem (m)
        volume: volume armazenado (m3)
        metodo: 'ana_lnec', 'froehlich_1995', 'froehlich_2008',
                'mmc', 'webby_1996', 'vtg_1990'

    Returns:
        vazao maxima (m3/s)
    """
    if metodo == 'ana_lnec':
        return max(_qmax_froehlich(altura, volume), _qmax_mmc(volume))
    elif metodo in ('froehlich_1995', 'froehlich_2008'):
        return _qmax_froehlich(altura, volume)
    elif metodo == 'mmc':
        return _qmax_mmc(volume)
    elif metodo == 'webby_1996':
        return _qmax_webby(altura, volume)
    elif metodo == 'vtg_1990':
        return _qmax_vtg(altura, volume)
    else:
        raise ValueError(f"Metodo desconhecido: {metodo}")


def qmax_todos_metodos(altura, volume):
    """Calcula Qmax com TODOS os metodos para comparativo.

    Returns:
        dict {metodo: qmax}
    """
    return {m: qmax_barragem(altura, volume, m) for m in METODOS_QMAX}


# ── Parametros de brecha ────────────────────────────────────────────────────

def parametros_brecha(altura, volume, modo_ruptura='overtopping'):
    """Calcula parametros da brecha de ruptura.

    Args:
        altura: altura da barragem (m)
        volume: volume armazenado (m3)
        modo_ruptura: 'overtopping' ou 'piping'

    Returns:
        dict com Tf (tempo de formacao, horas), B (largura, m), K (fator)
    """
    k_fator = 1.0 if modo_ruptura == 'overtopping' else 1.4

    # Froehlich 2008
    tf = 0.0176 * (volume / (G * altura ** 2)) ** 0.5  # horas
    b = 0.27 * k_fator * volume ** 0.5  # metros

    # Froehlich 1995 (alternativo)
    tf_95 = 0.00254 * volume ** 0.53 * altura ** (-0.9)  # horas

    return {
        'Tf_2008_h': tf,
        'Tf_1995_h': tf_95,
        'B_m': b,
        'K': k_fator,
        'modo': modo_ruptura,
    }


# ── Hidrograma de brecha ─────────────────────────────────────────────────────

def hidrograma_brecha(qmax, tf_h, dt_min=1.0, duracao_fator=3.0):
    """Gera hidrograma triangular/exponencial de ruptura da barragem.

    Fase de subida: linear ate Qmax em t = Tf
    Fase de recessao: exponencial decaindo ate ~1% de Qmax

    Args:
        qmax: vazao de pico (m3/s)
        tf_h: tempo de formacao da brecha (horas)
        dt_min: passo de tempo em minutos
        duracao_fator: duracao total = fator * Tf

    Returns:
        tempos: array de tempos em segundos
        vazoes: array de vazoes em m3/s
    """
    tf_s = tf_h * 3600  # converter horas -> segundos
    dt_s = dt_min * 60  # converter minutos -> segundos
    duracao_s = duracao_fator * tf_s

    # Garantir duracao e passo minimos
    tf_s = max(tf_s, 60)
    dt_s = max(dt_s, 10)
    duracao_s = max(duracao_s, 3 * tf_s)

    tempos = np.arange(0, duracao_s + dt_s, dt_s)
    vazoes = np.zeros_like(tempos, dtype=float)

    # Coeficiente de recessao: Q cai a 1% de Qmax no final
    # Q(t) = Qmax * exp(-alpha * (t - tf))
    # Em t = duracao: Q = 0.01*Qmax -> alpha = ln(100) / (duracao - tf)
    t_recessao = duracao_s - tf_s
    if t_recessao > 0:
        alpha = math.log(100) / t_recessao
    else:
        alpha = 1.0

    for i, t in enumerate(tempos):
        if t <= tf_s:
            # Subida linear
            vazoes[i] = qmax * (t / tf_s) if tf_s > 0 else qmax
        else:
            # Recessao exponencial
            vazoes[i] = qmax * math.exp(-alpha * (t - tf_s))

    return tempos, vazoes


# ── Comprimento do rio ──────────────────────────────────────────────────────

def comprimento_rio(volume, piso_adaptavel=True):
    """Calcula o comprimento do rio a ser modelado (km) a partir do volume (m3).

    Args:
        volume: volume armazenado (m3)
        piso_adaptavel: se True, usa piso proporcional ao volume
                        (resolve superestimacao para barragens pequenas).
                        Se False, usa piso fixo de 5.0 km (legado ANA).
    """
    volume_hm = volume * 1e-06
    c = (0.0000000887 * float(volume_hm) ** 3
         - 0.00026 * float(volume_hm) ** 2
         + 0.265 * float(volume_hm)
         + 6.74)

    if piso_adaptavel:
        piso = max(1.0, 0.5 * (volume_hm ** 0.4))
    else:
        piso = 5.0

    return max(piso, min(100.0, c))


# ── Atenuacao de vazao ──────────────────────────────────────────────────────

def qmax_secao(x, q_max_barr, volume, metodo_atenuacao='ana'):
    """Calcula a vazao maxima atenuada em cada secao a distancia x (m).

    Args:
        x: distancia da barragem (m)
        q_max_barr: vazao maxima na barragem (m3/s)
        volume: volume armazenado (m3)
        metodo_atenuacao: 'ana' (padrao) ou 'muskingum_cunge'
    """
    if x == 0:
        return q_max_barr

    volume_hm = volume * 1e-6

    if metodo_atenuacao == 'ana':
        if volume_hm > 6.2:
            return q_max_barr * 10 ** (-0.02 / 1609 * x)
        else:
            a = 0.002 * np.log(volume_hm) + 0.9626
            b = -0.20047 * (volume_hm + 25000) ** -0.5979
            return q_max_barr * a * np.exp(b * x)
    else:
        # Muskingum-Cunge: usa atenuacao ANA para estimativa pontual
        # (o routing completo e feito por muskingum_cunge_routing)
        if volume_hm > 6.2:
            return q_max_barr * 10 ** (-0.02 / 1609 * x)
        else:
            a = 0.002 * np.log(volume_hm) + 0.9626
            b = -0.20047 * (volume_hm + 25000) ** -0.5979
            return q_max_barr * a * np.exp(b * x)


# ── Muskingum-Cunge routing ─────────────────────────────────────────────────

def _mc_celeridade(q_ref, a_ref, bw_ref, r_ref, slope, k_mann):
    """Calcula celeridade da onda de cheia (Ponce & Simons, 1977).

    c = (1/n) * (5/3) * R^(2/3) * S^(1/2)  [aproximacao cinematica]

    Ou equivalente: c = (5/3) * (Q / A) quando dQ/dA ≈ (5/3)*v.
    """
    if a_ref <= 0 or bw_ref <= 0:
        return 1.0  # fallback
    v = q_ref / a_ref
    # Para canais largos: c ≈ (5/3)*v (onda cinematica)
    c = (5.0 / 3.0) * v
    return max(0.5, c)


def muskingum_cunge_routing(hidrograma_entrada, dx, slope, k_mann,
                            areas, raios, larguras, alturas_h):
    """Propaga um hidrograma por um trecho usando Muskingum-Cunge.

    O metodo Muskingum-Cunge e um esquema de diferencas finitas que
    aproxima a equacao de difusao cinematica. Os coeficientes C1, C2, C3
    dependem da geometria do canal e variam com a vazao de referencia.

    Args:
        hidrograma_entrada: tuple (tempos_s, vazoes) do trecho a montante
        dx: comprimento do trecho (m)
        slope: declividade do trecho (m/m)
        k_mann: coeficiente de Manning (1/n)
        areas: lista de areas molhadas para diferentes alturas
        raios: lista de raios hidraulicos para diferentes alturas
        larguras: lista de larguras de superficie para diferentes alturas
        alturas_h: array de alturas (0 a h_max)

    Returns:
        (tempos_s, vazoes_saida): hidrograma propagado na saida do trecho
    """
    tempos, q_in = hidrograma_entrada
    nt = len(tempos)
    q_out = np.zeros(nt)
    q_out[0] = q_in[0]

    # Pre-computar curvas Q-A-B-R para interpolacao rapida
    # Manning: Q = k * A * R^(2/3) * S^(1/2)
    qs_curva = np.array([
        k_mann * areas[i] * raios[i] ** (2.0 / 3.0) * slope ** 0.5
        for i in range(len(areas))
    ])

    for n in range(nt - 1):
        dt = tempos[n + 1] - tempos[n]
        if dt <= 0:
            q_out[n + 1] = q_out[n]
            continue

        # Vazao de referencia: media entre entrada e saida no passo anterior
        q_ref = max(1.0, 0.5 * (q_in[n] + q_out[n]))

        # Interpolar propriedades hidraulicas para q_ref
        if q_ref <= qs_curva[0] or len(qs_curva) < 2:
            a_ref = max(1.0, areas[0])
            r_ref = max(0.01, raios[0])
            bw_ref = max(1.0, larguras[0])
        elif q_ref >= qs_curva[-1]:
            a_ref = areas[-1]
            r_ref = raios[-1]
            bw_ref = larguras[-1]
        else:
            idx = np.searchsorted(qs_curva, q_ref) - 1
            idx = max(0, min(idx, len(qs_curva) - 2))
            frac = (q_ref - qs_curva[idx]) / max(1e-6, qs_curva[idx + 1] - qs_curva[idx])
            frac = max(0.0, min(1.0, frac))
            a_ref = areas[idx] + frac * (areas[idx + 1] - areas[idx])
            r_ref = raios[idx] + frac * (raios[idx + 1] - raios[idx])
            bw_ref = larguras[idx] + frac * (larguras[idx + 1] - larguras[idx])

        # Celeridade da onda
        c = _mc_celeridade(q_ref, a_ref, bw_ref, r_ref, slope, k_mann)

        # Parametros Muskingum
        K_mc = dx / c if c > 0 else dx  # tempo de transito do trecho
        # X: peso entre prisma e cunha (0 a 0.5)
        # X = 0.5 * (1 - Q / (B * S * c * dx))  — Ponce & Yevjevich 1978
        denom = bw_ref * slope * c * dx
        if denom > 0:
            X_mc = 0.5 * (1.0 - q_ref / denom)
        else:
            X_mc = 0.0
        # Limitar X entre 0 e 0.5 (estabilidade)
        X_mc = max(0.0, min(0.5, X_mc))

        # Coeficientes de routing
        denom2 = 2.0 * K_mc * (1.0 - X_mc) + dt
        if denom2 <= 0:
            q_out[n + 1] = q_in[n + 1]
            continue

        C1 = (dt - 2.0 * K_mc * X_mc) / denom2
        C2 = (dt + 2.0 * K_mc * X_mc) / denom2
        C3 = (2.0 * K_mc * (1.0 - X_mc) - dt) / denom2

        # Routing
        q_out[n + 1] = C1 * q_in[n + 1] + C2 * q_in[n] + C3 * q_out[n]
        q_out[n + 1] = max(0.0, q_out[n + 1])

    return tempos, q_out


# ── Funcoes auxiliares de secao transversal ──────────────────────────────────

def _line_coef(p1, p2):
    """Calcula coeficientes a, b de uma reta dados dois pontos."""
    x1, y1 = p1
    x2, y2 = p2
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    return a, b


def _increase_resolution(x, y, n=100):
    """Aumenta a resolucao do perfil da secao por interpolacao linear."""
    x, y = np.array(x), np.array(y)
    ixs, iys = [], []
    for i in range(len(x) - 1):
        ix = np.linspace(x[i], x[i + 1], n)
        a, b = _line_coef((x[i], y[i]), (x[i + 1], y[i + 1]))
        iy = a * ix + b
        ixs.extend(ix)
        iys.extend(iy)
    return np.array(ixs), np.array(iys)


def raio_hidraulico(y, x, h_max, n_alturas=11):
    """Calcula area molhada, raio hidraulico e largura de superficie.

    Returns:
        areas, raios, larguras, alturas
    """
    x, y = _increase_resolution(x, y)
    yt = -1 * y + max(y)
    hs = np.linspace(0, h_max, n_alturas)
    areas = []
    raios = []
    larguras = []

    for h in hs[1:]:
        ytt = yt - (max(yt) - h)
        f = ytt > 0
        ytt, xt = ytt[f], x[f]

        area = np.trapz(y=ytt, x=xt)

        distances = [
            math.hypot(xt[i + 1] - xt[i], ytt[i + 1] - ytt[i])
            for i in range(len(ytt) - 1)
        ]
        perimetro = np.sum(distances)
        areas.append(area)
        raios.append(area / perimetro)

        # Largura de superficie = extensao horizontal da agua
        largura = xt[-1] - xt[0] if len(xt) > 1 else 1.0
        larguras.append(max(1.0, largura))

    return areas, raios, larguras, hs


def manning(a, r, j, k=15):
    """Equacao de Manning: Q = k * A * R^(2/3) * J^(1/2)."""
    a, r = np.array(a), np.array(r)
    return k * a * r ** (2 / 3) * j ** (1 / 2)


def _polyfit(x, y, x_i):
    """Ajusta polinomial de terceiro grau e avalia em x_i."""
    coefs = poly.polyfit(x, y, 3)
    return poly.polyval(x_i, coefs)


# ── Calculo principal ────────────────────────────────────────────────────────

def _declividade_local(ct, ds, h_barr):
    """Calcula declividade local entre secoes consecutivas.

    Para a primeira secao, usa a declividade da barragem ate a secao seguinte.
    Para as demais, usa a declividade entre a secao anterior e a atual.
    Garante um piso minimo de 0.0001 (0.01%) para evitar divisao por zero.
    """
    n = len(ct)
    js = []
    for i in range(n):
        if i == 0:
            # Da barragem (topo = ct[0] + h_barr) ate secao 1
            if n > 1 and ds[1] > ds[0]:
                j = (ct[0] + h_barr - ct[1]) / (ds[1] - ds[0])
            else:
                j = h_barr / max(1, ds[-1])
        else:
            dx = ds[i] - ds[i - 1]
            if dx > 0:
                j = (ct[i - 1] - ct[i]) / dx
            else:
                j = 0.0001
        js.append(max(0.0001, abs(j)))
    return js


def altura_de_agua_secoes(ds, dp, cotas, qmax_barr, volume, h_barr,
                          fc=1, k=15, k_por_secao=None,
                          metodo_atenuacao='ana', tf_h=None,
                          modo_ruptura='overtopping'):
    """Calcula a altura de agua em cada secao transversal.

    Args:
        ds: distancias das secoes ao longo do rio
        dp: distancias dos pontos ao longo de cada secao
        cotas: elevacoes do terreno em cada secao
        qmax_barr: vazao maxima na barragem (m3/s)
        volume: volume armazenado (m3)
        h_barr: altura da barragem (m)
        fc: fator de correcao (1-6)
        k: coeficiente de Manning uniforme
        k_por_secao: lista de K por secao (sobrescreve k uniforme se fornecido)
        metodo_atenuacao: 'ana' ou 'muskingum_cunge'
        tf_h: tempo de formacao da brecha em horas (necessario para M-C)
        modo_ruptura: 'overtopping' ou 'piping'

    Returns:
        alturas_secoes: elevacao da agua em cada secao
        qs: vazao maxima em cada secao
        velocidades: velocidade media (m/s) em cada secao
        tempos_chegada: tempo de chegada da onda (minutos) em cada secao
        hidrogramas: lista de (tempos_s, vazoes) por secao (ou None se ANA)
    """
    n_pontos_perfil = len(cotas[0])
    idx_central = n_pontos_perfil // 2
    ct = [c[idx_central] for c in cotas]

    # Declividade local entre secoes (mais preciso que global)
    js = _declividade_local(ct, ds, h_barr)

    alt_max = h_barr / fc

    # Pre-calcular geometria hidraulica de todas as secoes
    geom_secoes = []
    for idx, elevacoes in enumerate(cotas):
        a, r, bw, h = raio_hidraulico(elevacoes, dp, alt_max)
        geom_secoes.append((a, r, bw, h))

    # ── Determinar Qmax por secao ──────────────────────────────────────────
    hidrogramas = None

    if metodo_atenuacao == 'muskingum_cunge':
        # Calcular Tf se nao fornecido
        if tf_h is None:
            brecha = parametros_brecha(h_barr, volume, modo_ruptura)
            tf_h = brecha['Tf_2008_h']

        # Gerar hidrograma de brecha
        tempos_brecha, vazoes_brecha = hidrograma_brecha(qmax_barr, tf_h)

        # Propagar secao a secao
        hidrogramas = [(tempos_brecha, vazoes_brecha)]  # secao 0 = barragem
        hidrograma_atual = (tempos_brecha, vazoes_brecha)

        qs = [float(np.max(vazoes_brecha))]  # Qmax na secao 0

        for idx in range(1, len(ds)):
            dx = ds[idx] - ds[idx - 1]
            if dx <= 0:
                hidrogramas.append(hidrograma_atual)
                qs.append(qs[-1])
                continue

            k_secao = k_por_secao[idx] if k_por_secao else k
            a_s, r_s, bw_s, h_s = geom_secoes[idx]

            _, q_propagado = muskingum_cunge_routing(
                hidrograma_atual, dx, js[idx], k_secao,
                a_s, r_s, bw_s, h_s,
            )
            hidrograma_atual = (tempos_brecha, q_propagado)
            hidrogramas.append(hidrograma_atual)
            qs.append(float(np.max(q_propagado)))
    else:
        # Atenuacao empirica ANA
        qs = [qmax_secao(d, qmax_barr, volume, 'ana') for d in ds]

    # ── Calcular altura, velocidade por secao ──────────────────────────────
    alturas_ref = None
    alturas_secoes = []
    velocidades = []
    for idx in range(len(cotas)):
        a, r, bw, h = geom_secoes[idx]
        if alturas_ref is None:
            alturas_ref = h

        k_secao = k_por_secao[idx] if k_por_secao else k
        qs_s = [manning(a[i], r[i], js[idx], k_secao) for i in range(len(a))]
        qs_s = np.insert(qs_s, 0, 0)

        nivel = _polyfit(qs_s, alturas_ref, qs[idx])
        nivel = nivel + ct[idx]
        alturas_secoes.append(nivel)

        # Velocidade media: v = Q / A
        h_agua = nivel - ct[idx]
        if h_agua > 0 and len(a) > 0:
            a_interp = np.interp(h_agua, alturas_ref[1:], a)
            if a_interp > 0:
                velocidades.append(qs[idx] / a_interp)
            else:
                velocidades.append(0.0)
        else:
            velocidades.append(0.0)

    # ── Tempo de chegada ───────────────────────────────────────────────────
    if metodo_atenuacao == 'muskingum_cunge' and hidrogramas:
        # Tempo de chegada real: quando Q atinge 5% do pico local
        tempos_chegada = []
        for idx, (t_hid, q_hid) in enumerate(hidrogramas):
            q_pico = np.max(q_hid)
            limiar = 0.05 * q_pico
            indices = np.where(q_hid >= limiar)[0]
            if len(indices) > 0:
                tempos_chegada.append(t_hid[indices[0]] / 60.0)  # seg -> min
            else:
                tempos_chegada.append(0.0)
    else:
        # Estimativa por celeridade (metodo ANA)
        tempos_chegada = [0.0]
        t_acum = 0.0
        for i in range(1, len(ds)):
            dx = ds[i] - ds[i - 1]
            v_medio = (velocidades[i - 1] + velocidades[i]) / 2
            if v_medio > 0:
                celeridade = 1.5 * v_medio
                t_acum += dx / celeridade
            else:
                t_acum += dx / 1.0
            tempos_chegada.append(t_acum / 60)

    return alturas_secoes, qs, velocidades, tempos_chegada, hidrogramas


# ── DPA (Dano Potencial Associado) — CNRH 241/2024 ──────────────────────────

def classificar_dpa(pontuacao_total):
    """Classifica DPA conforme CNRH 241/2024.

    Args:
        pontuacao_total: soma dos pontos dos criterios preenchidos

    Returns:
        'Alto', 'Medio' ou 'Baixo'
    """
    if pontuacao_total >= 13:
        return 'Alto'
    elif pontuacao_total >= 7:
        return 'Medio'
    else:
        return 'Baixo'
