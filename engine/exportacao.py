"""Exportacao de resultados: shapefile da mancha e relatorio CSV.

Substitui a abordagem PyVista (intersecao 3D) por Shapely 2D.
"""

import io
import zipfile
import numpy as np
import pandas as pd
import geopandas
from shapely.geometry import Point, MultiPoint, Polygon, LineString
from shapely.ops import unary_union

from .config import DEFAULT_DATUM


def _limites_inundacao(cotas_secao, nivel_agua, n_pts):
    """Encontra limites de inundacao expandindo a partir do talvegue real.

    Usa o ponto mais baixo proximo ao centro (nao o centro geometrico)
    como ponto de partida, garantindo que a inundacao comece no rio real.
    Expande para cada lado e para quando encontra terreno acima da agua.

    Returns:
        (i_esq, i_dir, centro) ou None se secao nao esta inundada.
    """
    # Buscar o talvegue real: ponto mais baixo na metade central da secao
    margem = n_pts // 4
    ini = max(0, n_pts // 2 - margem)
    fim = min(n_pts, n_pts // 2 + margem + 1)
    centro = ini + int(np.argmin(cotas_secao[ini:fim]))

    if cotas_secao[centro] > nivel_agua:
        return None

    # Expandir para ESQUERDA — parar no primeiro ponto acima da agua
    i_esq = centro
    for i in range(centro - 1, -1, -1):
        if cotas_secao[i] > nivel_agua:
            break
        i_esq = i

    # Expandir para DIREITA — mesma logica
    i_dir = centro
    for i in range(centro + 1, n_pts):
        if cotas_secao[i] > nivel_agua:
            break
        i_dir = i

    return i_esq, i_dir, centro


def gerar_poligono_mancha(cotas, alturas_secoes, xs, ys, buffer_m=50):
    """Gera o poligono da mancha de inundacao.

    Abordagem: buffer individual de cada segmento inundado + corredor
    central ao longo do rio. Isso evita que linhas retas entre secoes
    cruzem terreno elevado (problema da mancha "subindo morro").

    Args:
        cotas: elevacoes do terreno por secao
        alturas_secoes: nivel da agua por secao
        xs: coordenadas X dos pontos por secao
        ys: coordenadas Y dos pontos por secao
        buffer_m: raio de buffer em metros para suavizacao

    Returns:
        geometry: poligono Shapely da mancha de inundacao, ou None
    """
    partes = []
    centros_rio = []

    for idx, nivel_agua in enumerate(alturas_secoes):
        n_pts = len(xs[idx])
        lim = _limites_inundacao(cotas[idx], nivel_agua, n_pts)
        if lim is None:
            continue

        i_esq, i_dir, centro = lim

        # Segmento inundado nesta secao (LineString)
        coords = [(xs[idx][i], ys[idx][i]) for i in range(i_esq, i_dir + 1)]
        if len(coords) >= 2:
            partes.append(LineString(coords).buffer(buffer_m))
        elif len(coords) == 1:
            partes.append(Point(coords[0]).buffer(buffer_m))

        # Ponto central do rio (para conectar secoes)
        centros_rio.append((xs[idx][centro], ys[idx][centro]))

    if not partes:
        return None

    # Corredor ao longo do rio conecta as secoes entre si
    if len(centros_rio) >= 2:
        partes.append(LineString(centros_rio).buffer(buffer_m * 0.5))

    return unary_union(partes)


def mancha_to_shapefile(poligono, out_path, datum=DEFAULT_DATUM):
    """Exporta o poligono da mancha como shapefile."""
    gdf = geopandas.GeoDataFrame(
        {'descricao': ['Mancha de inundacao']},
        geometry=[poligono],
        crs=f'EPSG:{datum}'
    )
    gdf = gdf.to_crs(epsg=4326)
    gdf.to_file(out_path)


def mancha_to_zip_bytes(poligono, datum=DEFAULT_DATUM):
    """Exporta o poligono da mancha como ZIP em memoria (para download Streamlit).

    Returns:
        bytes do arquivo ZIP contendo .shp, .shx, .dbf, .prj
    """
    import tempfile
    import os
    import glob

    gdf = geopandas.GeoDataFrame(
        {'descricao': ['Mancha de inundacao']},
        geometry=[poligono],
        crs=f'EPSG:{datum}'
    )
    gdf = gdf.to_crs(epsg=4326)

    with tempfile.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, 'mancha_inundacao')
        gdf.to_file(f'{base}.shp')

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            for f in glob.glob(f'{base}.*'):
                zf.write(f, os.path.basename(f))
        buf.seek(0)
        return buf.getvalue()


def secoes_to_zip_bytes(secoes_gdf, datum=DEFAULT_DATUM):
    """Exporta as secoes como ZIP em memoria (para download Streamlit)."""
    import tempfile
    import os
    import glob

    gdf = secoes_gdf.set_crs(f'EPSG:{datum}', allow_override=True)
    gdf = gdf.to_crs(epsg=4326)

    with tempfile.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, 'secoes_transversais')
        gdf.to_file(f'{base}.shp')

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            for f in glob.glob(f'{base}.*'):
                zf.write(f, os.path.basename(f))
        buf.seek(0)
        return buf.getvalue()


def gerar_relatorio_csv(ds, qs, alturas_secoes, cotas, n_secoes,
                        velocidades=None, tempos_chegada=None):
    """Gera relatorio hidraulico como DataFrame.

    Args:
        ds: distancias das secoes
        qs: vazoes maximas por secao
        alturas_secoes: nivel da agua por secao
        cotas: elevacoes do terreno por secao
        n_secoes: numero de secoes
        velocidades: velocidade media (m/s) por secao (opcional)
        tempos_chegada: tempo de chegada da onda (min) por secao (opcional)

    Returns:
        DataFrame com colunas hidraulicas
    """
    n_pontos = len(cotas[0])
    idx_central = n_pontos // 2
    ct = [c[idx_central] for c in cotas]

    nomes = [f'secao {i}' for i in range(n_secoes)]
    alturas_agua = [alturas_secoes[i] - ct[i] for i in range(n_secoes)]

    data = {
        'Secao': nomes,
        'Distancia (m)': ds,
        'Vazao (m3/s)': qs,
        'Altura de agua (m)': alturas_agua,
    }

    if velocidades is not None:
        data['Velocidade (m/s)'] = [round(v, 2) for v in velocidades]
    if tempos_chegada is not None:
        data['Tempo de chegada (min)'] = [round(t, 1) for t in tempos_chegada]

    return pd.DataFrame(data)


def relatorio_to_csv_bytes(df):
    """Converte DataFrame do relatorio para bytes CSV (para download Streamlit)."""
    return df.to_csv(index=False).encode('utf-8')
