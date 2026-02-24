"""Exportacao de resultados: shapefile da mancha e relatorio CSV.

Substitui a abordagem PyVista (intersecao 3D) por Shapely 2D.
"""

import io
import zipfile
import numpy as np
import pandas as pd
import geopandas
from shapely.geometry import Point, MultiPoint, Polygon
from shapely.ops import unary_union

from .config import DEFAULT_DATUM


def gerar_poligono_mancha(cotas, alturas_secoes, xs, ys, buffer_m=50):
    """Gera o poligono da mancha de inundacao.

    Conecta os limites laterais de inundacao de cada secao (margem esquerda
    e margem direita) formando um poligono continuo ao longo do rio.
    Aplica buffer para suavizar e fechar gaps entre secoes.

    Fallback: se menos de 2 secoes tem inundacao, usa buffer de pontos.

    Args:
        cotas: elevacoes do terreno por secao
        alturas_secoes: nivel da agua por secao
        xs: coordenadas X dos pontos por secao
        ys: coordenadas Y dos pontos por secao
        buffer_m: raio de buffer em metros para suavizacao

    Returns:
        geometry: poligono Shapely da mancha de inundacao, ou None
    """
    # Para cada secao, encontrar limites esquerdo e direito da inundacao
    margem_esq = []
    margem_dir = []

    for idx, nivel_agua in enumerate(alturas_secoes):
        inundados = [i for i in range(len(xs[idx])) if cotas[idx][i] <= nivel_agua]
        if not inundados:
            continue
        i_esq = inundados[0]
        i_dir = inundados[-1]
        margem_esq.append((xs[idx][i_esq], ys[idx][i_esq]))
        margem_dir.append((xs[idx][i_dir], ys[idx][i_dir]))

    # Se pelo menos 2 secoes tem inundacao, criar poligono por contorno
    if len(margem_esq) >= 2:
        contorno = margem_esq + list(reversed(margem_dir))
        poly = Polygon(contorno)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly.buffer(buffer_m)

    # Fallback: buffer de pontos individuais
    pontos_inundados = []
    for idx, nivel_agua in enumerate(alturas_secoes):
        for i in range(len(xs[idx])):
            if cotas[idx][i] <= nivel_agua:
                pontos_inundados.append(Point(xs[idx][i], ys[idx][i]))

    if not pontos_inundados:
        return None

    buffered = [p.buffer(buffer_m) for p in pontos_inundados]
    return unary_union(buffered)


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
