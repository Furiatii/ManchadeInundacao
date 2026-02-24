"""Operacoes espaciais para modelagem de rompimento de barragens.

Inclui: transformacao de coordenadas, simplificacao de tracado,
secoes perpendiculares, leitura de cotas DEM, clip de raster.
"""

import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString, box
from shapely.ops import unary_union
from pyproj import Transformer
import rasterio.mask
import rasterio
import geopandas

from .config import DEFAULT_DATUM, DEFAULT_N_SECOES, DEFAULT_COMPRIMENTO_SECAO, DEFAULT_N_PONTOS_PERFIL


def transformacao(x, y, d_to_m, datum=DEFAULT_DATUM):
    """Transforma coordenadas entre graus (WGS84) e metros (SIRGAS 2000 UTM)."""
    if d_to_m:
        in_crs, out_crs = "epsg:4326", f"epsg:{datum}"
    else:
        in_crs, out_crs = f"epsg:{datum}", "epsg:4326"
    transformer = Transformer.from_crs(in_crs, out_crs)
    return transformer.transform(x, y)


def pontos_tracado(linha, n=DEFAULT_N_SECOES):
    """Retorna n pontos equidistantes ao longo de uma linha."""
    distances = np.linspace(0, linha.length, n)
    points = [linha.interpolate(d) for d in distances]
    return points, distances


def simplificar_tracado(tracado, n, datum=DEFAULT_DATUM):
    """Simplifica o tracado do rio em n segmentos de reta."""
    # Normalizar colunas para o formato esperado (Name, Description, geometry)
    tracado = tracado.to_crs(epsg=datum)
    if 'Name' not in tracado.columns:
        tracado['Name'] = ''
    if 'Description' not in tracado.columns:
        tracado['Description'] = ''
    # Manter apenas as colunas necessarias + geometry
    tracado = tracado[['Name', 'Description', 'geometry']].copy()

    distances = np.linspace(0, tracado.length, n)
    points = [tracado.interpolate(d) for d in distances]
    x = [p.x[0] for p in points]
    y = [p.y[0] for p in points]

    ls = LineString([Point(i, j) for i, j in zip(x, y)])
    row = geopandas.GeoDataFrame(
        [{'Name': 'tracado simplificado', 'Description': ''}],
        geometry=[ls],
        crs=tracado.crs,
    )
    tracado = pd.concat([tracado, row], ignore_index=True)
    return tracado


def _split_linha(tracado):
    """Segmenta o tracado simplificado em segmentos individuais."""
    coords = tracado.iloc[1]['geometry'].coords[:]
    return [
        LineString([coords[i], coords[i + 1]])
        for i in range(len(coords) - 1)
    ]


def _perpendicular(linha, ponto, comprimento):
    """Traca uma perpendicular a uma reta em um ponto dado.

    Usa vetor unitario ao inves de slope para evitar divisao por zero
    e funcionar com segmentos em qualquer direcao.
    """
    rp1, rp2 = linha.coords[:][0], linha.coords[:][1]
    dx = rp2[0] - rp1[0]
    dy = rp2[1] - rp1[1]
    length = np.sqrt(dx ** 2 + dy ** 2)
    if length < 1e-10:
        # Segmento degenerado — retorna perpendicular horizontal
        dx, dy = 1.0, 0.0
    else:
        dx /= length
        dy /= length

    # Vetor perpendicular (rotacao 90 graus)
    perp_dx = -dy
    perp_dy = dx

    B = ponto.coords[:][0]
    half = comprimento / 2
    C = (B[0] + perp_dx * half, B[1] + perp_dy * half)
    D = (B[0] - perp_dx * half, B[1] - perp_dy * half)
    return LineString([C, D])


def secoes_perpendiculares(tracado, n=DEFAULT_N_SECOES, comprimento=DEFAULT_COMPRIMENTO_SECAO):
    """Traca n secoes perpendiculares equidistantes ao longo do tracado simplificado."""
    p, d = pontos_tracado(tracado.iloc[1]['geometry'], n)
    segmentos = _split_linha(tracado)
    tol = 1e-8

    rows = []
    for i, point in enumerate(p):
        for line in segmentos:
            if line.distance(point) < tol:
                perp = _perpendicular(line, point, comprimento)
                rows.append({
                    'Name': f'secao {i}',
                    'Description': str(d[i]),
                    'geometry': perp,
                })

    if rows:
        secoes_gdf = geopandas.GeoDataFrame(rows, crs=tracado.crs)
        tracado = pd.concat([tracado, secoes_gdf], ignore_index=True)

    return tracado, d


def cotas_secoes(tracado, dem, datum=DEFAULT_DATUM, n_pontos=DEFAULT_N_PONTOS_PERFIL):
    """Calcula a altimetria das secoes a partir do DEM.

    Returns:
        cotas: lista de elevacoes por secao
        d: distancias ao longo de cada secao
        xs: coordenadas X por secao
        ys: coordenadas Y por secao
    """
    cotas = []
    xs = []
    ys = []
    for linha in tracado.iloc[2:]['geometry']:
        p, d = pontos_tracado(linha, n=n_pontos)
        p = [pt.coords[:][0] for pt in p]
        x = [pt[0] for pt in p]
        y = [pt[1] for pt in p]
        xs.append(x)
        ys.append(y)

        pt_wgs = transformacao(x, y, d_to_m=False, datum=datum)
        pt_pairs = [(lat, lon) for lon, lat in zip(pt_wgs[0], pt_wgs[1])]
        cota = [val[0] for val in dem.sample(pt_pairs)]
        cotas.append(cota)

    return cotas, d, xs, ys


def convex_hull(secoes_gdf):
    """Retorna o convex hull em torno das secoes transversais."""
    geoms = geopandas.GeoSeries(list(secoes_gdf.iloc[2:].geometry))
    return unary_union(geoms).convex_hull


def check_if_is_inside(chull, x, y):
    """Retorna mascara booleana dos pontos contidos no poligono."""
    pts = [Point(xi, yi) for xi, yi in zip(x, y)]
    return np.array([chull.contains(p) for p in pts])


def clip_raster(secs, dem, out_file, datum=DEFAULT_DATUM):
    """Corta o DEM ao bounding box das secoes."""
    secs = secs.set_crs(f'EPSG:{datum}', allow_override=True)
    secs_wgs = secs.to_crs(epsg=4326)
    bounds = secs_wgs.bounds
    bbox = box(
        bounds['minx'].min(), bounds['miny'].min(),
        bounds['maxx'].max(), bounds['maxy'].max()
    )
    out_image, out_transform = rasterio.mask.mask(dem, [bbox], crop=True, nodata=-999)
    out_meta = dem.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
    })
    with rasterio.open(out_file, 'w', **out_meta) as dest:
        dest.write(out_image)


def get_coordinates(clipado, datum=DEFAULT_DATUM):
    """Extrai coordenadas e cotas do raster cortado."""
    w, h = clipado.width, clipado.height
    ij = []
    for i in range(h):
        for j in range(w):
            ij.append(clipado.xy(i, j, offset='center'))

    z = clipado.read(1).flatten()
    valid = z > 0

    x = np.array([pt[0] for pt in ij])
    y = np.array([pt[1] for pt in ij])
    coords = transformacao(y, x, d_to_m=True, datum=datum)
    xcoords = np.array(coords[0])
    ycoords = np.array(coords[1])

    return xcoords[valid], ycoords[valid], z[valid]


def exportar_geopandas(gdf, nome_do_arquivo, datum=DEFAULT_DATUM):
    """Exporta GeoDataFrame como shapefile em WGS84."""
    gdf = gdf.set_crs(f'EPSG:{datum}', allow_override=True)
    gdf = gdf.to_crs(epsg=4326)
    gdf.to_file(nome_do_arquivo)
