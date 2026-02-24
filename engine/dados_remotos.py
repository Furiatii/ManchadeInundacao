"""Download automatico de DEM e hidrografia a partir de APIs publicas.

Fontes:
- DEM preferencial: ANADEM v1 (ANA/UFRGS) — bare-earth 30m, sem vegetacao
- DEM fallback: Copernicus GLO-30 via AWS Open Data (DSM 30m, com vegetacao)
- Hidrografia: ANA SNIRH (ArcGIS REST, sem autenticacao)
- Fallback hidrografia: OpenStreetMap Overpass API
"""

import math
import tempfile
import requests
import rasterio
from rasterio.merge import merge
from shapely.geometry import LineString, Point


# ── DEM — ANADEM v1 (ANA/UFRGS, bare-earth 30m) ─────────────────────────────
# Tiles organizados por MGRS Grid Zone Designator (ex: 23K, 22J)
# Bare-earth: vegetacao e edificacoes removidas → melhor para hidrologia

_ANADEM_BASE = "https://metadados.snirh.gov.br/files/anadem_v1_tiles"
_MGRS_BANDS = "CDEFGHJKLMNPQRSTUVWX"  # sem I, sem O


def _mgrs_gzd(lat, lon):
    """Retorna o Grid Zone Designator MGRS (ex: 23K) para lat/lon."""
    zone = int((lon + 180) / 6) + 1
    band_idx = int((lat + 80) / 8)
    band_idx = max(0, min(band_idx, len(_MGRS_BANDS) - 1))
    return f"{zone}{_MGRS_BANDS[band_idx]}"


def _tiles_for_bbox_anadem(south, north, west, east):
    """URLs dos tiles ANADEM para um bbox."""
    gzds = set()
    for lat in [south, north]:
        for lon in [west, east]:
            gzds.add(_mgrs_gzd(lat, lon))
    return [f"{_ANADEM_BASE}/anadem_v1_{gzd}.tif" for gzd in gzds]


def download_dem_anadem(lat, lon, buffer_deg=0.15):
    """Baixa DEM ANADEM (bare-earth 30m) ao redor de um ponto.

    Args:
        lat: latitude da barragem (graus decimais)
        lon: longitude da barragem (graus decimais)
        buffer_deg: margem em graus ao redor do ponto

    Returns:
        path: caminho do arquivo GeoTIFF temporario
    """
    south, north = lat - buffer_deg, lat + buffer_deg
    west, east = lon - buffer_deg, lon + buffer_deg

    urls = _tiles_for_bbox_anadem(south, north, west, east)

    datasets = []
    env = rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
        GDAL_HTTP_TIMEOUT="60",
        GDAL_HTTP_CONNECTTIMEOUT="15",
    )
    with env:
        for url in urls:
            try:
                ds = rasterio.open(url)
                datasets.append(ds)
            except Exception:
                continue

    if not datasets:
        raise RuntimeError("ANADEM indisponivel para esta regiao.")

    from rasterio.windows import from_bounds

    if len(datasets) == 1:
        src = datasets[0]
        window = from_bounds(west, south, east, north, src.transform)
        data = src.read(window=window)
        transform = src.window_transform(window)

        profile = src.profile.copy()
        profile.update(
            height=data.shape[1],
            width=data.shape[2],
            transform=transform,
        )
        tmp = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
        with rasterio.open(tmp.name, 'w', **profile) as dst:
            dst.write(data)
        src.close()
        tmp.close()
        return tmp.name
    else:
        mosaic, transform = merge(datasets, bounds=(west, south, east, north))
        profile = datasets[0].profile.copy()
        profile.update(
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform,
        )
        for ds in datasets:
            ds.close()

        tmp = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
        with rasterio.open(tmp.name, 'w', **profile) as dst:
            dst.write(mosaic)
        tmp.close()
        return tmp.name


def download_dem(lat, lon, buffer_deg=0.15):
    """Baixa DEM: tenta ANADEM (bare-earth) primeiro, depois Copernicus.

    Returns:
        (path, fonte): caminho do GeoTIFF e nome da fonte usada
    """
    try:
        path = download_dem_anadem(lat, lon, buffer_deg)
        return path, "ANADEM (bare-earth 30m)"
    except Exception:
        pass

    path = download_dem_copernicus(lat, lon, buffer_deg)
    return path, "Copernicus GLO-30 (DSM 30m)"


# ── DEM — Copernicus 30m via AWS (fallback) ──────────────────────────────────

_COP30_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


def _tile_name_cop30(lat_floor, lon_floor):
    """Monta o nome do tile Copernicus 30m a partir de lat/lon inteiros."""
    ns = "N" if lat_floor >= 0 else "S"
    ew = "E" if lon_floor >= 0 else "W"
    lat_str = f"{ns}{abs(lat_floor):02d}_00"
    lon_str = f"{ew}{abs(lon_floor):03d}_00"
    name = f"Copernicus_DSM_COG_10_{lat_str}_{lon_str}_DEM"
    return f"{_COP30_BASE}/{name}/{name}.tif"


def _tiles_for_bbox(south, north, west, east):
    """Retorna URLs de todos os tiles Copernicus que cobrem o bbox."""
    urls = []
    for lat in range(math.floor(south), math.floor(north) + 1):
        for lon in range(math.floor(west), math.floor(east) + 1):
            urls.append(_tile_name_cop30(lat, lon))
    return urls


def download_dem_copernicus(lat, lon, buffer_deg=0.15):
    """Baixa DEM Copernicus 30m ao redor de um ponto.

    Args:
        lat: latitude da barragem (graus decimais)
        lon: longitude da barragem (graus decimais)
        buffer_deg: margem em graus ao redor do ponto (~0.15 deg = ~16 km)

    Returns:
        path: caminho do arquivo GeoTIFF temporario
    """
    south, north = lat - buffer_deg, lat + buffer_deg
    west, east = lon - buffer_deg, lon + buffer_deg

    urls = _tiles_for_bbox(south, north, west, east)

    datasets = []
    env = rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    )
    with env:
        for url in urls:
            try:
                ds = rasterio.open(url)
                datasets.append(ds)
            except Exception:
                continue

    if not datasets:
        raise RuntimeError(
            "Nao foi possivel baixar nenhum tile DEM Copernicus. "
            "Verifique a conexao com a internet e as coordenadas."
        )

    if len(datasets) == 1:
        # Tile unico — salvar direto
        src = datasets[0]
        tmp = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
        from rasterio.windows import from_bounds
        window = from_bounds(west, south, east, north, src.transform)
        data = src.read(window=window)
        transform = src.window_transform(window)

        profile = src.profile.copy()
        profile.update(
            height=data.shape[1],
            width=data.shape[2],
            transform=transform,
        )
        with rasterio.open(tmp.name, 'w', **profile) as dst:
            dst.write(data)
        src.close()
        tmp.close()
        return tmp.name
    else:
        # Multiplos tiles — merge
        mosaic, transform = merge(datasets, bounds=(west, south, east, north))
        profile = datasets[0].profile.copy()
        profile.update(
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform,
        )
        for ds in datasets:
            ds.close()

        tmp = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
        with rasterio.open(tmp.name, 'w', **profile) as dst:
            dst.write(mosaic)
        tmp.close()
        return tmp.name


# ── Hidrografia — ANA SNIRH ─────────────────────────────────────────────────

_ANA_HIDRO_URL = (
    "https://www.snirh.gov.br/arcgis/rest/services/"
    "DADOSABERTOS/Hidrografia/MapServer/0/query"
)


def buscar_rios_ana(lat, lon, buffer_deg=0.1, timeout=60):
    """Consulta rios proximos via ANA SNIRH (ArcGIS REST).

    Args:
        lat, lon: coordenadas da barragem
        buffer_deg: margem de busca em graus

    Returns:
        lista de dicts com nome, coordenadas, area de drenagem
    """
    params = {
        "geometry": (
            f"{lon - buffer_deg},{lat - buffer_deg},"
            f"{lon + buffer_deg},{lat + buffer_deg}"
        ),
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "inSR": "4326",
        "outSR": "4326",
        "outFields": "NORIO,COBACIA,COCURSODAG,NUAREAMONT",
        "returnGeometry": "true",
        "f": "geojson",
    }

    resp = requests.get(_ANA_HIDRO_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    rios = []
    pt = Point(lon, lat)
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])
        if not coords or len(coords) < 2:
            continue

        linha = LineString(coords)
        rios.append({
            "nome": props.get("NORIO") or "Sem nome",
            "bacia": props.get("COBACIA", ""),
            "area_drenagem_km2": props.get("NUAREAMONT"),
            "coordenadas": coords,
            "distancia_km": pt.distance(linha) * 111,  # aprox graus -> km
        })

    rios.sort(key=lambda r: r["distancia_km"])
    return rios


# ── Fallback Hidrografia — OpenStreetMap Overpass ────────────────────────────

_OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]


def buscar_rios_osm(lat, lon, raio_m=10000, timeout=60):
    """Consulta rios proximos via Overpass API (OpenStreetMap).

    Args:
        lat, lon: coordenadas da barragem
        raio_m: raio de busca em metros

    Returns:
        lista de dicts com nome e coordenadas
    """
    # Buscar apenas rios (nao streams) para reduzir carga no servidor
    query = f"""
    [out:json][timeout:{timeout}];
    (
      way["waterway"="river"](around:{raio_m},{lat},{lon});
      relation["waterway"="river"](around:{raio_m},{lat},{lon});
    );
    out geom;
    """

    data = None
    last_err = None
    for mirror in _OVERPASS_MIRRORS:
        try:
            resp = requests.post(mirror, data={"data": query}, timeout=timeout + 10)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:
            last_err = e
            continue

    if data is None:
        raise last_err or RuntimeError("Todos os mirrors Overpass falharam")

    rios = []
    pt = Point(lon, lat)
    for elem in data.get("elements", []):
        nome = elem.get("tags", {}).get("name", "Sem nome")
        geom_pts = elem.get("geometry", [])
        if len(geom_pts) < 2:
            continue
        coords = [(p["lon"], p["lat"]) for p in geom_pts]
        linha = LineString(coords)
        rios.append({
            "nome": nome,
            "bacia": "",
            "area_drenagem_km2": None,
            "coordenadas": coords,
            "distancia_km": pt.distance(linha) * 111,
        })

    rios.sort(key=lambda r: r["distancia_km"])
    return rios


def buscar_rios(lat, lon, buffer_deg=0.1):
    """Busca rios usando ANA como primario, OSM como fallback.

    Returns:
        (lista de rios, fonte, erros)
    """
    erros = []

    try:
        rios = buscar_rios_ana(lat, lon, buffer_deg=buffer_deg)
        if rios:
            return rios, "ANA/SNIRH", ""
    except Exception as e:
        erros.append(f"ANA: {e}")

    # Fallback OSM com raio generoso
    try:
        raio_m = max(10000, int(buffer_deg * 111000))
        rios = buscar_rios_osm(lat, lon, raio_m=raio_m)
        if rios:
            return rios, "OpenStreetMap", ""
    except Exception as e:
        erros.append(f"OSM: {e}")

    return [], "nenhuma", "; ".join(erros)
