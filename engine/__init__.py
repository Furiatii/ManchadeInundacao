"""Engine de calculo para modelagem de rompimento de barragens."""

from .config import (
    DATUM_OPTIONS, DEFAULT_DATUM, DEFAULT_N_SECOES, DEFAULT_N_SIMPLIFICACAO,
    DEFAULT_COMPRIMENTO_SECAO, DEFAULT_MANNING_N, DEFAULT_MANNING_K, DEFAULT_FC,
    DEFAULT_PISO_ADAPTAVEL,
    METODOS_QMAX_LABELS, MODOS_RUPTURA, METODOS_ATENUACAO,
    DPA_LIMITE_ALTO, DPA_LIMITE_MEDIO,
    auto_datum, datum_label_from_epsg,
)
from .hidraulica import (
    comprimento_rio, qmax_barragem, qmax_todos_metodos,
    parametros_brecha, hidrograma_brecha, altura_de_agua_secoes,
    muskingum_cunge_routing,
    classificar_dpa, METODOS_QMAX,
)
from .geometria import (
    simplificar_tracado, secoes_perpendiculares, cotas_secoes,
    convex_hull, check_if_is_inside, clip_raster, get_coordinates,
    exportar_geopandas, transformacao,
)
from .exportacao import (
    gerar_poligono_mancha, mancha_to_shapefile, mancha_to_zip_bytes,
    secoes_to_zip_bytes, gerar_relatorio_csv, relatorio_to_csv_bytes,
)
from .dados_remotos import (
    download_dem, download_dem_anadem, download_dem_copernicus,
    buscar_rios, buscar_rios_ana, buscar_rios_osm,
)
from .auto_params import calcular_parametros_otimos, resumo_parametros
