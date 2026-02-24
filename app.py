"""Mancha de Inundacao v2 — Interface Streamlit."""

import streamlit as st
import tempfile
import os

import fiona
fiona.supported_drivers['KML'] = 'rw'

import rasterio
import geopandas
import pandas as pd
import folium
from shapely.geometry import LineString, Point
from shapely.ops import substring as shapely_substring
from streamlit_folium import st_folium

from engine import (
    DATUM_OPTIONS, DEFAULT_DATUM, DEFAULT_N_SECOES, DEFAULT_N_SIMPLIFICACAO,
    DEFAULT_COMPRIMENTO_SECAO, DEFAULT_MANNING_K, DEFAULT_FC,
    DEFAULT_PISO_ADAPTAVEL,
    METODOS_QMAX_LABELS, MODOS_RUPTURA, METODOS_ATENUACAO,
    DPA_LIMITE_ALTO, DPA_LIMITE_MEDIO,
    comprimento_rio, qmax_barragem, qmax_todos_metodos,
    parametros_brecha, altura_de_agua_secoes,
    classificar_dpa, METODOS_QMAX,
    simplificar_tracado, secoes_perpendiculares, cotas_secoes,
    convex_hull, check_if_is_inside, clip_raster, get_coordinates,
    gerar_poligono_mancha, mancha_to_zip_bytes, secoes_to_zip_bytes,
    gerar_relatorio_csv, relatorio_to_csv_bytes,
    download_dem, download_dem_copernicus, buscar_rios,
    auto_datum, datum_label_from_epsg,
    calcular_parametros_otimos, resumo_parametros,
)

st.set_page_config(page_title="Mancha de Inundacao", layout="wide")


# ── Helper: Recortar rio a jusante da barragem ───────────────────────────────

def _recortar_rio_jusante(rio_line, lat, lon, crio_km):
    """Recorta o rio a jusante da barragem, conectando barragem ao rio.

    1. Projeta a barragem sobre o rio (ponto mais proximo)
    2. Adiciona segmento reto barragem → rio (se a barragem nao esta sobre o rio)
    3. Recorta trecho a jusante com comprimento = crio_km

    Determina jusante pela extremidade mais distante da barragem.
    """
    dam_pt = Point(lon, lat)
    dist_proj = rio_line.project(dam_pt)
    nearest_pt = rio_line.interpolate(dist_proj)

    # Converter crio_km para graus (~111 km/grau)
    crio_deg = crio_km / 111.0

    # Determinar jusante: extremidade mais longe da barragem
    d_inicio = dam_pt.distance(Point(rio_line.coords[0]))
    d_fim = dam_pt.distance(Point(rio_line.coords[-1]))

    if d_fim >= d_inicio:
        # Jusante = em direcao ao fim da linha
        fim = min(dist_proj + crio_deg, rio_line.length)
        trecho = shapely_substring(rio_line, dist_proj, fim)
    else:
        # Jusante = em direcao ao inicio (rio invertido no dataset)
        inicio = max(0, dist_proj - crio_deg)
        trecho = shapely_substring(rio_line, inicio, dist_proj)
        trecho = LineString(list(trecho.coords)[::-1])

    # Se muito curto (rio pequeno), usar o que tiver
    if trecho.length < 1e-6:
        return rio_line

    # Conectar barragem ao rio: adicionar segmento reto barragem → ponto no rio
    dist_dam_to_river = dam_pt.distance(nearest_pt)
    if dist_dam_to_river > 1e-5:  # barragem nao esta sobre o rio (~1m)
        trecho = LineString([(lon, lat)] + list(trecho.coords))

    return trecho


# ── Helper: Pipeline automatico ──────────────────────────────────────────────

def _executar_modo_automatico(lat, lon, h, v):
    """Executa todas as etapas do modelo automaticamente."""
    progress = st.progress(0, text="Iniciando analise...")

    try:
        # ── Etapa 1: Parametros basicos ──────────────────────────────────
        progress.progress(5, text="Calculando parametros basicos...")
        datum = auto_datum(lon)
        st.session_state['datum_auto'] = datum

        crio_km = comprimento_rio(v, piso_adaptavel=True)
        qmax_val = qmax_barragem(h, v, metodo='ana_lnec')
        brecha = parametros_brecha(h, v, modo_ruptura='overtopping')
        todos_qmax = qmax_todos_metodos(h, v)

        st.session_state.update({
            'crio': crio_km, 'qmax_barr': qmax_val, 'brecha': brecha,
            'todos_qmax': todos_qmax, 'h': h, 'v': v, 'lat': lat, 'lon': lon,
        })

        # ── Etapa 2: DEM ─────────────────────────────────────────────────
        progress.progress(10, text="Baixando DEM (ANADEM bare-earth → Copernicus)...")
        buffer_deg = max(0.1, crio_km / 111 * 0.7)
        dem_path, dem_fonte = download_dem(lat, lon, buffer_deg=buffer_deg)
        st.session_state['dem_path'] = dem_path
        st.session_state['dem_fonte'] = dem_fonte

        with rasterio.open(dem_path) as src:
            dem_res = abs(src.transform[0])
            st.session_state['dem_res'] = (dem_res, abs(src.transform[4]))
            st.session_state['dem_crs'] = str(src.crs)

        # ── Etapa 3: Rio ─────────────────────────────────────────────────
        progress.progress(40, text="Buscando rio mais proximo (ANA/OSM)...")
        rios, fonte, erros = buscar_rios(lat, lon, buffer_deg=buffer_deg)

        if not rios:
            progress.progress(100, text="Erro: nenhum rio encontrado.")
            st.error(
                "Nenhum rio encontrado na regiao. "
                "Tente o **Modo Avancado** com upload manual do tracado."
            )
            if erros:
                st.caption(f"Detalhes: {erros}")
            return

        rio = rios[0]
        ls = LineString(rio['coordenadas'])

        # Recortar rio: comecar na barragem, ir a jusante por crio_km
        ls = _recortar_rio_jusante(ls, lat, lon, crio_km)

        gdf_rio = geopandas.GeoDataFrame(
            {'Name': [rio['nome']], 'Description': ['']},
            geometry=[ls], crs='EPSG:4326',
        )
        tmp_path = os.path.join(tempfile.gettempdir(), 'tracado_rio.kml')
        gdf_rio.to_file(tmp_path, driver='KML')
        st.session_state.update({
            'tracado_path': tmp_path, 'rio_nome': rio['nome'],
            'rios_encontrados': rios, 'rios_fonte': fonte,
        })

        progress.progress(50, text=f"Rio: {rio['nome']} — {crio_km:.1f} km a jusante")

        # ── Etapa 4: Parametros otimos ───────────────────────────────────
        progress.progress(55, text="Calculando parametros otimos...")
        params = calcular_parametros_otimos(h, v, crio_km, dem_resolution=dem_res)
        st.session_state['auto_params'] = params

        # ── Etapa 5: Secoes perpendiculares ──────────────────────────────
        progress.progress(60, text="Gerando secoes perpendiculares...")
        tracado = geopandas.read_file(tmp_path, driver='KML')
        tracado_simp = simplificar_tracado(tracado, params['n_simplificacao'], datum)

        linha_simp = tracado_simp.iloc[1]['geometry']
        rio_len_m = linha_simp.length

        comp_sec = params['comprimento_secao']
        n_sec = params['n_secoes']

        # Nao deixar secoes maiores que 60% do rio
        if comp_sec > rio_len_m * 0.6:
            comp_sec = max(200, int(rio_len_m * 0.4))

        # Espaco minimo entre secoes: 2x comprimento da secao
        max_secoes = max(3, int(rio_len_m / (comp_sec * 0.5)))
        if n_sec > max_secoes:
            n_sec = max_secoes

        secoes, ds = secoes_perpendiculares(tracado_simp, n=n_sec, comprimento=comp_sec)
        st.session_state['secoes'] = secoes
        st.session_state['ds'] = ds

        progress.progress(70, text=f"{n_sec} secoes x {comp_sec}m geradas")

        # ── Etapa 6: Calculo hidraulico ──────────────────────────────────
        progress.progress(75, text="Calculando mancha de inundacao...")
        dem = rasterio.open(dem_path)

        cotas_result, dp, xs, ys = cotas_secoes(
            secoes, dem, datum, n_pontos=params['n_pontos_perfil'],
        )

        brecha_auto = st.session_state.get('brecha', {})
        alturas, qs, velocidades, tempos_chegada, hidrogramas = altura_de_agua_secoes(
            ds, dp, cotas_result, qmax_val,
            v, h, fc=params['fc'], k=params['manning_k'],
            metodo_atenuacao=params['metodo_atenuacao'],
            tf_h=brecha_auto.get('Tf_2008_h'),
            modo_ruptura=params.get('modo_ruptura', 'overtopping'),
        )

        poligono = gerar_poligono_mancha(
            cotas_result, alturas, xs, ys, buffer_m=params['buffer_m'],
        )

        n_sec_real = len(alturas)
        df_relatorio = gerar_relatorio_csv(
            ds, qs, alturas, cotas_result, n_sec_real,
            velocidades=velocidades, tempos_chegada=tempos_chegada,
        )

        st.session_state.update({
            'alturas': alturas, 'qs': qs, 'cotas': cotas_result,
            'xs': xs, 'ys': ys, 'poligono': poligono,
            'df_relatorio': df_relatorio,
            'velocidades': velocidades, 'tempos_chegada': tempos_chegada,
            'hidrogramas': hidrogramas,
        })

        dem.close()

        # ── Etapa 7: Concluido ───────────────────────────────────────────
        progress.progress(100, text="Analise completa!")
        st.session_state['auto_resultado'] = True

    except Exception as e:
        progress.progress(100, text="Erro durante a analise.")
        st.error(f"Erro: {e}")
        st.info("Tente o **Modo Avancado** para diagnosticar o problema.")


# ══════════════════════════════════════════════════════════════════════════════
# Page header
# ══════════════════════════════════════════════════════════════════════════════

st.title("Mancha de Inundacao")
st.caption("Modelagem simplificada de ruptura de barragens")

modo_app = st.radio(
    "Modo de operacao",
    options=["Automatico", "Avancado"],
    index=0,
    horizontal=True,
    help="**Automatico**: 4 inputs, 1 botao — parametros otimos calculados pela literatura. "
         "**Avancado**: controle total de todos os parametros.",
)


# ══════════════════════════════════════════════════════════════════════════════
# MODO AUTOMATICO
# ══════════════════════════════════════════════════════════════════════════════

if modo_app == "Automatico":
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        auto_lat = st.number_input(
            "Latitude (graus decimais)", value=-20.0, format="%.6f", key="auto_lat",
        )
    with col2:
        auto_lon = st.number_input(
            "Longitude (graus decimais)", value=-44.0, format="%.6f", key="auto_lon",
        )
    with col3:
        auto_h = st.number_input(
            "Altura da barragem (m)", min_value=0.1, value=30.0, step=0.5, key="auto_h",
        )
    with col4:
        auto_v = st.number_input(
            "Volume armazenado (m3)", min_value=1.0, value=1000000.0, format="%.0f",
            key="auto_v",
        )

    col_btn, col_clear = st.columns([5, 1])
    with col_btn:
        executar = st.button(
            "Executar analise completa", type="primary", use_container_width=True,
        )
    with col_clear:
        limpar = st.button("Limpar", use_container_width=True)

    if limpar:
        keys_to_clear = [
            'auto_resultado', 'auto_params',
            'datum_auto', 'crio', 'qmax_barr', 'brecha', 'todos_qmax',
            'h', 'v', 'lat', 'lon',
            'dem_path', 'dem_res', 'dem_crs', 'dem_fonte',
            'tracado_path', 'rio_nome', 'rios_encontrados', 'rios_fonte',
            'secoes', 'ds', 'k_por_secao',
            'alturas', 'qs', 'cotas', 'xs', 'ys', 'poligono', 'df_relatorio',
        ]
        for k in keys_to_clear:
            st.session_state.pop(k, None)
        st.rerun()

    # Warnings de volume
    vol_hm = auto_v * 1e-6
    if vol_hm < 0.05:
        st.warning(
            "Volume muito pequeno (< 0.05 hm3). "
            "Os metodos empiricos podem nao ser confiaveis nesta faixa."
        )
    elif vol_hm > 500:
        st.warning(
            "Volume muito grande (> 500 hm3). "
            "Considere usar modelagem 2D completa (HEC-RAS, MIKE)."
        )

    if executar:
        _executar_modo_automatico(auto_lat, auto_lon, auto_h, auto_v)

    # ── Exibir resultados (persistem no session_state) ───────────────────
    if 'auto_resultado' in st.session_state:
        st.markdown("---")

        # Parametros usados (colapsavel)
        with st.expander("Parametros calculados automaticamente"):
            params = st.session_state.get('auto_params', {})
            if params:
                from engine.config import METODOS_QMAX_LABELS, MODOS_RUPTURA, METODOS_ATENUACAO
                _p = params
                _k = _p.get('manning_k', 20)
                tabela_params = pd.DataFrame([
                    {"Parametro": "Comprimento da secao",
                     "Valor": f"{_p.get('comprimento_secao', '?')} m",
                     "Justificativa": f"50 x H + 20 x sqrt(V_hm3) — HEC-RAS guidelines"},
                    {"Parametro": "Numero de secoes",
                     "Valor": str(_p.get('n_secoes', '?')),
                     "Justificativa": f"~1.5 secoes/km — ANA/LNEC (21 p/ ~14km)"},
                    {"Parametro": "Simplificacao",
                     "Valor": f"{_p.get('n_simplificacao', '?')} segmentos",
                     "Justificativa": "Metade das secoes preserva curvatura"},
                    {"Parametro": "Pontos por perfil",
                     "Valor": str(_p.get('n_pontos_perfil', '?')),
                     "Justificativa": f"comp_secao / res_DEM — nao amostrar mais fino que o DEM"},
                    {"Parametro": "Manning k (Strickler)",
                     "Valor": f"{_k} (n = {1/_k:.3f})",
                     "Justificativa": "Media rios BR — estudo Rio Doce (SciELO 2018)"},
                    {"Parametro": "Fator de correcao",
                     "Valor": str(_p.get('fc', '?')),
                     "Justificativa": "Conservador, sem reducao de altura (padrao ANA)"},
                    {"Parametro": "Metodo Qmax",
                     "Valor": METODOS_QMAX_LABELS.get(_p.get('metodo_qmax', ''), _p.get('metodo_qmax', '?')),
                     "Justificativa": "max(Froehlich, MMC) — menor incerteza (Wahl 2004)"},
                    {"Parametro": "Modo de ruptura",
                     "Valor": MODOS_RUPTURA.get(_p.get('modo_ruptura', ''), _p.get('modo_ruptura', '?')),
                     "Justificativa": "Pior caso conservador para analise automatica"},
                    {"Parametro": "Atenuacao",
                     "Valor": METODOS_ATENUACAO.get(_p.get('metodo_atenuacao', ''), _p.get('metodo_atenuacao', '?')),
                     "Justificativa": "Propagacao fisica do hidrograma (Ponce 1989)"},
                    {"Parametro": "Piso adaptavel",
                     "Valor": "Sim" if _p.get('piso_adaptavel') else "Nao",
                     "Justificativa": "Melhor para barragens pequenas"},
                    {"Parametro": "Buffer do poligono",
                     "Valor": f"{_p.get('buffer_m', '?'):.0f} m",
                     "Justificativa": "1.5 x resolucao DEM — fecha gaps entre pixels"},
                ])
                st.dataframe(tabela_params, use_container_width=True, hide_index=True)
                st.caption(
                    f"DEM: {st.session_state.get('dem_fonte', '?')} | "
                    f"UTM: {datum_label_from_epsg(st.session_state.get('datum_auto', 0))} | "
                    f"Rio: {st.session_state.get('rio_nome', '?')} ({st.session_state.get('rios_fonte', '?')})"
                )

        # Metricas resumo
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Comprimento do rio", f"{st.session_state.get('crio', 0):.2f} km")
        col_m2.metric("Qmax na barragem", f"{st.session_state.get('qmax_barr', 0):.1f} m3/s")
        brecha_r = st.session_state.get('brecha', {})
        col_m3.metric("Largura da brecha", f"{brecha_r.get('B_m', 0):.1f} m")
        col_m4.metric("Secoes geradas", f"{len(st.session_state.get('alturas', []))}")

        # Metricas de velocidade e tempo
        vels = st.session_state.get('velocidades', [])
        tempos = st.session_state.get('tempos_chegada', [])
        if vels and tempos:
            col_v1, col_v2, col_v3 = st.columns(3)
            v_max = max(vels) if vels else 0
            t_max = max(tempos) if tempos else 0
            col_v1.metric("Velocidade maxima", f"{v_max:.2f} m/s")
            col_v2.metric("Tempo de chegada (ultima secao)", f"{t_max:.1f} min")
            # Perigo: h*v > 1.0 m²/s
            alturas_r = st.session_state.get('alturas', [])
            cotas_r = st.session_state.get('cotas', [])
            if alturas_r and cotas_r:
                n_pontos_r = len(cotas_r[0])
                idx_c = n_pontos_r // 2
                ct_r = [c[idx_c] for c in cotas_r]
                hv_max = max(
                    (alturas_r[i] - ct_r[i]) * vels[i]
                    for i in range(len(vels)) if (alturas_r[i] - ct_r[i]) > 0
                ) if len(vels) > 0 else 0
                if hv_max > 1.0:
                    col_v3.metric("h x v maximo", f"{hv_max:.2f} m2/s", delta="ALTO PERIGO", delta_color="inverse")
                else:
                    col_v3.metric("h x v maximo", f"{hv_max:.2f} m2/s")

        # Layout principal: mapa + dados
        col_map, col_data = st.columns([3, 2])

        with col_map:
            st.subheader("Mapa da mancha de inundacao")
            lat_r = st.session_state['lat']
            lon_r = st.session_state['lon']
            datum_r = st.session_state.get('datum_auto', DEFAULT_DATUM)

            if st.session_state.get('poligono') is not None:
                mancha_gdf = geopandas.GeoDataFrame(
                    geometry=[st.session_state['poligono']],
                    crs=f'EPSG:{datum_r}',
                ).to_crs(epsg=4326)

                m = folium.Map(
                    location=[lat_r, lon_r], zoom_start=12, tiles="CartoDB positron",
                )
                folium.GeoJson(
                    mancha_gdf.__geo_interface__,
                    style_function=lambda x: {
                        'fillColor': '#3388ff', 'color': '#0033aa',
                        'weight': 2, 'fillOpacity': 0.4,
                    },
                ).add_to(m)

                # Secoes transversais no mapa
                if 'secoes' in st.session_state:
                    secoes_wgs = st.session_state['secoes'].set_crs(
                        f'EPSG:{datum_r}', allow_override=True,
                    ).to_crs(epsg=4326)
                    folium.GeoJson(
                        secoes_wgs.__geo_interface__,
                        style_function=lambda x: {
                            'color': 'orange', 'weight': 1, 'opacity': 0.5,
                        },
                    ).add_to(m)

                folium.Marker(
                    [lat_r, lon_r], popup="Barragem",
                    icon=folium.Icon(color='red', icon='info-sign'),
                ).add_to(m)
                st_folium(m, width=700, height=500)
            else:
                st.warning("Nenhuma mancha gerada (poligono vazio).")

        with col_data:
            st.subheader("Resultados por secao")
            if 'df_relatorio' in st.session_state:
                st.dataframe(st.session_state['df_relatorio'], height=350)

            # Downloads
            st.subheader("Downloads")
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                if 'df_relatorio' in st.session_state:
                    csv_bytes = relatorio_to_csv_bytes(st.session_state['df_relatorio'])
                    st.download_button(
                        "Relatorio CSV",
                        data=csv_bytes,
                        file_name="relatorio_hidraulico.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            with dcol2:
                if st.session_state.get('poligono') is not None:
                    shp_zip = mancha_to_zip_bytes(
                        st.session_state['poligono'],
                        st.session_state.get('datum_auto', DEFAULT_DATUM),
                    )
                    st.download_button(
                        "Shapefile mancha (.zip)",
                        data=shp_zip,
                        file_name="mancha_inundacao.zip",
                        mime="application/zip",
                        use_container_width=True,
                    )

        # Graficos hidraulicos
        has_hidrogramas = st.session_state.get('hidrogramas') is not None
        tab_names = ["Vazao", "Velocidade", "Tempo de chegada"]
        if has_hidrogramas:
            tab_names.append("Hidrogramas")

        graf_tabs = st.tabs(tab_names)

        dist_km = [d / 1000 for d in st.session_state.get('ds', [])]

        with graf_tabs[0]:
            st.subheader("Atenuacao de vazao ao longo do rio")
            if 'qs' in st.session_state and dist_km:
                chart_data = pd.DataFrame({
                    'Distancia (km)': dist_km,
                    'Vazao maxima (m3/s)': st.session_state['qs'],
                })
                st.line_chart(chart_data.set_index('Distancia (km)'))

        with graf_tabs[1]:
            st.subheader("Velocidade do escoamento")
            if 'velocidades' in st.session_state and dist_km:
                chart_vel = pd.DataFrame({
                    'Distancia (km)': dist_km,
                    'Velocidade (m/s)': st.session_state['velocidades'],
                })
                st.line_chart(chart_vel.set_index('Distancia (km)'))

        with graf_tabs[2]:
            st.subheader("Tempo de chegada da onda de cheia")
            if 'tempos_chegada' in st.session_state and dist_km:
                chart_tempo = pd.DataFrame({
                    'Distancia (km)': dist_km,
                    'Tempo (min)': st.session_state['tempos_chegada'],
                })
                st.line_chart(chart_tempo.set_index('Distancia (km)'))

        if has_hidrogramas:
            with graf_tabs[3]:
                st.subheader("Hidrogramas por secao (Muskingum-Cunge)")
                hidrogramas = st.session_state['hidrogramas']
                # Montar DataFrame com hidrogramas de secoes selecionadas
                n_sec = len(hidrogramas)
                # Mostrar secao 0, 25%, 50%, 75%, ultima
                indices_mostrar = sorted(set([
                    0, n_sec // 4, n_sec // 2, 3 * n_sec // 4, n_sec - 1
                ]))
                t_min = hidrogramas[0][0] / 60.0  # tempo em minutos
                hid_data = {'Tempo (min)': t_min}
                for idx in indices_mostrar:
                    d_km = dist_km[idx] if idx < len(dist_km) else 0
                    hid_data[f'Secao {idx} ({d_km:.1f} km)'] = hidrogramas[idx][1]
                df_hid = pd.DataFrame(hid_data)
                st.line_chart(df_hid.set_index('Tempo (min)'))

        # Comparativo de Qmax (colapsavel)
        with st.expander("Comparativo de metodos de vazao de pico"):
            todos = st.session_state.get('todos_qmax', {})
            if todos:
                df_qmax = pd.DataFrame([
                    {
                        'Metodo': METODOS_QMAX_LABELS.get(m, m),
                        'Qmax (m3/s)': f"{q:.1f}",
                        '': '  << usado' if m == 'ana_lnec' else '',
                    }
                    for m, q in todos.items()
                ])
                st.dataframe(df_qmax, hide_index=True)

        st.info(
            "Para classificacao DPA ou ajuste fino dos parametros, "
            "mude para o **Modo Avancado**."
        )


# ══════════════════════════════════════════════════════════════════════════════
# MODO AVANCADO
# ══════════════════════════════════════════════════════════════════════════════

else:
    # ── Sidebar: Parametros Globais ──────────────────────────────────────
    st.sidebar.header("Parametros Globais")

    if 'datum_auto' in st.session_state:
        datum_auto = st.session_state['datum_auto']
        datum_auto_label = datum_label_from_epsg(datum_auto)
        st.sidebar.info(f"UTM auto-detectado: {datum_auto_label}")
        datum = datum_auto
    else:
        datum_label = st.sidebar.selectbox(
            "Datum / Projecao",
            options=list(DATUM_OPTIONS.keys()),
            index=0,
        )
        datum = DATUM_OPTIONS[datum_label]

    override_datum = st.sidebar.checkbox("Sobrescrever datum manualmente", value=False)
    if override_datum:
        datum_label = st.sidebar.selectbox(
            "Datum / Projecao (manual)",
            options=list(DATUM_OPTIONS.keys()),
            index=list(DATUM_OPTIONS.values()).index(datum)
            if datum in DATUM_OPTIONS.values() else 0,
            key="datum_manual",
        )
        datum = DATUM_OPTIONS[datum_label]

    n_secoes = st.sidebar.number_input(
        "Numero de secoes transversais",
        min_value=5, max_value=100, value=DEFAULT_N_SECOES, step=1,
    )

    n_simplificacao = st.sidebar.number_input(
        "Segmentos para simplificacao do tracado",
        min_value=3, max_value=50, value=DEFAULT_N_SIMPLIFICACAO, step=1,
    )

    comprimento_secao = st.sidebar.number_input(
        "Comprimento das secoes (m)",
        min_value=500, max_value=20000, value=DEFAULT_COMPRIMENTO_SECAO, step=100,
    )

    fc = st.sidebar.number_input(
        "Fator de correcao (fc)",
        min_value=1, max_value=6, value=DEFAULT_FC, step=1,
    )

    # ── Sidebar: Metodo de Ruptura ───────────────────────────────────────
    st.sidebar.header("Metodo de Ruptura")

    metodo_qmax_label = st.sidebar.selectbox(
        "Vazao de pico",
        options=list(METODOS_QMAX_LABELS.values()),
        index=0,
    )
    metodo_qmax = [k for k, v in METODOS_QMAX_LABELS.items() if v == metodo_qmax_label][0]

    modo_ruptura_label = st.sidebar.selectbox(
        "Modo de ruptura",
        options=list(MODOS_RUPTURA.values()),
        index=0,
    )
    modo_ruptura = [k for k, v in MODOS_RUPTURA.items() if v == modo_ruptura_label][0]

    metodo_atenuacao_label = st.sidebar.selectbox(
        "Atenuacao",
        options=list(METODOS_ATENUACAO.values()),
        index=0,
    )
    metodo_atenuacao = [k for k, v in METODOS_ATENUACAO.items()
                        if v == metodo_atenuacao_label][0]

    piso_adaptavel = st.sidebar.checkbox(
        "Distancia minima adaptavel (barragens pequenas)",
        value=DEFAULT_PISO_ADAPTAVEL,
        help="Se ativado, o piso de distancia e proporcional ao volume "
             "(resolve superestimacao para barragens pequenas). "
             "Desativado usa piso fixo de 5 km (legado ANA).",
    )

    # ── Sidebar: Manning ─────────────────────────────────────────────────
    st.sidebar.header("Coeficiente de Manning")

    manning_modo = st.sidebar.radio(
        "Manning",
        options=["Uniforme", "Variavel por secao"],
        index=0,
    )

    if manning_modo == "Uniforme":
        manning_k = st.sidebar.number_input(
            "Coeficiente de Manning (k)",
            min_value=1.0, max_value=100.0, value=float(DEFAULT_MANNING_K), step=0.5,
        )
        k_por_secao = None
    else:
        manning_k = float(DEFAULT_MANNING_K)
        st.sidebar.info(
            "Defina os valores de K por secao na tabela abaixo (apos gerar secoes)."
        )
        if 'secoes' in st.session_state:
            n_sec = len(st.session_state['secoes'])
            if 'k_por_secao' not in st.session_state:
                st.session_state['k_por_secao'] = [float(DEFAULT_MANNING_K)] * n_sec
            df_k = pd.DataFrame({
                'Secao': [f"S{i+1}" for i in range(n_sec)],
                'K': st.session_state['k_por_secao'],
            })
            df_k_editado = st.sidebar.data_editor(
                df_k, hide_index=True, width="stretch",
                column_config={
                    "K": st.column_config.NumberColumn(min_value=1.0, max_value=100.0),
                },
            )
            k_por_secao = df_k_editado['K'].tolist()
            st.session_state['k_por_secao'] = k_por_secao
        else:
            k_por_secao = None

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "Passo 1: Dados da Barragem",
        "Passo 2: Tracado e Secoes",
        "Passo 3: Calculo e Resultados",
        "Passo 4: Classificacao DPA",
    ])

    # ── Tab 1: Dados da Barragem ─────────────────────────────────────────
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            latitude = st.number_input(
                "Latitude (graus decimais)", value=-20.0, format="%.6f",
                key="input_lat",
            )
            longitude = st.number_input(
                "Longitude (graus decimais)", value=-44.0, format="%.6f",
                key="input_lon",
            )
            altura = st.number_input(
                "Altura da barragem (m)", min_value=0.1, value=30.0, step=0.5,
                key="input_h",
            )
            volume = st.number_input(
                "Volume armazenado (m3)", min_value=1.0, value=1000000.0, format="%.0f",
                key="input_v",
            )

        with col2:
            if st.button("Calcular comprimento do rio", type="primary"):
                crio_val = comprimento_rio(volume, piso_adaptavel=piso_adaptavel)
                qmax_val = qmax_barragem(altura, volume, metodo=metodo_qmax)
                brecha = parametros_brecha(altura, volume, modo_ruptura=modo_ruptura)
                todos_qmax = qmax_todos_metodos(altura, volume)

                datum_auto = auto_datum(longitude)
                st.session_state['datum_auto'] = datum_auto

                st.session_state['crio'] = crio_val
                st.session_state['qmax_barr'] = qmax_val
                st.session_state['brecha'] = brecha
                st.session_state['todos_qmax'] = todos_qmax
                st.session_state['h'] = altura
                st.session_state['v'] = volume
                st.session_state['lat'] = latitude
                st.session_state['lon'] = longitude

            if 'crio' in st.session_state:
                st.metric(
                    "Comprimento do rio a modelar",
                    f"{st.session_state['crio']:.2f} km",
                )
                st.metric(
                    "Vazao maxima na barragem",
                    f"{st.session_state['qmax_barr']:.1f} m3/s",
                )
                st.caption(
                    f"Metodo: {METODOS_QMAX_LABELS.get(metodo_qmax, metodo_qmax)}"
                )

                # Comparativo de metodos
                st.subheader("Comparativo de Vazao de Pico")
                todos = st.session_state.get('todos_qmax', {})
                if todos:
                    df_qmax = pd.DataFrame([
                        {
                            'Metodo': METODOS_QMAX_LABELS.get(m, m),
                            'Qmax (m3/s)': f"{q:.1f}",
                            '': '  <<' if m == metodo_qmax else '',
                        }
                        for m, q in todos.items()
                    ])
                    st.dataframe(df_qmax, hide_index=True, width="stretch")

                # Parametros de brecha
                st.subheader("Parametros de Brecha")
                brecha = st.session_state.get('brecha', {})
                if brecha:
                    bc1, bc2, bc3 = st.columns(3)
                    bc1.metric("Tf (Froehlich 2008)", f"{brecha['Tf_2008_h']:.2f} h")
                    bc2.metric("Tf (Froehlich 1995)", f"{brecha['Tf_1995_h']:.2f} h")
                    bc3.metric("Largura da brecha (B)", f"{brecha['B_m']:.1f} m")
                    st.caption(
                        f"Modo de ruptura: "
                        f"{MODOS_RUPTURA.get(brecha['modo'], brecha['modo'])} "
                        f"(K = {brecha['K']})"
                    )

                # Warning para volumes fora do range calibrado
                vol_hm = volume * 1e-6
                if vol_hm < 0.05:
                    st.warning(
                        "Volume muito pequeno (< 0.05 hm3). "
                        "Os metodos empiricos podem nao ser confiaveis nesta faixa."
                    )
                elif vol_hm > 500:
                    st.warning(
                        "Volume muito grande (> 500 hm3). "
                        "Considere usar modelagem 2D completa (HEC-RAS, MIKE)."
                    )

    # ── Tab 2: Tracado e Secoes ──────────────────────────────────────────
    with tab2:
        col1, col2 = st.columns([1, 2])

        with col1:
            fonte_dados = st.radio(
                "Fonte de dados",
                options=["Automatico (internet)", "Upload manual"],
                index=0,
                horizontal=True,
                help="Automatico baixa DEM Copernicus 30m e busca rios na base da ANA/OSM.",
            )

            if fonte_dados == "Automatico (internet)":

                def _salvar_rio(rio):
                    """Salva rio selecionado como KML temporario."""
                    ls = LineString(rio['coordenadas'])
                    # Recortar a jusante se temos dados da barragem
                    if 'lat' in st.session_state and 'crio' in st.session_state:
                        ls = _recortar_rio_jusante(
                            ls, st.session_state['lat'],
                            st.session_state['lon'],
                            st.session_state['crio'],
                        )
                    gdf = geopandas.GeoDataFrame(
                        {'Name': [rio['nome']], 'Description': ['']},
                        geometry=[ls], crs='EPSG:4326',
                    )
                    tmp_path = os.path.join(tempfile.gettempdir(), 'tracado_rio.kml')
                    gdf.to_file(tmp_path, driver='KML')
                    st.session_state['tracado_path'] = tmp_path
                    st.session_state['rio_nome'] = rio['nome']

                if st.button("Baixar DEM e rio automaticamente", type="primary"):
                    if 'lat' not in st.session_state:
                        st.error(
                            "Preencha lat/lon e clique 'Calcular' no Passo 1 primeiro."
                        )
                    else:
                        lat = st.session_state['lat']
                        lon = st.session_state['lon']
                        crio_km = st.session_state.get('crio', 10)

                        with st.spinner("Baixando DEM Copernicus 30m..."):
                            try:
                                buffer = max(0.1, crio_km / 111 * 0.7)
                                dem_path = download_dem_copernicus(
                                    lat, lon, buffer_deg=buffer,
                                )
                                st.session_state['dem_path'] = dem_path
                                with rasterio.open(dem_path) as src:
                                    res_x = abs(src.transform[0])
                                    res_y = abs(src.transform[4])
                                    st.session_state['dem_res'] = (res_x, res_y)
                                    st.session_state['dem_crs'] = str(src.crs)
                                st.success("DEM Copernicus 30m baixado!")
                            except Exception as e:
                                st.error(f"Erro ao baixar DEM: {e}")

                        with st.spinner("Buscando rio mais proximo (ANA/OSM)..."):
                            try:
                                buf = max(0.1, crio_km / 111 * 0.7)
                                rios, fonte, erros = buscar_rios(
                                    lat, lon, buffer_deg=buf,
                                )
                                st.session_state['rios_encontrados'] = rios
                                st.session_state['rios_fonte'] = fonte
                                if rios:
                                    _salvar_rio(rios[0])
                                    st.success(
                                        f"Rio selecionado: **{rios[0]['nome']}** "
                                        f"({rios[0]['distancia_km']:.1f}km) via {fonte}"
                                    )
                                else:
                                    st.warning(
                                        "Nenhum rio encontrado. Use upload manual."
                                    )
                                    if erros:
                                        st.caption(f"Detalhes: {erros}")
                            except Exception as e:
                                st.error(f"Erro na busca de rios: {e}")

                # Status do que ja foi carregado
                s_dem = 'dem_path' in st.session_state
                s_rio = 'tracado_path' in st.session_state
                if s_dem or s_rio:
                    st.caption(
                        f"{'DEM carregado' if s_dem else 'DEM pendente'} | "
                        f"{'Rio: ' + st.session_state.get('rio_nome', 'carregado') if s_rio else 'Rio pendente'}"
                    )

                # Trocar rio (expander)
                if ('rios_encontrados' in st.session_state
                        and st.session_state['rios_encontrados']):
                    with st.expander("Trocar rio selecionado"):
                        rios = st.session_state['rios_encontrados']
                        for r in rios:
                            coords = r['coordenadas']
                            r['_comprimento_km'] = (
                                LineString(coords).length * 111
                                if len(coords) >= 2 else 0
                            )

                        opcoes = []
                        for i, r in enumerate(rios):
                            label = f"#{i+1} {r['nome']}"
                            label += f" | dist: {r['distancia_km']:.1f}km"
                            label += f" | trecho: {r['_comprimento_km']:.1f}km"
                            if r.get('area_drenagem_km2'):
                                label += f" | bacia: {r['area_drenagem_km2']:.0f}km2"
                            opcoes.append(label)

                        idx_rio = st.selectbox(
                            "Rio", options=range(len(opcoes)),
                            format_func=lambda i: opcoes[i],
                        )
                        if st.button("Usar este rio"):
                            _salvar_rio(rios[idx_rio])
                            st.success(f"Rio trocado para: {rios[idx_rio]['nome']}")

            else:
                # Upload manual
                dem_file = st.file_uploader(
                    "Carregar DEM (.tif)",
                    type=["tif", "tiff"],
                    help="Qualquer GeoTIFF: SRTM 30m, ALOS 12.5m, LiDAR, drone, etc.",
                )
                tracado_file = st.file_uploader(
                    "Carregar tracado do rio (.kml)", type=["kml"],
                )

                if dem_file is not None and 'dem_path' not in st.session_state:
                    tmp = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
                    tmp.write(dem_file.read())
                    tmp.close()
                    st.session_state['dem_path'] = tmp.name
                    with rasterio.open(tmp.name) as src:
                        res_x = abs(src.transform[0])
                        res_y = abs(src.transform[4])
                        st.session_state['dem_res'] = (res_x, res_y)
                        st.session_state['dem_crs'] = str(src.crs)
                    st.success("DEM carregado!")

                if tracado_file is not None and 'tracado_path' not in st.session_state:
                    tmp = tempfile.NamedTemporaryFile(suffix='.kml', delete=False)
                    tmp.write(tracado_file.read())
                    tmp.close()
                    st.session_state['tracado_path'] = tmp.name
                    st.success("Tracado carregado!")

            # Info do DEM carregado
            if 'dem_res' in st.session_state:
                rx, ry = st.session_state['dem_res']
                st.caption(
                    f"DEM: ~{rx:.1f}m x {ry:.1f}m | "
                    f"CRS: {st.session_state.get('dem_crs', '?')}"
                )
                if rx > 30 or ry > 30:
                    st.warning(
                        "Resolucao > 30m detectada. "
                        "Resultados podem ser comprometidos — considere um DEM mais fino."
                    )

            # Gerar secoes
            if st.button(
                "Gerar secoes perpendiculares", type="primary", key="btn_secoes",
            ):
                if ('dem_path' not in st.session_state
                        or 'tracado_path' not in st.session_state):
                    st.error("Carregue o DEM e o tracado primeiro.")
                else:
                    with st.spinner("Calculando secoes..."):
                        tracado = geopandas.read_file(
                            st.session_state['tracado_path'], driver='KML',
                        )
                        tracado_simp = simplificar_tracado(
                            tracado, n_simplificacao, datum,
                        )

                        linha_simp = tracado_simp.iloc[1]['geometry']
                        rio_len_m = linha_simp.length

                        n_sec_eff = n_secoes
                        comp_sec_eff = comprimento_secao

                        if comp_sec_eff > rio_len_m * 0.6:
                            comp_sec_eff = max(200, int(rio_len_m * 0.4))
                            st.info(
                                f"Comprimento de secao ajustado para {comp_sec_eff}m "
                                f"(rio tem {rio_len_m:.0f}m)."
                            )

                        max_secoes = max(3, int(rio_len_m / (comp_sec_eff * 0.5)))
                        if n_sec_eff > max_secoes:
                            n_sec_eff = max_secoes
                            st.info(
                                f"Numero de secoes ajustado para {n_sec_eff} "
                                f"(evitar sobreposicao)."
                            )

                        n_simp_eff = min(n_simplificacao, n_sec_eff - 1)

                        secoes, ds = secoes_perpendiculares(
                            tracado_simp, n=n_sec_eff, comprimento=comp_sec_eff,
                        )
                        st.session_state['secoes'] = secoes
                        st.session_state['ds'] = ds
                        st.success(
                            f"{n_sec_eff} secoes geradas ({comp_sec_eff}m cada)!"
                        )

            if 'secoes' in st.session_state:
                shp_zip = secoes_to_zip_bytes(st.session_state['secoes'], datum)
                st.download_button(
                    "Baixar secoes (.zip)",
                    data=shp_zip,
                    file_name="secoes_transversais.zip",
                    mime="application/zip",
                )

                secoes_editadas = st.file_uploader(
                    "Reimportar secoes editadas (.shp)", type=["shp"],
                    key="secoes_editadas",
                )
                if secoes_editadas is not None:
                    tmp = tempfile.NamedTemporaryFile(suffix='.shp', delete=False)
                    tmp.write(secoes_editadas.read())
                    tmp.close()
                    st.session_state['secoes'] = geopandas.read_file(tmp.name)
                    st.session_state['secoes'] = st.session_state['secoes'].to_crs(
                        epsg=datum,
                    )
                    st.success("Secoes reimportadas!")

        with col2:
            if 'secoes' in st.session_state and 'lat' in st.session_state:
                lat = st.session_state['lat']
                lon = st.session_state['lon']
                secoes_wgs = st.session_state['secoes'].set_crs(
                    f'EPSG:{datum}', allow_override=True,
                ).to_crs(epsg=4326)

                m = folium.Map(
                    location=[lat, lon], zoom_start=12, tiles="CartoDB positron",
                )
                folium.Marker(
                    [lat, lon], popup="Barragem",
                    icon=folium.Icon(color='red', icon='info-sign'),
                ).add_to(m)
                folium.GeoJson(
                    secoes_wgs.__geo_interface__,
                    style_function=lambda x: {'color': 'red', 'weight': 2},
                ).add_to(m)
                st_folium(m, width=700, height=500)
            else:
                st.info(
                    "Carregue os dados e calcule as secoes para visualizar o mapa."
                )

    # ── Tab 3: Calculo e Resultados ──────────────────────────────────────
    with tab3:
        if st.button("Executar calculo hidraulico", type="primary"):
            required = ['secoes', 'ds', 'dem_path', 'h', 'v']
            missing = [k for k in required if k not in st.session_state]
            if missing:
                st.error("Complete os passos 1 e 2 primeiro.")
            else:
                with st.spinner("Calculando mancha de inundacao..."):
                    dem = rasterio.open(st.session_state['dem_path'])
                    secoes = st.session_state['secoes']
                    ds = st.session_state['ds']
                    h = st.session_state['h']
                    v = st.session_state['v']

                    cotas_result, dp, xs, ys = cotas_secoes(secoes, dem, datum)

                    brecha_avanc = st.session_state.get('brecha', {})
                    alturas, qs, velocidades, tempos_chegada, hidrogramas = altura_de_agua_secoes(
                        ds, dp, cotas_result, st.session_state['qmax_barr'],
                        v, h, fc=fc, k=manning_k,
                        k_por_secao=k_por_secao,
                        metodo_atenuacao=metodo_atenuacao,
                        tf_h=brecha_avanc.get('Tf_2008_h'),
                        modo_ruptura=modo_ruptura,
                    )

                    poligono = gerar_poligono_mancha(cotas_result, alturas, xs, ys)

                    n_sec_real = len(alturas)
                    df_relatorio = gerar_relatorio_csv(
                        ds, qs, alturas, cotas_result, n_sec_real,
                        velocidades=velocidades, tempos_chegada=tempos_chegada,
                    )

                    st.session_state['alturas'] = alturas
                    st.session_state['qs'] = qs
                    st.session_state['cotas'] = cotas_result
                    st.session_state['xs'] = xs
                    st.session_state['ys'] = ys
                    st.session_state['poligono'] = poligono
                    st.session_state['df_relatorio'] = df_relatorio
                    st.session_state['velocidades'] = velocidades
                    st.session_state['tempos_chegada'] = tempos_chegada
                    st.session_state['hidrogramas'] = hidrogramas

                    dem.close()
                    st.success("Calculo finalizado!")

        if 'df_relatorio' in st.session_state:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Resultados por secao")
                st.dataframe(st.session_state['df_relatorio'], width="stretch")

                csv_bytes = relatorio_to_csv_bytes(st.session_state['df_relatorio'])
                st.download_button(
                    "Baixar relatorio CSV",
                    data=csv_bytes,
                    file_name="relatorio_hidraulico.csv",
                    mime="text/csv",
                )

                if st.session_state['poligono'] is not None:
                    shp_zip = mancha_to_zip_bytes(
                        st.session_state['poligono'], datum,
                    )
                    st.download_button(
                        "Baixar shapefile da mancha (.zip)",
                        data=shp_zip,
                        file_name="mancha_inundacao.zip",
                        mime="application/zip",
                    )

            with col2:
                if st.session_state['poligono'] is not None:
                    st.subheader("Mapa da mancha de inundacao")
                    lat = st.session_state['lat']
                    lon = st.session_state['lon']

                    mancha_gdf = geopandas.GeoDataFrame(
                        geometry=[st.session_state['poligono']],
                        crs=f'EPSG:{datum}',
                    ).to_crs(epsg=4326)

                    m2 = folium.Map(location=[lat, lon], zoom_start=11)
                    folium.GeoJson(
                        mancha_gdf.__geo_interface__,
                        style_function=lambda x: {
                            'fillColor': '#3388ff',
                            'color': '#0033aa',
                            'weight': 2,
                            'fillOpacity': 0.4,
                        },
                    ).add_to(m2)
                    folium.Marker(
                        [lat, lon], popup="Barragem",
                        icon=folium.Icon(color='red'),
                    ).add_to(m2)
                    st_folium(m2, width=650, height=500)

                # Grafico de atenuacao de vazao
                st.subheader("Atenuacao de vazao ao longo do rio")
                chart_data = pd.DataFrame({
                    'Distancia (km)': [d / 1000 for d in st.session_state['ds']],
                    'Vazao maxima (m3/s)': st.session_state['qs'],
                })
                st.line_chart(chart_data.set_index('Distancia (km)'))

    # ── Tab 4: Classificacao DPA ─────────────────────────────────────────
    with tab4:
        st.subheader("Classificacao de Dano Potencial Associado (DPA)")
        st.caption("Conforme Resolucao CNRH 241/2024 (substitui 143/2012)")

        st.markdown("---")
        st.markdown(
            "**Preencha os criterios abaixo conforme vistoria e dados disponiveis.**"
        )

        # Criterio 1: Potencial de perdas de vidas humanas
        st.markdown("#### 1. Potencial de perdas de vidas humanas")
        vidas = st.radio(
            "Existencia de populacao a jusante potencialmente afetada",
            options=[
                "INEXISTENTE (nao ha populacao a jusante) — 0 pts",
                "POUCO FREQUENTE (area rural, sem nucleos habitacionais) — 4 pts",
                "FREQUENTE (area rural com nucleos habitacionais ou urbana sem ocupacao densa) — 8 pts",
                "EXISTENTE (area urbana com ocupacao densa ou infraestrutura critica) — 12 pts",
            ],
            index=0,
            key="dpa_vidas",
        )
        pts_vidas = [0, 4, 8, 12][
            ["INEXISTENTE", "POUCO FREQUENTE", "FREQUENTE", "EXISTENTE"].index(
                vidas.split(" (")[0].split(" —")[0]
            )
        ]

        # Criterio 2: Impacto economico, social e ambiental
        st.markdown("#### 2. Impacto economico, social e ambiental")

        eco = st.radio(
            "Impacto nas atividades economicas, infraestrutura e servicos",
            options=[
                "INEXISTENTE — 0 pts",
                "BAIXO (area de pastagem/agricultura de subsistencia) — 1 pt",
                "MEDIO (area agricola produtiva ou estrada vicinal) — 3 pts",
                "ALTO (area industrial, estrada principal, ferrovia, utilidades) — 5 pts",
            ],
            index=0,
            key="dpa_eco",
        )
        pts_eco = [0, 1, 3, 5][
            ["INEXISTENTE", "BAIXO", "MEDIO", "ALTO"].index(
                eco.split(" (")[0].split(" —")[0]
            )
        ]

        amb = st.radio(
            "Dano ambiental",
            options=[
                "INEXISTENTE — 0 pts",
                "BAIXO (area sem interesse ambiental relevante) — 1 pt",
                "MEDIO (area de interesse ambiental nao protegida legalmente) — 3 pts",
                "ALTO (area de protecao ambiental, APP, unidade de conservacao) — 5 pts",
            ],
            index=0,
            key="dpa_amb",
        )
        pts_amb = [0, 1, 3, 5][
            ["INEXISTENTE", "BAIXO", "MEDIO", "ALTO"].index(
                amb.split(" (")[0].split(" —")[0]
            )
        ]

        # Pontuacao total e classificacao
        st.markdown("---")
        total = pts_vidas + pts_eco + pts_amb
        classe = classificar_dpa(total)

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Perdas de vidas", f"{pts_vidas} pts")
        col_r2.metric("Impacto eco/social/ambiental", f"{pts_eco + pts_amb} pts")
        col_r3.metric("Pontuacao total", f"{total} pts")

        cor_classe = {"Alto": "red", "Medio": "orange", "Baixo": "green"}
        st.markdown(
            f"### Classificacao DPA: "
            f"<span style='color:{cor_classe[classe]};font-weight:bold'>"
            f"{classe}</span>",
            unsafe_allow_html=True,
        )

        st.caption(
            f"Limites CNRH 241/2024: "
            f"Alto >= {DPA_LIMITE_ALTO} pts | "
            f"Medio >= {DPA_LIMITE_MEDIO} pts | "
            f"Baixo < {DPA_LIMITE_MEDIO} pts"
        )
