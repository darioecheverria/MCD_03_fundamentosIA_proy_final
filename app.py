"""App de Streamlit: tipología de celdas y probabilidad de red vial nueva.

Carga el paquete entrenado en 01_entrenamiento.ipynb y calcula las variables de cada
ciudad con urbano.py, el mismo módulo que usó el entrenamiento.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

import urbano as u

RAIZ = Path(__file__).parent
DATOS = RAIZ / "datos"
COLOR_TIPO = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COLOR_COMPARA = {"Acierto: señalada y con red nueva": "#2a78d6", "Señalada, sin red nueva": "#eda100",
                 "Con red nueva, no señalada": "#eb6834", "Sin señal y sin red nueva": "#d9d8d2"}
RAMPA = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]      # un solo tono, claro -> oscuro

st.set_page_config(page_title="Red vial nueva por celdas", page_icon="🗺️", layout="wide")


# ------------------------------------------------------------------ carga (en caché)
@st.cache_resource
def cargar_paquete():
    try:
        return joblib.load(RAIZ / "modelo" / "paquete.joblib")
    except Exception as e:      # casi siempre: scikit-learn distinto al del entrenamiento
        st.error(f"No se pudo leer `modelo/paquete.joblib` ({type(e).__name__}). El paquete solo se lee con la versión "
                 "de scikit-learn con la que se entrenó: revisa `requirements.txt` y la versión de Python del despliegue, "
                 "o vuelve a ejecutar `01_entrenamiento.ipynb`.")
        st.stop()


def ciudades_disponibles():
    """Ciudades con capa antigua en datos/. La capa actual es opcional."""
    return sorted(p.stem.removeprefix("ejes_antigua_") for p in DATOS.glob("ejes_antigua_*.gpkg"))


@st.cache_data(max_entries=3, show_spinner="Calculando variables de la red…")
def calcular(ciudad):
    """Tabla de celdas con tipo y probabilidad. max_entries limita la memoria del caché."""
    actual = DATOS / f"ejes_actual_{ciudad}.gpkg"
    celdas, info = u.tabla_celdas(DATOS / f"ejes_antigua_{ciudad}.gpkg", actual if actual.exists() else None)
    celdas = u.predecir(celdas, cargar_paquete())
    celdas["poligono"] = u.esquinas_wgs84(celdas)
    return celdas, info


def rgb(hexa, alfa=190):
    return [int(hexa[i:i + 2], 16) for i in (1, 3, 5)] + [alfa]


def color_rampa(valores, vmax):
    """Interpola la rampa secuencial; devuelve una lista [r, g, b, a] por valor."""
    paradas = np.array([rgb(c)[:3] for c in RAMPA], float)
    t = np.clip(np.asarray(valores) / vmax, 0, 1) * (len(RAMPA) - 1)
    i = np.minimum(t.astype(int), len(RAMPA) - 2)
    c = paradas[i] + (paradas[i + 1] - paradas[i]) * (t - i)[:, None]
    return np.c_[c, np.full(len(c), 190)].astype(int).tolist()


def leyenda(pares):
    cajas = "".join(f"<span style='display:inline-block;width:12px;height:12px;border-radius:3px;background:{c};"
                    f"margin:0 6px 0 14px;vertical-align:middle'></span>{n}" for n, c in pares)
    st.markdown(f"<div style='font-size:0.85rem;margin-left:-14px'>{cajas}</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ datos
paquete = cargar_paquete()
nombres = paquete["nombres_tipos"]
anio_ant, anio_act = paquete["anios"]["antigua"], paquete["anios"]["actual"]
ciudades = ciudades_disponibles()

st.title("¿Dónde aparece red vial nueva?")
st.caption(f"Tipología de celdas de {u.TAM_CELDA} m y probabilidad de que reciban red vial nueva, "
           f"a partir de la forma de la red de {anio_ant}. Ejes viales del INEC.")

if not ciudades:
    st.error("No hay geopaquetes en la carpeta `datos/`. Se espera `ejes_antigua_<ciudad>.gpkg`.")
    st.stop()

with st.sidebar:
    st.header("Controles")
    ciudad = st.selectbox("Ciudad", ciudades, format_func=lambda c: c.replace("_", " ").title())
    umbral = st.slider("Umbral de probabilidad", 0.05, 0.80, float(paquete["umbral"]), 0.05,
                       help="Una celda se señala si su probabilidad es igual o mayor que el umbral. "
                            "Un umbral bajo detecta más celdas con red nueva a costa de más falsas alarmas.")
    st.divider()
    st.caption(f"Modelo: {paquete['elegido']}, entrenado con {paquete['n_celdas']:,} celdas de "
               f"{', '.join(c.title() for c in paquete['ciudades'])}.")

celdas, info = calcular(ciudad)
hay_obs = "crece" in celdas.columns
de_entrenamiento = ciudad in paquete["ciudades"]
celdas = celdas.assign(senalada=celdas.prob >= umbral)

k = st.columns(4)
k[0].metric("Celdas analizadas", f"{len(celdas):,}")
k[1].metric(f"Red {anio_ant}", f"{info['km_ant']:,.0f} km")
k[2].metric("Celdas señaladas", f"{int(celdas.senalada.sum()):,}", help="Probabilidad igual o mayor que el umbral")
if hay_obs:
    k[3].metric(f"Celdas con red nueva en {anio_act}", f"{int(celdas.crece.sum()):,}",
                help=f"Al menos {u.MIN_NUEVA_M} m de ejes a más de {u.DIST_NUEVA} m de la red {anio_ant}")
else:
    k[3].metric("Versión actual", "No cargada", help="Sin capa actual solo se muestra la predicción")

t_mapa, t_tipos, t_sim, t_modelo = st.tabs(["Mapa", "Tipología", "Simulador", "Modelo y límites"])

# ------------------------------------------------------------------ mapa
with t_mapa:
    capas = ["Probabilidad de red nueva", "Tipología"] + ([f"Predicción frente a lo observado en {anio_act}"] if hay_obs else [])
    capa = st.radio("Capa", capas, horizontal=True, label_visibility="collapsed")
    m = celdas[["poligono", "prob", "tipo", "dens", "dist_red"]].copy()
    m["nombre_tipo"] = [nombres[t] for t in m.tipo]
    m["prob_txt"] = (100 * m.prob).round(0).astype(int).astype(str) + " %"
    m["dens_txt"] = m.dens.round(1).astype(str)
    m["estado"] = ""
    if capa == capas[0]:
        vmax = max(0.5, float(np.quantile(celdas.prob, 0.99)))
        m["color"] = color_rampa(m.prob, vmax)
        leyenda([(f"{100 * v * vmax:.0f} %", c) for v, c in zip(np.linspace(0, 1, len(RAMPA)), RAMPA)])
    elif capa == capas[1]:
        m["color"] = [rgb(COLOR_TIPO[t % len(COLOR_TIPO)]) for t in m.tipo]
        leyenda([(n, COLOR_TIPO[i % len(COLOR_TIPO)]) for i, n in enumerate(nombres)])
    else:
        estados = list(COLOR_COMPARA)
        cod = np.select([celdas.senalada & (celdas.crece == 1), celdas.senalada, celdas.crece == 1], [0, 1, 2], 3)
        m["estado"] = [estados[i] for i in cod]
        m["color"] = [rgb(COLOR_COMPARA[e], 90 if e == estados[3] else 200) for e in m.estado]
        leyenda(list(COLOR_COMPARA.items()))

    centro = np.array(m.poligono.iloc[int(celdas.dens_v4.to_numpy().argmax())][0])      # zona más densa
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(longitude=float(centro[0]), latitude=float(centro[1]), zoom=11.5),
        layers=[pdk.Layer("PolygonLayer", m.drop(columns=["tipo"]), get_polygon="poligono", get_fill_color="color",
                          stroked=False, pickable=True, auto_highlight=True)],
        tooltip={"html": "<b>{nombre_tipo}</b><br/>Probabilidad: {prob_txt}<br/>Densidad vial: {dens_txt} km/km²<br/>{estado}"},
    ), height=560)

    if hay_obs:
        vp = int((celdas.senalada & (celdas.crece == 1)).sum())
        prec = vp / max(int(celdas.senalada.sum()), 1)
        exh = vp / max(int(celdas.crece.sum()), 1)
        c = st.columns(3)
        c[0].metric("Precisión", f"{prec:.0%}", help="De las celdas señaladas, cuántas tienen red nueva")
        c[1].metric("Exhaustividad", f"{exh:.0%}", help="De las celdas con red nueva, cuántas fueron señaladas")
        c[2].metric("Proporción base", f"{celdas.crece.mean():.0%}", help="Precisión que se obtendría señalando celdas al azar")
        if de_entrenamiento:
            st.warning("Esta ciudad se usó para entrenar el modelo, así que estas dos cifras son optimistas. "
                       "El desempeño sobre zonas no vistas está en la pestaña **Modelo y límites**.", icon="⚠️")

# ------------------------------------------------------------------ tipología
with t_tipos:
    st.markdown("Cada celda se asigna a un tipo de tejido con **KMeans** sobre seis variables de la red "
                f"de {anio_ant}. Los tipos ordenan las celdas del núcleo a la periferia.")
    r = celdas.groupby("tipo").agg(celdas=("prob", "size"), prob_media=("prob", "mean"),
                                   dens=("dens", "median"), dist_nucleo=("dist_nucleo", "median"))
    if hay_obs:
        r["con_red_nueva"] = celdas.groupby("tipo").crece.mean()
    r = r.reindex(range(len(nombres))).fillna(0)
    r.insert(0, "Tipo", nombres)
    st.dataframe(r, hide_index=True, width="stretch", column_config={
        "celdas": st.column_config.NumberColumn("Celdas", format="%d"),
        "prob_media": st.column_config.ProgressColumn("Probabilidad media", format="percent", min_value=0, max_value=1),
        "con_red_nueva": st.column_config.ProgressColumn(f"Con red nueva en {anio_act}", format="percent", min_value=0, max_value=1),
        "dens": st.column_config.NumberColumn("Densidad mediana (km/km²)", format="%.1f"),
        "dist_nucleo": st.column_config.NumberColumn("Distancia mediana al núcleo (m)", format="%d"),
    })
    st.caption("Los tipos son cortes de un gradiente continuo, no grupos separados: sirven para describir, "
               "y el modelo de probabilidad usa las variables continuas.")

# ------------------------------------------------------------------ simulador
with t_sim:
    st.markdown("Describe una celda y su entorno; el modelo devuelve su tipo y la probabilidad de red nueva.")
    base = st.selectbox("Partir de la celda típica de", nombres)
    med, rangos = paquete["medianas_tipo"].loc[nombres.index(base)], paquete["rangos"]
    pasos = {"frac_v1": 0.125, "coher_v2": 0.05, "dist_red": float(u.TAM_CELDA), "dist_nucleo": float(u.TAM_CELDA)}
    x, cols = {}, st.columns(2)
    for i, v in enumerate(u.VARIABLES):
        paso = pasos.get(v, 0.5)
        tope = float(np.ceil(rangos.loc[v, "p99"] / paso) * paso)
        tope = min(tope, 10000.0) if v == "dist_nucleo" else tope
        valor = float(np.clip(round(med[v] / paso) * paso, 0.0, tope))
        # la clave incluye el tipo de partida: al cambiarlo, los controles vuelven a sus medianas
        x[v] = cols[i % 2].slider(u.DESCRIPCION[v], 0.0, tope, valor, paso, key=f"sim_{v}_{base}")
    if x["dens"] > 0 and x["dist_red"] > 0:
        st.info("Una celda con red está a distancia 0 de la red: se usa 0 m.")
        x["dist_red"] = 0.0
    fila = u.predecir(pd.DataFrame([x]), paquete).iloc[0]
    c = st.columns(3)
    c[0].metric("Probabilidad de red nueva", f"{fila.prob:.0%}")
    c[1].metric("Tipo asignado", nombres[int(fila.tipo)])
    c[2].metric("Proporción base en el entrenamiento", f"{paquete['prevalencia']:.0%}")
    st.caption("Los controles son independientes; algunas combinaciones no existen en una red real y el modelo "
               "las resuelve por cercanía a lo que vio en el entrenamiento.")

# ------------------------------------------------------------------ modelo
with t_modelo:
    met = paquete["metricas"]
    st.subheader("Desempeño en validación cruzada espacial")
    st.markdown("Bloques de 2 km: cada modelo se evalúa sobre zonas que no vio. "
                "AUC 0,5 equivale al azar; en Brier, menor es mejor.")
    st.dataframe(met, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.3f") for c in met.columns})
    a, b = st.columns(2)
    with a:
        st.subheader("Umbral")
        st.dataframe(paquete["umbrales"].reset_index(), hide_index=True, width="stretch", column_config={
            "umbral": st.column_config.NumberColumn("Umbral", format="%.2f"),
            "celdas señaladas": st.column_config.NumberColumn("Celdas señaladas", format="%d"),
            "precisión": st.column_config.NumberColumn("Precisión", format="%.2f"),
            "exhaustividad": st.column_config.NumberColumn("Exhaustividad", format="%.2f"),
            "F1": st.column_config.NumberColumn(format="%.2f")})
    with b:
        st.subheader("Importancia de las variables")
        imp = paquete["importancia"].sort_values("caída_AUC", ascending=False)
        st.dataframe(pd.DataFrame({"Variable": [u.DESCRIPCION[v] for v in imp.index], "Caída del AUC": imp.caída_AUC.to_numpy()}),
                     hide_index=True, width="stretch", column_config={
                         "Caída del AUC": st.column_config.ProgressColumn(format="%.3f", min_value=0,
                                                                          max_value=float(max(imp.caída_AUC.max(), 0.01)))})
    st.subheader("Límites")
    p = paquete["parametros"]
    st.markdown(f"""
- **Entrenado con {len(paquete['ciudades'])} ciudad(es).** Aplicarlo a otra ciudad es una extrapolación cuyo error no está medido.
- **Modelo de contigüidad.** Predice *dónde* a partir de la forma de la red; no usa pendiente, población ni uso de suelo.
- **Red nueva y levantamiento.** Es "nuevo" un eje de {anio_act} a más de {p['DIST_NUEVA']} m de la red {anio_ant}. Como la base {anio_act} es más detallada, parte de lo nuevo dentro del núcleo son vías que ya existían.
- **Universo.** Solo celdas a {p['RADIO_UNIVERSO']} m o menos de la red {anio_ant}: no anticipa urbanizaciones aisladas.
""")
    st.caption(f"Entrenado con scikit-learn {paquete['versiones']['scikit-learn']} · Python {paquete['versiones']['python']}")
