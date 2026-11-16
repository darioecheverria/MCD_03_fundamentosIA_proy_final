"""Funciones compartidas por el notebook de entrenamiento y por la app de Streamlit.

Todo el cálculo de variables vive aquí para que el modelo reciba en la app
exactamente las mismas variables con las que se entrenó.

Flujo: ejes (GPKG) -> segmentos -> malla de celdas -> variables 2014 por celda
       -> tipología (KMeans) -> probabilidad de red nueva (clasificador).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from scipy import ndimage

CRS_TRABAJO = "EPSG:31992"   # SIRGAS 1995 / UTM 17S, el CRS nativo de la capa MGN
TAM_CELDA = 250              # m; lado de la celda
RADIO_UNIVERSO = 500         # m; solo se modelan celdas a esta distancia o menos de la red 2014
DIST_NUEVA = 30              # m; un segmento 2021 es "nuevo" si su punto medio está más lejos que esto de la red 2014
MIN_NUEVA_M = 50             # m; longitud nueva mínima para que la celda cuente como "con red nueva"
DENS_NUCLEO = 8.0            # km/km²; densidad de vecindad (5x5) que define un núcleo consolidado

# Variables del modelo, en el orden en que se entrenó
VARIABLES = ["dens", "dens_v1", "dens_v2", "dens_v4", "frac_v1", "coher_v2", "dist_red", "dist_nucleo"]
# Variables que definen la tipología (las de distancia y densidad entran en logaritmo)
VARIABLES_CLUSTER = ["dens", "dens_v2", "frac_v1", "coher_v2", "dist_red", "dist_nucleo"]
EN_LOG = ["dens", "dens_v2", "dist_red", "dist_nucleo"]

DESCRIPCION = {
    "dens": "Densidad vial de la celda (km/km²)",
    "dens_v1": "Densidad media en la vecindad 3×3 (km/km²)",
    "dens_v2": "Densidad media en la vecindad 5×5 (km/km²)",
    "dens_v4": "Densidad media en la vecindad 9×9 (km/km²)",
    "frac_v1": "Fracción de las 8 celdas vecinas que tienen red",
    "coher_v2": "Coherencia de orientación en la vecindad 5×5 (0 = sin orden, 1 = calles paralelas)",
    "dist_red": "Distancia a la celda con red más cercana (m)",
    "dist_nucleo": "Distancia al núcleo consolidado más cercano (m)",
}


# ---------------------------------------------------------------- carga
def cargar_ejes(ruta, capa=None):
    """Lee una capa de ejes y devuelve segmentos LineString simples en CRS_TRABAJO."""
    g = gpd.read_file(ruta, layer=capa, engine="pyogrio")
    if g.crs is None:
        raise ValueError(f"{Path(ruta).name} no tiene CRS declarado")
    if g.crs.to_string() != CRS_TRABAJO:
        g = g.to_crs(CRS_TRABAJO)
    geoms = shapely.force_2d(g.geometry.values)
    geoms = geoms[~(shapely.is_missing(geoms) | shapely.is_empty(geoms))]
    partes = shapely.get_parts(shapely.line_merge(geoms))
    return partes[shapely.get_type_id(partes) == 1]          # 1 = LineString


def metricas_segmento(geoms):
    """Por segmento: longitud, punto medio y sumas de cos 2θ y sin 2θ ponderadas por longitud.

    El ángulo se duplica porque los ejes no tienen sentido: la orientación es axial (θ ≡ θ + 180°).
    """
    n = len(geoms)
    coords, idx = shapely.get_coordinates(geoms, return_index=True)
    mismo = idx[1:] == idx[:-1]
    d = (coords[1:] - coords[:-1])[mismo]
    sl, th, sid = np.hypot(d[:, 0], d[:, 1]), np.arctan2(d[:, 0], d[:, 1]), idx[:-1][mismo]
    medio = shapely.line_interpolate_point(geoms, 0.5, normalized=True)
    return pd.DataFrame({
        "L": np.bincount(sid, sl, minlength=n),
        "c2": np.bincount(sid, sl * np.cos(2 * th), minlength=n),
        "s2": np.bincount(sid, sl * np.sin(2 * th), minlength=n),
        "x": shapely.get_x(medio), "y": shapely.get_y(medio),
    })


def marcar_red_nueva(geoms_act, geoms_ant, dist=DIST_NUEVA):
    """True para los segmentos actuales cuyo punto medio está a más de `dist` m de la red antigua.

    Mide la red nueva por geometría, sin depender de las delimitaciones de área.
    """
    medio = shapely.line_interpolate_point(geoms_act, 0.5, normalized=True)
    arbol = shapely.STRtree(geoms_ant)
    i_act, _ = arbol.query(medio, predicate="dwithin", distance=dist)
    nueva = np.ones(len(geoms_act), bool)
    nueva[np.unique(i_act)] = False
    return nueva


# ---------------------------------------------------------------- malla y variables
def _vecindad(a, r):
    """Media de `a` en la ventana cuadrada de radio r celdas (lado 2r + 1)."""
    return ndimage.uniform_filter(a, size=2 * r + 1, mode="constant", cval=0.0)


def variables_celdas(seg_ant, tam=TAM_CELDA, radio=RADIO_UNIVERSO, margen=None):
    """Variables de la red antigua por celda, para las celdas a `radio` m o menos de esa red.

    seg_ant: DataFrame de `metricas_segmento` de la versión antigua.
    Devuelve un DataFrame con ix, iy (índices de celda), x, y (centro, m) y VARIABLES.
    """
    margen = margen if margen is not None else int(np.ceil(radio / tam)) + 4
    ix = np.floor(seg_ant.x.to_numpy() / tam).astype(np.int64)
    iy = np.floor(seg_ant.y.to_numpy() / tam).astype(np.int64)
    x0, y0 = ix.min() - margen, iy.min() - margen
    nx, ny = ix.max() - x0 + margen + 1, iy.max() - y0 + margen + 1
    plano = (iy - y0) * nx + (ix - x0)

    def raster(valores):
        return np.bincount(plano, valores, minlength=nx * ny).reshape(ny, nx)

    L, c2, s2 = raster(seg_ant.L.to_numpy()), raster(seg_ant.c2.to_numpy()), raster(seg_ant.s2.to_numpy())
    km2 = (tam / 1000) ** 2
    dens = L / 1000 / km2
    con_red = L > 0

    dens_v1, dens_v2, dens_v4 = _vecindad(dens, 1), _vecindad(dens, 2), _vecindad(dens, 4)
    frac_v1 = (_vecindad(con_red.astype(float), 1) * 9 - con_red) / 8
    Lv = _vecindad(L, 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        coher = np.where(Lv > 0, np.hypot(_vecindad(c2, 2), _vecindad(s2, 2)) / Lv, 0.0)
    dist_red = ndimage.distance_transform_edt(~con_red) * tam
    nucleo = dens_v2 >= DENS_NUCLEO
    # sin núcleos (red muy dispersa) la distancia queda en la diagonal de la malla
    dist_nucleo = ndimage.distance_transform_edt(~nucleo) * tam if nucleo.any() else np.full(L.shape, np.hypot(nx, ny) * tam)

    jy, jx = np.nonzero(dist_red <= radio)
    return pd.DataFrame({
        "ix": jx + x0, "iy": jy + y0,
        "x": (jx + x0 + 0.5) * tam, "y": (jy + y0 + 0.5) * tam,
        "dens": dens[jy, jx], "dens_v1": dens_v1[jy, jx], "dens_v2": dens_v2[jy, jx], "dens_v4": dens_v4[jy, jx],
        "frac_v1": frac_v1[jy, jx], "coher_v2": coher[jy, jx],
        "dist_red": dist_red[jy, jx], "dist_nucleo": dist_nucleo[jy, jx],
    })


def longitud_por_celda(seg, celdas, tam=TAM_CELDA):
    """Suma la longitud de `seg` (m) en cada celda de `celdas`; 0 si la celda no tiene segmentos."""
    s = (pd.DataFrame({"ix": np.floor(seg.x.to_numpy() / tam).astype(np.int64),
                       "iy": np.floor(seg.y.to_numpy() / tam).astype(np.int64), "L": seg.L.to_numpy()})
         .groupby(["ix", "iy"]).L.sum())
    return s.reindex(pd.MultiIndex.from_frame(celdas[["ix", "iy"]])).fillna(0).to_numpy()


def tabla_celdas(ruta_ant, ruta_act=None, tam=TAM_CELDA):
    """Tabla de celdas con las variables 2014 y, si hay versión actual, la red nueva observada.

    Columnas extra con versión actual: `m_nuevos` (m de red nueva) y `crece` (1 si m_nuevos >= MIN_NUEVA_M).
    """
    g_ant = cargar_ejes(ruta_ant)
    seg_ant = metricas_segmento(g_ant)
    celdas = variables_celdas(seg_ant, tam)
    info = {"segmentos_ant": len(g_ant), "km_ant": seg_ant.L.sum() / 1000}
    if ruta_act is not None:
        g_act = cargar_ejes(ruta_act)
        seg_act = metricas_segmento(g_act)
        nueva = marcar_red_nueva(g_act, g_ant)
        celdas["m_nuevos"] = longitud_por_celda(seg_act[nueva], celdas, tam)
        celdas["crece"] = (celdas.m_nuevos >= MIN_NUEVA_M).astype(int)
        info.update({"segmentos_act": len(g_act), "km_act": seg_act.L.sum() / 1000,
                     "km_nuevos": seg_act.L[nueva].sum() / 1000,
                     "km_nuevos_en_universo": celdas.m_nuevos.sum() / 1000})
    return celdas, info


# ---------------------------------------------------------------- tipología y predicción
def matriz_cluster(celdas):
    """Variables de la tipología, con logaritmo en las de cola larga."""
    X = celdas[VARIABLES_CLUSTER].copy()
    X[EN_LOG] = np.log1p(X[EN_LOG])
    return X


def predecir(celdas, paquete):
    """Añade `tipo` (cluster) y `prob` (probabilidad de red nueva) a la tabla de celdas."""
    celdas = celdas.copy()
    # el KMeans numera los grupos al azar; `orden` los renumera de más a menos consolidado
    celdas["tipo"] = np.asarray(paquete["orden"])[paquete["tipologia"].predict(matriz_cluster(celdas))]
    celdas["prob"] = paquete["modelo"].predict_proba(celdas[VARIABLES])[:, 1]
    return celdas


def esquinas_wgs84(celdas, tam=TAM_CELDA):
    """Polígono de cada celda en longitud/latitud, para el mapa. Devuelve una lista de anillos."""
    from pyproj import Transformer
    t = Transformer.from_crs(CRS_TRABAJO, "EPSG:4326", always_xy=True)
    x, y, h = celdas.x.to_numpy(), celdas.y.to_numpy(), tam / 2
    anillos = []
    for dx, dy in [(-h, -h), (h, -h), (h, h), (-h, h)]:
        lon, lat = t.transform(x + dx, y + dy)
        anillos.append(np.c_[lon, lat])
    return np.stack(anillos, axis=1).round(6).tolist()
