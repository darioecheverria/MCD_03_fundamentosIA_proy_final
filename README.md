# ¿Dónde aparece red vial nueva?

App de Streamlit que clasifica celdas de 250 m según el tejido vial de 2014 (tipología por KMeans) y estima la probabilidad de que reciban red vial nueva en 2021 (random forest). Proyecto final de Fundamentos de Inteligencia Artificial.

## Contenido

| Archivo | Para qué sirve |
|---|---|
| `01_entrenamiento.ipynb` | Tabla de celdas, tipología, comparación de modelos y exportación de `modelo/paquete.joblib` |
| `02_despliegue.ipynb` | Genera `app.py`, `requirements.txt` y `.streamlit/config.toml`, prueba la app y documenta el despliegue |
| `urbano.py` | Cálculo de variables; lo importan el entrenamiento y la app |
| `app.py` | La app |
| `modelo/paquete.joblib` | Tipología + modelo + métricas (0,8 MB) |
| `datos/` | Geopaquetes de ejes por ciudad |
| `requirements.txt` | Dependencias con scikit-learn y numpy fijados |

## Insumos: qué geopaquetes preparar

Dos archivos por ciudad en `datos/`:

```
datos/ejes_antigua_<ciudad>.gpkg     # base 2014 (obligatorio)
datos/ejes_actual_<ciudad>.gpkg      # CPV2021 (opcional en la app, obligatorio para entrenar)
```

| Característica | Requisito |
|---|---|
| Nombre | `<ciudad>` en minúsculas, sin espacios ni tildes (`santo_domingo`). La app lista las ciudades a partir de estos nombres |
| Capa | Una sola capa por archivo (se lee la primera) |
| Geometría | `LineString` o `MultiLineString`; se descarta Z/M |
| CRS | Declarado y proyectado en metros. Ideal: EPSG:31992; si es otro, se reproyecta |
| Atributos | No se usa ninguno. Conviene exportar solo la geometría: el archivo pesa menos |
| **Misma zona en las dos versiones** | Las dos capas deben salir del **mismo filtro** (mismas parroquias). Si la capa actual cubre más territorio, todo lo que sobra se contará como red nueva |
| Tamaño | Hasta 25 MB por archivo si subes por la web de GitHub (100 MB con `git push`). Ibarra pesa 2,0 y 3,3 MB |

Exportación desde los parquet nacionales (por cantón: los cuatro primeros dígitos de `parroquia`):

```python
import geopandas as gpd

CIUDAD, CANTON = "ibarra", "1001"
for version in ["antigua", "actual"]:
    g = gpd.read_parquet(f"insumos/ejes_{version}.parquet", columns=["parroquia", "geometry"],
                         filters=[("provincia", "==", CANTON[:2])])
    g = g[g.parroquia.astype(str).str.zfill(6).str[:4] == CANTON]
    g[["geometry"]].to_file(f"datos/ejes_{version}_{CIUDAD}.gpkg", driver="GPKG", engine="pyogrio")
```

Los parquet nacionales no van al repositorio ni a la app.

## Uso local

```bash
pip install -r requirements.txt
jupyter execute 01_entrenamiento.ipynb     # opcional: vuelve a entrenar
streamlit run app.py
```

Para añadir una ciudad **solo a la app**, se deben copiar sus geopaquetes a `datos/`. Para **entrenar** con ella, añádela además a `CIUDADES` en `01_entrenamiento.ipynb` y vuelve a ejecutarlo.

## Límites del modelo

- Entrenado con una sola ciudad (Ibarra); su error en otras ciudades no está medido.
- AUC 0,78 en validación cruzada espacial: capacidad moderada.
- La red "nueva" incluye diferencias de levantamiento entre las dos bases.
- Solo se modelan celdas a 500 m o menos de la red 2014.
