---
marp: true
paginate: true
size: 16:9
---

# ¿Dónde aparece red vial nueva?
## Tipología de celdas y modelo de crecimiento urbano con ejes viales del INEC

**Fundamentos de Inteligencia Artificial**

Darío Echeverría · Noviembre de 2026

App: `https://<nombre>.streamlit.app` · Repositorio: `https://github.com/<usuario>/<repositorio>`

---

## 1. Introducción

**Problema.** La difusión de los datos geográficos del Marco Geoestadístico Nacional (MGN) puede ser más dinámica, y a la vez presentar comparaciones con los conjuntos de datos del censo previo.

**Pregunta.** Con la forma de la red de 2014, ¿se puede anticipar en qué lugares aparece red nueva en 2021?

**Punto de partida.** Un proyecto anterior modeló el crecimiento por área amanzanada (763 áreas del país) y ningún modelo superó a predecir la media:

| Modelo por área | R² en validación cruzada |
|---|---|
| Referencia (media) | −0,010 |
| Regresión lineal | 0,002 |
| Random forest | −0,055 |

---

## 2. Temática: de áreas a celdas

**Por qué falló el modelo por área**

- Unidades no comparables: Guayaquil pesa lo mismo que una localidad de 50 segmentos.
- El crecimiento se medía con la delimitación de 2014, no con la red.
- El 35 % de las áreas tenía expansión cero y se usó regresión ordinaria.

**Reformulación**

| | Antes | Ahora |
|---|---|---|
| Unidad | Área amanzanada | Celda de 250 m |
| Red nueva | Ejes fuera del polígono 2014 | Ejes 2021 a más de 30 m de la red 2014 |
| Tarea | Regresión | Clasificación: la celda recibe red nueva o no |
| Validación | Aleatoria | Espacial, por bloques de 2 km |

**Caso de estudio:** cantón Ibarra. 662 km de red en 2014, 813 km en 2021 y 118 km identificados como nuevos.

---

## 3. Metodología

1. **Segmentos.** Limpieza de los ejes y cálculo de longitud, punto medio y orientación axial.
2. **Malla de 250 m.** Se modelan las 1.564 celdas a 500 m o menos de la red 2014; el 23 % recibe red nueva.
3. **Ocho variables de 2014** por celda: densidad vial propia y de la vecindad (3×3, 5×5, 9×9), fracción de vecinas con red, coherencia de orientación, distancia a la red y distancia al núcleo consolidado.
4. **Tipología (no supervisado).** KMeans con k = 4: núcleo consolidado, borde urbano, periferia inmediata y entorno disperso.
5. **Clasificación (supervisado).** Regresión logística y random forest, en tres variantes: global, con el cluster como variable y un modelo por cluster.
6. **Evaluación.** Validación cruzada espacial de 5 pliegues; métricas AUC, precisión media y Brier.
7. **Despliegue.** Modelo serializado con `joblib` y app pública en Streamlit, que calcula las variables con el mismo módulo del entrenamiento (`urbano.py`).

---

## 4. Resultados

| Modelo | AUC | Precisión media |
|---|---|---|
| Referencia (proporción) | 0,500 | 0,201 |
| Solo tipología | 0,695 | 0,310 |
| Regresión logística global | 0,774 | 0,435 |
| **Random forest global** | **0,782** | **0,479** |
| Random forest + cluster como variable | 0,781 | 0,473 |
| Random forest, un modelo por cluster | 0,777 | 0,473 |

- **El modelo por celda sí supera a la referencia.** Con umbral 0,25 señala 655 celdas: el 42 % recibe red nueva (23 % al azar) y detecta el 77 % de las que crecen.
- **Los clusters describen, el modelo global predice.** Celdas con red nueva por tipo: 37 %, 37 %, 11 % y 3 %. Modelar por cluster no mejora al modelo global.
- **Pesan la densidad de la celda y la de su vecindad inmediata:** la red nueva aparece junto a la red existente.

![w:760 Tipología de celdas y red nueva observada en Ibarra](img/tipologia_y_red_nueva.png)

---

## 5. Conclusiones y recomendaciones

**Conclusiones**

- Cambiar la unidad de análisis dio capacidad predictiva a los mismos datos: de R² ≈ 0 por área a AUC 0,78 por celda.
- La capacidad es moderada y el modelo es de contigüidad: indica dónde es más probable la red nueva, no por qué.
- La tipología no forma grupos naturales (silueta máxima en k = 2); es un gradiente del centro a la periferia, útil para describir.
- La validación aleatoria infla el resultado (AUC 0,80 frente a 0,78): con datos espaciales hay que validar por bloques.

**Límites**

- Entrenado con una sola ciudad; su error en otras no está medido.
- Parte de la red "nueva" es diferencia de levantamiento entre las dos bases.

**Recomendaciones**

- Entrenar con varias ciudades y validar dejando ciudades completas fuera.
- Añadir pendiente, población censal y uso de suelo.
- Excluir ejes no vehiculares para reducir el efecto del levantamiento.
