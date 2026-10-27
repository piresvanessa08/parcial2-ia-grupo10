# Segundo Parcial de Inteligencia Artificial — Grupo 10

**Estudiante:** Tatiana Vanessa Alvis  
**Tema:** Comercio — clientes de tienda en Cartago  
**Archivo asignado:** `grupo_10.csv`, tomado de `Datasets (3).zip`

El proyecto utiliza exclusivamente los datos asignados al grupo 10. El CSV original
se conserva sin modificar en `data/grupo_10.csv`; las correcciones se guardan en
`data/grupo_10_limpio.csv`. El identificador y el barrio se conservan para trazabilidad,
pero el perceptrón usa únicamente `visitas_mes` y `ticket_promedio_miles`.

## Entorno

Requisitos: Docker con Docker Compose. La imagen usa Python 3.12 e instala NumPy,
Pandas, Matplotlib y Seaborn desde `requirements.txt`.

### Reproducir desde WSL2 (Ubuntu)

Abre Ubuntu en WSL2 y, desde la carpeta del curso, clona el repositorio si aún no
está allí. Si ya tienes el proyecto, entra a su carpeta:

```bash
mkdir -p ~/CursoIA
cd ~/CursoIA
git clone https://github.com/piresvanessa08/parcial2-ia-grupo10.git parcial2
cd parcial2
```

Si el repositorio ya existe en `~/CursoIA/parcial2`, omite `git clone` y ejecuta
`cd ~/CursoIA/parcial2`. Luego construye el contenedor y reproduce todo el análisis:

```bash
docker compose up -d --build
docker compose exec parcial2 python src/parcial2.py
```

El montaje del proyecto hace que el CSV limpio y los gráficos generados queden
guardados en el equipo anfitrión. Para detener el contenedor:

```bash
docker compose down
```

También se puede ejecutar sin Docker con Python 3.12: instalar
`requirements.txt` en un entorno virtual y ejecutar `python src/parcial2.py`.

## Inspección

El script carga el archivo original con Pandas e imprime las primeras cinco filas,
la forma, los tipos, los nulos y las filas duplicadas antes de limpiar. La lectura
observada del CSV tiene **81 filas y 5 columnas**: 80 registros identificados como
clientes (`K001` a `K080`) y una fila final de resumen (`TOTAL`).

La columna `cliente_id` y `barrio` son texto; las otras tres columnas aparecen como
`float64`, aunque visitas y clase representan enteros. En el archivo hay campos
vacíos y una fila resumen con valores como `832.0` y `203413`; Pandas promueve las
columnas numéricas con faltantes a `float64`. No se detectó texto dentro de las
columnas de entrada numéricas. La salida completa de inspección, incluidos los
conteos de nulos y duplicados, se reproduce ejecutando el script.

## Limpieza

Las decisiones se basan en observaciones del CSV, no en reglas aprendidas del
objetivo:

| Evidencia reproducible | Decisión | Justificación |
|---|---|---|
| Una fila tiene `cliente_id=TOTAL`, sin barrio ni clase y con valores agregados (`832.0`, `203413`). | Excluir esa fila resumen del conjunto de clientes. | No representa a una persona cliente; incluirla sesgaría las entradas y no tiene una etiqueta válida. |
| `visitas_mes` está vacío en K009, K027 y K046 (3 de 80 clientes). | Imputar esos tres valores con la mediana observada entre clientes, **10 visitas**. | Conserva los 80 registros sin usar un valor extremo ni crear una categoría nueva. |
| El ticket de K058 es **191000.0**, mientras los otros tickets originales de clientes están entre **12 y 299** y su mediana es **144**. | Corregir K058 de **191000** a **191** miles de pesos. | La magnitud aislada es compatible con un error de ceros finales; 191 queda en el rango observado. La regla y la corrección explícita están en el script para que no se corrija silenciosamente otro archivo. |
| Hay variantes con mayúsculas/minúsculas o espacios en `barrio` (por ejemplo, `guadalupe`, `GUADALUPE`, `Zaragoza ` y ` Centro`). | Quitar espacios externos y unificar mayúsculas/minúsculas del nombre. | Son variantes de escritura, no categorías distintas. |

No se encontraron filas duplicadas entre los clientes, ni identificadores repetidos:
no se elimina ningún cliente por duplicación. `cliente_id` y `barrio` tampoco se
convierten en entradas del modelo. El número de clientes antes de limpiar es **80**
(81 filas contando el resumen); quedan **80** registros limpios. Sus cantidades por
clase son **47 de clase 0** y **33 de clase 1**. La media del ticket antes/después
de corregir K058 es **2542.6625** y **157.5500** miles de pesos, respectivamente.
El archivo resultante se guarda en `data/grupo_10_limpio.csv`.

### Valor sospechoso y escalado

El valor elegido es el ticket `191000.0` de K058. La media del ticket entre los
80 clientes antes de la corrección es **2542.6625** miles de pesos; después de
reemplazarlo por `191`, es **157.5500**. Sin corregir, ese valor elevaría fuertemente
la media y la desviación estándar usadas por el estandarizador, comprimiendo los
tickets normales alrededor de cero y reduciendo su influencia relativa sobre el
perceptrón. Esas dos estadísticas que se usan para escalar se calculan sobre las
filas de entrenamiento y se imprimen para poder reproducir el resultado.

## Visualizaciones

El script guarda gráficos legibles, con título y ejes etiquetados, en `graficos/`:

- `dispersion_clientes.png`: visitas al mes frente a ticket promedio, con color por
  clase (azul = ocasional, naranja = recurrente). En estos datos, la clase recurrente
  tiene medianas mayores de visitas y ticket (15 y 254) que la ocasional (9 y 90),
  aunque hay solapamiento entre clases. Es una asociación descriptiva, no evidencia
  que una variable cause la otra.
- `boxplot_visitas_clase.png`: distribución de las visitas mensuales para las clases
  ocasional (0) y recurrente (1). La mediana es 9 visitas para ocasionales y 15 para
  recurrentes; las distribuciones se solapan, por lo que las visitas por sí solas no
  separan perfectamente a los clientes. Esto no implica causalidad.

## Modelo

El perceptrón logístico está implementado desde cero con operaciones de NumPy:
sigmoide, probabilidades `sigmoid(X @ w + b)`, pérdida logarítmica, exactitud y
actualización por descenso del gradiente. No utiliza Scikit-Learn ni otra biblioteca
de aprendizaje automático. Usa solo las dos entradas numéricas; ni `cliente_id` ni
`barrio` se usan como predictores.

Las medias y desviaciones estándar se calculan sobre los datos usados para entrenar,
se imprimen y se reutilizan para estandarizar los ejemplos nuevos. El script entrena
**1000 épocas** y registra pérdida y exactitud en cada época. Guarda además:

- `graficos/curva_aprendizaje.png`: pérdida logarítmica por época.
- `graficos/frontera_decision.png`: clases y frontera p=0.5 representadas en las
  unidades originales de visitas y miles de pesos; los errores del umbral se
  resaltan.

## Resultados e interpretación

Al ejecutar `python src/parcial2.py`, la consola informa los pesos iniciales y
finales, pérdida inicial y final, exactitud, registros mal clasificados, estadísticas
de escalado y probabilidades de ejemplo. Los resultados detallados dependen de la
versión instalada de las bibliotecas y son reproducibles ejecutando el script sobre
el CSV incluido.

- **Signos de los pesos:** el signo de cada coeficiente final indica la dirección de
  asociación con la probabilidad estimada cuando la otra entrada se mantiene fija.
  En esta ejecución `w1 = 2.726407` (visitas mensuales) y
  `w2 = 2.618601` (ticket promedio) son positivos: al aumentar cualquiera de las
  entradas, manteniendo fija la otra, aumenta la probabilidad estimada de
  recurrencia. Para una tienda de Cartago, la dirección de esta asociación es
  razonable como hipótesis descriptiva: más visitas y un ticket mayor pueden
  acompañar la recurrencia. Sin embargo, estos 80 registros no demuestran que el
  patrón represente a toda Cartago ni que esas variables causen la recurrencia.
- **Separación y errores:** la frontera es una recta en estas dos entradas. El script
  muestra cuántos clientes quedan al lado equivocado usando el umbral 0.5. Con
  pérdida final **0.229269**, exactitud de entrenamiento **0.9125 (91.25 %)** y
  **7 de 80** errores, la separación es aproximadamente lineal, aunque no perfecta.
  Este conteo es sobre los mismos registros usados para entrenar, no una evaluación
  independiente.
- **Registros de prueba plausibles:** se prueban `(15 visitas, ticket 250)` como
  ejemplo esperado de recurrente y `(5 visitas, ticket 60)` como ejemplo esperado
  de ocasional. Ambos están dentro de los rangos observados tras limpiar
  (1–19 visitas y 12–299 miles de pesos por ticket); uno representa valores
  relativamente altos y el otro valores más bajos. Con la media de entrenamiento
  `[10.775, 157.550]` y la desviación estándar `[5.3221, 90.7689]`, sus
  probabilidades calculadas fueron **0.971525** (clase 1) y **0.000846** (clase 0);
  ambas coinciden con lo esperado. El script realiza el escalado y muestra las
  predicciones al ejecutarse.
- **Recomendación:** priorizar una prueba piloto de fidelización dirigida a clientes
  ocasionales, midiendo cambios en visitas y recurrencia antes de ampliar la
  estrategia. La decisión se sustenta en que el archivo contiene **47 clientes
  ocasionales y 33 recurrentes**; las probabilidades, errores y pesos impresos
  permiten precisar el alcance, sin asumir que una visita o un ticket cause
  recurrencia.
- **Limitaciones:** son solo 80 clientes de un archivo y una fotografía del sistema;
  la limpieza del valor K058 es una inferencia que debe contrastarse con el registro
  de origen; las clases se solapan y hay desbalance (47 frente a 33); el perceptrón
  solo representa una frontera lineal y no usa el barrio; la exactitud y los errores
  reportados son de entrenamiento y no garantizan desempeño con clientes futuros.
  No debe automatizar decisiones individuales sin validación temporal/externa,
  seguimiento y revisión humana.

## Repositorio y publicación en GitHub

El nombre requerido es **`parcial2-ia-grupo10`** y la visibilidad debe ser pública.
El repositorio público ya está publicado bajo la cuenta `piresvanessa08`:
[github.com/piresvanessa08/parcial2-ia-grupo10](https://github.com/piresvanessa08/parcial2-ia-grupo10).
La rama `main` contiene los avances del parcial, con más de los tres commits
requeridos.

```bash
git log --oneline -4
```

El directorio `parcial2` es un repositorio Git independiente del repositorio del
curso. Para reproducir las entregas futuras, usa esta carpeta como directorio de
trabajo. La entrega del enlace del repositorio en AVACO queda a cargo del estudiante.
