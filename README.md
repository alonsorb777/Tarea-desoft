# Proyecto Desarrollo de Software 2026-1
Repositorio Desoft: Segmentación de estructuras en discos protoplanetarios usando DSHARP + SAM

## Descripción

Este programa permite identificar y caracterizar estructuras (anillos, gaps, asimetrías) en discos protoplanetarios, utilizando datos reales del survey DSHARP de ALMA y técnicas modernas de segmentación de imágenes basadas en el modelo Segment Anything Model (SAM).  

## Objetivos

-Integrar modelo de SAM ya entrenado
-Cargar las imágenes .fits de los discos desde DSHARP
-Integrar el modelo SAM para realizar la segmentación 
-Poder guardar máscaras ya generadas
-Visualización de imágenes mediante interfaz gráfica
-Acercar la SAM con investigadores y estudiantes

## Tecnologías

-Python
-Astropy
-Matplotlib
-Segment Anything (SAM)
-Git y GitHub
-Pytest
-Torch

Cliente: Sebastán Peréz. Academico de la Universidad De Santiago especializado en formación exo-planetaria.

Como utilizarlo:
- Colocar la el archivo que se desea analizar en la carpeta de data
- En scr/main.py ingresar la ruta del archivo que se desea analizar
- Ejecutar main.py

Estado del Desarrollo: Sprint 2.

### Colaboradores: Christian Álvarez, Lindsay Rincón Torres, Alonso Robredo Bovet, Simona Serqueira y Sebastián Torres Carrasco.
