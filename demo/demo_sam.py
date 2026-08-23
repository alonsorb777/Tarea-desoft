import os
import glob
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch
from astropy.io import fits

# ---------------------------------------------------------
# IMPORTACIÓN DIRECTA DE MÓDULOS (src/)
# ---------------------------------------------------------
from src.descarga import descargar_continuum_dsharp, DEFAULT_DATA_DIR
from src.segmentacion.sam_segmentacion import (
    obtener_checkpoint,
    cargar_modelo,
    preparar_imagen,
    reducir_imagen_para_sam,
    generar_mascaras,
    ordenar_mascaras
)

#Función para filtrar máscaras que ocupan demasiado espacio (fondo) y limitar a las top 10 más relevantes
def filtrar_y_limitar_mascaras(masks, max_area_ratio=0.40, top_n=10):
    """
    Filtra máscaras que corresponden al fondo (ocupan un área excesiva)
    y retiene solo las top 10 máscaras más relevantes.
    """
    if not masks:
        return []
    
    # Obtener dimensiones de la imagen desde la primera máscara
    h, w = masks[0]["segmentation"].shape
    area_total = h * w
    
    mascaras_filtradas = []
    for m in masks:
        area_mask = m.get("area", np.sum(m["segmentation"]))
        # Descartar si la máscara ocupa más del 40% de la imagen total (Fondo)
        if (area_mask / area_total) <= max_area_ratio:
            mascaras_filtradas.append(m)
            
    # Retorna las primeras 10 máscaras más relevantes (según el orden original)
    mascaras_filtradas.sort(key=lambda x: x.get("predicted_iou", 0), reverse=True)  # Ordenar por IoU predicho
    return mascaras_filtradas[:top_n]

st.set_page_config(page_title="DSHARP + SAM Analyzer", layout="wide")

# 1. DETECCIÓN DE HARDWARE (CPU / GPU)
@st.cache_resource
def detectar_hardware():
    if torch.cuda.is_available():
        return "cuda", f"GPU CUDA ({torch.cuda.get_device_name(0)})"
    return "cpu", "CPU (Procesamiento por software)"

device, device_label = detectar_hardware()

st.title("Segmentación de Discos Protoplanetarios (DSHARP + SAM)")
st.caption(f"**Dispositivo detectado:** {device_label}")

# 2. SELECCIÓN O DESCARGA DEL DISCO PROTOPLANETARIO
st.sidebar.header("1. Selección del Disco")

local_fits = glob.glob(os.path.join(DEFAULT_DATA_DIR, "*.fits"))

origen = st.sidebar.radio(
    "Fuente de datos:",
    ["Discos en carpeta local (data/)", "Descargar de DSHARP (Web)", "Subir archivo .fits local"]
)

fits_file_path = None
raw_fits_data = None

if origen == "Discos en carpeta local (data/)":
    if local_fits:
        selected = st.sidebar.selectbox("Selecciona un disco:", local_fits)
        fits_file_path = selected
        with fits.open(selected) as hdul:
            raw_fits_data = hdul[0].data
    else:
        st.sidebar.warning("No hay archivos .fits en la carpeta 'data/'.")

elif origen == "Descargar de DSHARP (Web)":
    # Lista de discos representativos de DSHARP
    discos_dsharp = ["AS209", "HD163296", "Elias24", "GWLup", "HTLup", "IMLup", "MYLup", "WaOph6"]
    disco_target = st.sidebar.selectbox("Selecciona el objetivo DSHARP:", discos_dsharp)
    
    if st.sidebar.button(f"Descargar {disco_target}"):
        with st.spinner(f"Descargando continuum de {disco_target} desde ALMA..."):
            descargar_continuum_dsharp(disco_target)
            st.sidebar.success(f"¡{disco_target} descargado en data/!")
            st.rerun()

elif origen == "Subir archivo .fits local":
    uploaded = st.sidebar.file_uploader("Cargar imagen .fits", type=["fits"])
    if uploaded is not None:
        fits_file_path = uploaded
        with fits.open(io.BytesIO(uploaded.getvalue())) as hdul:
            raw_fits_data = hdul[0].data

# Ajustar dimensiones de la matriz astronómica (4D/3D a 2D)
if raw_fits_data is not None:
    while raw_fits_data.ndim > 2:
        raw_fits_data = raw_fits_data[0]

# 3. GENERACIÓN DE MÁSCARAS CON SAM
if fits_file_path is not None and raw_fits_data is not None:
    st.sidebar.success("Imagen FITS cargada correctamente")

    st.sidebar.header("2. Parámetros de SAM")
    criterio_orden = st.sidebar.selectbox(
        "Ordenar máscaras por:",
        ["predicted_iou", "stability_score", "area"]
    )

    if st.sidebar.button("Ejecutar Segmentación SAM"):
        with st.spinner("Procesando estructura con SAM..."):
            # A. Obtener Checkpoint
            checkpoint_path = os.path.join("models", "sam_vit_b_01ec64.pth")
            actual_checkpoint = obtener_checkpoint(checkpoint_path)

            # B. Cargar Modelo usando la función de tu equipo
            predictor = cargar_modelo(actual_checkpoint, device=device)

            # C. Preparar y Reducir Imagen usando tu normalización astronómica (asinh)
            rgb_image = preparar_imagen(raw_fits_data)
            rgb_reducida, escala = reducir_imagen_para_sam(rgb_image, max_size=1500)

            # D. Generar, Ordenar y Filtrar Máscaras
            masks = generar_mascaras(predictor, rgb_reducida)
            masks_ordenadas = ordenar_mascaras(masks, criterio=criterio_orden)
            
            # Aplicar filtro de fondo y limitar a las top 10 estructuras principales
            masks_finales = filtrar_y_limitar_mascaras(masks_ordenadas, max_area_ratio=0.40, top_n=10)

            # Guardar en Estado de Sesión de Streamlit
            st.session_state["masks"] = masks_finales
            st.session_state["rgb_display"] = rgb_reducida
            st.session_state["raw_fits"] = raw_fits_data

# 4. INTERACCIÓN Y ANÁLISIS DE MÁSCARAS
if "masks" in st.session_state:
    masks = st.session_state["masks"]
    rgb_display = st.session_state["rgb_display"]
    raw_fits = st.session_state["raw_fits"]

    st.subheader("Análisis y Caracterización de Estructuras Identificadas")

    # Extraer métricas y características de cada máscara
    stats_list = []
    for i, m in enumerate(masks):
        mask_array = m["segmentation"]
        area_px = int(m.get("area", np.sum(mask_array)))
        iou = round(float(m.get("predicted_iou", 0)), 4)
        stability = round(float(m.get("stability_score", 0)), 4)

        stats_list.append({
            "ID Máscara": i + 1,
            "Área (px)": area_px,
            "IoU Predicho": iou,
            "Estabilidad": stability,
            "BBox [x, y, w, h]": str(m.get("bbox", []))
        })

    df_stats = pd.DataFrame(stats_list)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.write("### Propiedades de las Máscaras")
        selected_ids = st.multiselect(
            "Selecciona las máscaras a visualizar en el mapa del disco:",
            options=df_stats["ID Máscara"].tolist(),
            default=[1] if len(df_stats) > 0 else []
        )
        st.dataframe(df_stats, use_container_width=True)

    with col2:
        st.write("### Visualización de Estructuras")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(rgb_display)

        # Superponer las máscaras seleccionadas
        overlay = np.zeros_like(rgb_display)
        for m_id in selected_ids:
            mask_bin = masks[m_id - 1]["segmentation"]
            overlay[mask_bin] = [255, 100, 0]  # Resaltado en naranja

        ax.imshow(overlay, alpha=0.5)
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)

    # 5. EXPORTACIÓN Y GUARDADO DE RESULTADOS
    st.markdown("---")
    st.subheader(" Exportar Resultados")

    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        csv_bytes = df_stats.to_csv(index=False).encode("utf-8")
        st.download_button(
            " Guardar Métricas (.CSV)",
            data=csv_bytes,
            file_name="caracteristicas_disco.csv",
            mime="text/csv"
        )

    with col_dl2:
        if selected_ids:
            mascaras_dict = {
                f"mask_{m_id}": masks[m_id - 1]["segmentation"]
                for m_id in selected_ids
            }
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **mascaras_dict)
            buffer.seek(0)

            st.download_button(
                " Guardar Máscaras Seleccionadas (.NPZ)",
                data=buffer,
                file_name="mascaras_disco.npz",
                mime="application/octet-stream"
            )
else:
    st.info("Selecciona o descarga un disco protoplanetario para iniciar la segmentación.")