import os
import glob
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch
from astropy.io import fits

# -------------------------------
# Importación de módulos de src/
# -------------------------------
from src.descarga import descargar_continuum_dsharp, DEFAULT_DATA_DIR
from src.segmentacion.sam_segmentacion import (
    obtener_checkpoint,
    cargar_modelo,
    preparar_imagen,
    reducir_imagen_para_sam,
    generar_mascaras,
    ordenar_mascaras
)

# 1. Configuración de la página y estilo 
st.set_page_config(
    page_title="ALMA + SAM Analisis de discos protoplanetarios", 
    page_icon="🌌",
    layout="wide"
)

# Estilo personalizado: Inspirado en el latte cosmico.
st.markdown("""
    <style>
    .stApp {
        background-color: #0d0c10;
        color: #FFF8E7;
    }
    .stSidebar {
        background-color: #16141c;
    }
    h1, h2, h3 {
        color: #FFF8E7 !important;
        font-family: 'Georgia', serif;
    }
    stCaption, .stCaption p {
        color: #d4c5b9 !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #d4a373 0%, #FFF8E7 100%);
        color: #1a1412;
        border: none;
        border-radius: 8px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Helper: Reproducir un sonido discreto al finalizar la segmentación
def reproducir_sonido_exito():
    # Frecuencia de audio corta y limpia (Beep/Chime)
    sound_html = """
        <audio autoplay style="display:none;">
            <source src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3" type="audio/mpeg">
        </audio>
    """
    st.html(sound_html)

# Helper: Filtrar fondo y retener Top 10 máscaras más relevantes según el criterio de ordenamiento
def filtrar_y_limitar_mascaras(masks, max_area_ratio=0.40, top_n=10):
    if not masks:
        return []
    h, w = masks[0]["segmentation"].shape
    area_total = h * w
    mascaras_filtradas = []
    for m in masks:
        area_mask = m.get("area", np.sum(m["segmentation"]))
        if (area_mask / area_total) <= max_area_ratio:
            mascaras_filtradas.append(m)
    return mascaras_filtradas[:top_n]

# 2. CPU/GPU Selección automática y visualización del hardware activo
@st.cache_resource
def detectar_hardware():
    if torch.cuda.is_available():
        return "cuda", f"⚡ GPU CUDA ({torch.cuda.get_device_name(0)})"
    return "cpu", "💻 CPU (Procesamiento por software)"

device, device_label = detectar_hardware()

st.title("🌌 Segmentación de Discos Protoplanetarios")
st.caption(f"**Hardware activo:** {device_label}")

# 3. SELECCIÓN DE ARCHIVO (Limpio y legible)
st.sidebar.header("🛸 1. Selección del Disco")

local_fits = glob.glob(os.path.join(DEFAULT_DATA_DIR, "*.fits"))

# Diccionario para mapear solo el nombre corto del archivo
nombres_limpios_fits = {os.path.basename(f): f for f in local_fits}

origen = st.sidebar.radio(
    "Fuente de datos:",
    ["Discos locales (data/)", "Descargar de DSHARP (Web)", "Subir archivo .fits"]
)

fits_file_path = None
raw_fits_data = None
nombre_disco_display = "disco_desconocido"

if origen == "Discos locales (data/)":
    if nombres_limpios_fits:
        seleccion_limpia = st.sidebar.selectbox("Disco seleccionado:", list(nombres_limpios_fits.keys()))
        fits_file_path = nombres_limpios_fits[seleccion_limpia]
        nombre_disco_display = seleccion_limpia.replace(".fits", "")
        with fits.open(fits_file_path) as hdul:
            raw_fits_data = hdul[0].data
        st.sidebar.info(f"**Archivo activo:** `{seleccion_limpia}`")
    else:
        st.sidebar.warning("No hay archivos .fits en 'data/'.")

elif origen == "Descargar de DSHARP (Web)":
    discos_dsharp = ["AS209", "HD163296", "Elias24", "GWLup", "HTLup", "IMLup", "MYLup", "WaOph6"]
    disco_target = st.sidebar.selectbox("Objetivo DSHARP:", discos_dsharp)
    
    if st.sidebar.button(f"📥 Descargar {disco_target}"):
        with st.spinner(f"Descargando datos de ALMA para {disco_target}..."):
            descargar_continuum_dsharp(disco_target)
            st.sidebar.success(f"¡{disco_target} listo!")
            st.rerun()

elif origen == "Subir archivo .fits":
    uploaded = st.sidebar.file_uploader("Cargar .fits local", type=["fits"])
    if uploaded is not None:
        fits_file_path = uploaded
        nombre_disco_display = uploaded.name.replace(".fits", "")
        with fits.open(io.BytesIO(uploaded.getvalue())) as hdul:
            raw_fits_data = hdul[0].data

# Ajustar dimensiones astronómicas 4D/3D a 2D
if raw_fits_data is not None:
    while raw_fits_data.ndim > 2:
        raw_fits_data = raw_fits_data[0]

# 4. Parametros y procesamiento de segmentación
if fits_file_path is not None and raw_fits_data is not None:
    st.sidebar.header("⚙️ 2. Parámetros SAM")
    criterio_orden = st.sidebar.selectbox(
        "Clasificar máscaras por:",
        ["predicted_iou", "stability_score", "area"]
    )

    if st.sidebar.button("🚀 Ejecutar Segmentación"):
        with st.spinner("🔍 Analizando estructuras del disco protoplanetario..."):
            checkpoint_path = os.path.join("models", "sam_vit_b_01ec64.pth")
            actual_checkpoint = obtener_checkpoint(checkpoint_path)

            predictor = cargar_modelo(actual_checkpoint, device=device)
            rgb_image = preparar_imagen(raw_fits_data)
            rgb_reducida, escala = reducir_imagen_para_sam(rgb_image, max_size=1500)

            masks = generar_mascaras(predictor, rgb_reducida)
            masks_ordenadas = ordenar_mascaras(masks, criterio=criterio_orden)
            masks_finales = filtrar_y_limitar_mascaras(masks_ordenadas, max_area_ratio=0.40, top_n=10)

            st.session_state["masks"] = masks_finales
            st.session_state["rgb_display"] = rgb_reducida
            st.session_state["raw_fits"] = raw_fits_data
            st.session_state["nombre_disco"] = nombre_disco_display
            
            # Reproducir el sonidito al finalizar
            st.session_state["play_sound"] = True

# Reproducir audio si está activado
if st.session_state.get("play_sound", False):
    reproducir_sonido_exito()
    st.session_state["play_sound"] = False

# 5. VVizualizacion interactiva de resultados
if "masks" in st.session_state:
    masks = st.session_state["masks"]
    rgb_display = st.session_state["rgb_display"]
    nombre_disco = st.session_state.get("nombre_disco", "disco")

    st.subheader(f"✨ Análisis de Estructuras Principal: {nombre_disco}")

    stats_list = []
    for i, m in enumerate(masks):
        mask_array = m["segmentation"]
        stats_list.append({
            "ID Máscara": i + 1,
            "Área (px)": int(m.get("area", np.sum(mask_array))),
            "IoU Predicho": round(float(m.get("predicted_iou", 0)), 4),
            "Estabilidad": round(float(m.get("stability_score", 0)), 4)
        })

    df_stats = pd.DataFrame(stats_list)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.write("### 📊 Top 10 Estructuras Identificadas")
        selected_ids = st.multiselect(
            "Selecciona máscaras a inspeccionar:",
            options=df_stats["ID Máscara"].tolist(),
            default=df_stats["ID Máscara"].tolist()
        )
        st.dataframe(df_stats, width='stretch')

    # Figura estilo Latte Cósmico (#FFF8E7)
    COSMIC_LATTE = "#FFF8E7"
    DARK_BG = "#0d0c10"

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), facecolor=DARK_BG)
    
    # Subplot 1: Imagen Original DSHARP
    ax[0].set_facecolor(DARK_BG)
    ax[0].imshow(rgb_display, origin='lower')
    ax[0].set_title(f"{nombre_disco} - Imagen DSHARP", color=COSMIC_LATTE, fontsize=12, pad=10)
    ax[0].set_xlabel("Píxeles", color=COSMIC_LATTE)
    ax[0].set_ylabel("Píxeles", color=COSMIC_LATTE)
    ax[0].tick_params(colors=COSMIC_LATTE)

    # Subplot 2: Contornos de las máscaras
    ax[1].set_facecolor(DARK_BG)
    ax[1].imshow(rgb_display, origin='lower')
    
    # Paleta de colores cálidos estilo café / espresso / latte
    latte_colors = ["#FFF8E7", "#f4e1d2", "#e6ccb2", "#ddb892", "#b08968", "#7f5539", "#9c6644", "#d4a373", "#ccd5ae", "#e9edc9"]

    for idx, m_id in enumerate(selected_ids):
        mask_bin = masks[m_id - 1]["segmentation"]
        color = latte_colors[idx % len(latte_colors)]
        ax[1].contour(mask_bin, levels=[0.5], colors=[color], linewidths=1.3, origin='lower')

    ax[1].set_title(f"Top {len(selected_ids)} máscaras SAM", color=COSMIC_LATTE, fontsize=12, pad=10)
    ax[1].set_xlabel("Píxeles", color=COSMIC_LATTE)
    ax[1].set_ylabel("Píxeles", color=COSMIC_LATTE)
    ax[1].tick_params(colors=COSMIC_LATTE)

    plt.tight_layout()
    
    with col2:
        st.write("### 🔭 Comparativa Visual")
        st.pyplot(fig)

    # 6. DESCARGAS Y EXPORTACIÓN DE RESULTADOS
    st.markdown("---")
    st.subheader("💾 Exportar Resultados del Análisis")

    col_dl1, col_dl2, col_dl3 = st.columns(3)

    # Descarga 1: Imagen Comparativa (PNG)
    with col_dl1:
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", bbox_inches="tight", facecolor='#0b0d17')
        img_buffer.seek(0)
        
        st.download_button(
            "🖼️ Guardar Comparativa (.PNG)",
            data=img_buffer,
            file_name=f"{nombre_disco}_comparativa.png",
            mime="image/png"
        )
    plt.close(fig)

    # Descarga 2: Métricas (CSV)
    with col_dl2:
        st.download_button(
            "📊 Guardar Métricas (.CSV)",
            data=df_stats.to_csv(index=False).encode("utf-8"),
            file_name=f"{nombre_disco}_metricas.csv",
            mime="text/csv"
        )

    # Descarga 3: Máscaras (NPZ)
    with col_dl3:
        if selected_ids:
            mascaras_dict = {f"mask_{m_id}": masks[m_id - 1]["segmentation"] for m_id in selected_ids}
            npz_buffer = io.BytesIO()
            np.savez_compressed(npz_buffer, **mascaras_dict)
            npz_buffer.seek(0)

            st.download_button(
                "📦 Guardar Máscaras (.NPZ)",
                data=npz_buffer,
                file_name=f"{nombre_disco}_mascaras.npz",
                mime="application/octet-stream"
            )
else:
    st.info("👈 Selecciona o descarga un disco en el menú lateral para iniciar el análisis.")