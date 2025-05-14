import streamlit as st
import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import io
import base64
import json
from pathlib import Path
from fpdf import FPDF
from datetime import datetime
import os
import time
import tempfile
import re
import pandas as pd
import threading
import requests
import os
import torch

# Crear la carpeta si no existe
os.makedirs("modelo", exist_ok=True)

# URL del modelo en Google Drive (ID obtenido del enlace)
url = "https://drive.google.com/uc?id=1YC5V2r-zGBH0VvvuDCH5nnWFEy2hwUEP"
modelo_path = "modelo/modelo.ckpt"

# Descargar el modelo si no existe o si la descarga anterior fue corrupta
if not os.path.exists(modelo_path) or os.path.getsize(modelo_path) < 100000:
    print("Descargando modelo desde Google Drive...")
    response = requests.get(url, stream=True)
    with open(modelo_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Modelo descargado exitosamente.")

# Verificar que el archivo se descargó correctamente
if not os.path.exists(modelo_path) or os.path.getsize(modelo_path) < 100000:
    raise FileNotFoundError(f"El archivo descargado parece estar corrupto o incompleto: {modelo_path}")

# Cargar el modelo en PyTorch sin errores de serialización
try:
    checkpoint = torch.load(modelo_path, map_location=torch.device("cpu"))
except Exception as e:
    print("Error al cargar el archivo `.ckpt`, convirtiéndolo a `.pth`...")
    checkpoint = torch.load(modelo_path, map_location=torch.device("cpu"), encoding="latin1")

# Guardar el modelo en un formato más seguro (`.pth`)
modelo_pth = "modelo/modelo.pth"
torch.save(checkpoint, modelo_pth)
print("Modelo convertido y guardado en formato .pth correctamente.")

# Ahora, cargar el modelo `.pth`
checkpoint = torch.load(modelo_pth, map_location=torch.device("cpu"))
print("Modelo `.pth` cargado correctamente en PyTorch.")




class CerebellumModelSegmentation(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes
        )
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        self.loss_fn = smp.losses.DiceLoss(mode="multiclass", from_logits=True)

    def forward(self, image):
        image = (image - self.mean) / self.std
        return self.model(image)

def normalize_filename(name):
    # Elimina puntos y guiones y convierte a minúsculas para comparación flexible.
    return re.sub(r"[-.]", "", name).lower()


def generate_groundtruth_mask(uploaded_filename, coco_json_path, input_dir, category_colors):
    """
    Genera la máscara de segmentación basada en anotaciones COCO, ignorando diferencias entre puntos y guiones.
    """
    with open(coco_json_path, "r") as f:
        annotations = json.load(f)

    # Normalizar nombre de imagen subida
    normalized_uploaded = normalize_filename(uploaded_filename)

    # Buscar imagen en JSON por coincidencia flexible
    image_info = next(
        (img for img in annotations["images"]
         if normalize_filename(img["extra"].get("name", "")) == normalized_uploaded),
        None
    )

    if image_info is None:
        return None

    image_path = Path(input_dir) / image_info["file_name"]
    if not image_path.exists():
        return None

    # Crear la máscara
    with Image.open(image_path) as img:
        width, height = img.size

    mask = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(mask)

    valid_category_ids = set(category_colors.keys())

    for ann in annotations["annotations"]:
        if ann["image_id"] == image_info["id"] and ann["category_id"] in valid_category_ids:
            for seg in ann["segmentation"]:
                points = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
                draw.polygon(points, outline=category_colors[ann["category_id"]],
                             fill=category_colors[ann["category_id"]])
    return mask


def image_to_base64(img):
    """
    Convierte una imagen en formato Base64.
    """
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def pad_image_to_multiple(image_pil, multiple=32):
    """
    Rellena una imagen PIL para que sus dimensiones sean divisibles por `multiple`.
    """
    width, height = image_pil.size
    new_width = (width + multiple - 1) // multiple * multiple
    new_height = (height + multiple - 1) // multiple * multiple

    padding_left = (new_width - width) // 2
    padding_top = (new_height - height) // 2
    padding_right = new_width - width - padding_left
    padding_bottom = new_height - height - padding_top

    padded_image = transforms.functional.pad(
        image_pil, (padding_left, padding_top, padding_right, padding_bottom), fill=0
    )
    return padded_image


def predict_mask(image_pil, model):
    """
    Genera una máscara segmentada a partir de una imagen PIL y un modelo.
    """
    # Ajustar las dimensiones de la imagen
    padded_image = pad_image_to_multiple(image_pil)

    # Preprocesar la imagen
    image_tensor = transforms.ToTensor()(padded_image).unsqueeze(0)
    image_tensor = (image_tensor - model.mean) / model.std

    # Realizar la predicción
    with torch.no_grad():
        output = model(image_tensor)
        pred_mask = output.softmax(dim=1).argmax(dim=1).squeeze(0).cpu().numpy()

    # Mapear los valores de la máscara a colores
    color_map = {
        0: (0, 0, 0),       # background: black
        1: (255, 0, 0),     # cerebellum: red
        2: (0, 255, 0),     # cisterna magna: green
        3: (0, 0, 255),     # vermis: blue
    }

    color_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for label, color in color_map.items():
        color_mask[pred_mask == label] = color

    mask_image = Image.fromarray(color_mask)
    return padded_image, mask_image

def load_model():
    arch = "Unetplusplus"
    encoder_name = "resnext50_32x4d"
    in_channels = 3
    out_classes = 4

    model = CerebellumModelSegmentation(arch, encoder_name, in_channels, out_classes)

    checkpoint = torch.load("modelo/da_cerebelum_model-epoch=20-val_loss=0.27.ckpt", map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    model.eval()
    return model


def generate_pdf(patient_name, record_number, segmented_img, original_img, groundtruth_img=None, week=None,
                 logo_sacyl_path="logo_sacyl.png", logo_junta_path="logo_junta.png"):

    pdf = FPDF("P", "mm", "A4")
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()

    # === CONFIGURACIONES ===
    page_width = 210
    left_margin = 10
    top_margin = 10
    rect_width = 190  # 19 cm
    rect_height = 34  # 3.4 cm
    line_v_x = left_margin + 70  # Línea vertical a 7 cm del margen izquierdo
    line_h_y = top_margin + 17  # Línea horizontal a 1.7 cm del margen superior

    # === RECUADRO PRINCIPAL ===
    pdf.set_line_width(0.4)
    pdf.rect(left_margin, top_margin, rect_width, rect_height)  # Recuadro completo
    pdf.line(line_v_x, top_margin, line_v_x, top_margin + rect_height)  # Línea vertical
    pdf.line(left_margin, line_h_y, line_v_x, line_h_y)  # Línea horizontal izquierda

    # === LOGOS ===
    if os.path.exists(logo_sacyl_path):
        pdf.image(logo_sacyl_path, x=left_margin + 2, y=top_margin + 2, w=65, h=14)

    # === DIRECCIÓN (parte inferior izquierda) ===
    pdf.set_xy(left_margin, line_h_y + 1)
    pdf.set_font("Arial", "", 8)
    pdf.multi_cell(70, 4.5,
    "Hospital Universitario de Burgos.\nAvda. Islas Baleares, s/n 09006 -BURGOS-\nTfno:947256246 Fax:null",
    align="C")


    # === DATOS DEL PACIENTE (parte derecha del recuadro) ===
    pdf.set_xy(line_v_x + 2, top_margin + 2)
    pdf.set_font("Arial", "", 9)
    pdf.cell(60, 6, f"N Historia: {record_number}", ln=1)
    pdf.set_x(line_v_x + 2)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(60, 6, patient_name.upper(), ln=1)
    pdf.set_font("Arial", "", 9)
    pdf.set_x(line_v_x + 2)
    pdf.cell(60, 6, f"Fecha informe: {datetime.now().strftime('%Y-%m-%d')}", ln=1)
    if week:
        pdf.set_x(line_v_x + 2)
        pdf.cell(60, 6, f"Semana embarazo: {week}", ln=1)

    def save_temp(img):
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        img.save(temp.name)
        return temp.name

    y_start = 60  # Ajustar posición inicial para dejar espacio debajo del encabezado
    image_width = 160  # Ancho ajustado para alinearse correctamente
    image_height = 100  # Alto ajustado para que ambas entren en el espacio disponible
    spacing = 10  # Espacio entre títulos

    if original_img:
        path = save_temp(original_img)
        pdf.set_font("Arial", "B", 11)
        pdf.set_xy(left_margin, y_start - 10)  # Título ajustado
        pdf.cell(0, 6, "IMAGEN ORIGINAL", ln=1, align="C")
        pdf.image(path, x=(page_width - image_width) / 2, y=y_start, w=image_width, h=image_height)

    if segmented_img:
        y_start += image_height + spacing  # Mover la imagen segmentada debajo de la original
        path = save_temp(segmented_img)
        pdf.set_font("Arial", "B", 11)
        pdf.set_xy(left_margin, y_start - 10)  # Título ajustado
        pdf.cell(0, 6, "IMAGEN SEGMENTADA", ln=1, align="C")
        pdf.image(path, x=(page_width - image_width) / 2, y=y_start, w=image_width, h=image_height)


    # === PIE DE PÁGINA ===
    footer_y = 297 - 15  # 1.5 cm del borde inferior
    pdf.set_line_width(0.3)
    pdf.line(left_margin, footer_y, 210 - left_margin, footer_y)

    if os.path.exists(logo_junta_path):
        pdf.image(logo_junta_path, x=(210 - 12) / 2, y=footer_y - 6, h=12)

    # === EXPORTAR PDF ===
    pdf_buffer = io.BytesIO()
    pdf_buffer.write(pdf.output(dest='S').encode('latin1'))
    pdf_buffer.seek(0)
    return pdf_buffer


def convert_rgb_to_classes(rgb_mask):
    """
    Convierte una imagen RGB con colores específicos a una máscara de clases (entero por píxel).
    """
    if rgb_mask.ndim == 3 and rgb_mask.shape[-1] == 3:
        class_mapping = {
            (255, 0, 0): 1,   # Rojo → Cerebelo
            (0, 255, 0): 2,   # Verde → Cisterna Magna
            (0, 0, 255): 3,   # Azul → Vermis
            (0, 0, 0): 0      # Negro → Fondo
        }

        class_mask = np.zeros((rgb_mask.shape[0], rgb_mask.shape[1]), dtype=np.uint8)

        for color, class_id in class_mapping.items():
            match = np.all(rgb_mask == np.array(color), axis=-1)
            class_mask[match] = class_id

        return class_mask
    else:
        return rgb_mask  # Ya es máscara de clases


def calculate_metrics(true_mask, pred_mask, num_classes=3):
    """
    Calcula Precisión (%), Sensibilidad (%) e IoU (%) por clase, eliminando TP, FP y FN.
    """

    # Convertir ambas a clases desde RGB si fuera necesario
    true_mask = convert_rgb_to_classes(true_mask)
    pred_mask = convert_rgb_to_classes(pred_mask)

    precision_per_class = []
    recall_per_class = []
    iou_per_class = []

    for cls in range(1, num_classes + 1):
        true_class_pixels = (true_mask == cls)
        pred_class_pixels = (pred_mask == cls)

        tp = np.sum(np.logical_and(true_class_pixels, pred_class_pixels))
        fp = np.sum(np.logical_and(~true_class_pixels, pred_class_pixels))
        fn = np.sum(np.logical_and(true_class_pixels, ~pred_class_pixels))

        # Métricas por clase con solo 1 decimal
        precision = round((tp / (tp + fp) * 100), 1) if (tp + fp) > 0 else 0
        recall = round((tp / (tp + fn) * 100), 1) if (tp + fn) > 0 else 0
        iou = round((tp / (tp + fp + fn) * 100), 1) if (tp + fp + fn) > 0 else 0

        precision_per_class.append(precision)
        recall_per_class.append(recall)
        iou_per_class.append(iou)

    class_labels = {1: "Cerebelo", 2: "Cisterna Magna", 3: "Vermis"}

    # 📌 Solo mantener columnas de Precisión, Sensibilidad e IoU
    table_data = pd.DataFrame({
        "Estructura": [class_labels[cls] for cls in range(1, num_classes + 1)],
        "Precisión (%)": precision_per_class,
        "Sensibilidad (%)": recall_per_class,
        "IoU (%)": iou_per_class
    })

    # 📌 Filas con valores promedio
    mean_row = {
        "Estructura": "Media",
        "Precisión (%)": round(np.mean(precision_per_class), 1),
        "Sensibilidad (%)": round(np.mean(recall_per_class), 1),
        "IoU (%)": round(np.mean(iou_per_class), 1)
    }

    table_data = pd.concat([table_data, pd.DataFrame([mean_row])], ignore_index=True)

    return table_data

def run_prediction(input_image, model, result_dict):
    resized_image, mask_image = predict_mask(input_image, model)
    result_dict["resized"] = resized_image
    result_dict["mask"] = mask_image



model = load_model()


st.set_page_config(page_title="Fetal Brain Segmentation", layout="wide")


st.markdown("""
    <div style="text-align: center; padding: 10px 0;">
        <h1 style="font-size: 2.5em; font-weight: 600; color: #2c3e50;">SegmentFetal</h1>
        <p style="font-size: 1.1em; color: #555;">Sube una imagen de ecografía fetal para segmentar automáticamente el cerebelo, la cisterna magna y el vermis.</p>
    </div>
    <hr style="border: none; border-top: 1px solid #ccc; margin-top: 5px;">
    <div style="text-align: center; max-width: 900px; margin: auto; padding-top: 10px;">
        <p style="font-size: 1.05em; line-height: 1.6; color: #333;">
            Esta aplicación web está diseñada para ayudar a los profesionales de la salud en el análisis de imágenes de ecografía 2D del cerebro fetal, centrándose específicamente en la segmentación de estructuras clave del cerebelo. Una vez que se carga una imagen, el sistema identifica y resalta automáticamente el cerebelo, la cisterna magna y el vermis utilizando técnicas de aprendizaje profundo.
            <br><br>
            La herramienta busca proporcionar referencias anatómicas rápidas y confiables para apoyar la evaluación clínica del desarrollo de la fosa posterior en diagnósticos prenatales.
        </p>
    </div>
""", unsafe_allow_html=True)


uploaded_file = st.file_uploader("Sube una imagen de ecografía fetal (JPG o PNG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")
    
    result = {}
    
    preview_container = st.empty()
    preview_base64 = image_to_base64(input_image)
    preview_container.markdown(f"""
        <div style="display: flex; justify-content: center; margin-bottom: 10px;">
            <img src="data:image/png;base64,{preview_base64}" style="max-height: 400px; border: 1px solid #ccc; border-radius: 8px;" />
        </div>
    """, unsafe_allow_html=True)

    thread = threading.Thread(target=run_prediction, args=(input_image, model, result))
    thread.start()

    # Mostrar barra de progreso durante 30 segundos
    progress_bar = st.progress(0)
    progress_text = st.empty()
    progress_text.text("Procesando...")

    for i in range(31):  # 0 a 30
        time.sleep(1)
        progress_bar.progress(i * 100 // 30)

    # Esperar a que la predicción termine
    thread.join()

    # Ocultar barra y texto
    progress_bar.empty()
    progress_text.empty()
    preview_container.empty()

    # Recuperar el resultado
    resized_image = result["resized"]
    mask_image = result["mask"]

        # Mostrar Ground Truth si está disponible
    COCO_JSON_PATH = 'data\\_annotations.coco.json'
    INPUT_IMAGES_DIR = 'data'
    CATEGORY_COLORS = {
        1: (255, 0, 0),    # Cerebelo
        4: (0, 255, 0),    # Cisterna magna
        6: (0, 0, 255),    # Vermis
    }

    # Resultados de Segmentación
    st.markdown("""
        <h4 style='text-align: center; margin-top: 30px; font-size: 2em; color: #2c3e50;'>
            Resultados de Segmentación
        </h4>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style='display: flex; justify-content: center; gap: 30px; margin-top: 10px; margin-bottom: 30px; color: #333; font-size: 0.9em;'>
            <div style='display: flex; align-items: center; gap: 8px;'>
                <div style='width: 15px; height: 15px; background-color: rgb(255,0,0); border: 1px solid #000;'></div>
                <span>Cerebelo</span>
            </div>
            <div style='display: flex; align-items: center; gap: 8px;'>
                <div style='width: 15px; height: 15px; background-color: rgb(0,255,0); border: 1px solid #000;'></div>
                <span>Cisterna Magna</span>
            </div>
            <div style='display: flex; align-items: center; gap: 8px;'>
                <div style='width: 15px; height: 15px; background-color: rgb(0,0,255); border: 1px solid #000;'></div>
                <span>Vermis</span>
            </div>
            <div style='display: flex; align-items: center; gap: 8px;'>
                <div style='width: 15px; height: 15px; background-color: rgb(0,0,0); border: 1px solid #000;'></div>
                <span>Fondo</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)  # Añadir margen antes de las imágenes

    gt_mask = generate_groundtruth_mask(uploaded_file.name, COCO_JSON_PATH, INPUT_IMAGES_DIR, CATEGORY_COLORS)
    if gt_mask:
        
        gt_mask = gt_mask.resize(mask_image.size)

        # Mostrar imágenes lado a lado
        col1, col2, col3 = st.columns(3, gap="large")

        with col1:
            st.image(input_image, caption="Imagen Original", use_container_width=True)

        with col2:
            st.image(mask_image, caption="Imagen Segmentada", use_container_width=True)

        with col3:
            st.image(gt_mask, caption="Máscara Ground Truth", use_container_width=True, clamp=True)

        
        pred_mask_np = np.array(mask_image)
        gt_mask_np = np.array(gt_mask)

        
        num_classes = 3
        table_data = calculate_metrics(gt_mask_np, pred_mask_np, num_classes)
        st.markdown("<h5 style='text-align: center; margin-top: 30px; font-size: 1.5em; color: #2c3e50;'>Métricas de Segmentación</h5>", unsafe_allow_html=True)
        
        html_table = table_data.to_html(index=False, classes="styled-table")

        st.markdown("""
            <style>
            .styled-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 17px;
            }
            .styled-table th, .styled-table td {
                border: 1px solid #ddd;
                padding: 5px;  /* <-- Aquí ajustas la altura */
                text-align: center;
            }
            .styled-table tr:last-child {
                background-color: #f0f0f0;
                font-weight: bold;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown(html_table, unsafe_allow_html=True)

    else:
    # Mostrar imágenes lado a lado
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.image(input_image, caption="Imagen Original", use_container_width=True)

        with col2:
            st.image(mask_image, caption="Imagen Segmentada", use_container_width=True)


    # Formulario de información del paciente
    with st.form("patient_info_form"):
        st.markdown("""
            <h4 style='text-align: center; margin-top: 40px; font-size: 1.8em; color: #2c3e50;'>
                Información del Paciente
            </h4>
        """, unsafe_allow_html=True)
        patient_name = st.text_input("Nombre del paciente", placeholder="Ejemplo: Ana Gómez Ruiz")
        record_number = st.text_input("Número de historia clínica", placeholder="Ejemplo: HCU123456")
        week = st.number_input("Semana de gestación", min_value=10, max_value=40, step=1)
        submitted = st.form_submit_button("Generar Informe")

    if submitted:
        if not patient_name or not record_number:
            st.error("Por favor, completa todos los campos antes de descargar el informe.")
        else:
            pdf_result = {}

            # Función para generar el PDF y guardarlo en el diccionario
            def generate_pdf_thread():
                pdf_buffer = generate_pdf(
                    patient_name=patient_name,
                    record_number=record_number,
                    segmented_img=mask_image,
                    week=week,
                    original_img=input_image,
                    logo_sacyl_path="logo_sacyl.png",
                    logo_junta_path="logo_junta.png"
                )
                pdf_result["buffer"] = pdf_buffer

            # Lanzar hilo
            pdf_thread = threading.Thread(target=generate_pdf_thread)
            pdf_thread.start()

            # Barra de progreso durante 40 segundos
            progress_bar = st.progress(0)
            progress_text = st.empty()
            progress_text.text("Generando informe PDF...")

            for i in range(41):
                time.sleep(1)
                progress_bar.progress(i * 100 // 40)

            # Esperar a que el hilo termine si aún no lo ha hecho
            pdf_thread.join()

            # Ocultar barra
            progress_bar.empty()
            progress_text.empty()

            st.success("Informe generado con éxito. Haz clic en el botón para descargar.")
            # Mostrar botón de descarga cuando esté listo
            st.download_button(
                label="Descargar informe PDF",
                data=pdf_result["buffer"],
                file_name=f"{record_number}_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
            