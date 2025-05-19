import gradio as gr
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
from huggingface_hub import hf_hub_download

def descargar_modelo():
    # Ruta donde se almacenará el modelo descargado
    os.makedirs("modelo", exist_ok=True)
    modelo_path = hf_hub_download(
        repo_id="eirarodriguez/segmentation_checkpoint",  # El ID del repositorio que creaste
        filename="da_cerebelum_model-epoch=20-val_loss=0.27.ckpt",  # El nombre del archivo
    )

    # Verificar que el archivo ha sido descargado
    if not os.path.exists(modelo_path):
        raise FileNotFoundError("El modelo no se descargó desde Hugging Face.")

    size = os.path.getsize(modelo_path)
    print(f"Tamaño del modelo descargado: {size / 1024:.2f} KB")

    if size < 100_000:  # Si es menor a 100 KB, probablemente sea una página de error HTML
        with open(modelo_path, "r", encoding="utf-8", errors="ignore") as f:
            print("\nContenido del archivo descargado (primeras líneas):")
            for _ in range(10):
                print(f.readline())
        raise ValueError("El archivo descargado no es un checkpoint válido. Es probable que sea una página de error HTML.")

    return modelo_path

modelo_descargado = descargar_modelo()
print(f"Modelo descargado en: {modelo_descargado}")

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
        0: (0, 0, 0),     # background: black
        1: (255, 0, 0),   # cerebellum: red
        2: (0, 255, 0),   # cisterna magna: green
        3: (0, 0, 255),   # vermis: blue
    }

    color_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for label, color in color_map.items():
        color_mask[pred_mask == label] = color

    mask_image = Image.fromarray(color_mask)
    return padded_image, mask_image

def cargar_modelo():
    modelo_path = 'cerebelum_model-epoch=25-val_loss=0.27.ckpt'

    if not os.path.exists(modelo_path):
        raise FileNotFoundError(f"El archivo del modelo no se encuentra en la ruta: {modelo_path}")

    # Crear el modelo con la arquitectura esperada
    model = CerebellumModelSegmentation(
        arch="unet",
        encoder_name="resnext50_32x4d",
        in_channels=3,
        out_classes=4
    )

    # Cargar pesos del state_dict
    try:
        state_dict = torch.load(modelo_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict, strict=True)
        print("Modelo cargado correctamente.")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        raise

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
            (0, 0, 0): 0     # Negro → Fondo
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
    Calcula Precisión (%), Sensibilidad (%) e IoU (%) por clase.
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

def run_prediction(input_image, model):
    resized_image, mask_image = predict_mask(input_image, model)
    return {
        "resized": resized_image,
        "mask": mask_image
    }

model = cargar_modelo()

def segment_and_report(image, patient_name, record_number, week):
    if image is not None:
        input_image = Image.open(image).convert("RGB")
        result = run_prediction(input_image, model)
        resized_image = result["resized"]
        mask_image = result["mask"]

        # Mostrar Ground Truth si está disponible
        COCO_JSON_PATH = 'data/_annotations.coco.json'
        INPUT_IMAGES_DIR = 'data'
        CATEGORY_COLORS = {
            1: (255, 0, 0),    # Cerebelo
            4: (0, 255, 0),    # Cisterna magna
            6: (0, 0, 255),    # Vermis
        }

        gt_mask = generate_groundtruth_mask(image.name, COCO_JSON_PATH, INPUT_IMAGES_DIR, CATEGORY_COLORS)
        metrics_df = None
        if gt_mask:
            gt_mask = gt_mask.resize(mask_image.size)
            pred_mask_np = np.array(mask_image)
            gt_mask_np = np.array(gt_mask)
            num_classes = 3
            metrics_df = calculate_metrics(gt_mask_np, pred_mask_np, num_classes)
            return resized_image, mask_image, gt_mask, metrics_df
        else:
            return resized_image, mask_image, None, None
    return None, None, None, None

def generate_report_gradio(patient_name, record_number, week, original_image, segmented_image):
    if original_image is not None and segmented_image is not None:
        pdf_buffer = generate_pdf(
            patient_name=patient_name,
            record_number=record_number,
            segmented_img=segmented_image,
            original_img=original_image,
            week=week,
            logo_sacyl_path="logo_sacyl.png",
            logo_junta_path="logo_junta.png"
        )
        return pdf_buffer
    else:
        return None

with gr.Blocks() as iface:
    gr.Markdown("""
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
    """)

    with gr.Row():
        input_image = gr.Image(label="Imagen de Ecografía Fetal (JPG o PNG)")

    with gr.Row():
        with gr.Column():
            patient_name_input = gr.Textbox(label="Nombre del paciente", placeholder="Ejemplo: Ana Gómez Ruiz")
            record_number_input = gr.Textbox(label="Número de historia clínica", placeholder="Ejemplo: HCU123456")
            week_input = gr.Number(label="Semana de gestación", minimum=10, maximum=40, step=1)
            segment_button = gr.Button("Segmentar Imagen")

        with gr.Column():
            original_output = gr.Image(label="Imagen Original")
            segmented_output = gr.Image(label="Imagen Segmentada")
            groundtruth_output = gr.Image(label="Máscara Ground Truth (si disponible)")
            metrics_output = gr.DataFrame(label="Métricas de Segmentación (si Ground Truth disponible)")

    gr.Markdown("""
    <h4 style='text-align: center; margin-top: 30px; font-size: 2em; color: #2c3e50;'>
        Resultados de Segmentación
    </h4>
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
    """)

    with gr.Row():
        generate_report_button = gr.Button("Generar Informe PDF")
        pdf_output = gr.File(label="Descargar Informe PDF")

    segment_output = segment_button.click(
        segment_and_report,
        inputs=[input_image, patient_name_input, record_number_input, week_input],
        outputs=[original_output, segmented_output, groundtruth_output, metrics_output]
    )

    generate_report_button.click(
        generate_report_gradio,
        inputs=[patient_name_input, record_number_input, week_input, original_output, segmented_output],
        outputs=[pdf_output],
        preprocess=False,
        postprocess=False
    )

iface.launch()