from __future__ import annotations

import json
import logging
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO

from lib.schemas import ClassifyResult, DetectResult, DogDetection
from lib.services.classifier_service import ClassifierService

logger = logging.getLogger(__name__)


class DetectionService:
    """Etapa 3: pipeline de deteccion y clasificacion.

    Funciones a implementar por el estudiante:
      - detect_dogs(image)
      - classify_detected_dog(crop)

    La orquestacion (predict: deteccion -> recorte -> clasificacion -> JSON)
    ya esta provista.
    """

    def __init__(
        self,
        classifier: ClassifierService,
        yolo_model: str,
        conf_threshold: float,
        dog_class_id: int,
    ) -> None:
        self.classifier = classifier
        self.yolo_model_name = yolo_model
        self.yolo_model = YOLO(self.yolo_model_name)
        self.conf_threshold = conf_threshold
        self.dog_class_id = dog_class_id
        self.class_names = self._load_class_names()
        self.classification_preprocess = transforms.Compose(
            [
                transforms.Resize(self.classifier.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @staticmethod
    def _clip_xyxy(
        x1: int, y1: int, x2: int, y2: int, height: int, width: int
    ) -> tuple[int, int, int, int]:
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))
        if x2 <= x1:
            x2 = min(x1 + 1, width)
        if y2 <= y1:
            y2 = min(y1 + 1, height)
        return x1, y1, x2, y2

    def _load_image(self, source_path: str) -> np.ndarray:
        image = cv2.imread(str(source_path))
        if image is None:
            raise ValueError(f"Could not read image: {source_path}")
        # BGR uint8 (convencion OpenCV / ultralytics)
        return image

    def _load_class_names(self) -> list[str]:
        for split in ("train", "valid", "test"):
            split_path = self.classifier.dataset_path / split
            if split_path.exists():
                class_names = sorted(path.name for path in split_path.iterdir() if path.is_dir())
                if class_names:
                    return class_names
        return []

    # ------------------------------------------------------------------
    # Etapa 3: funciones a implementar
    # ------------------------------------------------------------------

    def detect_dogs(self, image: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        """
        Detecta todos los perros presentes en la imagen usando YOLOv8.

        Retorna una lista con el formato:
            [((x1, y1, x2, y2), confidence), ...]

        Las coordenadas están expresadas en píxeles.
        """

        resultados = self.yolo_model.predict(
            source=image,
            classes=[self.dog_class_id],
            conf=self.conf_threshold,
            device="cpu",
            verbose=False,
        )

        cajas = resultados[0].boxes

        if cajas is None or len(cajas) == 0:
            return []

        perros_detectados = []

        for coordenadas, confianza in zip(cajas.xyxy, cajas.conf):
            x1, y1, x2, y2 = map(int, coordenadas.tolist())

            perros_detectados.append(
                (
                    (x1, y1, x2, y2),
                    float(confianza.item()),
                )
            )

        return perros_detectados

    def classify_detected_dog(self, crop: np.ndarray) -> tuple[str, float]:
        """
        Clasifica la raza del recorte de un perro detectado usando el modelo
        entrenado en la Etapa 2 (self.classifier.load_model()).

        El recorte llega en BGR (OpenCV). Retorna (raza, score).
        """
        if crop is None or crop.size == 0:
            return "unknown", 0.0

        model = self.classifier._prepare_model()
        model.eval()

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        input_tensor = self.classification_preprocess(pil_img).unsqueeze(0)

        with torch.inference_mode():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output.squeeze(0), dim=0)
            top_prob, top_cat = torch.max(probabilities, dim=0)

        class_idx = int(top_cat.item())
        if class_idx >= len(self.class_names):
            logger.warning(
                "Predicted class index %s but only %s class names were found.",
                class_idx,
                len(self.class_names),
            )
            return "unknown", float(top_prob.item())

        return self.class_names[class_idx], float(top_prob.item())

    # ------------------------------------------------------------------
    # Orquestacion provista
    # ------------------------------------------------------------------

    def classify_image(
        self, source_path: str, output_path: Path, model_name: str | None = None
    ) -> str:
        """Clasifica la imagen completa con el modelo entrenado (pestaña Etapa 2).

        Reutiliza classify_detected_dog tratando la imagen entera como recorte,
        por lo que requiere la Etapa 2 (modelo entrenado) y classify_detected_dog.
        Escribe el resultado como JSON en `output_path` y retorna su ruta.
        """
        image = self._load_image(source_path)
        if model_name:
            self.classifier.set_active_model(model_name)
        breed, score = self.classify_detected_dog(image)
        payload = ClassifyResult(
            source_path=source_path,
            model=model_name or self.classifier.active_model_name,
            breed=breed,
            score=round(float(score), 4),
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)

    def predict(self, source_path: str, output_path: Path) -> str:
        """Flujo completo: deteccion -> bounding boxes -> recortes -> clasificacion.

        Escribe el resultado como JSON en `output_path` y retorna su ruta.
        """
        image = self._load_image(source_path)
        height, width = image.shape[:2]

        detections: list[DogDetection] = []
        for (box, det_score) in self.detect_dogs(image):
            x1, y1, x2, y2 = self._clip_xyxy(*[int(v) for v in box], height, width)
            crop = image[y1:y2, x1:x2]
            breed, breed_score = self.classify_detected_dog(crop)
            detections.append(
                DogDetection(
                    bbox=[x1, y1, x2, y2],
                    det_score=round(float(det_score), 4),
                    breed=breed,
                    breed_score=round(float(breed_score), 4),
                )
            )

        detected_breeds = sorted({item.breed for item in detections if item.breed != "unknown"})
        payload = DetectResult(
            source_path=source_path,
            detections=detections,
            detected_breeds=detected_breeds,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)
