from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import onnxruntime

import cv2
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

logger = logging.getLogger(__name__)


class CNNPropia(nn.Module):
    """CNN custom usada en la notebook de Etapa 2."""

    def __init__(self, n_clases: int = 70, dropout_rate: float = 0.3) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.cabeza = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, n_clases),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cabeza(self.backbone(x))


class ClassifierService:
    """Etapa 2: entrenamiento y comparacion de modelos de clasificacion.

    Funciones a implementar por el estudiante:
      - train_classifier()
      - evaluate_classifier()
      - extract_custom_embedding(image)

    La carga de checkpoints (.pth / .onnx) y la seleccion del modelo activo
    ya estan provistas.
    """

    def __init__(
        self,
        checkpoints: dict[str, Path],
        image_size: int,
        dataset_path: Path,
        output_path: Path,
        active_model: str = "resnet18_finetuned",
    ) -> None:
        # checkpoints: nombre logico -> ruta del archivo (ej. resnet18_finetuned -> models/resnet18_finetuned.pth)
        self.checkpoints = checkpoints
        self.image_size = image_size
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.active_model_name = active_model
        self._loaded: dict[str, Any] = {}

#---------------------------------------------------------

        self._prepared_models: dict[str, nn.Module] = {}
        self._preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    # ------------------------------------------------------------------
    # Infraestructura provista
    # ------------------------------------------------------------------

    def set_active_model(self, name: str) -> None:
        """Define que checkpoint usan extract_custom_embedding y la clasificacion.

        Valores esperados: resnet18_finetuned | cnn_custom.
        """
        if name not in self.checkpoints:
            raise ValueError(f"Unknown model '{name}'. Expected one of: {sorted(self.checkpoints)}")
        self.active_model_name = name

    @property
    def active_checkpoint(self) -> Path:
        return self.checkpoints[self.active_model_name]

    def load_model(self, name: str | None = None) -> Any:
        """Carga (con cache) el checkpoint del modelo indicado o del activo.

        Soporta modelos PyTorch (.pth) y exportados a ONNX (.onnx).
        """
        key = name or self.active_model_name
        if key in self._loaded:
            return self._loaded[key]
        path = self.checkpoints[key]
        if not path.exists():
            raise ValueError(
                f"Checkpoint not found: {path}. Entrena el modelo (Etapa 2) y guardalo en esa ruta."
            )
        suf = path.suffix.lower()
        if suf == ".pth":
            model = torch.load(path, map_location="cpu", weights_only=False)
        elif suf == ".onnx":
            model = onnxruntime.InferenceSession(str(path))
        else:
            raise ValueError(f"Unsupported model format (expected .pth or .onnx): {path}")
        self._loaded[key] = model
        return model

    # ------------------------------------------------------------------
    # Etapa 2: funciones a implementar
    # ------------------------------------------------------------------

    def train_classifier(self) -> None:
        """
        Entrena el clasificador de razas sobre el dataset (self.dataset_path).

        Modelo A (obligatorio): fine-tuning de ResNet18 pre-entrenado.
        Modelo B (opcional, recomendado): CNN propia.

        Debe:
          - Usar los splits train/valid definidos en la notebook.
          - Aplicar el preprocesamiento y data augmentation justificados.
          - Guardar el checkpoint resultante en self.active_checkpoint
            (ej: models/resnet18_finetuned.pth).
        """
        raise NotImplementedError("Etapa 2: implementar train_classifier")

    def evaluate_classifier(self) -> dict[str, float]:
        """
        Evalua el modelo activo sobre el conjunto de prueba.

        Debe reportar: accuracy, precision, recall (sensibilidad),
        specificity (especificidad) y F1-Score. La matriz de confusion y las
        curvas de entrenamiento se documentan en la notebook.

        Retorna un dict con las metricas, ej:
          {"accuracy": 0.91, "precision": 0.90, "recall": 0.89,
           "specificity": 0.99, "f1": 0.90}
        """
        raise NotImplementedError("Etapa 2: implementar evaluate_classifier")

#--------------------------------------
    def _prepare_model(self) -> nn.Module:
        if self.active_model_name in self._prepared_models:
            return self._prepared_models[self.active_model_name]

        checkpoint = self.load_model()
        if isinstance(checkpoint, nn.Module):
            model = checkpoint
        else:
            if self.active_model_name == "resnet18_finetuned":
                state_dict = checkpoint
                model = models.resnet18(weights=None)
                model.fc = nn.Linear(model.fc.in_features, state_dict["fc.weight"].shape[0])
            elif self.active_model_name == "cnn_custom":
                state_dict = checkpoint["estado_modelo"] if "estado_modelo" in checkpoint else checkpoint
                model = CNNPropia(n_clases=state_dict["cabeza.4.weight"].shape[0])
            else:
                raise ValueError(f"Unsupported active model: {self.active_model_name}")
            model.load_state_dict(state_dict)

        model.eval()
        self._prepared_models[self.active_model_name] = model
        return model

    def _extract_resnet18_embedding(
        self,
        model: nn.Module,
        image_tensor: torch.Tensor,
    ) -> torch.Tensor:
        required_layers = (
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "avgpool",
        )
        if not all(hasattr(model, layer) for layer in required_layers):
            raise ValueError("El modelo resnet18_finetuned no tiene la arquitectura ResNet esperada.")

        x = model.conv1(image_tensor)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        return torch.flatten(x, 1).squeeze(0)

    def _extract_cnn_custom_embedding(
        self,
        model: nn.Module,
        image_tensor: torch.Tensor,
    ) -> torch.Tensor:
        if not hasattr(model, "backbone") or not hasattr(model, "cabeza"):
            raise ValueError("El modelo cnn_custom debe exponer 'backbone' y 'cabeza'.")

        x = model.backbone(image_tensor)
        for layer in list(model.cabeza.children())[:-1]:
            x = layer(x)
        return x.squeeze(0)

    def extract_custom_embedding(self, image: np.ndarray) -> list[float]:
        """
        Genera el embedding de una imagen usando el modelo propio activo
        (penultima capa del ResNet18 fine-tuned o de la CNN custom).

        Se usa cuando EMBEDDING_MODEL != baseline para que la busqueda por
        similitud (Etapa 1) funcione con los modelos entrenados.
        La imagen llega en BGR (OpenCV). Retorna una lista de floats de
        dimension EMBEDDING_DIM.
        """
        if image is None or image.size == 0:
            raise ValueError("Image is empty.")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tensor = self._preprocess(image_pil).unsqueeze(0)
        model = self._prepare_model()

        with torch.inference_mode():
            if self.active_model_name == "resnet18_finetuned":
                embedding = self._extract_resnet18_embedding(model, image_tensor)
            elif self.active_model_name == "cnn_custom":
                embedding = self._extract_cnn_custom_embedding(model, image_tensor)
            else:
                raise ValueError(f"Unsupported active model: {self.active_model_name}")

        embedding = embedding.detach().cpu().flatten().to(torch.float32)
        return embedding.tolist()


#---------------------------