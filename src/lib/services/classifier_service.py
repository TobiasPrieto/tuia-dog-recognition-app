from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import onnxruntime

import cv2
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

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

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

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
        train_loader, val_loader = self._build_train_val_loaders()
        num_classes = len(train_loader.dataset.classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = self._build_training_model(num_classes).to(device)
        if self.active_model_name == "resnet18_finetuned":
            self._train_resnet18(model, train_loader, val_loader, device)
        elif self.active_model_name == "cnn_custom":
            self._train_cnn_custom(model, train_loader, val_loader, device)
        else:
            raise ValueError(f"Modelo no soportado: {self.active_model_name}")

        self._loaded.pop(self.active_model_name, None)
        self._prepared_models.pop(self.active_model_name, None)

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
        test_loader = self._build_test_loader()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self._prepare_model().to(device)
        model.eval()

        y_true: list[int] = []
        y_pred: list[int] = []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluando"):
                images = images.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().tolist()

                y_pred.extend(preds)
                y_true.extend(labels.tolist())

        return self._calculate_classification_metrics(
            y_true,
            y_pred,
            num_classes=len(test_loader.dataset.classes),
        )

#--------------------------------------
    def _build_training_transforms(self) -> tuple[transforms.Compose, transforms.Compose]:
        train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.image_size,
                    scale=(0.85, 1.0),
                    ratio=(0.9, 1.1),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.02,
                ),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        return train_transforms, self._preprocess

    def _build_train_val_loaders(self) -> tuple[DataLoader, DataLoader]:
        train_transforms, val_transforms = self._build_training_transforms()

        balanced_train_path = self.dataset_path.parent / "dataset_balanced" / "train"
        train_path = balanced_train_path if balanced_train_path.is_dir() else self.dataset_path / "train"
        val_path = self.dataset_path / "valid"

        for path in (train_path, val_path):
            if not path.is_dir():
                raise FileNotFoundError(f"No existe el directorio: {path}")

        train_dataset = datasets.ImageFolder(root=str(train_path), transform=train_transforms)
        val_dataset = datasets.ImageFolder(root=str(val_path), transform=val_transforms)

        if train_dataset.classes != val_dataset.classes:
            raise ValueError("Los splits train y valid no tienen las mismas clases.")

        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=2,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=2,
        )
        return train_loader, val_loader

    def _build_test_loader(self) -> DataLoader:
        test_path = self.dataset_path / "test"
        if not test_path.is_dir():
            raise FileNotFoundError(f"No existe el directorio requerido: {test_path}")

        test_dataset = datasets.ImageFolder(root=str(test_path), transform=self._preprocess)
        return DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=2,
        )

    def _calculate_classification_metrics(
        self,
        y_true: list[int],
        y_pred: list[int],
        num_classes: int,
    ) -> dict[str, float]:
        truth = np.array(y_true, dtype=np.int64)
        pred = np.array(y_pred, dtype=np.int64)

        if truth.size == 0:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "specificity": 0.0,
                "f1": 0.0,
            }

        accuracy = float((truth == pred).mean())
        precisions: list[float] = []
        recalls: list[float] = []
        specificities: list[float] = []
        f1_scores: list[float] = []

        for class_idx in range(num_classes):
            tp = int(np.sum((truth == class_idx) & (pred == class_idx)))
            fp = int(np.sum((truth != class_idx) & (pred == class_idx)))
            fn = int(np.sum((truth == class_idx) & (pred != class_idx)))
            tn = int(np.sum((truth != class_idx) & (pred != class_idx)))

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            specificity = tn / (tn + fp) if (tn + fp) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

            precisions.append(precision)
            recalls.append(recall)
            specificities.append(specificity)
            f1_scores.append(f1)

        return {
            "accuracy": accuracy,
            "precision": float(np.mean(precisions)),
            "recall": float(np.mean(recalls)),
            "specificity": float(np.mean(specificities)),
            "f1": float(np.mean(f1_scores)),
        }

    def _build_training_model(self, num_classes: int) -> nn.Module:
        if self.active_model_name == "resnet18_finetuned":
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model

        if self.active_model_name == "cnn_custom":
            return CNNPropia(n_clases=num_classes)

        raise ValueError(f"Modelo no soportado: {self.active_model_name}")

    def _train_resnet18(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4,
            weight_decay=1e-4,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )

        best_model_weights = copy.deepcopy(model.state_dict())
        best_val_acc = -1.0
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        num_epochs = 20
        for epoch in range(num_epochs):
            print(f"\nEpoca {epoch + 1}/{num_epochs}")
            print("-" * 40)

            train_loss, train_acc = self._run_training_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                use_tqdm=True,
                clip_gradients=False,
            )
            val_loss, val_acc = self._run_validation_epoch(
                model,
                val_loader,
                criterion,
                device,
                use_tqdm=True,
            )

            scheduler.step(val_loss)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_weights = copy.deepcopy(model.state_dict())
                self._save_resnet18_checkpoint(best_model_weights)
                print(f"Nuevo mejor modelo guardado con Val Acc: {best_val_acc:.4f}")

        model.load_state_dict(best_model_weights)
        print("\nEntrenamiento finalizado.")
        print(f"Mejor accuracy de validacion: {best_val_acc:.4f}")
        print(f"Checkpoint guardado en: {self.active_checkpoint}")

    def _train_cnn_custom(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )

        epochs = 500
        patience = 50
        start_epoch = 1
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }
        best_val_loss = float("inf")
        best_weights: dict[str, torch.Tensor] | None = None
        wait = 0

        if self.active_checkpoint.is_file():
            print(f"[Checkpoint] Cargando desde '{self.active_checkpoint}'...")
            try:
                checkpoint = torch.load(self.active_checkpoint, map_location=device, weights_only=False)
                if isinstance(checkpoint, dict) and "estado_modelo" in checkpoint:
                    model.load_state_dict(checkpoint["estado_modelo"])
                    if "estado_optimizador" in checkpoint:
                        optimizer.load_state_dict(checkpoint["estado_optimizador"])
                    if checkpoint.get("estado_scheduler") is not None:
                        scheduler.load_state_dict(checkpoint["estado_scheduler"])
                    start_epoch = int(checkpoint.get("epoca", 0)) + 1
                    best_val_loss = float(checkpoint.get("mejor_val_loss", best_val_loss))
                    history = checkpoint.get("historial", history)
                    best_weights = {key: value.clone() for key, value in model.state_dict().items()}
                    print(
                        f"[Checkpoint] Reanudando desde epoca {start_epoch} "
                        f"(mejor val_loss registrada: {best_val_loss:.4f})"
                    )
                else:
                    model.load_state_dict(checkpoint)
                    best_weights = {key: value.clone() for key, value in model.state_dict().items()}
                    print("[Checkpoint] Checkpoint de pesos cargado. Continuando sin estado de optimizador.")
            except Exception as exc:
                print(f"[Checkpoint] Error al cargar '{self.active_checkpoint}': {exc}. Iniciando desde cero.")
        else:
            print("[Checkpoint] Sin checkpoint. Iniciando desde cero.")

        if start_epoch > epochs:
            print(
                f"[Info] Entrenamiento ya completado hasta epoca {start_epoch - 1} "
                f"(epochs solicitadas: {epochs})."
            )
            return

        print(
            f"\nEntrenamiento 'cnn_custom' | Epocas [{start_epoch}-{epochs}] | Paciencia: {patience}\n"
            f"  {'Epoca':>6} | {'Train Loss':>10} | {'Val Loss':>10} | "
            f"{'Train Acc':>10} | {'Val Acc':>9} | {'LR':>8} | {'Estado':>10}"
        )
        print("  " + "-" * 74)

        for epoch in range(start_epoch, epochs + 1):
            train_loss, train_acc = self._run_training_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                use_tqdm=False,
                clip_gradients=True,
            )
            val_loss, val_acc = self._run_validation_epoch(
                model,
                val_loader,
                criterion,
                device,
                use_tqdm=False,
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = {key: value.clone() for key, value in model.state_dict().items()}
                wait = 0
                status = "guardado"
                self._save_cnn_custom_checkpoint(
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    best_val_loss,
                    history,
                )
            else:
                wait += 1
                status = f"espera {wait}/{patience}"

            print(
                f"  {epoch:6d} | {train_loss:10.4f} | {val_loss:10.4f} | "
                f"{train_acc:10.4f} | {val_acc:9.4f} | {current_lr:.2e} | {status:>10}"
            )

            if wait >= patience:
                print(
                    f"\n[Early Stopping] Detenido en epoca {epoch} "
                    f"(mejor val_loss: {best_val_loss:.4f})."
                )
                break

        if best_weights is not None:
            model.load_state_dict(best_weights)
            print(f"\n[Finalizado] Mejores pesos restaurados (val_loss: {best_val_loss:.4f}).")
        else:
            print("\n[Advertencia] No hay pesos guardados para restaurar.")
        print(f"Checkpoint guardado en: {self.active_checkpoint}")

    def _run_training_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        *,
        use_tqdm: bool,
        clip_gradients: bool,
    ) -> tuple[float, float]:
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        batches = tqdm(loader, desc="Entrenando") if use_tqdm else loader

        for images, labels in batches:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            if clip_gradients:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            running_corrects += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

        return running_loss / total_samples, running_corrects / total_samples

    def _run_validation_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        *,
        use_tqdm: bool,
    ) -> tuple[float, float]:
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        batches = tqdm(loader, desc="Validando") if use_tqdm else loader

        with torch.no_grad():
            for images, labels in batches:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                batch_size = images.size(0)
                running_loss += loss.item() * batch_size
                running_corrects += (outputs.argmax(dim=1) == labels).sum().item()
                total_samples += batch_size

        return running_loss / total_samples, running_corrects / total_samples

    def _save_resnet18_checkpoint(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> None:
        self.active_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, self.active_checkpoint)

    def _save_cnn_custom_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.ReduceLROnPlateau,
        best_val_loss: float,
        history: dict[str, list[float]],
    ) -> None:
        self.active_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoca": epoch,
                "estado_modelo": model.state_dict(),
                "estado_optimizador": optimizer.state_dict(),
                "estado_scheduler": scheduler.state_dict(),
                "mejor_val_loss": best_val_loss,
                "historial": history,
            },
            self.active_checkpoint,
        )

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
