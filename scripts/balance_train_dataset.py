"""Balancea el split de entrenamiento mediante oversampling con data augmentation.

Ejemplo:
    python scripts/balance_train_dataset.py \
        --input data/dataset/train \
        --output data/dataset_balanced/train \
        --target 200

Los originales se copian a la salida y se generan imágenes aumentadas hasta que
cada clase tenga exactamente ``--target`` elementos. Nunca modifica el dataset
de entrada.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import albumentations as A
import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_augmentation() -> A.Compose:
    """Aumentos moderados que conservan los rasgos principales del perro."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.90, 1.10),
                translate_percent=(-0.06, 0.06),
                rotate=(-15, 15),
                shear=(-5, 5),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.8,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.15,
                        contrast_limit=0.15,
                        p=1.0,
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=5,
                        sat_shift_limit=12,
                        val_shift_limit=10,
                        p=1.0,
                    ),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.GaussNoise(std_range=(0.01, 0.04), p=1.0),
                ],
                p=0.15,
            ),
        ]
    )


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/dataset/train"),
        help="Carpeta train original, organizada en una subcarpeta por clase.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/dataset_balanced/train"),
        help="Nueva carpeta donde se escribirá el train balanceado.",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=200,
        help="Cantidad final de imágenes por clase.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.input.resolve()
    output_root = args.output.resolve()

    if args.target <= 0:
        raise ValueError("--target debe ser mayor que cero.")
    if not source_root.is_dir():
        raise FileNotFoundError(f"No existe el dataset de entrada: {source_root}")
    if source_root == output_root or source_root in output_root.parents:
        raise ValueError("La salida no puede ser igual ni estar dentro del train original.")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"La salida no está vacía: {output_root}. "
            "Usa otra ruta o elimina esa carpeta conscientemente."
        )

    random.seed(args.seed)
    transform = build_augmentation()
    class_directories = sorted(path for path in source_root.iterdir() if path.is_dir())

    if not class_directories:
        raise ValueError(f"No se encontraron carpetas de clases en {source_root}")

    counts = {directory.name: len(image_files(directory)) for directory in class_directories}
    empty_classes = [name for name, count in counts.items() if count == 0]
    oversized_classes = [name for name, count in counts.items() if count > args.target]

    if empty_classes:
        raise ValueError(f"Hay clases sin imágenes: {', '.join(empty_classes)}")
    if oversized_classes:
        raise ValueError(
            "Estas clases superan el objetivo y no serán recortadas automáticamente: "
            + ", ".join(oversized_classes)
        )

    output_root.mkdir(parents=True, exist_ok=True)
    generated_total = 0

    for class_directory in class_directories:
        originals = image_files(class_directory)
        class_output = output_root / class_directory.name
        class_output.mkdir(parents=True, exist_ok=False)

        for index, source_path in enumerate(originals):
            destination = class_output / f"original_{index:04d}{source_path.suffix.lower()}"
            shutil.copy2(source_path, destination)

        missing = args.target - len(originals)
        for index in range(missing):
            source_path = random.choice(originals)
            image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"No se pudo leer la imagen: {source_path}")

            augmented = transform(image=image)["image"]
            destination = class_output / f"aug_{index:04d}.jpg"
            if not cv2.imwrite(
                str(destination),
                augmented,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            ):
                raise OSError(f"No se pudo escribir la imagen: {destination}")

        generated_total += missing
        print(
            f"{class_directory.name}: {len(originals)} originales + "
            f"{missing} aumentadas = {args.target}"
        )

    final_counts = {
        directory.name: len(image_files(directory))
        for directory in output_root.iterdir()
        if directory.is_dir()
    }
    incorrect = {
        name: count for name, count in final_counts.items() if count != args.target
    }
    if incorrect:
        raise RuntimeError(f"Conteos finales incorrectos: {incorrect}")

    print(
        f"\nListo: {len(final_counts)} clases x {args.target} imágenes "
        f"= {len(final_counts) * args.target} imágenes."
    )
    print(f"Imágenes sintéticas generadas: {generated_total}")
    print(f"Dataset balanceado: {output_root}")


if __name__ == "__main__":
    main()
