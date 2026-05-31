#!/usr/bin/env python3
"""Download public blood-cell microscopy images and annotate them with pretrained models."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


HF_DATASET_TEST_ZIP = (
    "https://huggingface.co/datasets/keremberke/blood-cell-object-detection/"
    "resolve/main/data/test.zip"
)

DEFAULT_MODELS = {
    "yolov8n_bccd": {
        "kind": "yolo",
        "repo": "keremberke/yolov8n-blood-cell-detection",
        "weights_url": "https://huggingface.co/keremberke/yolov8n-blood-cell-detection/resolve/main/best.pt",
        "description": "YOLOv8n pretrained on BCCD for Platelets/RBC/WBC detection.",
    },
    "yolov8s_bccd": {
        "kind": "yolo",
        "repo": "keremberke/yolov8s-blood-cell-detection",
        "weights_url": "https://huggingface.co/keremberke/yolov8s-blood-cell-detection/resolve/main/best.pt",
        "description": "YOLOv8s pretrained on BCCD for Platelets/RBC/WBC detection.",
    },
}

WIKIMEDIA_FILES = [
    {
        "title": "File:Peripheral blood smear.jpg",
        "preferred_name": "wikimedia_peripheral_blood_smear.jpg",
    },
    {
        "title": "File:Redbloodcells.jpg",
        "preferred_name": "wikimedia_redbloodcells.jpg",
    },
    {
        "title": "File:Red White Blood cells.jpg",
        "preferred_name": "wikimedia_red_white_blood_cells.jpg",
    },
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HTTP_HEADERS = {
    "User-Agent": "blood-cell-annotation-example/1.0 (research demo; https://github.com/jaon1234/GPT-GNN)"
}
CLASS_COLORS = {
    "Platelets": (255, 196, 0),
    "platelets": (255, 196, 0),
    "RBC": (230, 57, 70),
    "rbc": (230, 57, 70),
    "WBC": (29, 78, 216),
    "wbc": (29, 78, 216),
    "cellpose_cell": (46, 204, 113),
}


@dataclass
class ImageSource:
    path: Path
    source: str
    title: str
    url: str
    license: str | None = None
    attribution: str | None = None

    def as_json(self, output_dir: Path) -> dict[str, Any]:
        return {
            "path": str(self.path.relative_to(output_dir)),
            "source": self.source,
            "title": self.title,
            "url": self.url,
            "license": self.license,
            "attribution": self.attribution,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download blood-cell microscopy samples and annotate cells with pretrained models."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/blood_cell_annotation/runs/demo"),
        help="Directory for downloaded images, annotations, and JSON outputs.",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=Path("examples/blood_cell_annotation/models"),
        help="Directory for downloaded pretrained weights.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(DEFAULT_MODELS),
        default=sorted(DEFAULT_MODELS),
        help="Pretrained YOLO models to run.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--device", default="cpu", help="Inference device, for example 'cpu' or '0'.")
    parser.add_argument(
        "--max-dataset-images",
        type=int,
        default=5,
        help="Maximum number of Hugging Face BCCD test images to include.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=8,
        help="Maximum total number of images to annotate after downloads.",
    )
    parser.add_argument(
        "--skip-wikimedia",
        action="store_true",
        help="Only use Hugging Face BCCD dataset samples.",
    )
    parser.add_argument(
        "--skip-hf-dataset",
        action="store_true",
        help="Only use Wikimedia Commons images.",
    )
    parser.add_argument(
        "--run-cellpose",
        action="store_true",
        help="Also run the optional Cellpose pretrained segmentation model if cellpose is installed.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download images, dataset zip, and model weights even if local files already exist.",
    )
    return parser.parse_args()


def safe_stem(name: str) -> str:
    stem = Path(name).stem.lower()
    return re.sub(r"[^a-z0-9]+", "_", stem).strip("_")


def download_file(url: str, destination: Path, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return destination

    tmp = destination.with_suffix(destination.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=60, headers=HTTP_HEADERS) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with tmp.open("wb") as handle, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {destination.name}",
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    bar.update(len(chunk))
    tmp.replace(destination)
    return destination


def fetch_wikimedia_source(item: dict[str, str], image_dir: Path, force: bool) -> ImageSource:
    title = item["title"]
    api_url = (
        "https://commons.wikimedia.org/w/api.php"
        f"?action=query&titles={quote(title)}&prop=imageinfo"
        "&iiprop=url|extmetadata&format=json"
    )
    response = requests.get(api_url, timeout=30, headers=HTTP_HEADERS)
    response.raise_for_status()
    pages = response.json()["query"]["pages"]
    page = next(iter(pages.values()))
    image_info = page["imageinfo"][0]
    source_url = image_info["url"]
    metadata = image_info.get("extmetadata", {})

    license_short = metadata.get("LicenseShortName", {}).get("value")
    attribution = metadata.get("Artist", {}).get("value") or metadata.get("Credit", {}).get("value")
    filename = item["preferred_name"]
    path = download_file(source_url, image_dir / filename, force=force)
    return ImageSource(
        path=path,
        source="wikimedia_commons",
        title=title,
        url=source_url,
        license=license_short,
        attribution=attribution,
    )


def collect_wikimedia_images(image_dir: Path, force: bool) -> list[ImageSource]:
    return [fetch_wikimedia_source(item, image_dir, force) for item in WIKIMEDIA_FILES]


def collect_hf_dataset_images(
    output_dir: Path,
    image_dir: Path,
    force: bool,
    max_images: int,
) -> list[ImageSource]:
    if max_images <= 0:
        return []

    cache_dir = output_dir / "cache"
    zip_path = cache_dir / "hf_bccd_test.zip"
    extract_dir = cache_dir / "hf_bccd_test"

    download_file(HF_DATASET_TEST_ZIP, zip_path, force=force)
    if force and extract_dir.exists():
        shutil.rmtree(extract_dir)
    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)

    image_paths = sorted(path for path in extract_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    sources: list[ImageSource] = []
    for source_path in image_paths[:max_images]:
        target = image_dir / f"hf_bccd_{safe_stem(source_path.name)}{source_path.suffix.lower()}"
        if force or not target.exists():
            shutil.copy2(source_path, target)
        sources.append(
            ImageSource(
                path=target,
                source="huggingface_dataset",
                title=source_path.name,
                url=HF_DATASET_TEST_ZIP,
                license="See dataset card",
                attribution="keremberke/blood-cell-object-detection; Team Roboflow",
            )
        )
    return sources


def ensure_models(model_names: list[str], model_cache_dir: Path, force: bool) -> dict[str, Path]:
    weights: dict[str, Path] = {}
    for model_name in model_names:
        config = DEFAULT_MODELS[model_name]
        destination = model_cache_dir / f"{model_name}.pt"
        weights[model_name] = download_file(config["weights_url"], destination, force=force)
    return weights


def color_for(label: str) -> tuple[int, int, int]:
    return CLASS_COLORS.get(label, (20, 184, 166))


def draw_detections(
    image_path: Path,
    detections: list[dict[str, Any]],
    output_path: Path,
    model_label: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    width, height = image.size
    line_width = max(2, round(min(width, height) / 250))
    for detection in detections:
        x1, y1, x2, y2 = detection["box_xyxy"]
        label = detection["class_name"]
        score = detection.get("confidence")
        text = f"{label} {score:.2f}" if isinstance(score, float) else label
        color = color_for(label)

        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = right - left, bottom - top
        label_y = max(0, y1 - text_h - 4)
        draw.rectangle((x1, label_y, x1 + text_w + 6, label_y + text_h + 4), fill=color)
        draw.text((x1 + 3, label_y + 2), text, fill=(255, 255, 255), font=font)

    footer = f"{model_label}: {len(detections)} detections"
    footer_bbox = draw.textbbox((0, 0), footer, font=font)
    footer_w = footer_bbox[2] - footer_bbox[0]
    footer_h = footer_bbox[3] - footer_bbox[1]
    draw.rectangle((0, 0, footer_w + 8, footer_h + 8), fill=(0, 0, 0))
    draw.text((4, 4), footer, fill=(255, 255, 255), font=font)
    image.save(output_path)


def run_yolo_models(
    image_sources: list[ImageSource],
    weights: dict[str, Path],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'ultralytics'. Install it with: "
            "python -m pip install -r examples/blood_cell_annotation/requirements.txt"
        ) from exc

    results_json: list[dict[str, Any]] = []
    for model_name, weight_path in weights.items():
        model = YOLO(str(weight_path))
        model_output_dir = args.output_dir / "annotations" / model_name
        for source in image_sources:
            predictions = model.predict(
                source=str(source.path),
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )
            result = predictions[0]
            detections: list[dict[str, Any]] = []
            names = result.names
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    xyxy = [round(float(value), 2) for value in box.xyxy[0].tolist()]
                    detections.append(
                        {
                            "class_id": class_id,
                            "class_name": str(names[class_id]),
                            "confidence": round(confidence, 4),
                            "box_xyxy": xyxy,
                        }
                    )

            annotated_path = model_output_dir / source.path.name
            draw_detections(source.path, detections, annotated_path, model_name)
            results_json.append(
                {
                    "image": str(source.path.relative_to(args.output_dir)),
                    "model": model_name,
                    "model_repo": DEFAULT_MODELS[model_name]["repo"],
                    "annotation": str(annotated_path.relative_to(args.output_dir)),
                    "detections": detections,
                }
            )
    return results_json


def bbox_from_mask(mask: Any, value: int) -> list[float] | None:
    import numpy as np

    ys, xs = np.where(mask == value)
    if xs.size == 0 or ys.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def run_cellpose(image_sources: list[ImageSource], args: argparse.Namespace) -> list[dict[str, Any]]:
    try:
        import numpy as np
        from cellpose import models
    except ImportError as exc:
        raise SystemExit(
            "Cellpose is optional and not installed. Install it with: python -m pip install cellpose"
        ) from exc

    gpu = args.device != "cpu"
    model = models.CellposeModel(gpu=gpu, model_type="cyto3")
    results_json: list[dict[str, Any]] = []
    model_output_dir = args.output_dir / "annotations" / "cellpose_cyto3"

    for source in image_sources:
        image = np.array(Image.open(source.path).convert("RGB"))
        masks, _flows, _styles = model.eval(image, diameter=None, channels=[0, 0])
        detections: list[dict[str, Any]] = []
        for mask_id in sorted(int(value) for value in np.unique(masks) if value != 0):
            bbox = bbox_from_mask(masks, mask_id)
            if bbox is None:
                continue
            detections.append(
                {
                    "class_id": 0,
                    "class_name": "cellpose_cell",
                    "confidence": None,
                    "box_xyxy": [round(value, 2) for value in bbox],
                    "mask_id": mask_id,
                }
            )

        annotated_path = model_output_dir / source.path.name
        draw_detections(source.path, detections, annotated_path, "cellpose_cyto3")
        results_json.append(
            {
                "image": str(source.path.relative_to(args.output_dir)),
                "model": "cellpose_cyto3",
                "model_repo": "MouseLand/cellpose pretrained cyto3",
                "annotation": str(annotated_path.relative_to(args.output_dir)),
                "detections": detections,
            }
        )
    return results_json


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    image_sources: list[ImageSource] = []
    if not args.skip_hf_dataset:
        image_sources.extend(
            collect_hf_dataset_images(
                output_dir=args.output_dir,
                image_dir=image_dir,
                force=args.force_download,
                max_images=args.max_dataset_images,
            )
        )
    if not args.skip_wikimedia:
        image_sources.extend(collect_wikimedia_images(image_dir=image_dir, force=args.force_download))

    image_sources = image_sources[: args.max_images]
    if not image_sources:
        raise SystemExit("No images were collected. Check --skip-* and --max-* arguments.")

    write_json(
        args.output_dir / "sources.json",
        [source.as_json(args.output_dir) for source in image_sources],
    )

    weights = ensure_models(args.models, args.model_cache_dir, args.force_download)
    detection_results = run_yolo_models(image_sources, weights, args)
    if args.run_cellpose:
        detection_results.extend(run_cellpose(image_sources, args))

    write_json(args.output_dir / "detections.json", detection_results)

    print(f"Annotated {len(image_sources)} images.")
    print(f"JSON results: {args.output_dir / 'detections.json'}")
    print(f"Annotated images: {args.output_dir / 'annotations'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
