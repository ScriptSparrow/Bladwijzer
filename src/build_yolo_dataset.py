"""
build_yolo_dataset.py
=====================

Bouwt een schone YOLO-dataset uit de bestaande CVAT XML-annotaties en de
nieuwe CVAT YOLO-export.

Bronnen:
    - berenklauwen.zip                -> annotations.xml voor oude berenklauw
    - madeliefjes.zip                 -> annotations.xml voor oude madeliefjes
    - data/raw/berenklauw/berenklauw_set2.zip -> YOLO-labels voor nieuwe berenklauw

Canonical class-ids in de output:
    0 = madeliefje
    1 = berenklauw

Class-ids uit elke bron worden expliciet geremapt via labelnaam. Daardoor wordt
label_berenklauw altijd 1, ook als een CVAT-export die klasse lokaal als 0 heeft.

Output:
    data/yolo/
        data.yaml
        images/{train,val,test}/...
        labels/{train,val,test}/...

Gebruik:
    python src/build_yolo_dataset.py
"""
from __future__ import annotations

from collections import Counter, defaultdict
import random
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from PIL import Image, ImageOps

# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "yolo"
EXTRACTED_ANNOTATION_DIR = RAW_DIR / "annotations"
NEGATIVES_DIR = RAW_DIR / "negatives"

SOURCES = [
    {
        "name": "oude_berenklauw_xml",
        "kind": "cvat_xml_zip",
        "img_dir": RAW_DIR / "berenklauw",
        "zip": ROOT / "berenklauwen.zip",
        "xml": EXTRACTED_ANNOTATION_DIR / "berenklauwen_annotations.xml",
    },
    {
        "name": "oude_madeliefjes_xml",
        "kind": "cvat_xml_zip",
        "img_dir": RAW_DIR / "madeliefje",
        "zip": ROOT / "madeliefjes.zip",
        "xml": EXTRACTED_ANNOTATION_DIR / "madeliefjes_annotations.xml",
    },
    {
        "name": "nieuwe_berenklauw_set2_yolo",
        "kind": "cvat_yolo_zip",
        "img_dir": RAW_DIR / "berenklauw",
        "zip": RAW_DIR / "berenklauw" / "berenklauw_set2.zip",
    },
]

LABEL_TO_CLS = {
    "label_madeliefje": 0,
    "label_berenklauw": 1,
}
CLASS_NAMES = {0: "madeliefje", 1: "berenklauw"}

SEED = 42
SPLIT_RATIOS = (0.8, 0.1, 0.1)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


# --------------------------------------------------------------------------- #
def extract_annotations_xml(zip_path: Path, out_path: Path) -> Path:
    """Pak annotations.xml uit een CVAT XML-export ZIP naar een stabiel pad."""
    if not zip_path.is_file():
        raise FileNotFoundError(f"Zip niet gevonden: {zip_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        if "annotations.xml" not in zf.namelist():
            raise FileNotFoundError(f"{zip_path} bevat geen annotations.xml")
        out_path.write_bytes(zf.read("annotations.xml"))
    return out_path


def parse_zip_yaml_names(yaml_bytes: bytes) -> dict[int, str]:
    """Minimale parser voor het 'names:' blok in een CVAT data.yaml."""
    text = yaml_bytes.decode("utf-8")
    names: dict[int, str] = {}
    in_names = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("names:"):
            in_names = True
            continue
        if in_names:
            m = re.match(r"^\s*(\d+)\s*:\s*(.+?)\s*$", line)
            if m:
                names[int(m.group(1))] = m.group(2)
            elif stripped and not line.startswith((" ", "\t")):
                in_names = False
    return names


def yolo_box_from_cvat(box: ET.Element, width: float, height: float) -> tuple[float, float, float, float] | None:
    xtl = max(0.0, min(width, float(box.attrib["xtl"])))
    ytl = max(0.0, min(height, float(box.attrib["ytl"])))
    xbr = max(0.0, min(width, float(box.attrib["xbr"])))
    ybr = max(0.0, min(height, float(box.attrib["ybr"])))

    if xbr <= xtl or ybr <= ytl:
        return None

    cx = ((xtl + xbr) / 2.0) / width
    cy = ((ytl + ybr) / 2.0) / height
    w = (xbr - xtl) / width
    h = (ybr - ytl) / height
    return cx, cy, w, h


def load_labels_from_xml(xml_path: Path) -> tuple[list[str], dict[str, list[tuple[int, float, float, float, float]]]]:
    """Lees CVAT XML en converteer rectangle boxes naar canonical YOLO labels."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_names: list[str] = []
    labels: dict[str, list[tuple[int, float, float, float, float]]] = {}

    for image in root.findall("image"):
        name = image.attrib["name"]
        width = float(image.attrib["width"])
        height = float(image.attrib["height"])
        image_names.append(name)

        entries: list[tuple[int, float, float, float, float]] = []
        for box in image.findall("box"):
            label_name = box.attrib["label"]
            if label_name not in LABEL_TO_CLS:
                print(f"  WARN: onbekend label '{label_name}' in {xml_path.name}, wordt overgeslagen")
                continue

            yolo_box = yolo_box_from_cvat(box, width, height)
            if yolo_box is None:
                continue
            entries.append((LABEL_TO_CLS[label_name], *yolo_box))

        labels[name] = entries

    return image_names, labels


def load_labels_from_yolo_zip(zip_path: Path) -> tuple[list[str], dict[str, list[tuple[int, float, float, float, float]]], dict[int, int]]:
    """
    Returnt image names uit train.txt, canonical YOLO labels per image name, en
    de bronklasse -> canonical klasse remap.
    """
    if not zip_path.is_file():
        raise FileNotFoundError(f"Zip niet gevonden: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        names_in_zip = parse_zip_yaml_names(zf.read("data.yaml"))
        remap: dict[int, int] = {}
        for src_id, label_name in names_in_zip.items():
            if label_name in LABEL_TO_CLS:
                remap[src_id] = LABEL_TO_CLS[label_name]
            else:
                print(f"  WARN: onbekend label '{label_name}' in {zip_path.name}, wordt overgeslagen")

        if "train.txt" in zf.namelist():
            image_names = [
                Path(line.strip()).name
                for line in zf.read("train.txt").decode("utf-8").splitlines()
                if line.strip()
            ]
        else:
            image_names = []

        labels_by_stem: dict[str, list[tuple[int, float, float, float, float]]] = defaultdict(list)
        for n in zf.namelist():
            if not (n.startswith("labels/") and n.endswith(".txt")):
                continue

            stem = Path(n).stem
            for line in zf.read(n).decode("utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                src_cls = int(parts[0])
                if src_cls not in remap:
                    continue

                cx, cy, w, h = map(float, parts[1:5])
                labels_by_stem[stem].append((remap[src_cls], cx, cy, w, h))

    if not image_names:
        image_names = sorted(f"{stem}.jpg" for stem in labels_by_stem)

    labels = {name: labels_by_stem.get(Path(name).stem, []) for name in image_names}
    return image_names, labels, remap


def find_image(img_dir: Path, image_name: str) -> Path | None:
    img_path = img_dir / image_name
    if img_path.is_file() and img_path.suffix.lower() in IMAGE_SUFFIXES:
        return img_path

    matches = [p for p in img_dir.iterdir() if p.is_file() and p.name.lower() == image_name.lower()]
    if matches:
        return matches[0]
    return None


def source_class_counts(labels: dict[str, list[tuple[int, float, float, float, float]]]) -> Counter:
    cnt: Counter = Counter()
    for boxes in labels.values():
        cnt.update(c for c, *_ in boxes)
    return cnt


def collect_negative_records() -> list[dict]:
    """Verzamel hard-negative images (geen boxes) uit data/raw/negatives."""
    if not NEGATIVES_DIR.is_dir():
        return []
    records: list[dict] = []
    for p in sorted(NEGATIVES_DIR.iterdir(), key=lambda x: x.name.lower()):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            records.append({
                "name": p.name,
                "src": p,
                "boxes": [],
                "sources": ["negatives"],
            })
    if records:
        print(f"  negatives: {len(records)} images uit {NEGATIVES_DIR.relative_to(ROOT)}")
    return records


def collect_records() -> list[dict]:
    """Maak een unieke record per image en merge boxes als bronnen overlappen."""
    records_by_path: dict[str, dict] = {}

    for src in SOURCES:
        img_dir = src["img_dir"]
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Image-map niet gevonden: {img_dir}")

        if src["kind"] == "cvat_xml_zip":
            xml_path = extract_annotations_xml(src["zip"], src["xml"])
            image_names, labels = load_labels_from_xml(xml_path)
            print(f"  {src['name']}: uitgepakt naar {xml_path.relative_to(ROOT)}")
        elif src["kind"] == "cvat_yolo_zip":
            image_names, labels, remap = load_labels_from_yolo_zip(src["zip"])
            print(f"  {src['name']}: class-remap = {remap}")
        else:
            raise ValueError(f"Onbekende source kind: {src['kind']}")

        cnt = source_class_counts(labels)
        print(
            f"    {len(image_names)} images, "
            f"{sum(cnt.values())} boxes "
            f"(madeliefje={cnt[0]}, berenklauw={cnt[1]})"
        )

        missing = 0
        for image_name in image_names:
            img_path = find_image(img_dir, image_name)
            if img_path is None:
                missing += 1
                continue

            key = str(img_path.resolve()).lower()
            if key not in records_by_path:
                records_by_path[key] = {
                    "name": img_path.name,
                    "src": img_path,
                    "boxes": [],
                    "sources": [],
                }

            records_by_path[key]["boxes"].extend(labels.get(image_name, []))
            records_by_path[key]["sources"].append(src["name"])

        if missing:
            print(f"    WARN: {missing} images uit annotaties niet gevonden in {img_dir}")

    records = sorted(records_by_path.values(), key=lambda r: r["name"].lower())
    validate_records(records)

    # Voeg hard-negatives toe (alleen als ze geen bestaande key overlappen).
    existing_keys = {str(r["src"].resolve()).lower() for r in records}
    for neg in collect_negative_records():
        key = str(neg["src"].resolve()).lower()
        if key not in existing_keys:
            records.append(neg)
            existing_keys.add(key)
    records.sort(key=lambda r: r["name"].lower())
    return records


def validate_records(records: list[dict]) -> None:
    for rec in records:
        for cls_id, cx, cy, w, h in rec["boxes"]:
            if cls_id not in CLASS_NAMES:
                raise ValueError(f"Onbekende class-id {cls_id} in {rec['name']}")
            if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
                raise ValueError(f"Box buiten YOLO-range in {rec['name']}: {(cx, cy, w, h)}")

        is_berenklauw_source = "berenklauw" in rec["src"].parent.name.lower() or "berenklauw" in rec["name"].lower()
        if is_berenklauw_source and any(cls_id == 0 for cls_id, *_ in rec["boxes"]):
            raise ValueError(f"Berenklauw-image bevat class 0 label: {rec['src']}")


def stratified_split(records: list[dict]) -> dict[str, list[dict]]:
    rnd = random.Random(SEED)
    # Stratum-key: 'negative' (geen boxes), 'berenklauw' (heeft berenklauw),
    # of 'madeliefje_only' (alleen madeliefje). Zo komen negatives ook in val/test.
    strata: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        if not rec["boxes"]:
            stratum = "negative"
        elif any(b[0] == 1 for b in rec["boxes"]):
            stratum = "berenklauw"
        else:
            stratum = "madeliefje_only"
        strata[stratum].append(rec)

    splits = {"train": [], "val": [], "test": []}
    for _, items in sorted(strata.items()):
        items = items[:]
        rnd.shuffle(items)
        n = len(items)
        n_val = max(1, round(n * SPLIT_RATIOS[1])) if n >= 3 else 0
        n_test = max(1, round(n * SPLIT_RATIOS[2])) if n >= 3 else 0
        if n - n_val - n_test < 1:
            n_test = max(0, n - n_val - 1)
        splits["val"].extend(items[:n_val])
        splits["test"].extend(items[n_val:n_val + n_test])
        splits["train"].extend(items[n_val + n_test:])

    for split in splits.values():
        split.sort(key=lambda r: r["name"].lower())
    return splits


def reset_output_dir() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for split in ("train", "val", "test"):
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_dataset(splits: dict[str, list[dict]]) -> None:
    print("\nKopieer images + schrijf labels...")
    for split, recs in splits.items():
        img_out = OUT_DIR / "images" / split
        lbl_out = OUT_DIR / "labels" / split
        for rec in recs:
            with Image.open(rec["src"]) as im:
                im = ImageOps.exif_transpose(im)
                out_path = img_out / rec["name"]
                save_kwargs = {}
                if out_path.suffix.lower() in {".jpg", ".jpeg"}:
                    if im.mode not in {"RGB", "L"}:
                        im = im.convert("RGB")
                    save_kwargs["quality"] = 95
                im.save(out_path, **save_kwargs)

            stem = Path(rec["name"]).stem
            lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for (c, cx, cy, w, h) in rec["boxes"]]
            (lbl_out / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")


def write_data_yaml() -> None:
    yaml_path = OUT_DIR / "data.yaml"
    yaml_path.write_text(
        f"path: {OUT_DIR.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n"
        f"  0: {CLASS_NAMES[0]}\n"
        f"  1: {CLASS_NAMES[1]}\n",
        encoding="utf-8",
    )
    print(f"\nGeschreven: {yaml_path}")


def print_split_stats(splits: dict[str, list[dict]]) -> None:
    print("Stratified split (seed=42, 80/10/10)...")
    for split in ("train", "val", "test"):
        recs = splits[split]
        image_cnt = len(recs)
        with_bk = sum(1 for r in recs if any(b[0] == 1 for b in r["boxes"]))
        negatives = sum(1 for r in recs if not r["boxes"])
        box_cnt = Counter(c for r in recs for c, *_ in r["boxes"])
        ratio = (box_cnt[0] / box_cnt[1]) if box_cnt[1] else float("inf")
        print(
            f"  {split:5s}: {image_cnt:4d} images, "
            f"{with_bk:4d} met berenklauw, "
            f"{negatives:4d} negatives, "
            f"madeliefje={box_cnt[0]:4d}, berenklauw={box_cnt[1]:4d} "
            f"(ratio {ratio:.2f}:1)"
        )


def main() -> None:
    print("Bronnen inspecteren, ZIPs uitpakken en labels laden...")
    records = collect_records()
    n_total = len(records)
    n_with_box = sum(1 for r in records if r["boxes"])
    total_boxes = Counter(c for r in records for c, *_ in r["boxes"])
    print(
        f"\nTotaal: {n_total} images "
        f"({n_with_box} met labels), "
        f"madeliefje={total_boxes[0]}, berenklauw={total_boxes[1]}"
    )

    print("\nReset output:", OUT_DIR)
    reset_output_dir()

    splits = stratified_split(records)
    print_split_stats(splits)
    write_dataset(splits)
    write_data_yaml()

    print("\nKlaar.")


if __name__ == "__main__":
    main()
