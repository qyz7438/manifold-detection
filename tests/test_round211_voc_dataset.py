from pathlib import Path

import torch
from PIL import Image

from spectral_detection_posttrain.datasets.voc_detection import VOC_CLASS_TO_LABEL, VOCDetectionSubset, parse_voc_annotation


def test_parse_voc_annotation_keeps_selected_classes(tmp_path):
    xml_path = tmp_path / "sample.xml"
    xml_path.write_text(
        """
<annotation>
  <object><name>person</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>10</xmax><ymax>20</ymax></bndbox></object>
  <object><name>bottle</name><bndbox><xmin>3</xmin><ymin>4</ymin><xmax>8</xmax><ymax>9</ymax></bndbox></object>
</annotation>
""",
        encoding="utf-8",
    )
    target = parse_voc_annotation(xml_path, classes=["person", "car", "dog"])
    assert target["boxes"].shape == (1, 4)
    assert target["labels"].tolist() == [VOC_CLASS_TO_LABEL["person"]]


def test_voc_subset_returns_faster_rcnn_target(tmp_path):
    root = tmp_path / "VOCdevkit" / "VOC2007"
    (root / "JPEGImages").mkdir(parents=True)
    (root / "Annotations").mkdir(parents=True)
    (root / "ImageSets" / "Main").mkdir(parents=True)
    Image.new("RGB", (32, 32), color=(127, 127, 127)).save(root / "JPEGImages" / "000001.jpg")
    (root / "Annotations" / "000001.xml").write_text(
        "<annotation><object><name>dog</name><bndbox><xmin>5</xmin><ymin>6</ymin><xmax>20</xmax><ymax>24</ymax></bndbox></object></annotation>",
        encoding="utf-8",
    )
    (root / "ImageSets" / "Main" / "train.txt").write_text("000001\n", encoding="utf-8")
    dataset = VOCDetectionSubset(tmp_path, image_set="train", classes=["person", "car", "dog"], download=False)
    image, target = dataset[0]
    assert image.shape[0] == 3
    assert target["boxes"].dtype == torch.float32
    assert target["labels"].tolist() == [VOC_CLASS_TO_LABEL["dog"]]
