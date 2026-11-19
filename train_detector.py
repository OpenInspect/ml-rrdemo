"""Fine-tune torchvision Faster R-CNN (ResNet50-FPN v2, COCO-pretrained) on Pascal VOC-style solder defects.

Layout (both dirs contain Annotations/*.xml and JPEGImages/*):
    train_data/   -> split 80/20 into train/val (by image, seeded)
    test_data/    -> held-out test set, only evaluated at the end

    python train_detector.py --train-dir train_data --test-dir test_data --epochs 40
"""
import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, Subset
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.transforms.functional import to_tensor

CLASSES = ["__background__", "Bad_podu", "Bad_qiaojiao"]


class VOCDefects(Dataset):
    def __init__(self, root, hflip=False, vflip=False, rot90=False):
        self.root = Path(root)
        self.hflip, self.vflip, self.rot90 = hflip, vflip, rot90
        self.ids = sorted(p.stem for p in (self.root / "Annotations").glob("*.xml"))

    def __len__(self):
        return len(self.ids)

    def _image_path(self, stem):
        for ext in (".jpeg", ".jpg", ".png"):
            p = self.root / "JPEGImages" / (stem + ext)
            if p.exists():
                return p
        raise FileNotFoundError(stem)

    def __getitem__(self, i):
        stem = self.ids[i]
        img = Image.open(self._image_path(stem)).convert("RGB")
        boxes, labels = [], []
        for obj in ET.parse(self.root / "Annotations" / f"{stem}.xml").getroot().iter("object"):
            name = obj.find("name").text
            if name not in CLASSES:
                continue
            b = obj.find("bndbox")
            x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(CLASSES.index(name))
        img = to_tensor(img)
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(labels, dtype=torch.int64)
        # NOTE: flips/rotations change which direction the lighting appears to come from, so all are off by default.
        if self.hflip and random.random() < 0.5:
            img = img.flip(-1)
            w = img.shape[-1]
            boxes = torch.stack([w - boxes[:, 2], boxes[:, 1], w - boxes[:, 0], boxes[:, 3]], 1)
        if self.vflip and random.random() < 0.5:
            img = img.flip(-2)
            h = img.shape[-2]
            boxes = torch.stack([boxes[:, 0], h - boxes[:, 3], boxes[:, 2], h - boxes[:, 1]], 1)
        if self.rot90:
            img, boxes = rotate90(img, boxes, random.randint(0, 3))
        return img, {"boxes": boxes, "labels": labels}


def rotate90(img, boxes, k):
    """Rotate a CHW image and its xyxy boxes by k * 90 degrees counter-clockwise."""
    for _ in range(k % 4):
        w = img.shape[-1]
        img = torch.rot90(img, 1, dims=(1, 2))
        # (x, y) -> (y, w - x) in the rotated frame
        boxes = torch.stack([boxes[:, 1], w - boxes[:, 2], boxes[:, 3], w - boxes[:, 0]], 1)
    return img, boxes


def collate(batch):
    return tuple(zip(*batch))


def build_model(img_size, pretrained=True):
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None,
        weights_backbone=None,
        min_size=img_size, max_size=img_size,   # upscale 600px images so ~15px defects get more pixels
        box_detections_per_img=300,
    )
    # Anchors: one size per FPN level (small, since defects are ~3-75px), 5 aspect ratios (h/w) for thin boxes.
    ratios = (0.25, 0.5, 1.0, 2.0, 4.0)
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),  # in resized-image pixels (~1.7x original)
        aspect_ratios=(ratios,) * 5,
    )
    # Anchors/location went 3 -> 5, so the RPN's final 1x1 convs must be re-created (shared 3x3 conv stays pretrained).
    head = model.rpn.head
    n = len(ratios)
    head.cls_logits = torch.nn.Conv2d(256, n, 1)
    head.bbox_pred = torch.nn.Conv2d(256, n * 4, 1)
    for l in (head.cls_logits, head.bbox_pred):
        torch.nn.init.normal_(l.weight, std=0.01)
        torch.nn.init.constant_(l.bias, 0)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, len(CLASSES))
    return model


@torch.no_grad()
def evaluate(model, loader, device, desc="eval"):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    for imgs, targets in tqdm(loader, desc=desc, leave=False):
        outs = model([i.to(device) for i in imgs])
        preds = [{k: v.cpu() for k, v in o.items()} for o in outs]
        metric.update(preds, [{k: v for k, v in t.items()} for t in targets])
    return metric.compute()


COLORS = {"Bad_podu": (255, 64, 64), "Bad_qiaojiao": (64, 160, 255)}


@torch.no_grad()
def find_annotation(image_path, gt):
    """gt == 'auto' -> look for ../Annotations/<stem>.xml next to the image's JPEGImages dir; else treat as a path."""
    if gt != "auto":
        return Path(gt)
    img = Path(image_path)
    return img.parent.parent / "Annotations" / f"{img.stem}.xml"


def draw_ground_truth(draw, xml_path):
    n = 0
    for obj in ET.parse(xml_path).getroot().iter("object"):
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        # Bright green, drawn slightly outside the prediction boxes so both stay visible when they overlap.
        draw.rectangle([x1 - 5, y1 - 5, x2 + 5, y2 + 5], outline=(0, 230, 0), width=1)
        draw.text((x1, y2 + 6), obj.find("name").text, fill=(0, 230, 0))
        n += 1
    return n


def predict(image_path, weights, score_thresh, save_path, show, gt=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(weights, map_location=device)
    model = build_model(ckpt["img_size"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    img = Image.open(image_path).convert("RGB")
    out = model([to_tensor(img).to(device)])[0]
    keep = out["scores"] >= score_thresh
    boxes, labels, scores = out["boxes"][keep].cpu(), out["labels"][keep].cpu(), out["scores"][keep].cpu()

    draw = ImageDraw.Draw(img)
    if gt:
        xml_path = find_annotation(image_path, gt)
        if xml_path.exists():
            print(f"{draw_ground_truth(draw, xml_path)} ground-truth box(es) from {xml_path} (green)")
        else:
            print(f"no annotation file found at {xml_path}; skipping ground truth")
    for (x1, y1, x2, y2), l, sc in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
        name = ckpt["classes"][l]
        color = COLORS.get(name, (255, 255, 0))
        draw.rectangle([x1 - 2, y1 - 2, x2 + 2, y2 + 2], outline=color, width=2)  # small pad so tiny boxes stay visible
        draw.text((x1, max(0, y1 - 12)), f"{name} {sc:.2f}", fill=color)
        print(f"{name:14s} {sc:.2f}  [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
    print(f"{len(boxes)} detection(s) at score >= {score_thresh}")
    img.save(save_path)
    print(f"saved {save_path}")
    if show:
        img.show()


def fmt(m):
    per = m["map_per_class"].tolist() if m["map_per_class"].ndim else [float(m["map_per_class"])]
    return (f"mAP@.5 {m['map_50'].item():.3f}  mAP@.5:.95 {m['map'].item():.3f}  "
            f"AR@100 {m['mar_100'].item():.3f}  per-class mAP {[round(x, 3) for x in per]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="train_data")
    ap.add_argument("--test-dir", default="test_data")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--hflip", action="store_true", help="random horizontal flip (train only)")
    ap.add_argument("--vflip", action="store_true", help="random vertical flip (train only)")
    ap.add_argument("--rot90", action="store_true", help="random 0/90/180/270 rotation (train only)")
    ap.add_argument("--out", default="detector_best.pt")
    ap.add_argument("--predict", metavar="IMAGE", help="run detection on this image and exit (no training)")
    ap.add_argument("--weights", default=None, help="checkpoint for --predict (default: --out)")
    ap.add_argument("--score-thresh", type=float, default=0.5)
    ap.add_argument("--save", default="prediction.png", help="where --predict writes the annotated image")
    ap.add_argument("--gt", nargs="?", const="auto", default=None, metavar="XML",
                    help="also draw ground-truth boxes in green; optional XML path, "
                         "default: ../Annotations/<image>.xml next to the image")
    ap.add_argument("--show", action="store_true", help="open the annotated image in the default viewer")
    args = ap.parse_args()

    if args.predict:
        predict(args.predict, args.weights or args.out, args.score_thresh, args.save, args.show, args.gt)
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    # 80/20 split by image
    n = len(VOCDefects(args.train_dir))
    idx = list(range(n))
    random.Random(args.seed).shuffle(idx)
    n_val = max(1, int(n * args.val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_ds = Subset(VOCDefects(args.train_dir, hflip=args.hflip, vflip=args.vflip, rot90=args.rot90), train_idx)
    val_ds = Subset(VOCDefects(args.train_dir), val_idx)
    test_ds = VOCDefects(args.test_dir)
    print(f"train {len(train_ds)}  val {len(val_ds)}  test {len(test_ds)}")

    kw = dict(collate_fn=collate, num_workers=args.workers, persistent_workers=args.workers > 0, pin_memory=amp)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, 4, **kw)
    test_loader = DataLoader(test_ds, 4, **kw)

    model = build_model(args.img_size).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    total = args.epochs * len(train_loader)
    warm = min(100, len(train_loader))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + torch.cos(torch.tensor((s - warm) / max(1, total - warm) * 3.14159265)).item()))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        run = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1:02d}/{args.epochs}", leave=False)
        for step, (imgs, targets) in enumerate(pbar, 1):
            imgs = [i.to(device) for i in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss = sum(model(imgs, targets).values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
            pbar.set_postfix(loss=f"{run / step:.3f}", lr=f"{sched.get_last_lr()[0]:.4f}")
        m = evaluate(model, val_loader, device, desc="val")
        mem = torch.cuda.max_memory_allocated() / 2**30 if amp else 0
        tqdm.write(f"epoch {epoch + 1:02d}/{args.epochs} loss {run / len(train_loader):.3f} | val {fmt(m)} | VRAM {mem:.1f} GiB")
        if m["map_50"].item() > best:
            best = m["map_50"].item()
            torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "img_size": args.img_size}, args.out)

    print(f"\nbest val mAP@.5 {best:.3f}; evaluating that checkpoint on held-out test set")
    model.load_state_dict(torch.load(args.out)["state_dict"])
    print("TEST", fmt(evaluate(model, test_loader, device, desc="test")))


if __name__ == "__main__":
    main()
