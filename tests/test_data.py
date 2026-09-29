import json
import os

import numpy as np
import pytest
import torch

from mqda.data.datasets import BraTSDataset, FigShareDataset, FolderDataset, collate
from mqda.data.transforms import Augment, crop_or_pad
from mqda.utils.config import load_config


def _write_brats_case(root, cid, naming="2023", shape=(48, 52, 40), seg_values=(1, 2, 3)):
    import nibabel as nib

    mods = ["t1n", "t1c", "t2w", "t2f"] if naming == "2023" else ["t1", "t1ce", "t2", "flair"]
    sep = "-" if naming == "2023" else "_"
    d = os.path.join(root, cid)
    os.makedirs(d)
    rng = np.random.default_rng(0)
    for m in mods:
        vol = np.zeros(shape, np.float32)
        vol[4:-4, 4:-4, 4:-4] = rng.random((shape[0] - 8, shape[1] - 8, shape[2] - 8)) + 1
        nib.save(nib.Nifti1Image(vol, np.eye(4)), os.path.join(d, f"{cid}{sep}{m}.nii.gz"))
    seg = np.zeros(shape, np.int16)
    seg[10:20, 10:20, 10:20] = seg_values[1]
    seg[12:18, 12:18, 12:18] = seg_values[2]
    seg[14:16, 14:16, 14:16] = seg_values[0]
    nib.save(nib.Nifti1Image(seg, np.eye(4)), os.path.join(d, f"{cid}{sep}seg.nii.gz"))


def test_brats_2023(tmp_path):
    pytest.importorskip("nibabel")
    _write_brats_case(tmp_path, "BraTS-GLI-00001-000")
    rep = tmp_path / "reports.jsonl"
    rep.write_text(json.dumps({"case_id": "BraTS-GLI-00001-000", "report": "Findings: x"}) + "\n")
    ds = BraTSDataset(str(tmp_path), size=(32, 32, 32), reports_jsonl=str(rep))
    item = ds[0]
    assert item["image"].shape == (4, 32, 32, 32)
    assert set(item["seg"].unique().tolist()) <= {0, 1, 2, 3}
    assert item["report"] == "Findings: x"
    tr = BraTSDataset(str(tmp_path), size=(32, 32, 32), train=True, augment={"p_affine": 1.0})
    assert tr[0]["image"].shape == (4, 32, 32, 32)
    full = BraTSDataset(str(tmp_path), size=None)
    assert full[0]["image"].shape == (4, 48, 52, 40)


def test_brats_2021_label_remap(tmp_path):
    pytest.importorskip("nibabel")
    _write_brats_case(tmp_path, "BraTS2021_00000", naming="2021", seg_values=(1, 2, 4))
    ds = BraTSDataset(str(tmp_path), naming="2021", size=None)
    assert set(ds[0]["seg"].unique().tolist()) == {0, 1, 2, 3}


def test_figshare_mat(tmp_path):
    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "1.mat", "w") as f:
        g = f.create_group("cjdata")
        img = np.zeros((64, 64), np.int16)
        img[10:50, 10:50] = 100
        mask = np.zeros((64, 64), np.uint8)
        mask[20:30, 20:30] = 1
        g["image"] = img.T
        g["label"] = np.array([[2.0]])
        g["tumorMask"] = mask.T
    ds = FigShareDataset(str(tmp_path), ["glioma", "meningioma", "pituitary", "no tumor"], size=(32, 32))
    item = ds[0]
    assert item["cls"] == 0 and item["image"].shape == (1, 32, 32) and item["seg"].sum() > 0


def test_folder_dataset(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    for folder in ("yes", "no"):
        os.makedirs(tmp_path / folder)
        Image.fromarray((np.random.rand(40, 40) * 255).astype(np.uint8)).save(tmp_path / folder / "a.png")
    ds = FolderDataset(str(tmp_path), {"no": 3, "yes": -100}, size=(24, 24))
    batch = collate([ds[0], ds[1]])
    assert batch["image"].shape == (2, 1, 24, 24)
    assert sorted(batch["cls"].tolist()) == [-100, 3]
    assert not batch["has_mask"].any()


def test_augment_and_crop_keep_shapes():
    img, lab = torch.randn(2, 20, 20), torch.randint(0, 4, (20, 20))
    a = Augment(p_affine=1, p_elastic=1, p_shift=1, p_noise=1, p_flip=0.5)
    x, y = a(img, lab)
    assert x.shape == img.shape and y.shape == lab.shape and y.dtype == torch.long
    x, y = crop_or_pad(torch.randn(1, 10, 30), torch.zeros(10, 30), (16, 16))
    assert x.shape == (1, 16, 16) and y.shape == (16, 16)


def test_config_inheritance():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config(os.path.join(root, "configs", "nickparvar.yaml"), ["train.epochs=3"])
    assert cfg["model"]["spatial_dims"] == 2          # from figshare_br35h.yaml
    assert cfg["model"]["n_qubits"] == 12             # from base.yaml
    assert cfg["data"]["type"] == "folder" and cfg["train"]["epochs"] == 3


def test_folder_split_files_select_the_listed_images(tmp_path):
    Image = pytest.importorskip("PIL.Image")

    from mqda.data.build import build_datasets
    from mqda.models.mqda_net import ModelConfig

    for cls in ("glioma", "notumor"):
        os.makedirs(tmp_path / "Training" / cls)
        for i in range(4):
            Image.fromarray((np.random.rand(20, 20) * 255).astype("uint8")).save(
                tmp_path / "Training" / cls / f"{i}.png")
    train_list, test_list = tmp_path / "tr.txt", tmp_path / "te.txt"
    train_list.write_text("Training/glioma/0.png\nTraining/notumor/0.png\n")
    test_list.write_text("Training/glioma/1.png\n")
    cfg = {"type": "folder", "size": [16, 16],
           "folder": {"class_map": {"glioma": 0, "notumor": 3}, "root": str(tmp_path),
                      "train_split": str(train_list), "val_split": str(test_list)}}
    tr, va = build_datasets(cfg, ModelConfig(spatial_dims=2, in_channels=1))
    assert len(tr) == 2 and len(va) == 1
    assert sorted(tr[i]["cls"] for i in range(len(tr))) == [0, 3]
