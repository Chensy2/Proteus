# Proteus

This repository accompanies our **NDSS 2026** paper:
**Enhancing Website Fingerprinting Attacks against Traffic Drift**.

Proteus introduces an adaptive fine-tuning framework that significantly enhances the robustness of website fingerprinting (WF) attacks against **temporal, version, network, behavioral, and open-world traffic drift scenarios**, by leveraging **unlabeled traffic** for continual adaptation.

---

## 📂 Datasets

We collected **over 350,000 real-world Tor traffic traces**, covering diverse drift scenarios.

* The dataset is organized into **six categories** corresponding to the experimental settings in our paper.
* Download the datasets via [link](https://drive.google.com/drive/folders/1bAqAvvDrY2wrY4EU-Rxm9mv9hsIvKwGk).

```bash
mkdir datasets
```

Extract all datasets and place them under the `datasets/` directory.

---

## ⚙️ Installation

Clone this repository and install the dependencies:

```bash
cd wflib_copy
pip install --user .
cd ..
```

More details can be found in [WFlib](https://github.com/Xinhao-Deng/Website-Fingerprinting-Library).

---

## 🧪 Running Experiments

We provide scripts to reproduce the main experiments reported in our NDSS 2026 paper.

* **Temporal Drift**

```bash
bash scripts/TemporalDrift/Proteus.sh
```

* **Version Drift**

```bash
bash scripts/VersionDrift/Proteus.sh
```

* **Network Drift**

```bash
bash scripts/NetworkDrift/Proteus.sh
```

* **Behavioral Drift**

```bash
bash scripts/BehaviorDrift/Proteus.sh
```

* **Open-World Setting**

```bash
bash scripts/OpenWorld/Proteus.sh
```

* **Against Defenses**

```bash
bash scripts/Defense/Proteus.sh
```

---

## 📖 Citation

If you use this code or dataset in your research, please cite our paper:

```bibtex
@inproceedings{proteus2026,
  title     = {Enhancing Website Fingerprinting Attacks against Traffic Drift},
  author    = {Xinhao Deng, Yixiang Zhang, Qi Li, Zhuotao Liu, Yabo Wang, Ke Xu},
  booktitle = {Network and Distributed System Security (NDSS) Symposium},
  year      = {2026}
}
```

---

## 📜 License

This project is released under the **MIT License**. See [LICENSE](./LICENSE) for details.
