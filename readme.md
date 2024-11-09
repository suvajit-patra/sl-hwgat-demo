[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hierarchical-windowed-graph-attention-network/sign-language-recognition-on-fdmse-isl)](https://paperswithcode.com/sota/sign-language-recognition-on-fdmse-isl?p=hierarchical-windowed-graph-attention-network)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hierarchical-windowed-graph-attention-network/sign-language-recognition-on-lsa64)](https://paperswithcode.com/sota/sign-language-recognition-on-lsa64?p=hierarchical-windowed-graph-attention-network)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hierarchical-windowed-graph-attention-network/sign-language-recognition-on-autsl)](https://paperswithcode.com/sota/sign-language-recognition-on-autsl?p=hierarchical-windowed-graph-attention-network)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hierarchical-windowed-graph-attention-network/sign-language-recognition-on-wlasl-2000)](https://paperswithcode.com/sota/sign-language-recognition-on-wlasl-2000?p=hierarchical-windowed-graph-attention-network)

# Sign Language Recognition System

This repository contains a Sign Language Recognition System using the HWGAT model. It includes a Jupyter Notebook demo for model inference for sign language recognition and a Flask web application API.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Notebook Demo](#notebook-demo)
  - [Flask App](#flask-app)
- [Model](#model)
- [License](#license)

## Overview
The Sign Language Recognition System leverages the HWGAT (Hierarchical Windowed Graph Attention Network) model to recognize sign language gestures. The system provides a notebook demo for testing the model with sample data and a Flask web application API for making web based application.

## Installation
1. Clone this repository and install the necessary dependencies.

    ```bash
    git clone https://github.com/yourusername/sl-hwgat-demo.git
    cd sl-hwgat-demo
    pip install -r requirements.txt
    ```

2. Get the pretrained model and sign class map from [here](https://drive.google.com/file/d/1UlR1MBgmPRezvZ3n191tyOdtZL4cTbdx/view?usp=sharing) and update the `save_model_path` and `class_map_path` in the `demo_utils.py` file.

## Usage

### Notebook Demo
The Jupyter Notebook demo allows you to test the HWGAT model on real-time camera feed.

1. Open the `demo.ipynb` notebook.
2. Follow the instructions in the notebook to run the demo.

### Flask App
The Flask web application API provides an interface for making web based sign language recognition application.

1. Run the Flask application.

    ```
    python api_app.py
    ```

2. Make web application use the API endpoint `http://127.0.0.1:5000/upload` for sign language recognition.

## Model
The HWGAT model is a Hierarchical Windowed Graph Attention Network, designed for sign language recognition. It is pre-trained and included in this repository. You can go the main [HWGAT repository](https://github.com/suvajit-patra/sl-hwgat) to get information about training the model.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation
If you find this project useful in your research, please consider cite:
```bibtex
@misc{patra2024hierarchicalwindowedgraphattention,
      title={Hierarchical Windowed Graph Attention Network and a Large Scale Dataset for Isolated Indian Sign Language Recognition}, 
      author={Suvajit Patra and Arkadip Maitra and Megha Tiwari and K. Kumaran and Swathy Prabhu and Swami Punyeshwarananda and Soumitra Samanta},
      year={2024},
      eprint={2407.14224},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.14224}, 
}
```

---

Thank you for using this repository. For any questions or support, please open an issue in this repository.

---
