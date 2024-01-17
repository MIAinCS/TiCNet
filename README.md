# TiCNet: Transformer in Convolutional Neural Network for Pulmonary Nodule Detection on CT images
## Usage

```
@article{ma2024ticnet,
  title={TiCNet: Transformer in Convolutional Neural Network for Pulmonary Nodule Detection on CT Images},
  author={Ma, Ling and Li, Gen and Feng, Xingyu and Fan, Qiliang and Liu, Lizhi},
  journal={Journal of Imaging Informatics in Medicine},
  pages={1--13},
  year={2024},
  publisher={Springer}
}
```

The project structure and intention are as follows :

```
TiCNet                             
    ├── config.py                   # Configuration information
    ├── dataset
    │   ├── bbox_reader.py          # Custom Dataset Loader
    │   └── collate.py              # Data collate function
    ├── net                         # All models are created in this folder
    │   ├── __init__.py
    │   ├── layer                   # Loss function and other
    │   ├── feature_net.py          # Backbone encoder
    │   ├── main_net.py             # Base network architecture
    │   ├── module.py               # Attention module
    │   ├── multi_scale.py          # multi-scale fusion module
    │   ├── position_encoding.py    # Build position encoding
    │   └── transformer.py          # Transformer module
    ├── utils
    │   ├── cvrt_annos_to_npy.py    # Annotation tranform
    │   ├── preprocess.py           # Preprocess the CT images
    │   ├── split.py                # cross val test split
    │   └── ...
    ├── test.py                     # Test file
    ├── train.py                    # Train file
    ├── visualize.py                # Generate visualized detection image
    └── ...
```

## Experiments steps

- Install dependencies

  ```
  pip install -r requirements.txt
  ```

- Build a custom module for bounding box NMS and overlap calculation

  ```
  cd build/box
  python setup.py install
  ```

- Data preprocessing

  ```
  cd utils
  python preprocess.py
  ```

- Generating new annotations

  ```
  cd utils
  python cvrt_annos_to_npy.py
  ```

- Training

  ```
  python train.py [--args]
  ```

- Evaluation

  ```
  python test.py [--args]
  ```

- Using followed script to perform a 10 fold cross validation

  ```
  cd scripts
  bash cross_val_10fold.sh
  ```

## Results

We conducted a 10-fold cross-validation test on the [LUNA16](https://luna16.grand-challenge.org/) dataset, and the results are shown in the following table:
|        | 0.125 | 0.25  | 0.5   | 1     | 2     | 4     | 8     | Avg.  |
| :----- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 0_Fold | 83.03 | 91.07 | 93.75 | 96.42 | 97.32 | 97.32 | 97.32 | 93.75 |
| 1_Fold | 88.28 | 89.06 | 90.62 | 92.96 | 96.09 | 96.87 | 96.87 | 92.96 |
| 2_Fold | 84.37 | 88.28 | 92.18 | 94.53 | 96.09 | 96.09 | 96.09 | 92.52 |
| 3_Fold | 73.1  | 82.35 | 89.91 | 93.27 | 97.47 | 98.31 | 98.31 | 90.39 |
| 4_Fold | 78.12 | 81.25 | 88.28 | 92.96 | 96.09 | 96.87 | 96.87 | 90.06 |
| 5_Fold | 78.7  | 82.4  | 85.18 | 86.11 | 92.59 | 94.44 | 94.44 | 87.69 |
| 6_Fold | 77.51 | 82.94 | 86.04 | 89.14 | 92.24 | 96.89 | 97.67 | 88.92 |
| 7_Fold | 77.47 | 84.68 | 86.48 | 90.09 | 94.59 | 95.49 | 95.49 | 89.18 |
| 8_Fold | 79.66 | 86.44 | 88.98 | 91.52 | 94.06 | 95.76 | 98.3  | 90.67 |
| 9_Fold | 75.23 | 85.71 | 89.52 | 95.23 | 97.14 | 97.14 | 98.09 | 91.15 |
| Avg.   | 79.55 | 85.42 | 89.09 | 92.22 | 95.37 | 96.52 | 96.95 | 90.73 |


## References
The code for this work is referenced from [https://github.com/uci-cbcl/NoduleNet](https://github.com/uci-cbcl/NoduleNet) and [https://github.com/facebookresearch/detr](https://github.com/facebookresearch/detr).

## Contact
Any questions about this paper please contact [ligen@mail.nankai.edu.cn](mailto:ligen@mail.nankai.edu.cn).
