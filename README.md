# GAM
The implementation of "Source-Free Multi-Domain Adaptation with Generally Auxiliary Model Training" in Python. 

Code for IJCNN(2022) publication. The full paper can be found [here](https://doi.org/10.1109/IJCNN55064.2022.9892718).

## Contribution

- A multi-source data-free domain adaptation method which learns both specific and general source models. 
- A general model based on auxiliary training that can fit multiple domains by sharing source parameters.
- An algorithm that introduces class balance coefficients to source-free domain adaptation.

## Overview
![Framework-Source](https://github.com/el3518/GAM/blob/main/img/frame-s.JPG)
![Framework-Target](https://github.com/el3518/GAM/blob/main/img/frame-t.JPG)

## Setup
Ensure that you have Python 3.7.4 and PyTorch 1.1.0

## Dataset
You can find the datasets [here](https://github.com/jindongwang/transferlearning/tree/master/data).

## Usage

Run "mda2s_cb.py" for source domains, "mds2sf_cb.py" for target domain. 

## Results

| Task  | D | W  | A | Avg  | 
| ---- | ---- | ---- | ---- | ---- |
| MSCLDA  | 99.5  | 98.7  | 75.4  | 91.2 |


Please consider citing if you find this helpful or use this code for your research.

Citation
```
@inproceedings{li2022source,
  title={Source-free multi-domain adaptation with generally auxiliary model training},
  author={Li, Keqiuyin and Lu, Jie and Zuo, Hua and Zhang, Guangquan},
  booktitle={2022 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--8},
  year={2022},
  organization={IEEE}
}
