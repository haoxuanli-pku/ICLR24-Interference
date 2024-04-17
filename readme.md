
# This is the official pytorch implementation of the ICLR 24 "Be Aware of the Neighborhood Effect: Modeling Selection Bias under Interference for Recommendation" paper.
We use three public real-world datasets (Coat, Yahoo! R3 and KuaiRec) for real-world experiments, and ML-100K dataset for semi synthetic experiments. 

## Environment
The code runs well at python 3.7, and we use the version of pytorch = 1.9.1 + cu111

## For KuaiRec dataset:
- user.txt: biased data collected by normal policy of recommendation platform. Each line is user ID, item ID, rating of the user to the item. 
- random.txt: unbiased data collected by stochastic policy where items are assigned to users randomly. Each line in the file is user ID, item ID, rating of the user to the item. 

## Run the Code for semi synthetic experiments
First run the complete_final.ipynb to reconstruct the whole rating matrix, second run the convert_final.ipynb to obtain six prediction matrices. Finally run the synthetic_interference_final.ipynb file to obtain the results with varying mask numbers.

## Run the real-world experiments code
Please refer to coat.ipynb, yahoo.ipynb and kuai.ipynb for the results of the corresponding datasets.

## 
If you find this code useful for your work, please consider to cite our work as
```
@inproceedings{li2024interference,
  title={Be Aware of the Neighborhood Effect: Modeling Selection Bias under Interference for Recommendation},
  author={Li, Haoxuan and Zheng, Chunyuan and Ding, Sihao and Wu, Peng and Geng, Zhi and Feng, Fuli and He, Xiangnan},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2024}
}
```
