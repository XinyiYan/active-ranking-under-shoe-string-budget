# **PARWiS**: Winner determination from Active Pairwise Comparisons under a Shoestring Budget
Provides code for the **AAAI 2020** submission number **5581**. All the algorithms implemented and datasets used have their corresponding references in the paper. For the ones which require external code or data, the links have been included in the comments section of the code file. Please find help section regarding the command line arguments in the `utils.py` file. The results corresponding to each run will be printed on screen once the code is run completely. Please refer to the `requirements.txt` file for additional libraries and packages necessary.

## Experiments on Synthetic Dataset
For all the algorithms, we resort to using a randomly generated ground truth score vector where topper has score 100 and rest in the range [0, 75]. To ensure fairness across comparing algorithms, we have precomputed a score vector and stored for values of n = 10, 25, 50.

### Reproducing Numbers
Say you want to reproduce the numbers corresponding to **n = 25** for *Synthetic-Data* by the **PARWiS** Algorithm, please use the following command:
```shell
python PARWiS.py \
       --n 25 \
       --experiments 10 \
       --iterations 100 \
       --precomputed \
       --budget 25
```

### Separation vs Recovery experiments
For the algorithms **PARWiS** and **SELECT**, we performed experiments when the topper had score **x** and the rest has **100-x**. Say you want to reproduce the experiments for **SELECT**, you can use the following command:
```shell
python PARWiS.py \
       --n 50 \
       --experiments 10 \
       --iterations 100 \
       --budget 3 \
       --toppers 55 60 65 70 75 80
```

## Experiments on Real World Dataset
We have evaluated our algorithm on five real-world datasets.
- SUSHI Item Set A & B ("sushi-A" & "sushi-B")
- Jester Joke Dataset 1 ("jester")
- Netflix Prize Dataset ("netflix")
- MovieLens 100k Dataset ("movielens")

### Reproducing Numbers
Say you want to reproduce the numbers corresponding to **Jester** Dataset by the **PARWiS** Algorithm, please use the following command:
```shell
python PARWiS.py \
       --experiments 10 \
       --iterations 100 \
       --dataset "jester" \
       --budget 7
```

## Datasets
The code for generating the BTL score vectors for all the datasets can be found in `/datasets/` folder. We have already generated and stored these numbers, the code can be run for verification purposes. The download links for the dataset have been mentioned in the code comments, please change the data-paths in the code wherever required as per your system before running. The corresponding score vectors will be printed on the screen as you run the file.

## References
- [choix](https://github.com/lucasmaystre/choix)
- ["Active Ranking from Pairwise Comparisons and When Parametric Assumptions Don't Help" Paper Supplement](https://github.com/reinhardh/supplement_active_ranking)
