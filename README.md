# **PARWiS**: Winner determination from Active Pairwise Comparisons under a Shoestring Budget
All the algorithms implemented and datasets used have their corresponding references in the paper. For the ones which require external code or data, the links have been included in the comments section of the code file and the references below. Please find help section regarding the command line arguments in the `utils.py` file. The results corresponding to each run will be printed on screen once the code is run completely. Please refer to the `requirements.txt` file for a list of libraries and packages necessary to run the code.

## Experiments on Synthetic Dataset
For all the simulation experiments, we resort to using a randomly generated ground truth score vector where topper has score *100* and rest in the range *[0, 75]*. To ensure fairness across comparing algorithms, we have precomputed a score vector and stored for values of *n = 10, 25, 50*. Examples for reproducing the numbers reported in the paper for various experiments are given below.

### Reproducing Numbers
Say you want to reproduce the numbers corresponding to **n = 25** for *Synthetic-Data* by the **PARWiS** Algorithm, please use the following command:
```shell
python PARWiS.py --n 25 --experiments 10 --iterations 100 --budget 25 --precomputed
```

### Separation vs Recovery experiments
For the algorithms **PARWiS** and **SELECT**, we performed experiments when the topper had score **x** and the rest has **100-x**. Say you want to reproduce the experiments for **SELECT**, you can use the following command:
```shell
python SELECT.py --n 50 --experiments 10 --iterations 100 --budget 3 --toppers 55 60 65 70 75 80
```

## Experiments on Real World Dataset
We have evaluated our algorithm on five real-world datasets.
- SUSHI Item Set A & B (*"sushi-A"* & *"sushi-B"*)
- Jester Joke Dataset 1 (*"jester"*)
- Netflix Prize Dataset (*"netflix"*)
- MovieLens 100k Dataset (*"movielens"*)

### Reproducing Numbers
Say you want to reproduce the numbers corresponding to **Jester** Dataset by the **PARWiS** Algorithm, please use the following command:
```shell
python PARWiS.py --experiments 10 --iterations 100 --budget 7 --dataset "jester"
```

## Datasets
The code for generating the BTL score vectors for all the datasets can be found in `/datasets/` folder. We have already generated and stored these numbers, the code can be run for verification purposes. The download links for the dataset have been mentioned in the code comments and references below. Please change the data-paths in the code wherever required as per your system before running. The corresponding score vectors will be printed on the screen as you run the file.

## References
- [choix](https://github.com/lucasmaystre/choix)
- ["Active Ranking from Pairwise Comparisons and When Parametric Assumptions Don't Help" Paper Supplement](https://github.com/reinhardh/supplement_active_ranking)
- [SUSHI Preference Data Sets](http://www.kamishima.net/sushi/)
- [Jester Joke Dataset 1](http://eigentaste.berkeley.edu/dataset/)
- [Netflix Prize](https://www.netflixprize.com/)
- [Netflix Prize data on kaggle](https://www.kaggle.com/netflix-inc/netflix-prize-data)
- [MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/)
