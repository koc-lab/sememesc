# Sememe-Based Semantic Communications
T. Ozates, U. Kargı and A. Koç, "Sememe-Based Semantic Communications," in IEEE Communications Letters, vol. 28, no. 10, pp. 2308-2312, Oct. 2024
Paper link: https://ieeexplore.ieee.org/document/10648841

To reproduce the results, follow these steps:

1- Download and extract EuroParl dataset from: https://drive.google.com/file/d/1l-Q5GiH8oc27Ifse9JYOFUEEKrIJ7Z3-/view?usp=drive_link

2- Run the following commands to pre-process and generate data for symbol decoder training and simulations:

```
python dataset_preprocess.py
python semantic_symbol_extractor.py
python noisy_dataset_generation.py
```
3- Train symbol decoder by running `train_symbol_decoder.ipynb`

4- Run physical channel simulations by following command:

```
python run_simulation.py
```

5- Test and evaluate results by running `evaluate.ipynb`
