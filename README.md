## Usage

Install kerpy (a package adapted from https://github.com/oxcsml/kerpy)

```(bash)
cd kerpy
python setup.py develop
cd ..
```

Predict the causal structure

```(bash)
cd algorithm
python discrimination.py
cd ..
```

Estimate the causal effect

```(bash)
cd algorithm
python estimation.py
cd ..
```

## Requirements

pytorch 2.1.2

numpy 1.24.3

scipy 1.10.1

networkx 3.1

matplotlib 3.7.1
