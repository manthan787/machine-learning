## PA 3

Report is in `manthan_thakar_PA3.pdf`.

### EM

The code for `Expectation Maximization Algorithm for GMM` implementation can be found in `models.py` in class `GMM`.

To build a model using `GMM` you can use following code:

```
# create a GMM object
gmm = GMM(k) -> k is number of mixutres

# Fit the model for input X
gmm.fit(X)
```

There are other parameters (optional) to the GMM class that you can look at by looking at the GMM class definition.

`models.py` also contains `KMeans` class which is responsible for fitting a `KMeans` clustering moddel on input X. The API for this class is similar to `GMM` class.


### Model selection

The model selection code lives inside `model_selection.py` file.