# mlalgos
Machine Learning algorithms written from scratch

# Why?

While its true that a number of ML libraries exist, a lot of my personal projects revolve around creating my own implementations and using those, especially in julia because of compile times. This package is going to be written in julia using base libraries.

# How will it look like

Currently, training a lasso regression is done like this.

```{julia}
model = clf() |> sse |> l1 |> train(X, y)
```

We can also train primal svms in the same compositional manner

```{julia}
model = clf() |> hinge |> l2 |> C(10.0) |> train(X, y)
```

My goal is to implement most linear models and dual kernel models with some implementation in this manner.

