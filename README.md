# mlalgos
Composable Machine Learning Models, written from scratch

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

# What it looks like right now

If you chose not to use the pipe syntax, setting up a lasso regression would look like this.

```{julia}
model = train(X_train, y_train)(mse(l1(sse(clf()))))
mean_squared_acc = score(X_test, y_test)(model)
```

Written with the pipe syntax it looks less like lisp

```{julia}
model = clf() |> sse() |> l1 |> mse |> train(X_train, y_train)
mean_squared_acc = model |> score(X_test, y_test)
```

You can create a model, apply methods that change the internal structure of the model, train and then score or predict. Methods like sse and l1 return models, score and predict return outputs. Every model has an objective, regularization, metric, learning rate, maxiter, cost.

An advantage of this compositional structure is in the creation of search grids. For example,

```
(sg([sae, sse]) * sg([l1, l2]) * sg(λ.([1, 5, 10])) * sg(η.([0.01, 0.1, 1.0]))).v
```
Outputs a structure sg (search grid) with a vector of composition functions that can be applied to the model

```{julia}
36-element Vector{ComposedFunction{O, Main.sgd.var"#η_h#6"{Float64}} where O}:
 Main.sgd.sae ∘ Main.sgd.l1 ∘ Main.sgd.var"#λ_h#8"{Int64}(1) ∘ Main.sgd.var"#η_h#6"{Float64}(0.01)
 Main.sgd.sae ∘ Main.sgd.l1 ∘ Main.sgd.var"#λ_h#8"{Int64}(1) ∘ Main.sgd.var"#η_h#6"{Float64}(0.1)
 Main.sgd.sae ∘ Main.sgd.l1 ∘ Main.sgd.var"#λ_h#8"{Int64}(1) ∘ Main.sgd.var"#η_h#6"{Float64}(1.0)
 ⋮
 Main.sgd.sse ∘ Main.sgd.l2 ∘ Main.sgd.var"#λ_h#8"{Int64}(10) ∘ Main.sgd.var"#η_h#6"{Float64}(0.1)
 Main.sgd.sse ∘ Main.sgd.l2 ∘ Main.sgd.var"#λ_h#8"{Int64}(10) ∘ Main.sgd.var"#η_h#6"{Float64}(1.0)
```

Multiply this search grid by a model and each function gets applied to the model

```{julia}
sg([sae, sse]) * sg([l1, l2]) * sg(λ.([1, 5, 10])) * sg(η.([0.01, 0.1, 1.0])) * clf()
```

```{julia}
clf[clf(Main.sgd.sse_obj, Main.sgd.sse_dobj, Main.sgd.sse_prob, Main.sgd.sse_prob, Main.sgd.l2reg, Main.sgd.l2dreg, 10, 1.0, 1000, 1.0, NaN, Main.sgd.mse_metric), clf(Main.sgd.sse_obj, Main.sgd.sse_dobj, Main.sgd.sse_prob, Main.sgd.sse_prob, Main.sgd.l2reg, Main.sgd.l2dreg, 10, 1.0, 1000, 1.0, NaN, Main.sgd.mse_metric), clf(Main.sgd.sse_obj, Main.sgd.sse_dobj, Main.sgd.sse_prob, Main.sgd.sse_prob, Main.sgd.l2reg, Main.sgd.l2dreg, 10, 1.0, 1000, 1.0, NaN, Main.sgd.mse_metric), clf...
```

Which can be randomly selected from to train/validate, test, or modify. While not less verbose than sklearn I think this approach is a lot simpler and scalable to more complex model types.

## Objective functions

hinge, cross_entropy, sse, sae, surrogate

## Regularization

l0, l1, l2, elasticnet

## Metrics

mae, mse

For right now I plan on adding more and expanding from the initial SGD to minibatch options, alternate learning rate optimizers (ADAM), different learning algorithms (Newton Ralphson, Pegasos), implement kernel optimization, and limited feature engineering.