using LinearAlgebra

sigmoid(x) = 1/(1+exp(-x))
dsigmoid(x) = sigmoid(x)*(1-sigmoid(x))


ce_obj(w, x, y) = -1 * (y*log(sigmoid(w'*x)) + (1-y)*log(1-sigmoid(w'*x)))
ce_dobj(w, x, y) = (y / sigmoid(w' * x) + (1 - y) / (1 - sigmoid(w' * x))) * sigmoid(w' * x) * (1 - sigmoid(w' * x)) * x
ce_prob(w, x) = sigmoid(w' * x)

h_obj(w, x, y) = max(0, 1 - y * w' * x)
h_dobj(w, x, y) = -y * x * (y * w' * x < 1)
h_prob(w, x) = sign(w' * x)

sur_obj(w, x, y) = tanh(w' * x)^3 * y / 3 - tanh(w' * x)^2 * y^2 + tanh(w' * x) * y^3 / 3
sur_dobj(w, x, y) = -y * (tanh(w' * x) - y)^2 * x
sur_prob(w, x) = tanh(w' * x)

sse_obj(w, x, y) = (w' * x - y)^2
sse_dobj(w, x, y) = 2 * (w' * x - y) * x
sse_prob(w, x) = w' * x

sae_obj(w, x, y) = abs(w' * x - y)
sae_dobj(w, x, y) = sign(w' * x - y) * x
sae_prob(w, x) = w' * x


l2reg(w, λ) = λ * w' * w
l2dreg(w, λ) = 2 * λ * w
l1reg(w, λ) = λ * norm(w, 1)
l1dreg(w, λ) = λ * sign.(w)
elasticnetreg(w, λ) = λ * w' * w + λ * norm(w, 1)
elasticnetdreg(w, λ) = 2 * λ * w + λ * sign.(w)
l0reg(w, λ) = 0
l0dreg(w, λ) = zeros(length(w))

function C(c)
    function C_h(model)
        model.C = c
        return model
    end
    return C_h
end

function η(r)
    function η_h(model)
        model.η = r
        return model
    end
    return η_h
end


function cross_entropy(model) 
    model.obj = ce_obj
    model.dobj = ce_dobj
    model.prob = ce_prob
    return model
end

function hinge(model)
    model.obj = h_obj
    model.dobj = h_dobj
    model.prob = h_prob
    return model
end

function surrogate(model)
    model.obj = sur_obj
    model.dobj = sur_dobj
    model.prob = sur_prob
    return model
end

function sse(model)
    model.obj = sse_obj
    model.dobj = sse_dobj
    model.prob = sse_prob
    return model
end

function sae(model)
    model.obj = sae_obj
    model.dobj = sae_dobj
    model.prob = sae_prob
    return model
end

function l2(model)
    model.reg = l2reg
    model.dreg = l2dreg
    return model
end

function l1(model)
    model.reg = l1reg
    model.dreg = l1dreg
    return model
end

function elasticnet(model)
    model.reg = elasticnetreg
    model.dreg = elasticnetdreg
    return model
end

function l0(model)
    model.reg = l0reg
    model.dreg = l0dreg
    return model
end

function η(r)
    function η_h(model)
        model.η = r
        return model
    end
    return η_h
end

function C(c)
    function C_h(model)
        model.C = c
        return model
    end
    return C_h
end

function λ(l)
    function λ_h(model)
        model.λ = l
        return model
    end
    return λ_h
end

function maxiter(m)
    function maxiter_h(model)
        model.maxiter = m
        return model
    end
    return maxiter_h
end

module test
    mutable struct clf
        obj
        dobj
        prob
        reg
        dreg
        λ
        C
        maxiter
        η
        w
    end
    mutable struct sg
        v
    end
end

test.clf() = test.clf(sse_obj, sse_dobj, sse_prob, l0reg, l0dreg, 0.0, 1.0, 1000, 0.01, NaN)
sg(v) = test.sg(v)

cl = test.clf()

cl |> l2

function train(X, y)
    function train_h(model)
        if model.w == NaN
            model.w = zeros(size(X, 2))
        end
        w = model.w
        for _ in 1:model.maxiter
            i = rand(1:size(X, 1))
            w -= model.η * (model.dobj(w, X[i, :], y[i]) + model.C * model.dreg(w, model.λ))
        end
        model.w = w
        return model
    end
    return train_h
end


function aug(model)
    model.X = [ones(size(model.X, 1)) model.X]
    return model
end

function pred(X)
    function pred_h(model)
        yp = zeros(size(X, 1))
        for i in 1:size(X, 1)
            yp[i] = model.prob(model.w, X[i, :])
        end
        return yp
    end
    return pred_h
end

function aug(X)
    return [ones(size(X, 1)) X]
end

function to_0_1(Y)
    return (Y .+ 1) ./ 2 |> round
end

function to_neg_pos(Y)
    return 2 .* Y .- 1 |> round
end


clf = test.clf

X = 1:.1:10
y = 2 .* X .+ 1
aX = aug(X)

ηs = η.(10 .^ (-1.0 .* (1:3)))
Cs = C.(1:5:20)

function flatten(X)
    return [y for x in X for y in x]
end

ms = clf() |> η(.01)

ms = [clf()]
ms = [f.(ms) for f in ηs]
ms = flatten(ms)
ms = [f.(ms) for f in Cs]
ms = flatten(ms)

[f.([clf()]) for f in η.(10 .^ (-1.0 .* (1:3)))]


mdl = clf() |> l2 |> C(1) |> η(0.01)
