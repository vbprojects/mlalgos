using LinearAlgebra, Statistics, Random

module sgd

    export C, η, cross_entropy, hinge, surrogate, sse, sae, l2, l1, elasticnet, l0, score, clf, sg, sgc, score, train, mse, η, λ, C, maxiter, train, aug, to_0_1, to_neg_pos

    sigmoid(x) = 1/(1+exp(-x))
    dsigmoid(x) = sigmoid(x)*(1-sigmoid(x))

    ce_obj(w, x, y) = -1 * (y*log(sigmoid(w' * x)) + (1-y)*log(1-sigmoid(w'*x)))
    ce_dobj(w, x, y) = (y / sigmoid(w' * x) + (1 - y) / (1 - sigmoid(w' * x))) * sigmoid(w' * x) * (1 - sigmoid(w' * x)) * x
    ce_prob(w, x) = sigmoid(w' * x)
    ce_pred(w, x) = sigmoid(w' * x) > 0.5 |> Float64

    h_obj(w, x, y) = max(0, 1 - y * w' * x)
    h_dobj(w, x, y) = -y * x * (y * w' * x < 1)
    h_prob(w, x) = sign(w' * x) |> Float64
    h_pred(w, x) = sign(w' * x) |> Float64

    sur_obj(w, x, y) = tanh(w' * x)^3 * y / 3 - tanh(w' * x)^2 * y^2 + tanh(w' * x) * y^3 / 3
    sur_dobj(w, x, y) = -y * (tanh(w' * x) - y)^2 * x
    sur_prob(w, x) = tanh(w' * x)
    sur_pred(w, x) = tanh(w' * x) |> sign |> Float64

    sse_obj(w, x, y) = ((w' * x) - y)^2
    sse_dobj(w, x, y) = 2 * ((w' * x) - y) * x
    sse_prob(w, x) = w' * x
    sse_pred = sse_prob

    sae_obj(w, x, y) = abs((w' * x) - y)
    sae_dobj(w, x, y) = sign((w' * x) - y) * x
    sae_prob(w, x) = w' * x
    sae_pred = sae_prob

    l2reg(w, λ) = λ * w' * w
    l2dreg(w, λ) = 2 * λ * w
    l1reg(w, λ) = λ * norm(w, 1)
    l1dreg(w, λ) = λ * sign.(w)
    elasticnetreg(w, λ) = λ * w' * w + λ * norm(w, 1)
    elasticnetdreg(w, λ) = 2 * λ * w + λ * sign.(w)
    l0reg(w, λ) = 0
    l0dreg(w, λ) = zeros(length(w))

    

    function score(X, Y)
        function score_h(model)
            Yp = [model.pred(model.w, X[i, :]) for i in 1:size(X, 1)]
            return mse_metric(Y, Yp)
        end
    end


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
        model.pred = ce_pred
        return model
    end

    function hinge(model)
        model.obj = h_obj
        model.dobj = h_dobj
        model.prob = h_prob
        model.pred = h_pred
        return model
    end

    function surrogate(model)
        model.obj = sur_obj
        model.dobj = sur_dobj
        model.prob = sur_prob
        model.pred = sur_pred
        return model
    end

    function sse(model)
        model.obj = sse_obj
        model.dobj = sse_dobj
        model.prob = sse_prob
        model.pred = sse_pred
        return model
    end

    function sae(model)
        model.obj = sae_obj
        model.dobj = sae_dobj
        model.prob = sae_prob
        model.pred = sae_pred
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

    function mse(m)
        m.metric = mse_metric
        return m
    end

    function mse_metric(Y, Yp)
        return sum(abs2, (Y .- Yp)) / length(Y)
    end


    mutable struct clf
        obj
        dobj
        prob
        pred
        reg
        dreg
        λ
        C
        maxiter
        η
        w
        metric
    end
    mutable struct sg
        v
    end
    import Base

    function Base.:*(a::sg, b::sg)
        sg([f ∘ g for f in a.v for g in b.v])
    end

    function Base.:*(a::sg, b::Function)
        sg([f ∘ b for f in a.v])
    end

    function Base.:*(a::Function, b::sg)
        sg([a ∘ f for f in b.v])
    end

    mutable struct sgc
        cs
    end

    function Base.:*(a::sg, b::clf)
        sgc([f(b) for f in a.v])
    end

    function Base.:*(a::clf, b::sg)
        sgc([f(a) for f in b.v])
    end

    function train(X, y)
        function train_h(model)
            if isnan(model.w)
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

    clf() = clf(sse_obj, sse_dobj, sse_prob, sse_pred, l0reg, l0dreg, 0.0, 1.0, 1000, 0.01, NaN, mse_metric)

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
end

using .sgd


D = 1:.1:10
Y = (2 .* D) .+ 1
D = sgd.aug(D)

model = clf() |> sae |> l1 |> train(D, Y) |> score(D, Y)
print((sg([sae, sse]) * sg([l1, l2]) * sg(λ.([1, 5, 10])) * sg(η.([0.01, 0.1, 1.0])) * clf()).cs)


(sg([sae, sse]) * sg([l1, l2]) * sg(λ.([1, 5, 10])) * sg(η.([0.01, 0.1, 1.0]))).v
