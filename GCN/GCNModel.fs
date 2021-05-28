module GCNModel
open TorchSharp.Tensor
open TorchSharp.NN
open type TorchSharp.NN.Modules
open TorchSharp.Fun 
let inline (!>) (x:^a) : ^b = ((^a or ^b) : (static member op_Implicit : ^a -> ^b) x)

///Single graph convolutional layer
let gcnLayer in_features out_features hasBias (adj:TorchTensor) =
    let weight = Parameter(randName(),Float32Tensor.empty([|in_features; out_features|],requiresGrad=true))
    let bias = if hasBias then Parameter(randName(),Float32Tensor.empty([|out_features|],requiresGrad=true)) |> Some else None
    let parms = [| yield weight; if hasBias then yield bias.Value|]
    Init.kaiming_uniform(weight.Tensor) |> ignore
    if hasBias then Init.uniform(bias.Value.Tensor,0.,1.0) |> ignore

    F' [] parms (fun t ->
        use support = t.mm(weight.Tensor)
        let output = adj.mm(support)
        if hasBias then
           output.add(bias.Value.Tensor)
        else
            output)

///Create two layer GCN model with dropout
let create nfeat nhid nclass dropout adj =
    let gc1 = gcnLayer nfeat nhid true adj
    let gc2 = gcnLayer nhid nclass true adj        
    let drp = Dropout(dropout) |> M

    F [gc1;gc2;drp] (fun t ->
        use t = gc1.forward(t)
        use t = Functions.ReLU(t)
        use t = drp.forward(t)
        use t = gc2.forward(t)
        let t = Functions.LogSoftmax(t, dimension=1L)
        t)   
