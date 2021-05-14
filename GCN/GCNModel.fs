module GCNModel
open TorchSharp.Tensor
open TorchSharp.NN
open type TorchSharp.NN.Modules
open TorchSharp.Fun 
let inline (!>) (x:^a) : ^b = ((^a or ^b) : (static member op_Implicit : ^a -> ^b) x)

let gcnLayer in_features out_features hasBias (adj:TorchTensor) =
    let weight = Parameter(randName(),Float32Tensor.empty([|in_features; out_features|]))
    let bias = if hasBias then Parameter(randName(),Float32Tensor.empty([|out_features|])) |> Some else None
    let parms = [| yield weight; if hasBias then yield bias.Value|]
    Init.kaiming_uniform(weight.Tensor) |> ignore

    Model.create(parms,fun wts t -> 
        let support = t.mm(wts.[0])
        let output = adj.mm(support)
        if hasBias then
           output.add(wts.[1])
        else
            output)

let create nfeat nhid nclass dropout adj =
    let gc1 = gcnLayer nfeat nhid true adj
    let gc2 = gcnLayer nhid nclass true adj        
    let relu = ReLU()
    let logm = LogSoftmax(1L)
    let drp = if dropout then Dropout() |> M else Model.nop
    
    fwd3 gc1 gc2 drp (fun t g1 g2 drp -> 
        use t = gc1.forward(t)
        use t = relu.forward(t)
        use t = drp.forward(t)
        use t = gc2.forward(t)
        let t = logm.forward(t)
        t)
