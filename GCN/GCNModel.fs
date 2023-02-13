module GCNModel
open TorchSharp
open TorchSharp.Fun 
let inline (!>) (x:^a) : ^b = ((^a or ^b) : (static member op_Implicit : ^a -> ^b) x)

///Single graph convolutional layer
let gcnLayer in_features out_features hasBias (adj:torch.Tensor) =    
    let bias_ = if hasBias then torch.nn.Parameter(torch.zeros([|out_features|])) else null 
    let bias = ref bias_ //parameters have to be 'ref' as the underlying tensor may move to another device
    let weight = ref (torch.nn.Parameter(torch.zeros([|in_features; out_features|])))
    let parms :obj list = [ yield weight; if hasBias then yield bias ]
    torch.nn.init.kaiming_uniform_(weight.Value) |> ignore
    if hasBias then  torch.nn.init.uniform_(bias.Value,0.,1.0) |> ignore
    
    F [] parms (fun t ->
        use support = t.mm(weight.Value)
        let output = adj.mm(support)
        if hasBias then
           output.add(bias.Value)
        else
            output)

///Create two layer GCN model with dropout
let create nfeat nhid nclass dropout adj =
    let gc1 = gcnLayer nfeat nhid true adj 
    let gc2 = gcnLayer nhid nclass true adj      
    let drp = torch.nn.Dropout(dropout)  

    F [] [gc1;gc2;drp] (fun t ->
        use t = gc1.forward(t)
        use t = torch.nn.functional.relu(t)
        use t = drp.forward(t)
        use t = gc2.forward(t)
        let t = torch.nn.functional.log_softmax(t, dim=1L)
        t)   
