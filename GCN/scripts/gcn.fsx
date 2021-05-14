#load "packages.fsx"
open System
open System.IO
open MathNet.Numerics.LinearAlgebra

let dataFolder = @"C:\Users\fwaris\Downloads\pygcn-master\data\cora"
let contentFile = $"{dataFolder}/cora.content"
let citesFile = $"{dataFolder}/cora.cites"
let yourself x = x

let dataCntnt =
    contentFile
    |> File.ReadLines
    |> Seq.map(fun x -> x.Split('\t'))
    |> Seq.map(fun xs -> 
        {|
            Id = xs.[0]
            Features = xs.[1 .. xs.Length-2] |> Array.map float32
            Label = xs.[xs.Length-1]
        |})

let dataCites =
    citesFile
    |> File.ReadLines
    |> Seq.map (fun x->x.Split('\t'))
    |> Seq.map (fun xs -> xs.[0],xs.[1])
    |> Seq.toArray

let citationIdx = dataCites |> Seq.collect (fun (a,b)->[a;b]) |> Seq.distinct |> Seq.mapi (fun i x->x,i) |> dict

let ftrs = Matrix.Build.DenseOfRows(dataCntnt |> Seq.map (fun x->Array.toSeq x.Features))

let graph = Matrix.Build.SparseFromCoordinateFormat
                (
                    dataCites.Length, dataCites.Length, dataCites.Length,
                    dataCites |> Array.map (fun x -> citationIdx.[fst x]),
                    dataCites |> Array.map (fun x -> citationIdx.[snd x]),
                    dataCites |> Array.map (fun _ -> 1.0f)
                )

let normalize (m:Matrix<float32>) =
    let rowsum = m.RowSums()
    let r_inv = rowsum.PointwisePower(-1.0f)
    let r_inv = r_inv.Map(fun x-> if Single.IsInfinity x then 0.0f else x)
    let r_mat_inv = Matrix.Build.SparseOfDiagonalVector(r_inv)
    let mx = r_mat_inv.Multiply(m)
    mx
      
let graph_n = Matrix.Build.SparseIdentity(graph.RowCount) + graph |> normalize
let ftrs_n = normalize ftrs

open TorchSharp.Tensor
let sparse_mx_to_torch_sparse_tensor (m:Matrix<float32>) =
    let coo = m.EnumerateIndexed(Zeros.AllowSkip)
    let rows = coo |> Seq.map (fun (r,c,v) -> int64 r)
    let cols = coo |> Seq.map (fun (r,c,v) -> int64 c)
    let idxs = Seq.append rows cols |> Seq.toArray
    let idx1 = idxs |> Int64Tensor.from |> fun x -> x.view(2L,-1L)
    let vals = coo |> Seq.map(fun (r,c,v) -> v) |> Seq.toArray |> Float32Tensor.from     
    Float32Tensor.sparse(idx1,vals,[|int64 m.RowCount; int64 m.ColumnCount|])

let adj = sparse_mx_to_torch_sparse_tensor(graph_n)

module GCNModel = 
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



