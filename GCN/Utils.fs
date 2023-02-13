module Utils
open System
open System.IO
open MathNet.Numerics.LinearAlgebra
open TorchSharp

let encode_onehot labels =
    let classes_dict = 
        labels
        |> Seq.distinct
        |> Seq.mapi (fun i c -> c,i)
        |> dict
    let labels_onehot =
        labels
        |> Seq.map (fun l -> 
            let array = Array.zeroCreate classes_dict.Count
            array.[classes_dict.[l]] <- 1L
            array)
    labels_onehot,classes_dict.Count

let maxIdx xs = xs |> Seq.mapi (fun i x-> i,x) |> Seq.maxBy snd |> fst |> int64

let normalize (m:Matrix<float32>) =
    let rowsum = m.RowSums()
    let r_inv = rowsum.PointwisePower(-1.0f)
    let r_inv = r_inv.Map(fun x-> if Single.IsInfinity x then 0.0f else x)
    let r_mat_inv = Matrix.Build.SparseOfDiagonalVector(r_inv)                  //Laplacian
    let mx = r_mat_inv.Multiply(m)
    mx

let sparse_mx_to_torch_sparse_tensor (m:Matrix<float32>) =
    let coo = m.EnumerateIndexed(Zeros.AllowSkip)
    let rows = coo |> Seq.map (fun struct(r,c,v) -> int64 r)   
    let cols = coo |> Seq.map (fun struct(r,c,v) -> int64 c)
    let vals = coo |> Seq.map (fun struct(r,c,v) -> v)
    let idxs = Seq.append rows cols |> Seq.toArray
    let idxT = idxs |> torch.tensor |> fun x -> x.view(2L, idxs.Length / 2 |> int64)
    let valsT = vals |> Seq.toArray |> torch.tensor
    let t = torch.sparse(idxT,valsT,[|int64 m.RowCount; int64 m.ColumnCount|])
    t

let accuracy(output:torch.Tensor, labels:torch.Tensor) = 
    let predsData = TorchSharp.Fun.Tensor.getData<float32>(output)
    let preds  = predsData |> Array.chunkBySize (int output.shape.[1]) |> Array.map maxIdx
    let lbls = TorchSharp.Fun.Tensor.getData<int64>(labels)
    let correct = Array.zip preds lbls |> Array.filter (fun (a,b) -> a = b) |> Array.length |> float
    let total = float lbls.Length 
    correct / total

let loadData (dataFolder:string) dataset =
    let dataset = defaultArg dataset "cora"
    let featuresFile = $"{dataFolder}/{dataset}.content"
    let edgesFile = $"{dataFolder}/{dataset}.cites"
    let yourself x = x

    let dataFeatures =
        featuresFile
        |> File.ReadLines
        |> Seq.map(fun x -> x.Split('\t'))
        |> Seq.map(fun xs -> 
            {|
                Id = xs.[0]
                Features = xs.[1 .. xs.Length-2] |> Array.map float32
                Label = xs.[xs.Length-1]
            |})

    let idx_map = dataFeatures |> Seq.mapi (fun i x-> x.Id,i) |> Map.ofSeq

    let edges_unordered =
        edgesFile
        |> File.ReadLines
        |> Seq.map (fun x->x.Split('\t'))
        |> Seq.map (fun xs -> xs.[0],xs.[1])
        |> Seq.toArray

    let edges = 
        edges_unordered 
        |> Array.map (fun (a,b) -> idx_map.[a],idx_map.[b])

    let ftrs =  Matrix.Build.SparseOfRowArrays(dataFeatures |> Seq.map (fun x-> x.Features) |> Seq.toArray)

    let graph = Matrix.Build.SparseFromCoordinateFormat
                    (
                        idx_map.Count, idx_map.Count, edges.Length,     //rows,cols,num vals
                        edges |> Array.map fst,                         //hot row idx
                        edges |> Array.map snd,                         //hot col idx
                        edges |> Array.map (fun _ -> 1.0f)              //values
                    )

    //symmetric graph 
    let graphSym = (graph + graph.Transpose()).Map(fun v -> if v > 0.f then 1.0f else 0.0f)

    let graph_n = Matrix.Build.SparseIdentity(graph.RowCount) + graphSym |> normalize
    let ftrs_n = normalize ftrs
    let labels,numClasses = dataFeatures |> Seq.map (fun x->x.Label) |> encode_onehot

    let idx_train = [|0L   .. 139L|]
    let idx_val   = [|200L .. 499L|]
    let idx_test  = [|500L .. 1499L|]

    let features = torch.tensor(ftrs_n.Enumerate() |> Seq.toArray).view(-1L, int64 ftrs_n.ColumnCount)
    let labels = torch.tensor(labels |> Seq.map maxIdx |> Seq.toArray ).view(-1L)
    let adj = sparse_mx_to_torch_sparse_tensor graph_n
    let idx_train = torch.tensor idx_train 
    let idx_val = torch.tensor idx_val     
    let idx_test = torch.tensor idx_test   
    
    adj, features, labels, idx_train, idx_val, idx_test
