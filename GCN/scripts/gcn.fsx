#load "packages.fsx"
#load "../Utils.fs"
open TorchSharp.Fun

let datafolder =  __SOURCE_DIRECTORY__ + @"../../../data/cora"
let  adj, features, labels, idx_train, idx_val, idx_test = Utils.loadData datafolder None

let v1 = adj.[0L,50L] |> float

let idx = adj.SparseIndices |> Tensor.getData<int64>
let rc = idx |> Array.chunkBySize (idx.Length/2)
let vals = adj.SparseValues |> Tensor.getData<float32>

let i = 500
let r,c = rc.[0].[i],rc.[1].[i]
let vx = adj.[r,c] |> float

let df = features |> Tensor.getData<float32> |> Array.chunkBySize (int features.shape.[1])

let f1 = features.[1L,12L] |> float




