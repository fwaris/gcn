module Program
open System
open Argu
open System.Diagnostics

let DATA_FOLDER    =  __SOURCE_DIRECTORY__ + @"/../data/cora"

module Defs =
    let no_cuda =  false
    let fastmode = false
    let seed = 42
    let epochs = 200
    let lr = 0.01
    let weight_decay = 5e-4
    let hidden = 16
    let dropout = 0.5

    type Args =
        //| [<Mandatory>] Datafolder of string
        | Datafolder of string
        | No_CUDA of bool
        | Fastmode of bool
        | Seed of int
        | Epochs of int
        | Lr of float
        | Weight_Decay of float
        | Hidden of int
        | Dropout of float
        with 
        interface IArgParserTemplate with
            member s.Usage = 
                match s with
                | Datafolder _  -> "folder containing the data files"
                | No_CUDA x     -> $"Disables CUDA training [{no_cuda}]"
                | Fastmode x    -> $"Validate during training pass [{fastmode}]"
                | Seed x        -> $"Random seed [{seed}]"
                | Epochs x      -> $"Number of epochs to train [{epochs}])"
                | Lr x          -> $"Initial learning rate [{lr}]"
                | Weight_Decay x-> $"Weight decay (L2 loss on parameters) [{weight_decay}]"
                | Hidden x      -> $"Number of hidden units [{hidden}]"
                | Dropout x     -> $"Droput rate (1-keepProb) [{dropout}]"

    let parse args =
        let parser = ArgumentParser.Create<Args>(programName = "gcn.exe")
        let args = parser.Parse(args)
        let datafolder = args.GetResult (Args.Datafolder, defaultValue = DATA_FOLDER)
        let no_cuda = args.GetResult (Args.No_CUDA, defaultValue=no_cuda)
        let fastmode = args.GetResult (Args.Fastmode, defaultValue=fastmode)
        let epochs = args.GetResult (Args.Epochs, defaultValue=epochs)
        let dropout = args.GetResult (Args.Dropout, defaultValue=dropout)
        let lr = args.GetResult (Args.Lr, defaultValue=lr)
        let hidden = args.GetResult (Args.Hidden, defaultValue=hidden)
        let seed = args.GetResult (Args.Seed, defaultValue=seed)
        let weight_decay = args.GetResult (Args.Weight_Decay, defaultValue=weight_decay)
        datafolder,no_cuda,fastmode,epochs,dropout,lr,hidden,seed,weight_decay
        
[<EntryPoint>]
let main args =
    let runParms = Defs.parse args
    Train.run runParms
    System.Console.ReadLine() |> ignore
    0
