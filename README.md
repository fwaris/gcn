# Graph Convolutional Networks in TorchSharp.Fun

TorchSharp.Fun is thin functional wrapper in F# over TorchSharp (a .Net binding of PyTorch).

## TorchSharp.Fun Example

Below is a simple sequential model. It is a composition over standard TorchSharp 'modules'. The composition is performed with the '->>' operator.

```F#
let model = 
    Linear(10L,5L) 
    ->> Dropout(0.5)
    ->> Linear(5L,1L) 
    ->> RelU()
```

## GCN Model

The Graph Convolutional Network (GCN) model presented in this repo is based on the work of Thomas Kipf, [Graph Convolutional Networks](http://tkipf.github.io/graph-convolutional-networks/) (2016).

It is a port of the [Pytorch GCN model](http://github.com/tkipf/pygcn).

### For more context see this [LinkedIn article](https://www.linkedin.com/pulse/graph-convolutional-network-model-strongly-typed-faisal-waris-phd/?trackingId=i4c8eRsEUdaaaXP5KkFLdw%3D%3D)

## TorchSharp.Fun

The code for TorchSharp.Fun is included in the repo. At this stage it is expected to undergo considerable churn and therefore is not released as an independent package.

## Training the model

The data for the model included is however two changes to source are required to train the model. Both are in Program.fs file. These are:

- Path to libtorch native library - [download link](https://pytorch.org/)
- Path to the data folder

It is recommend to use Visual Studio code with F# / Ionide plug-in - just start the project after making the above changes.

## Why TorchSharp.Fun?

A function-compositional approach to deep learning models arose when I could not easily create a deep ResNet model with 'standard' TorchSharp.

An alternative F# library was also tried. The library supports an elegant API; it was easy to create a deep ResNet model. Unfortunately at its current stage of development, the training performance for deep models is not on par with that of basic TorchSharp.

TorchSharp.Fun is a very thin wrapper over TorchSharp does not suffer any noticable performance hits when compared with TorchSharp (or PyTorch for that matter).

Below is an example of a 30 layer ResNet regression model:

```F#
module Resnet =
    let FTR_DIM = 310L
    let RESNET_DIM = 10L
    let RESNET_DEPTH = 30
    let act() = SELU() //SiLU()// LeakyReLU(0.05) // GELU() // GELU()
    //residual block
    let resnetCell (input: Model) =
        let cell =  
            act()
            ->> Linear(RESNET_DIM, RESNET_DIM) //weight layer 1  
            ->> Dropout(0.1)
            ->> act()
            ->> Linear(RESNET_DIM, RESNET_DIM)                
        //skip connection
        let join =
            F [input; cell] (fun ``input tensor`` -> 
                    use t1 = input.forward ``input tensor``
                    use t2 = cell.forward t1
                    t1 + t2)
        join ->> act()
    //model
    let model =
        let emb = Linear(FTR_DIM, RESNET_DIM, hasBias=false) |> M
        let rsLayers =
            (emb, [ 1 .. RESNET_DEPTH ])
            ||> List.fold (fun emb _ -> resnetCell emb) //stack blocks
        rsLayers
        ->> Linear(RESNET_DIM,10L) 
        ->> Linear(10L, 1L)        
```
