module TorchSharp.Fun
open System
open TorchSharp
open TorchSharp.Tensor
open TorchSharp.NN

type IModel =
    abstract forward : TorchTensor -> TorchTensor
    abstract Module : Module

let randName() = Guid.NewGuid().ToString()

let registerNamed (parent:IModel) (name:string,child:IModel) =
    if child.Module <> null then
        parent.Module.RegisterModule(name,child.Module)

let register (parent:IModel) (child:IModel) = 
    registerNamed parent (randName(),child)

[<AbstractClass>]
type AbstractModel() =
    abstract member forward : TorchTensor->TorchTensor
    abstract member Module : Module
    interface IModel with
        member this.forward(x) = this.forward(x)
        member this.Module = this.Module
    //operators TBD
    static member (+) (a:IModel,b:TorchTensor) = {new IModel with
                                                        member _.forward(x) = use t1 = a.forward(x) in t1 + b
                                                        member _.Module = a.Module
                                                   }
                                                   |> Model   

and CustomModel(name,parameters,fwd:TorchTensor[]->TorchTensor->TorchTensor) =
    inherit CustomModule(name,parameters)
    override this.forward(t) = fwd (this.parameters()) t
    interface IModel with
        member this.forward(t) = this.forward(t)
        member this.Module = this :> _

and Model (a:IModel) =
    inherit AbstractModel()
    override _.forward(x) = a.forward(x)
    override _.Module = a.Module
    member this.dispose() = if this.Module <> null then this.Module.Dispose()
    static member create(parameters:Parameter[],fwd:TorchTensor[]->TorchTensor->TorchTensor,?name) =
        let name = defaultArg name (randName())
        new CustomModel(name,parameters,fwd)
    static member nop = {new IModel with
                            member _.forward t = t
                            member _.Module = null
                        }
                        |> Model
    
///Convert a TorchSharp module to a Model 
let inline M< ^T when  ^T : (member forward:TorchTensor->TorchTensor)> (mdl:^T) =
    match box mdl with
    | :? Model as m -> m
    | :? IModel as f ->  Model f
    | _ ->
        {new IModel with
            member _.forward (t) = (^T : (member forward:TorchTensor->TorchTensor)(mdl,t))
            member _.Module = 
                match (box mdl) with 
                | :? Module as m -> m 
                | :? IModel as f -> f.Module
                | _ -> failwith $"module or model expected but got {mdl.GetType()}"
        }
        |> Model


let inline fwd< ^T when  ^T : (member forward:TorchTensor->TorchTensor)> 
                (parent:^T) 
                (forwardFunc : TorchTensor -> IModel -> TorchTensor) =
    let p = M parent
    {new IModel with 
        member _.forward(t) = forwardFunc t p
        member _.Module = p.Module
    }
    |> Model  

/// <summary>
/// Returns a Model whose forward function is supplied by 'forwardFunc'.
/// <para>Parent and child can be TorchSharp modules; these will be converted to Models.</para>
/// <para>Child will be registered as submodule of parent.</para>
/// <para>forwardFunc signature: input tensor -> parent model -> child model -> output tensor</para>
/// </summary>
/// <remarks>forwardFunc input tensor is the input into the parent model </remarks>
let inline fwd2< ^A,^B 
                when  ^A : (member forward:TorchTensor->TorchTensor)
                and   ^B : (member forward:TorchTensor->TorchTensor)
                > 
                (parent:^A) 
                (child:^B) 
                (forwardFunc : TorchTensor -> IModel -> IModel -> TorchTensor) =
    let p = M parent
    let c = M child
    register p c    
    {new IModel with 
        member _.forward(t) = forwardFunc t p c
        member _.Module = p.Module
    }
    |> Model    

let inline fwd3< ^A,^B,^C  
                when  ^A : (member forward:TorchTensor->TorchTensor)
                and   ^B : (member forward:TorchTensor->TorchTensor)
                and   ^C : (member forward:TorchTensor->TorchTensor)
                > 
                (parent:^A) 
                (child1:^B) 
                (child2:^C)
                (forwardFunc : TorchTensor -> IModel -> IModel -> IModel -> TorchTensor) =
    let p = M parent
    let c1 = M child1
    let c2 = M child2
    register p c1
    register p c2
    {new IModel with 
        member _.forward(t) = forwardFunc t p c1 c2
        member _.Module = p.Module
    }
    |> Model   

let inline fwd4< ^A,^B,^C,^D  
                when  ^A : (member forward:TorchTensor->TorchTensor)
                and   ^B : (member forward:TorchTensor->TorchTensor)
                and   ^C : (member forward:TorchTensor->TorchTensor)
                and   ^D : (member forward:TorchTensor->TorchTensor)
                >  
                (parent:^A) 
                (child1:^B) 
                (child2:^C)
                (child3:^D)                
                (forwardFunc : TorchTensor -> IModel -> IModel -> IModel -> IModel -> TorchTensor) =
    let p = M parent
    let c1 = M child1
    let c2 = M child2
    let c3 = M child3 
    register p c1
    register p c2
    register p c3
    {new IModel with 
        member _.forward(t) = forwardFunc t p c1 c2 c3
        member _.Module = p.Module
    }
    |> Model 

let compose (m1:IModel) (name,m2:IModel) = 
    let name = defaultArg name (randName())
    registerNamed m1 (name,m2)    
    {new IModel with
        member _.forward (t) = 
            use t1 = m1.forward t 
            let t2 = m2.forward t1 
            t2
        member _.Module = m1.Module
    }
    |> Model

let inline (->>) m1 m2 = compose (M m1) (None,M m2)
let inline (=>>) m1 (n,m2) = compose (M m1) (Some n, M m2)

module Tensor = 
    //Note: ensure 't matches tensor datatype otherwise ToArray might crash the app (i.e. exception cannot be caught)
    let private _getData<'t> (t:TorchTensor) =
        let s = t.Data<'t>()
        s.ToArray()

    let getData<'t> (t:TorchTensor) =
        if t.device_type <> DeviceType.CPU then 
            //use t1 = t.clone()
            use t2 = t.cpu()
            _getData<'t> t2
        else 
            _getData<'t> t
  
    let setData<'t> (t:TorchTensor) (data:'t[]) =
        if t.device_type = DeviceType.CPU |> not then failwith "tensor has to be on cpu for setData"
        let s = t.Data<'t>()
        for i in 0 .. data.Length-1 do
            s.[i] <- data.[i]

module Model =
    open MBrace.FsPickler
    open Tensor

    let saveParms<'t> file (model:Model) =
        let values = model.Module.parameters() |> Array.map getData<'t>
        let ser = FsPickler.CreateBinarySerializer()
        use str = IO.File.Create(file:string)
        ser.Serialize(str,values)

    let loadParms<'t> file (model:Model) =
        let ser = FsPickler.CreateBinarySerializer()
        use str = IO.File.OpenRead(file:string)
        let values = ser.Deserialize<'t[][]>(str)
        Array.zip values (model.Module.parameters())
        |> Array.iter (fun (v,p) -> setData<'t> p v)
