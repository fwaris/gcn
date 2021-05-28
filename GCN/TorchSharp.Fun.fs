module rec TorchSharp.Fun
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

let inline private registerNamedChildren names childModules parent =      
    Seq.zip (Seq.tail names) childModules     
    |> Seq.iter (fun (n,c) -> registerNamed parent (n,M c))

let inline registerChildren< ^A,^B                
                when  ^A : (member forward:TorchTensor->TorchTensor)
                and   ^B : (member forward:TorchTensor->TorchTensor)
                >  
                (children:^B seq) (parent:^A) =
    let parent = M parent
    let childModules = children |> Seq.map M
    childModules |> Seq.iter (fun c -> register parent c)

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

type FuncModel(name,parameters,fwd:TorchTensor->TorchTensor) =
    inherit CustomModule(name,parameters)
    override this.forward(t) = fwd t
    member this.Module : Module = this :> _
    interface IModel with
        member this.forward(t) = this.forward(t)
        member this.Module = this :> _    

type Model (a:IModel) =
    inherit AbstractModel()
    override _.forward(x) = a.forward(x)
    override _.Module = a.Module
    member this.dispose() = if this.Module <> null then this.Module.Dispose()
    static member nop = {new IModel with
                            member _.forward t = t
                            member _.Module = null
                        }
                        |> Model
    
///Create a model (module) from the given function and register the childModules as children
let F (childModules:IModel seq) (fwd:TorchTensor -> TorchTensor) =
    let p = new FuncModel(randName(), [||],fwd) 
    registerChildren childModules p
    p

///Create a model (module) from the given function. Register the childModules as children and add the parameters to the model
let F' (childModules:IModel seq) (parameters:Parameter seq) (fwd:TorchTensor -> TorchTensor) =
    let p = new FuncModel(randName(), Seq.toArray parameters,fwd) 
    registerChildren childModules p
    p

let private checkNames names childModules =
    if Seq.length names <> Seq.length childModules + 1 then 
        failwithf $"number of names should be 1 + the-number-of-child-modules. The first name is for the module itself. Expecting {Seq.length childModules + 1} name(s) but got {Seq.length names}"

///<summary>Same as F but now assign names to all models (modules)</summary>
/// <seealso cref="F" />
let Fn names (childModules:IModel seq) (fwd:TorchTensor -> TorchTensor) =
    checkNames names childModules
    let p = new FuncModel(Seq.head names, [||],fwd) 
    registerNamedChildren names childModules p
    p

///<summary>Same as F' but now assign names to all models (modules)</summary>
/// <seealso cref="F'" />
let Fn' names (childModules:IModel seq) (parameters:Parameter seq) (fwd:TorchTensor -> TorchTensor) =
    checkNames names childModules
    let p = new FuncModel(Seq.head names, Seq.toArray parameters,fwd) 
    registerNamedChildren names childModules p
    p
    
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
