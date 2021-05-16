module Train
open TorchSharp
open TorchSharp.Tensor
open System

let run (datafolder,no_cuda,fastmode,epochs,dropout,lr,hidden,seed,weight_decay) =
    let cuda = not no_cuda && Torch.IsCudaAvailable()
    Torch.ManualSeed(int64 seed) |> ignore
    
    let  adj, features, labels, idx_train, idx_val, idx_test = Utils.loadData datafolder None

    let features = if cuda then features.cuda() else features
    let adj = if cuda then adj.cuda() else adj
    let labels = if cuda then labels.cuda() else labels
    let idx_train = (if cuda then idx_train.cuda() else idx_train) |> TorchTensorIndex.Tensor
    let idx_val = (if cuda then idx_val.cuda() else idx_val)       |> TorchTensorIndex.Tensor
    let idx_test = (if cuda then idx_test.cuda() else idx_test)    |> TorchTensorIndex.Tensor

    let nclass = labels.max().ToInt64() + 1L

    let model = GCNModel.create features.shape.[1] (int64 hidden) nclass dropout adj
    let loss = NN.Functions.nll_loss()

    if cuda then
        model.Module.cuda() |> ignore

    let optimizer = NN.Optimizer.Adam(model.Module.parameters(), learningRate = lr, weight_decay=weight_decay)

    let train epoch =
        let t = DateTime.Now
        model.Module.Train()        
        optimizer.zero_grad()
        let output = model.forward(features)
        let loss_train = loss.Invoke(output.[ idx_train], labels.[idx_train])
        let acc_train = Utils.accuracy(output.[idx_train], labels.[idx_train])
        loss_train.backward()
        optimizer.step()

        let parms = model.Module.parameters()
        let data = parms |> Array.map TorchSharp.Fun.Tensor.getData<float32>
        let i = 1

        let loss_val,acc_val =
            if not fastmode then
                model.Module.Eval()
                let y' = model.forward(features)
                let loss_val = loss.Invoke(y'.[idx_val], labels.[idx_val])
                let acc_val = Utils.accuracy(y'.[idx_val], labels.[idx_val])
                loss_val,acc_val
            else
                let loss_val = loss.Invoke(output.[idx_val], labels.[idx_val])
                let acc_val = Utils.accuracy(output.[idx_val], labels.[idx_val])
                loss_val,acc_val
                
        printf $"Epoch: {epoch}, loss_train: %0.4f{float loss_train}, acc_train: %0.4f{acc_train}, "
        printfn $"loss_val: %0.4f{float loss_val}, acc_val: %0.4f{acc_val}"

    let test() =
        model.Module.Eval()
        let y' = model.forward(features)
        let loss_test = loss.Invoke(y'.[idx_test], labels.[idx_test])
        let acc_test = Utils.accuracy(y'.[idx_test],labels.[idx_test])
        printfn $"""Test set results:
         loss: {float loss_test}, accuracy: {acc_test}
        """

    let t_total = DateTime.Now
    for i in 1 .. epochs-1 do
        train i
    printfn "Optimization done"
    printfn $"Time elapsed: %0.2f{(DateTime.Now - t_total).TotalSeconds} seconds"

    test()

