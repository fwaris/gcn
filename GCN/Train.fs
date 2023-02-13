module Train
open TorchSharp
open System

let run (datafolder,no_cuda,fastmode,epochs,dropout,lr,hidden,seed,weight_decay) =
    let cuda = not no_cuda && torch.cuda_is_available()
    torch.random.manual_seed(int64 seed) |> ignore
    
    let  adj, features, labels, idx_train, idx_val, idx_test = Utils.loadData datafolder None

    let features    = if cuda then features.cuda() else features
    let adj         = if cuda then adj.cuda() else adj
    let labels      = if cuda then labels.cuda() else labels
    let idx_train   = (if cuda then idx_train.cuda() else idx_train) |> torch.TensorIndex.Tensor
    let idx_val     = (if cuda then idx_val.cuda() else idx_val)     |> torch.TensorIndex.Tensor
    let idx_test    = (if cuda then idx_test.cuda() else idx_test)   |> torch.TensorIndex.Tensor

    let nclass = labels.max().ToInt64() + 1L

    let model = GCNModel.create features.shape.[1] (int64 hidden) nclass dropout adj
    let loss =  torch.nn.NLLLoss()

    if cuda then
        model.to'(torch.Device("cuda")) ///need to use IModel.to' to move mode to device to also move any buffers associated with the model

    let optimizer = torch.optim.Adam(model.Module.parameters(), lr=lr, weight_decay=weight_decay)

    let train epoch =
        let t = DateTime.Now
        model.Module.train() 
        optimizer.zero_grad()
        let output = model.forward(features)
        let loss_train = loss.forward(output.[ idx_train], labels.[idx_train])
        let acc_train = Utils.accuracy(output.[idx_train], labels.[idx_train])
        loss_train.backward()
        use r = optimizer.step() 

        let loss_val,acc_val =
            if not fastmode then
                model.Module.eval()
                let y' = model.forward(features)
                let loss_val = loss.forward(y'.[idx_val], labels.[idx_val])
                let acc_val = Utils.accuracy(y'.[idx_val], labels.[idx_val])
                loss_val,acc_val
            else
                let loss_val = loss.forward(output.[idx_val], labels.[idx_val])
                let acc_val = Utils.accuracy(output.[idx_val], labels.[idx_val])
                loss_val,acc_val
                
        printf $"Epoch: {epoch}, loss_train: %0.4f{float loss_train}, acc_train: %0.4f{acc_train}, "
        printfn $"loss_val: %0.4f{float loss_val}, acc_val: %0.4f{acc_val}"

    let test() =
        model.Module.eval()
        let y' = model.forward(features)
        let loss_test = loss.forward(y'.[idx_test], labels.[idx_test])
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

