#r "nuget: Sdk=Microsoft.NET.Sdk.WindowsDesktop" 

#r "nuget: TorchSharp"
#r "nuget: SixLabors.ImageSharp"
#r "nuget: Microsoft.ML.AutoML"
#r "nuget: FSharp.Data"
#r "nuget: FSharp.Plotly"
#r "nuget: Microsoft.Msagl.GraphViewerGDI"
#r "nuget: FsPickler"
#r "nuget: MathNet.Numerics"
#r "nuget: MathNet.Numerics.FSharp"

// Libtorch binaries
// Option A: you can use a platform-specific nuget package
//#r "nuget: libtorch-cuda-11.1-win-x64, 1.8.0.7"
System.Runtime.InteropServices.NativeLibrary.Load(@"D:\s\libtorch\lib\torch_cuda.dll")


#load @"..\TorchSharp.Fun.fs"
#I @"C:\Program Files\dotnet\shared\Microsoft.WindowsDesktop.App\5.0.4"
#r "System.Windows.Forms"
