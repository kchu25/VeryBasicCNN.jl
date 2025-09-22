using VeryBasicCNN
using Documenter

DocMeta.setdocmeta!(VeryBasicCNN, :DocTestSetup, :(using VeryBasicCNN); recursive=true)

makedocs(;
    modules=[VeryBasicCNN],
    authors="Shane Kuei-Hsien Chu (skchu@wustl.edu)",
    sitename="VeryBasicCNN.jl",
    format=Documenter.HTML(;
        canonical="https://kchu25.github.io/VeryBasicCNN.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/kchu25/VeryBasicCNN.jl",
    devbranch="main",
)
