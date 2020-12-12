using VectorizedRNG
using Documenter

makedocs(;
    modules=[VectorizedRNG],
    authors="Chris Elrod",
    repo="https://github.com/chriselrod/VectorizedRNG.jl/blob/{commit}{path}#L{line}",
    sitename="VectorizedRNG.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://chriselrod.github.io/VectorizedRNG.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
    strict=false,
)

deploydocs(;
    repo="github.com/chriselrod/VectorizedRNG.jl",
)
