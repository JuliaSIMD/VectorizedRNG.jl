using Documenter, VectorizedRNG

makedocs(;
    modules=[VectorizedRNG],
    format=:html,
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/chriselrod/VectorizedRNG.jl/blob/{commit}{path}#L{line}",
    sitename="VectorizedRNG.jl",
    authors="Chris Elrod",
    assets=[],
)

deploydocs(;
    repo="github.com/chriselrod/VectorizedRNG.jl",
    target="build",
    julia="1.0",
    deps=nothing,
    make=nothing,
)
