using VectorizedRNG
using Documenter

makedocs(;
  modules = [VectorizedRNG],
  authors = "Chris Elrod",
  repo = Remotes.GitHub("JuliaSIMD","VectorizedRNG.jl"),
  sitename = "VectorizedRNG.jl",
  format = Documenter.HTML(;
    prettyurls = get(ENV, "CI", "false") == "true",
    canonical = "https://JuliaSIMD.github.io/VectorizedRNG.jl",
    assets = String[]
  ),
  pages = ["Home" => "index.md"],
  warnonly = true
)

deploydocs(; repo = "github.com/JuliaSIMD/VectorizedRNG.jl")
