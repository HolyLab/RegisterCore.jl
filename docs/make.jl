using Documenter
using RegisterCore

makedocs(
    sitename = "RegisterCore",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    modules = [RegisterCore],
    checkdocs = :exports,
    pages = [
        "Overview" => "index.md",
        "API Reference" => "api.md",
    ]
)

deploydocs(
    repo = "github.com/HolyLab/RegisterCore.jl.git"
)
