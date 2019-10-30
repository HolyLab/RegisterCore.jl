using Documenter
using RegisterCore

makedocs(
    sitename = "RegisterCore",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    modules = [RegisterCore],
    pages = ["index.md", "api.md"]
)

deploydocs(
    repo = "github.com/HolyLab/RegisterCore.jl.git"
)
