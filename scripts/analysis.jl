using Turing
using StatsPlots
using BSON: @load
using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--path_input", "-i"
            help = "Input folder, where the chains are stored"
            arg_type = String
            required = true
    end

    return parse_args(s)
end

parsed_args = parse_commandline()

path_input  = string(parsed_args["path_input"])

@load path_input*"/chains_12.bson" chains_12
@load path_input*"/chains_14.bson" chains_14
@load path_input*"/chains_16.bson" chains_16
@load path_input*"/chains_18.bson" chains_18
@load path_input*"/chains_20.bson" chains_20

p = plot(chains_12)
savefig(path_input*"/traceplots_12.png")
savefig(path_input*"/traceplots_12.pdf")

p = plot(chains_14)
savefig(path_input*"/traceplots_14.png")
savefig(path_input*"/traceplots_14.pdf")

p = plot(chains_16)
savefig(path_input*"/traceplots_16.png")
savefig(path_input*"/traceplots_16.pdf")

p = plot(chains_18)
savefig(path_input*"/traceplots_18.png")
savefig(path_input*"/traceplots_18.pdf")

p = plot(chains_20)
savefig(path_input*"/traceplots_20.png")
savefig(path_input*"/traceplots_20.pdf")
