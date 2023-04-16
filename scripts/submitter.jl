using ArgParse
using JSON3

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--nsteps"
            help = "The number of steps for each chain"
            arg_type = Int
            default = 1000
        "--nadapts"
            help = "The number of steps to adapt the HMC mass matrix"
            arg_type = Int
            default = 500
        "--nchains"
            help = "The number of chains to run in parallel"
            arg_type = Int
            default = 16
        "--nprocs"
            help = "The number of process to ask for each run"
            arg_type = Int
            default = 8
        "--resum"
            help = "The kind of resummation to use"
            arg_type = String
            default = "lagrangian"
        "--rescale_cov"
            help = "Boolean, used to decided wether to rescale to cov or not"
            arg_type = Bool
            default = false
        "--path_output", "-o"
            help = "Output folder, where store the chains"
            arg_type = String
            required = true
        "--julia_command", "-j"
            help = "Command to run julia. Modify only if the default julia mustn't be used"
            arg_type = String
            default = "julia"
    end

    return parse_args(s)
end

parsed_args = parse_commandline()

nsteps  = string(parsed_args["nsteps"])
nadapts = string(parsed_args["nadapts"])
nchains = string(parsed_args["nchains"])
nprocs = string(parsed_args["nprocs"])
julia_command = string(parsed_args["julia_command"])
resum = parsed_args["resum"]
rescale_cov = string(parsed_args["rescale_cov"])
path_output = parsed_args["path_output"]

mkdir(path_output)

open(path_output*"/config.json", "w") do io
    JSON3.pretty(io, parsed_args)
end

function create_job_sh(nsteps, nadapts, nchains, nprocs, resum, rescale_cov, path_output,
    julia_command)
    touch(path_output*"/job.sh")
    file = open(path_output*"/job.sh", "w")
    write(file, "#!/bin/bash")
    write(file, "\n")
    write(file, julia_command*" -t "*nprocs*" "*pwd()*"/mcmc.jl --nsteps "*
    nsteps*" --nadapts "*nadapts*" --nchains "*nchains*" --resum "*resum*" --rescale_cov "*
    rescale_cov*" --path_output "*path_output)

    close(file)

    return nothing
end

function create_submission_sh(path_output, resum, nprocs)
    touch(path_output*"/submission.sh")
    file = open(path_output*"/submission.sh", "w")
    write(file, "bsub -P c7 -q medium -o " *path_output*"/job_"*resum*".out -e "*
    path_output*"/job_"*resum*".err -n "*nprocs*" -M 12000 "*path_output*"/job.sh")
    close(file)

    return nothing
end

create_submission_sh(path_output, resum, nprocs)
create_job_sh(nsteps, nadapts, nchains, nprocs, resum, rescale_cov, path_output,
julia_command)
cmd_sub = `chmod +x $path_output/submission.sh`
cmd_job = `chmod +x $path_output/job.sh`
cmd_move = `cd $path_output`

cmd_exec = `./submission.sh`
run(cmd_sub)
run(cmd_job)
cd(path_output)
run(cmd_exec)
