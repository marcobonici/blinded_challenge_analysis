using Statistics
using Distributions
using SimpleChains
using Static
using NPZ
using ForwardDiff
using LinearAlgebra
using Statistics
using ProgressMeter
using Turing
using Pathfinder
using Optim
using Transducers
using Effort
using BlindedChallenge
using JSON3
using BSON: @save, @load

using ArgParse
println(@__DIR__)

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--nsteps"
            help = "The number steps of the chains"
            arg_type = Int
            default = 1000
        "--nadapts"
            help = "The number of steps to adapt the NUTS algorithm"
            arg_type = Int
            default = 500
        "--nchains"
            help = "The number of chains to run in parallel"
            arg_type = Int
            default = 4
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
            required = true
    end

    return parse_args(s)
end

parsed_args = parse_commandline()

nsteps  = parsed_args["nsteps"]
nadapts = parsed_args["nadapts"]
nchains = parsed_args["nchains"]
resum = parsed_args["resum"]
rescale_cov = parsed_args["rescale_cov"]
path_output = parsed_args["path_output"]

if resum == "lagrangian"
    println("You choose lagrangian resummation!")
elseif resum == "optimal"
    println("You choose optimal resummation!")
else
    error("You didn't choose a viable resummation!")
end


if rescale_cov
    println("You decided to rescale the covariance cov!")
    scaling_factor = 4
    scaling_name = "_scaled_"
else
    println("You decided not to rescale the covariance!")
    scaling_factor = 1
    scaling_name = ""
end

function create_mlpd(nk)

    mlpd = SimpleChain(
            static(3),
            TurboDense(tanh, 64),
            TurboDense(tanh, 64),
            TurboDense(tanh, 64),
            TurboDense(tanh, 64),
            TurboDense(tanh, 64),
            TurboDense(identity, nk)
            )

    return mlpd
end

function load_emulator(path::String, architecture)
    return Effort.SimpleChainsEmulator(Architecture=architecture, Weights=npzread(path))
end

function load_emulator_PyBird(path::String, multipole)

    k_grid = npzread(path*"/k_grid.npy")
    in_minmax = npzread(path*"/inMinMax_l_"*multipole*".npy")

    emu_11 = load_emulator(path*"/weights_P_11_l_"*multipole*".npy", create_mlpd(60))
    out_minmax_11 = npzread(path*"/outMinMax_P_11_l_"*multipole*".npy")
    Emulator_11 = Effort.P11Emulator(TrainedEmulator=emu_11, kgrid = k_grid,
                                                                                                 InMinMax = in_minmax, OutMinMax = out_minmax_11)

    emu_loop = load_emulator(path*"/weights_P_loop_l_"*multipole*".npy", create_mlpd(240))
    out_minmax_loop = npzread(path*"/outMinMax_P_loop_l_"*multipole*".npy")
    Emulator_loop = Effort.PloopEmulator(TrainedEmulator=emu_loop, kgrid = k_grid,
                                                                                                       InMinMax = in_minmax, OutMinMax = out_minmax_loop)


    emu_ct = load_emulator(path*"/weights_P_ct_l_"*multipole*".npy", create_mlpd(120))
    out_minmax_ct = npzread(path*"/outMinMax_P_ct_l_"*multipole*".npy")
    Emulator_ct = Effort.PctEmulator(TrainedEmulator=emu_ct, kgrid = k_grid,
                                                                                                 InMinMax = in_minmax, OutMinMax = out_minmax_ct)

    return Effort.PℓEmulatorPyBird(P11=Emulator_11, Ploop=Emulator_loop, Pct=Emulator_ct)
end

if resum == "lagrangian"
    println("You choose lagrangian resummation!")
    Mono_Emu = load_emulator_PyBird(string(@__DIR__)*"/../trained_emulators/PyBird_061_10000_lagrangian_guido_spectra_check", "0")
    Quad_Emu = load_emulator_PyBird(string(@__DIR__)*"/../trained_emulators/PyBird_061_10000_lagrangian_guido_spectra_check", "2");
elseif resum == "optimal"
    println("You choose optimal resummation!")
    Mono_Emu = load_emulator_PyBird(string(@__DIR__)*"/../trained_emulators/PyBird_061_10000_optiresum_final", "0")
    Quad_Emu = load_emulator_PyBird(string(@__DIR__)*"/../trained_emulators/PyBird_061_10000_optiresum_final", "2");
else
    error("You didn't choose a viable resummation!")
end

function theory(θ, n, Mono_Emu, Quad_Emu)
    # θ[1:3] cosmoparams, ln_10_As, H0, ΩM
    # θ[4:9] bias
    # θ[10:11] stoch
    #f = fN(θ[3], 0.61)
    f = Effort._f_z(0.61, θ[3], -1., 0.);
    #conversion from c2-c4 to b2-b4
    b2 = (θ[5]+θ[7])/√2
    b4 = (θ[5]-θ[7])/√2
    my_θ = deepcopy(θ)
    my_θ[8] /= (0.7^2)
    my_θ[9] /= (0.7^2)
    n_bar = 3e-4 #value  that has been suggested by Guido himself
    k_bins = Effort.create_bin_edges(BlindedChallenge.k_grid)
    #stoch_0, stoch_2 = Effort.get_stoch_terms(0, θ[10], θ[11], n_bar, k_grid)
    stoch_0, stoch_2 = Effort.get_stoch_terms_binned_efficient(0., my_θ[10], my_θ[11], n_bar, k_bins)
    return vcat((Effort.get_Pℓ(my_θ[1:3], vcat(my_θ[4], b2, my_θ[6], b4, my_θ[8:9] , 0), f, Mono_Emu) .+ stoch_0)[1:n],
                                (Effort.get_Pℓ(my_θ[1:3], vcat(my_θ[4], b2, my_θ[6], b4, my_θ[8:9] , 0), f, Quad_Emu) .+ stoch_2)[1:n])
end

@model function pplmodel(data, cov, n, Mono_Emu, Quad_Emu)
    ln10As ~ Uniform(2.9, 3.15)
    H0     ~ Uniform(62.,68.)
    ΩM     ~ Uniform(0.30, 0.34)
    b1     ~ Uniform(0., 4.)
    c2     ~ Uniform(-4., 4.)
    b3     ~ Normal(0., 10.)
    c4     ~ Normal(0., 2.)
    cct    ~ Normal(0., 4.)
    cr1    ~ Normal(0., 8.)
    cϵm    ~ Normal(0., 2.)
    cϵq    ~ Normal(0., 4.)

    θ = [ln10As, H0, ΩM, b1, c2, b3, c4, cct, cr1, cϵm, cϵq]

    prediction = theory(θ, n, Mono_Emu, Quad_Emu)

    data ~ MvNormal(prediction, cov)
    return nothing
end

n = 20

data_20, k_20, cov_20, yerror_Mono_20, yerror_Quad_20 = BlindedChallenge.create_data(n)

model_20 = pplmodel(data_20, cov_20.*scaling_factor, n, Mono_Emu, Quad_Emu)

result_multi_20 = multipathfinder(model_20, 1000; nruns = 8, executor = Transducers.PreferParallel())
init_params_20 = collect.(eachrow(result_multi_20.draws_transformed.value[1:nchains, :, 1]))

chains_20 = sample(model_20, Turing.NUTS(nadapts, 0.65), MCMCThreads(), nsteps, nchains; init_params = init_params_20)

@save path_output*"/chains_20.bson" chains_20

n = 18

data_18, k_18, cov_18, yerror_Mono_18, yerror_Quad_18 = BlindedChallenge.create_data(n)

model_18 = pplmodel(data_18, cov_18.*scaling_factor, n, Mono_Emu, Quad_Emu)

result_multi_18 = multipathfinder(model_18, 1000; nruns = 8, executor = Transducers.PreferParallel())
init_params_18 = collect.(eachrow(result_multi_18.draws_transformed.value[1:nchains, :, 1]))

chains_18 = sample(model_18, Turing.NUTS(nadapts, 0.65), MCMCThreads(), nsteps, nchains; init_params = init_params_18)

@save path_output*"/chains_18.bson" chains_18

n = 16

data_16, k_16, cov_16, yerror_Mono_16, yerror_Quad_16 = BlindedChallenge.create_data(n)

model_16 = pplmodel(data_16, cov_16.*scaling_factor, n, Mono_Emu, Quad_Emu)

result_multi_16 = multipathfinder(model_16, 1000; nruns = 8, executor = Transducers.PreferParallel())
init_params_16 = collect.(eachrow(result_multi_16.draws_transformed.value[1:nchains, :, 1]))

chains_16 = sample(model_16, Turing.NUTS(nadapts, 0.65), MCMCThreads(), nsteps, nchains; init_params = init_params_16)

@save path_output*"/chains_16.bson" chains_16

n = 14

data_14, k_14, cov_14, yerror_Mono_14, yerror_Quad_14 = BlindedChallenge.create_data(n)

model_14 = pplmodel(data_14, cov_14.*scaling_factor, n, Mono_Emu, Quad_Emu)

result_multi_14 = multipathfinder(model_14, 1000; nruns = 8, executor = Transducers.PreferParallel())
init_params_14 = collect.(eachrow(result_multi_14.draws_transformed.value[1:nchains, :, 1]))

chains_14 = sample(model_14, Turing.NUTS(nadapts, 0.65), MCMCThreads(), nsteps, nchains; init_params = init_params_14)

@save path_output*"/chains_14.bson" chains_14

n = 12

data_12, k_12, cov_12, yerror_Mono_12, yerror_Quad_12 = BlindedChallenge.create_data(n)

model_12 = pplmodel(data_12, cov_12.*scaling_factor, n, Mono_Emu, Quad_Emu)

result_multi_12 = multipathfinder(model_12, 1000; nruns = 8, executor = Transducers.PreferParallel())
init_params_12 = collect.(eachrow(result_multi_12.draws_transformed.value[1:nchains, :, 1]))

chains_12 = sample(model_12, Turing.NUTS(nadapts, 0.65), MCMCThreads(), nsteps, nchains; init_params = init_params_12)

@save path_output*"/chains_12.bson" chains_12
