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
using Distributed
using ClusterManagers

n_proc = 20
n_adapt = 500
n_steps = 1000

addprocs_lsf(n_proc)


@everywhere using Statistics
@everywhere using Distributions
@everywhere using SimpleChains
@everywhere using Static
@everywhere using NPZ
@everywhere using ForwardDiff
@everywhere using LinearAlgebra
@everywhere using Statistics
@everywhere using ProgressMeter
@everywhere using Turing
@everywhere using Pathfinder
@everywhere using Optim
@everywhere using Transducers
@everywhere using Effort
@everywhere using BlindedChallenge

@everywhere resum = "optimal"
@everywhere if resum == "lagrangian"
    println("You choose lagrangian resummation!")
elseif resum == "optimal"
    println("You choose optimal resummation!")
else
    error("You didn't choose a viable resummation!")
end

@everywhere rescale_cov = false
@everywhere if rescale_cov
    println("You decided to rescale the covariance cov!")
    scaling_factor = 4
    scaling_name = "_scaled_"
else
    println("You decided not to rescale the covariance!")
    scaling_factor = 1
    scaling_name = ""
end

@everywhere function create_mlpd(nk)

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

@everywhere function load_emulator(path::String, architecture)
    return Effort.SimpleChainsEmulator(Architecture=architecture, Weights=npzread(path))
end

@everywhere function load_emulator_PyBird(path::String, multipole)

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

@everywhere if resum == "lagrangian"
    println("You choose lagrangian resummation!")
    Mono_Emu = load_emulator_PyBird("../trained_emulators/PyBird_061_10000_lagrangian_guido_spectra_check", "0")
    Quad_Emu = load_emulator_PyBird("../trained_emulators/PyBird_061_10000_lagrangian_guido_spectra_check", "2");
elseif resum == "optimal"
    println("You choose optimal resummation!")
    Mono_Emu = load_emulator_PyBird("../trained_emulators/PyBird_061_10000_optiresum_final", "0")
    Quad_Emu = load_emulator_PyBird("../trained_emulators/PyBird_061_10000_optiresum_final", "2");
else
    error("You didn't choose a viable resummation!")
end

@everywhere function theory(θ, n, Mono_Emu, Quad_Emu)
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

@everywhere @model function pplmodel(data, cov, n, Mono_Emu, Quad_Emu)
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

@everywhere n = 20

@everywhere data_20, k_20, cov_20, yerror_Mono_20, yerror_Quad_20 = BlindedChallenge.create_data(n)

@everywhere model_20 = pplmodel(data_20, cov_20.*scaling_factor, 20, Mono_Emu, Quad_Emu)

sample(model_20, NUTS(50, 0.65), MCMCDistributed(), 100, 4)
