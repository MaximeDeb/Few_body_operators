#include "itensor/all.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <tuple>

using namespace itensor;

// Paramètres
struct ModelParams {
    int L;
    double dt;
    double V, Uint, gamma, J, Jz;
    std::string model;
};

// Créer MPO identité (simple, sans QN)
MPO createIdentityMPO(const SiteSet& sites) {
    int L = length(sites);
    auto Op = MPO(sites);

    for (int i = 1; i <= L; ++i) {
        auto s = sites(i);
        auto sp = prime(s);

        auto Id = ITensor(s, sp);

        // Identité : somme sur tous les états
        for (int n = 1; n <= 4; ++n) {
            Id.set(s = n, sp = n, 1.0 / std::sqrt(2.0));
        }

        Op.ref(i) = Id;
    }

    return Op;
}

// Générer porte (simple, sans QN)
ITensor makeGate(const SiteSet& sites, int i, int j,
    const ModelParams& params, double dt) {

    auto Sp_i = op(sites, "S+", i);
    auto Sm_i = op(sites, "S-", i);
    auto Sz_i = op(sites, "Sz", i);

    auto Sp_j = op(sites, "S+", j);
    auto Sm_j = op(sites, "S-", j);
    auto Sz_j = op(sites, "Sz", j);

    ITensor Hamil;

    if (params.model == "IRLM") {
        if (i == 1) {
            Hamil = params.V * (Sp_i * Sm_j + Sm_i * Sp_j) +
                params.Uint * Sz_i * Sz_j;
        }
        else {
            Hamil = params.gamma * (Sp_i * Sm_j + Sm_i * Sp_j);
        }
    }
    else if (params.model == "Heis_nn") {
        Hamil = 0.5 * params.J * (Sp_i * Sm_j + Sm_i * Sp_j) +
            params.Jz * Sz_i * Sz_j;
    }

    auto gate = expHermitian(Hamil, -Cplx_i * dt);
    return gate;
}

// Appliquer porte sur MPO (d'après forum ITensor)
void applyGateMPO(MPO& Op, const ITensor& gate, int i, int j,
    const Args& args = Args::global()) {

    // Récupérer les indices
    auto si = siteIndex(Op, i);
    auto sj = siteIndex(Op, j);

    // Appliquer la porte SEULEMENT sur les indices non-primés
    auto AA = Op(i) * Op(j);
    auto gAA = gate * AA;

    // Déprimer seulement les indices non-primés
    gAA.noPrime(si);
    gAA.noPrime(sj);

    // SVD avec la signature correcte : svd(T, U, S, V, args)
    ITensor U, S, V;
    auto spec = svd(gAA, U, S, V, args);

    // Reconstruire le MPO
    Op.ref(i) = U;
    Op.ref(j) = S * V;
}

int main() {
    std::cout << "=== ITensor MPO Evolution (Simplified) ===" << std::endl;

    // Paramètres
    ModelParams params;
    params.L = 20;
    params.dt = 0.01;
    params.model = "IRLM";
    params.V = 0.2;
    params.Uint = 0.2;
    params.gamma = 0.5;
    params.J = 0.0;
    params.Jz = 0.0;

    std::cout << "Parameters:" << std::endl;
    std::cout << "  L = " << params.L << std::endl;
    std::cout << "  dt = " << params.dt << std::endl;
    std::cout << "  Model = " << params.model << std::endl;

    // Sites SANS conservation QN (simplifié)
    auto sites = Electron(params.L, { "ConserveQNs", false });

    std::cout << "\nCreating initial MPO..." << std::endl;
    auto Op = createIdentityMPO(sites);

    auto args = Args("Cutoff=", 1e-10, "MaxDim=", 1024);

    // Couches de portes
    std::vector<std::vector<std::pair<int, int>>> gatesLayers;

    if (params.model == "IRLM") {
        gatesLayers.push_back({ {1, 2} });

        std::vector<std::pair<int, int>> layer1, layer2;
        for (int i = 3; i < params.L; i += 2) layer1.push_back({ i, i + 1 });
        for (int i = 2; i < params.L; i += 2) {
            if (i != 2) layer2.push_back({ i, i + 1 });
        }
        if (!layer1.empty()) gatesLayers.push_back(layer1);
        if (!layer2.empty()) gatesLayers.push_back(layer2);
    }

    // Générer portes Trotter
    double dt = params.dt;
    double dt1 = dt / (4.0 - std::pow(4.0, 1.0 / 3.0));
    double dt2 = dt - 4.0 * dt1;

    std::cout << "Trotter dt1=" << dt1 << ", dt2=" << dt2 << std::endl;

    std::map<std::tuple<int, double>, ITensor> gates;

    for (double ddt : {dt1, dt2, dt1 / 2, dt2 / 2}) {
        for (size_t idx = 0; idx < gatesLayers.size(); ++idx) {
            for (const auto& [i, j] : gatesLayers[idx]) {
                gates[{idx, ddt}] = makeGate(sites, i, j, params, ddt);
            }
        }
    }

    std::cout << "Generated " << gates.size() << " gates" << std::endl;

    // Fonction U2
    auto applyU2 = [&](double ddt) {
        for (size_t idx = 0; idx < gatesLayers.size(); ++idx) {
            for (const auto& [i, j] : gatesLayers[idx]) {
                applyGateMPO(Op, gates.at({ idx, ddt / 2 }), i, j, args);
            }
        }
        for (int idx = gatesLayers.size() - 1; idx >= 0; --idx) {
            for (auto it = gatesLayers[idx].rbegin();
                it != gatesLayers[idx].rend(); ++it) {
                applyGateMPO(Op, gates.at({ idx, ddt / 2 }), it->first, it->second, args);
            }
        }
        };

    // Évolution
    double T = 0.5;  // Temps court pour test
    int nSteps = static_cast<int>(T / dt);
    int nStepsCalc = 10;

    std::cout << "\nStarting time evolution (" << nSteps << " steps)..." << std::endl;

    double t = 0.0;

    for (int step = 1; step <= nSteps; ++step) {

        // Trotter ordre 4
        applyU2(dt1);
        applyU2(dt1);
        applyU2(dt2);
        applyU2(dt1);
        applyU2(dt1);

        t += dt;

        if (step % nStepsCalc == 0 || step == nSteps) {
            std::cout << "\n========================================" << std::endl;
            std::cout << "Step " << step << "/" << nSteps << ", Time: " << t << std::endl;

            std::cout << "Bond dims: ";
            int maxDim = 0;
            for (int i = 1; i < params.L; ++i) {
                int d = dim(linkIndex(Op, i));
                std::cout << d << " ";
                maxDim = std::max(maxDim, d);
            }
            std::cout << "\nMax: " << maxDim << std::endl;
        }
    }

    std::cout << "\n=== Simulation completed! ===" << std::endl;

    return 0;
}