#ifndef TENSOR_NETWORK_HPP
#define TENSOR_NETWORK_HPP

#include <complex>
#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions> 

using Complex = std::complex<double>;
using Matrix = Eigen::MatrixXcd;
using Vector = Eigen::VectorXcd;

// Simple structure pour un tenseur 3D
struct Tensor3D {
    std::vector<Complex> data;
    int dim0, dim1, dim2;
    
    Tensor3D(int d0, int d1, int d2) : dim0(d0), dim1(d1), dim2(d2) {
        data.resize(d0 * d1 * d2, 0.0);
    }
    
    Complex& operator()(int i, int j, int k) {
        return data[i * dim1 * dim2 + j * dim2 + k];
    }
    
    const Complex& operator()(int i, int j, int k) const {
        return data[i * dim1 * dim2 + j * dim2 + k];
    }
};

// Simple structure pour un tenseur 4D
struct Tensor4D {
    std::vector<Complex> data;
    int dim0, dim1, dim2, dim3;
    
    Tensor4D(int d0, int d1, int d2, int d3) 
        : dim0(d0), dim1(d1), dim2(d2), dim3(d3) {
        data.resize(d0 * d1 * d2 * d3, 0.0);
    }
    
    Complex& operator()(int i, int j, int k, int l) {
        return data[i * dim1 * dim2 * dim3 + j * dim2 * dim3 + k * dim3 + l];
    }
    
    const Complex& operator()(int i, int j, int k, int l) const {
        return data[i * dim1 * dim2 * dim3 + j * dim2 * dim3 + k * dim3 + l];
    }
};

Matrix rotationFromGivens(const std::vector<std::pair<int, int>>& indices,
    const std::vector<Matrix>& givens,
    const std::vector<int>& sect);

void applyGateMPOBothSides(MPO& Op, const ITensor& Lgate, const ITensor& Rgate,
    int i, int j, const Args& args = Args::global());

// Opérateurs de base
namespace Operators {
    extern Matrix c, cdag, sz, sp, sm, n, Id;
    void initialize();
}

// Paramètres du modèle
struct ModelParams {
    int L;
    double dt;
    double V;
    double Uint;
    double gamma;
    double J;
    double Jz;
    double ed;
    std::string model;
};

// Étape de Trotter
struct TrotterStep {
    std::string layer;
    double dt;
    bool newTimeStep;
    bool computeObs;
};

// Classe MPS
class MPS {
public:
    MPS(int L);
    void print() const;
    
    std::vector<Tensor3D> B;
    std::vector<Vector> Lambda;
    std::vector<int> bondDim;
    int L;
};

// Classe MPO
class MPO {
public:
    MPO(int L);
    void compress(double threshold = 1e-12);
    void printBondDim() const;
    
    std::vector<Tensor4D> B;
    std::vector<Vector> Lambda;
    std::vector<int> bondDim;
    int L;
};

// Générer une porte
Tensor4D generateGate(const std::string& step, const ModelParams& params, double dt);

// Appliquer une porte sur MPO
void applyGateMPO(MPO& Op, const Tensor4D& gate, int l1, int l2);

// Créer la séquence de Trotter
std::vector<TrotterStep> createTrotterSequence(
    int order, int nSteps, double dt, int nParts, 
    int nStepsCalc, const std::vector<std::string>& Hsteps
);

#endif // TENSOR_NETWORK_HPP
