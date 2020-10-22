#include "knn.h"

double sigmoid(double d) {
    return 1/(1+exp(-d));
}

double SiLU(double d) {
    return d * sigmoid(d);
}

double dsigmoid(double d) {
    return sigmoid(d) * (1 - sigmoid(d));
}

double dSiLU(double d) {
    return sigmoid(d)*(1+d*(1-sigmoid(d)));
}



KNN::KNN(uint num_inputs, uint num_hidden, uint num_outputs, double learning_rate, uint num_depth) {
    this->num_inputs = num_inputs;
    this->num_depth = num_depth;
    this->num_hidden = num_hidden;
    this->num_outputs = num_outputs;

    this->learning_rate = learning_rate;

    this->weights = new Matrix[L];
    Weight(1) = Matrix(this->num_hidden, this->num_inputs).randomize();
    for(uint i = 2; i < L; i++) {
        Weight(i) = Matrix(this->num_hidden, this->num_hidden).randomize();
    }
    Weight(L) = Matrix(this->num_outputs, this->num_hidden).randomize();

    this->biases = new Matrix[L];
    Bias(1) = Matrix(this->num_hidden, 1).randomize();
    for(uint i = 2; i < L; i++) {
        Bias(i) = Matrix(this->num_hidden, 1).randomize();
    }
    Bias(L) = Matrix(this->num_outputs, 1).randomize();

    this->layer = new Matrix[L+1];

    this->gradients = new Matrix[L+1];
    this->deltas = new Matrix[L+1];

    this->activation = SiLU;
    this->dactivation = dSiLU;

    
}

KNN::~KNN() {
    delete [] this->weights;
    delete [] this->biases;
    delete [] this->layer;
    delete [] this->gradients;
    delete [] this->deltas;
}

Matrix KNN::feedforward(const Matrix& input) {
    Layer(0) = input;
    
    for(uint i = 1; i <= L; i++) {
        Layer(i) = Matrix::multiply(Weight(i), Layer(i-1));
        Layer(i) = Matrix::function(Layer(i), sigmoid);
    }
    return this->layer[L];
}

void KNN::train(const Matrix& input, const Matrix& target) {
    Layer(0) = input;
    for(uint i = 1; i <= L; i++) {
        Layer(i) = Matrix::multiply(Weight(i), Layer(i-1));
        Layer(i) = Matrix::function(Layer(i), this->activation);
    }
    Delta(L) = Matrix::multiply(Matrix::subtract(Layer(L), target), Matrix::function(Matrix::multiply(Weight(L), Layer(L-1)), this->dactivation), true);
    Gradient(L) = Matrix::multiply(Delta(L), Matrix::transpose(Layer(L-1)));
    Weight(L) = Matrix::subtract(Weight(L), Matrix::multiply(Matrix::multiply(Weight(L), this->learning_rate), Gradient(L), true));
    for(int i = L-1; i > 0; i--) {
        Delta(i) = Matrix::multiply(Matrix::multiply(Matrix::transpose(Weight(i+1)), Delta(i+1)), Matrix::function(Matrix::multiply(Weight(i), Layer(i-1)), this->dactivation), true);
        Gradient(i) = Matrix::multiply(Delta(i), Matrix::transpose(Layer(i-1)));
        Weight(i) = Matrix::subtract(Weight(i), Matrix::multiply(Matrix::multiply(Weight(i), this->learning_rate), Gradient(i), true));
    }
    std::cout << Matrix::subtract(Layer(L), target) << std::endl << std::endl;
}