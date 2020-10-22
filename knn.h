#pragma once

#include "mat.h"
#include "math.h"

#define L this->num_depth+1
#define Layer(x) this->layer[x]
#define Weight(x) this->weights[x-1]
#define Bias(x) this->biases[x-1]
#define Error(x) this->errors[x]
#define Delta(x) this->deltas[x]
#define Gradient(x) this->gradients[x]

double sigmoid(double d);
double SiLU(double d);
double dsigmoid(double d);
double dSiLU(double d);

class KNN {
    private:
        uint num_inputs;
        uint num_depth;
        uint num_hidden;
        uint num_outputs;

        double learning_rate;

        Matrix* weights;
        Matrix* biases;
        Matrix* layer;
        Matrix* gradients;
        Matrix* deltas;

        double (*activation)(double);
        double (*dactivation)(double);
        
    public:
        KNN(uint num_inputs, uint num_hidden, uint num_outputs, double learning_rate = 0.12, uint num_depth = 1);
        ~KNN();

        Matrix feedforward(const Matrix& input);
        void train(const Matrix& input, const Matrix& target);
};