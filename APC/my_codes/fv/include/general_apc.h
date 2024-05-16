#ifndef GENERAL_APC_H
#define GENERAL_APC_H
#include <armadillo>

using namespace arma;


class General_APC
{
public:
    General_APC();
    cx_mat algorithm(cx_mat s, cx_mat y_noisy, double N, mat alpha, double sigma);
	cx_mat algorithm2(cx_mat s, cx_mat y_noisy, double N, mat alpha, double sigma);
private:
    cx_mat elementwisePow(cx_mat input, double p);
};

#endif // GENERAL_APC_H
