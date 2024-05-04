#ifndef MATCHFILTER_H
#define MATCHFILTER_H
#include <armadillo>

using namespace arma;

class MatchFilter
{
public:
    MatchFilter();
    cx_mat algorithm(cx_mat s, cx_mat y_noisy, double N);
};

#endif // MATCHFILTER_H
