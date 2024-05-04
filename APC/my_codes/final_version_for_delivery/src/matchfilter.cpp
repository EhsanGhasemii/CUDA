#include "../include/matchfilter.h"

MatchFilter::MatchFilter()
{

}

cx_mat MatchFilter::algorithm(cx_mat s, cx_mat  y_noisy, double N)
{
    int L = y_noisy.size();
    cx_mat temp = s;

    for (int var = 0; var < s.size(); ++var) {
        temp(var,0) = s(N-var-1,0);
    }

    cx_mat result = conv(y_noisy,temp);
    return result.submat(N-1,0,L,0);
}
