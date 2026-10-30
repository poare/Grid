// Example_mg_refine.h
//
// The sloppy basis refresh, apart from Example_mg_common.h: that header's own
// PVdagMLinearOperator clashes with develop's, so a driver on PVdagMMultiGrid.h
// cannot include it.
#pragma once

#include <Grid/Grid.h>
#include <Grid/algorithms/iterative/PrecGeneralisedConjugateResidualNonHermitian.h>

NAMESPACE_BEGIN(Grid);

//////////////////////////////////////////////////////////////////////////////
// Subspace refresh by inverse iteration (strategy (c)).
//
// This is the non-Hermitian (GCR) analogue of Aggregation::RefineSubspace:
// each existing basis vector is used to seed ONE sloppy inverse-iteration
// pass with the CURRENT operator,
//     v_new  ~  [PV^dag M (U_t)]^{-1} v_old   (solved loosely),
// which contracts the O(dU) error of the stale vector back towards the
// near-null space of the updated operator. The structure mirrors the inner
// loop of Aggregation::CreateSubspaceGCR, but seeded with the old vector
// instead of fresh noise, and with a much smaller iteration budget.
// Returns the GCR steps taken for each vector.
//////////////////////////////////////////////////////////////////////////////
template<class Field>
std::vector<int> RefineBasisGCR(std::vector<Field> &V,
		       LinearOperatorBase<Field> &DiracOp,
		       int nn,
		       RealD tol, int maxouter)
{
  RealD scale;
  TrivialPrecon<Field> simple;
  PrecGeneralisedConjugateResidualNonHermitian<Field> GCR(tol,maxouter,DiracOp,simple,10,10);

  Field src(V[0].Grid());
  Field guess(V[0].Grid());
  Field Mn(V[0].Grid());
  std::vector<int> steps(nn);

  for(int b=0;b<nn;b++){
    scale = std::pow(norm2(V[b]),-0.5);
    src   = V[b]*scale;

    DiracOp.Op(src,Mn);
    std::cout<<GridLogMessage << "refine-in ["<<b<<"] <v|Op|v> "<<innerProduct(src,Mn)<<std::endl;

    guess = Zero();
    GCR(src,guess);
    steps[b] = GCR.Steps();

    scale = std::pow(norm2(guess),-0.5);
    V[b] = guess*scale;

    DiracOp.Op(V[b],Mn);
    std::cout<<GridLogMessage << "refine-out["<<b<<"] <v|Op|v> "<<innerProduct(V[b],Mn)<<std::endl;
  }
  return steps;
}

NAMESPACE_END(Grid);
