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

//////////////////////////////////////////////////////////////////////////////
// Orthonormalise the basis in place (modified Gram-Schmidt, in index order).
// Between RefineBasisGCR passes this turns independent inverse iteration,
// which drives every vector towards the same lowest modes, into subspace
// iteration, which converges to the low-mode space of the whole basis.
// Aggregation::GlobalOrthonormalise does this on its own copy of the basis,
// which costs nbasis more fine fields.
//////////////////////////////////////////////////////////////////////////////
template<class Field>
void OrthonormaliseBasis(std::vector<Field> &V)
{
  for(int i=0;i<(int)V.size();i++){
    basisOrthogonalize(V,V[i],i);
    V[i] = V[i]*std::pow(norm2(V[i]),-0.5);
  }
}

//////////////////////////////////////////////////////////////////////////////
// v -> v + kick |v| eta/|eta|, eta Gaussian, before an inverse-iteration pass,
// which smooths the noise into a fresh low-mode component.  Keeps the basis a
// random sample of the low-mode band instead of converging to the lowest
// eigenspace under repeated passes.
//////////////////////////////////////////////////////////////////////////////
template<class Field>
void KickBasis(std::vector<Field> &V, GridParallelRNG &RNG, RealD kick)
{
  Field eta(V[0].Grid());
  for(auto &v : V){
    gaussian(RNG,eta);
    v = v + (kick*std::sqrt(norm2(v)/norm2(eta)))*eta;
  }
}

NAMESPACE_END(Grid);
