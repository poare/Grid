// Example_coarse_coherence.h
//
// Coarse-level coherence diagnostics for Example_pvdagm_inverse_iteration
// (research hub, projects/multigrid, notes/inverse-iteration-test-design.md):
// Peter Boyle's census and coherence checks (Example_pvdagm_census.cc,
// Example_pvdagm_coherence.cc) run on the PRODUCTION coarse operator, the
// MultiGeneralCoarsenedOperator of PVdagMMultiGridCoarsening.
//
// Two facts about that operator shape this file.
//
//   1. It applies only on the D+1 grid given to SetGrid, while the coarse
//      basis images (rawPsi) and Peter's block primitives live on the 5D
//      coarse grid.  CoarseSliceOperator is the single-RHS view: insert
//      slice 0, apply, extract (the pattern of BuildDenseBottom).
//   2. Its Mdag is M (GeneralCoarsenedMatrixMultiRHS.h), so it has no
//      adjoint.  A_c^dag = P^dag A^dag P is obtained by coarsening A^dag on
//      the same basis into a second operator; PVdagMAdjointOperator is the
//      fine operator for that coarsening.  AdjointMismatch certifies the
//      pair before anything uses it.
//
// Include Example_basis_diagnostics.h first (BasisList).
#pragma once

#include <Grid/Grid.h>
#include <Grid/algorithms/multigrid/PVdagMOperators.h>

NAMESPACE_BEGIN(Grid);

/**
 * A^dag = M^dag PV as a fine operator, for coarsening A_c^dag on the same
 * basis as A_c.  Op and AdjOp of PVdagMLinearOperator swapped; SloppyComms
 * forwarded, so the coarsening follows the same halo policy as A_c's.
 */
template<class Matrix,class Field>
class PVdagMAdjointOperator : public LinearOperatorBase<Field> {
  PVdagMLinearOperator<Matrix,Field> &_A;
public:
  PVdagMAdjointOperator(PVdagMLinearOperator<Matrix,Field> &A) : _A(A) {};
  void SloppyComms(int sloppy) { _A.SloppyComms(sloppy); }
  void OpDiag (const Field &in, Field &out)                  { GRID_ASSERT(0); }
  void OpDir  (const Field &in, Field &out,int dir,int disp) { GRID_ASSERT(0); }
  void OpDirAll  (const Field &in, std::vector<Field> &out)  { GRID_ASSERT(0); };
  void Op     (const Field &in, Field &out) { _A.AdjOp(in,out); }
  void AdjOp  (const Field &in, Field &out) { _A.Op(in,out); }
  void HermOpAndNorm(const Field &in, Field &out,RealD &n1,RealD &n2)
  {
    HermOp(in,out);
    ComplexD d=innerProduct(in,out);
    n1=real(d);
    n2=norm2(out);
  }
  void HermOp(const Field &in, Field &out)
  {
    Field tmp(in.Grid());
    Op(in,tmp);
    AdjOp(tmp,out);
  }
};

/**
 * The single-RHS view of a production coarse operator and its separately
 * coarsened adjoint, on 5D coarse vectors.  Both operators must already be
 * set (SetGrid) to coarseOne, the D+1 grid with one right-hand side.
 *   Op     = A_c         (A.M through slice 0)
 *   AdjOp  = A_c^dag     (Adag.M through slice 0; never A.Mdag, which is A.M)
 *   HermOp = A_c^dag A_c
 */
template<class CoarseOp>
class CoarseSliceOperator : public LinearOperatorBase<typename CoarseOp::CoarseVector> {
public:
  typedef typename CoarseOp::CoarseVector Field;
private:
  CoarseOp      &_A;
  CoarseOp      &_Adag;
  GridCartesian *_coarse5d;
  GridCartesian *_coarseOne;
  Field          _in1, _out1;
  void Apply(CoarseOp &op, const Field &in, Field &out)
  {
    InsertSliceFast(in,_in1,0,0);
    op.M(_in1,_out1);
    ExtractSliceFast(out,_out1,0,0);
  }
public:
  CoarseSliceOperator(CoarseOp &A, CoarseOp &Adag, GridCartesian *coarse5d, GridCartesian *coarseOne)
    : _A(A), _Adag(Adag), _coarse5d(coarse5d), _coarseOne(coarseOne), _in1(coarseOne), _out1(coarseOne)
  {
    GRID_ASSERT( coarseOne->_fdimensions[0] == 1 );
  };
  void OpDiag (const Field &in, Field &out)                  { GRID_ASSERT(0); }
  void OpDir  (const Field &in, Field &out,int dir,int disp) { GRID_ASSERT(0); }
  void OpDirAll  (const Field &in, std::vector<Field> &out)  { GRID_ASSERT(0); };
  void Op     (const Field &in, Field &out) { Apply(_A,in,out); }
  void AdjOp  (const Field &in, Field &out) { Apply(_Adag,in,out); }
  void HermOpAndNorm(const Field &in, Field &out,RealD &n1,RealD &n2)
  {
    HermOp(in,out);
    ComplexD d=innerProduct(in,out);
    n1=real(d);
    n2=norm2(out);
  }
  void HermOp(const Field &in, Field &out)
  {
    Field tmp(_coarse5d);
    Op(in,tmp);
    AdjOp(tmp,out);
  }
};

/**
 * |<y|A x> - <A^dag y|x>| / |<y|A x>| for random x, y (Peter's CENSUS 1).
 * Rounding level (~1e-15 in fp64) when AdjOp really is the adjoint of Op.
 */
template<class Field>
RealD AdjointMismatch(LinearOperatorBase<Field> &A, GridParallelRNG &RNG, GridBase *grid)
{
  Field x(grid), y(grid), Ax(grid), Ady(grid);
  random(RNG,x);
  random(RNG,y);
  A.Op(x,Ax);
  A.AdjOp(y,Ady);
  ComplexD l = innerProduct(y,Ax);
  ComplexD r = innerProduct(Ady,x);
  return abs(l-r)/abs(l);
}

//////////////////////////////////////////////////////////////////////
// Copied from Peter Boyle's Example_pvdagm_coherence.cc, which is a .cc
// and cannot be included.  blockGSCensus returns the two remainder lists
// instead of only printing them; the arithmetic is unchanged.
//
// Block Gram-Schmidt with remainder census: one GS pass + normalise, and
// per vector the GLOBAL remainder fraction ||v_k^perp||^2/||v_k||^2 and
// the MIN over blocks of the same ratio.  A second silent pass tightens
// orthonormality for the completeness measurements.
//////////////////////////////////////////////////////////////////////
template<class Field, class CField>
void blockGSCensus(std::vector<Field> &basis, GridBase *ccGrid, const std::string tag,
                   std::vector<RealD> &glob, std::vector<RealD> &minblock)
{
  int n = basis.size();
  CField ip(ccGrid);
  CField n0(ccGrid), n1(ccGrid);
  glob.resize(n);
  minblock.resize(n);

  for(int v=0; v<n; v++){
    RealD before = norm2(basis[v]);
    blockInnerProduct(n0, basis[v], basis[v]);
    for(int u=0; u<v; u++){
      blockInnerProductD(ip, basis[u], basis[v]);
      ip = -ip;
      blockZAXPY(basis[v], ip, basis[u], basis[v]);
    }
    RealD after = norm2(basis[v]);
    blockInnerProduct(n1, basis[v], basis[v]);

    // min over blocks of remainder fraction n1/n0
    RealD locmin = 1.0e60;
    {
      typedef typename CField::scalar_object sobj;
      for(int64_t s=0; s<ccGrid->lSites(); s++){
        Coordinate lcoor(ccGrid->_ndimension);
        ccGrid->LocalIndexToLocalCoor(s,lcoor);
        sobj s0, s1;
        peekLocalSite(s0, n0, lcoor);
        peekLocalSite(s1, n1, lcoor);
        RealD r0 = real(TensorRemove(s0));
        RealD r1 = real(TensorRemove(s1));
        if ( r0 > 0.0 ) {
          RealD frac = r1/r0;
          if (frac < locmin) locmin = frac;
        }
      }
      RealD neg = -locmin;
      ccGrid->GlobalMax(neg);
      locmin = -neg;
    }
    glob[v]     = after/before;
    minblock[v] = locmin;
    if ( locmin < 1.0e-10 )
      std::cout << GridLogMessage << "GS[" << tag << "] vec " << v
                << "  min-block remainder " << locmin << "   <== RANK LOSS FLAG" << std::endl;

    blockNormalise(ip, basis[v]);
  }
  // Second (silent) pass for numerical hygiene of downstream projections
  for(int v=0; v<n; v++){
    for(int u=0; u<v; u++){
      blockInnerProductD(ip, basis[u], basis[v]);
      ip = -ip;
      blockZAXPY(basis[v], ip, basis[u], basis[v]);
    }
    blockNormalise(ip, basis[v]);
  }
}

// P P^dag v for a block-orthonormal basis, via the block primitives.
template<class Field, class CField>
void blockPPdag(const std::vector<Field> &q, GridBase *ccGrid, const Field &v, Field &w)
{
  CField ip(ccGrid);
  w = Zero();
  for(int j=0; j<(int)q.size(); j++){
    blockInnerProduct(ip, q[j], v);
    blockZAXPY(w, ip, q[j], w);
  }
}

NAMESPACE_END(Grid);
