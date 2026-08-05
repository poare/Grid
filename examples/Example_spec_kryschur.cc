/*************************************************************************************

    Runs the Krylov-Schur algorithm on a (pre-conditioned) domain-wall fermion operator 
    to determine part of its spectrum. 

    Usage : 
      $ ./Example_spec_kryschur <Nm> <Nk> <maxiter> <Nstop> <inFile> <outDir> <?rf>

      Nm = Maximum size of approximation subspace.
      Nk = Size of truncation subspace
      maxiter = Maximum number of iterations.
      Nstop   = Stop when Nstop eigenvalues have converged. 
      inFile  = Gauge configuration to read in.
      outDir  = Directory to write output to.
      rf      = (Optional) RitzFilter to sort with. Takes in any string in 
                  {EvalNormSmall, EvalNormLarge, EvalReSmall, EvalReLarge, EvalImSmall, EvalImLarge}
    
    Output:
      ${outDir}/evals.txt  = Contains all eigenvalues. Each line is formatted as `$idx $eval $ritz`, where:
                              - $idx is the index of the eigenvalue.
                              - $eval is the eigenvalue, formated as "(re,im)".
                              - $ritz is the Ritz estimate of the eigenvalue (deviation from being a true eigenvalue)
      ${outDir}/evec${idx} = Eigenvector $idx written out in SCIDAC format (if LIME is enabled).

    Grid physics library, www.github.com/paboyle/Grid 

    Source file: ./tests/Test_padded_cell.cc

    Copyright (C) 2023

    Author: Peter Boyle <paboyle@ph.ed.ac.uk>
    Author: Patrick Oare <poare@bnl.edu>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

    See the full license in the file "LICENSE" in the top level distribution directory
    *************************************************************************************/
    /*  END LEGAL */

#include <cstdlib>

#include <Grid/Grid.h>
#include <Grid/lattice/PaddedCell.h>
#include <Grid/stencil/GeneralLocalStencil.h>

#include <Grid/algorithms/iterative/PrecGeneralisedConjugateResidual.h>
#include <Grid/algorithms/iterative/PrecGeneralisedConjugateResidualNonHermitian.h>
#include <Grid/algorithms/iterative/BiCGSTAB.h>

#include <Grid/parallelIO/IldgIOtypes.h>
#include <Grid/parallelIO/IldgIO.h>

using namespace std;
using namespace Grid;

template <class T> void writeFile(T& in, std::string const fname){  
  #ifdef HAVE_LIME
    // Ref: https://github.com/paboyle/Grid/blob/feature/scidac-wp1/tests/debug/Test_general_coarse_hdcg_phys48.cc#L111
    std::cout << Grid::GridLogMessage << "Writes to: " << fname << std::endl;
    Grid::emptyUserRecord record;
    Grid::ScidacWriter WR(in.Grid()->IsBoss());
    WR.open(fname);
    WR.writeScidacFieldRecord(in,record,0); // Lexico
    WR.close();
  #endif
}

/**
 * Writes the eigensystem of a Krylov Schur object to a directory. 
 * 
 * Parameters
 * ----------
 * std::string path
 *    Directory to write to. 
 */
template <class Field>
void writeEigensystem(KrylovSchur<Field> KS, std::string outDir) {
  int Nk = KS.getNk();
  std::cout << GridLogMessage << "Writing output to directory: " << outDir << std::endl;
  
  // Write evals
  std::string evalPath = outDir + "/evals.txt";
  std::ofstream fEval;
  fEval.open(evalPath);
  Eigen::VectorXcd evals = KS.getEvals();
  std::vector<RealD> ritz  = KS.getRitzEstimates();
  for (int i = 0; i < Nk; i++) {
    // write eigenvalues and Ritz estimates
    fEval << i << " " << evals(i) << " " << ritz[i];
    if (i < Nk - 1) { fEval << "\n"; }
  }
  fEval.close();
  
  // Write evecs (TODO: very heavy on storage costs! Don't write them all out)
  int Nevecs_write = Nk;    // only do this for one run
  std::vector<Field> evecs = KS.getEvecs();
  for (int i = 0; i < Nevecs_write; i++) {
    std::string fName = outDir + "/evec" + std::to_string(i);
    writeFile(evecs[i], fName);     // using method from Grid/HMC/ComputeWilsonFlow.cc
  }
}

// Hermitize a DWF operator by squaring it
template<class Matrix,class Field>
class SquaredLinearOperator : public LinearOperatorBase<Field> {

  public:
  Matrix &_Mat;

  public:
    SquaredLinearOperator(Matrix &Mat): _Mat(Mat) {};

    void OpDiag (const Field &in, Field &out) {    assert(0);  }
    void OpDir  (const Field &in, Field &out,int dir,int disp) {    assert(0);  }
    void OpDirAll  (const Field &in, std::vector<Field> &out){    assert(0);  };
    void Op     (const Field &in, Field &out){
      // std::cout << "Op is overloaded as HermOp" << std::endl;
      HermOp(in, out);
    }
    void AdjOp     (const Field &in, Field &out){
      HermOp(in, out);
    }
    void _Op     (const Field &in, Field &out){
      // std::cout << "Op: M "<<std::endl;
      _Mat.M(in, out);
    }
    void _AdjOp     (const Field &in, Field &out){
      // std::cout << "AdjOp: Mdag "<<std::endl;
      _Mat.Mdag(in, out);
    }
    void HermOpAndNorm(const Field &in, Field &out,RealD &n1,RealD &n2){    assert(0);  }
    void HermOp(const Field &in, Field &out){
      // std::cout << "HermOp: Mdag M Mdag M"<<std::endl;
      Field tmp(in.Grid());
      _Op(in,tmp);
      _AdjOp(tmp,out);
    }
};

template<class Matrix,class Field>
class PVdagMLinearOperator : public LinearOperatorBase<Field> {
  Matrix &_Mat;
  Matrix &_PV;
public:
  PVdagMLinearOperator(Matrix &Mat,Matrix &PV): _Mat(Mat),_PV(PV){};

  void OpDiag (const Field &in, Field &out) {    assert(0);  }
  void OpDir  (const Field &in, Field &out,int dir,int disp) {    assert(0);  }
  void OpDirAll  (const Field &in, std::vector<Field> &out){    assert(0);  };
  void Op     (const Field &in, Field &out){
    std::cout << "Op: PVdag M "<<std::endl;
    Field tmp(in.Grid());
    _Mat.M(in,tmp);
    _PV.Mdag(tmp,out);
  }
  void AdjOp     (const Field &in, Field &out){
    std::cout << "AdjOp: Mdag PV "<<std::endl;
    Field tmp(in.Grid());
    _PV.M(in,tmp);
    _Mat.Mdag(tmp,out);
  }
  void HermOpAndNorm(const Field &in, Field &out,RealD &n1,RealD &n2){    assert(0);  }
  void HermOp(const Field &in, Field &out){
    std::cout << "HermOp: Mdag PV PVdag M"<<std::endl;
    Field tmp(in.Grid());
    //    _Mat.M(in,tmp);
    //    _PV.Mdag(tmp,out);
    //    _PV.M(out,tmp);
    //    _Mat.Mdag(tmp,out);
    Op(in,tmp);
    AdjOp(tmp,out);
    //    std::cout << "HermOp done "<<norm2(out)<<std::endl;
  }
};

template<class Matrix,class Field>
class ShiftedPVdagMLinearOperator : public LinearOperatorBase<Field> {
  Matrix &_Mat;
  Matrix &_PV;
  RealD shift;
public:
  ShiftedPVdagMLinearOperator(RealD _shift,Matrix &Mat,Matrix &PV): shift(_shift),_Mat(Mat),_PV(PV){};

  void OpDiag (const Field &in, Field &out) {    assert(0);  }
  void OpDir  (const Field &in, Field &out,int dir,int disp) {    assert(0);  }
  void OpDirAll  (const Field &in, std::vector<Field> &out){    assert(0);  };
  void Op     (const Field &in, Field &out){
    std::cout << "Op: PVdag M "<<std::endl;
    Field tmp(in.Grid());
    _Mat.M(in,tmp);
    _PV.Mdag(tmp,out);
    out = out + shift * in;
  }
  void AdjOp     (const Field &in, Field &out){
    std::cout << "AdjOp: Mdag PV "<<std::endl;
    Field tmp(in.Grid());
    _PV.M(tmp,out);
    _Mat.Mdag(in,tmp);
    out = out + shift * in;
  }
  void HermOpAndNorm(const Field &in, Field &out,RealD &n1,RealD &n2){    assert(0);  }
  void HermOp(const Field &in, Field &out){
    std::cout << "HermOp: Mdag PV PVdag M"<<std::endl;
    Field tmp(in.Grid());
    Op(in,tmp);
    AdjOp(tmp,out);
  }
};

template<class Matrix, class Field>
class ShiftedComplexPVdagMLinearOperator : public LinearOperatorBase<Field> {
  Matrix &_Mat;
  Matrix &_PV;
  ComplexD shift;
public:
ShiftedComplexPVdagMLinearOperator(ComplexD _shift,Matrix &Mat,Matrix &PV): shift(_shift),_Mat(Mat),_PV(PV){};

  void OpDiag (const Field &in, Field &out) {    assert(0);  }
  void OpDir  (const Field &in, Field &out,int dir,int disp) {    assert(0);  }
  void OpDirAll  (const Field &in, std::vector<Field> &out){    assert(0);  };
  void Op     (const Field &in, Field &out){
    std::cout << "Op: PVdag M "<<std::endl;
    Field tmp(in.Grid());
    _Mat.M(in,tmp);
    _PV.Mdag(tmp,out);
    out = out + shift * in;
  }
  void AdjOp     (const Field &in, Field &out){
    std::cout << "AdjOp: Mdag PV "<<std::endl;
    Field tmp(in.Grid());
    _PV.M(tmp,out);
    _Mat.Mdag(in,tmp);
    out = out + shift * in;
  }
  void HermOpAndNorm(const Field &in, Field &out,RealD &n1,RealD &n2){    assert(0);  }
  void HermOp(const Field &in, Field &out){
    std::cout << "HermOp: Mdag PV PVdag M"<<std::endl;
    Field tmp(in.Grid());
    Op(in,tmp);
    AdjOp(tmp,out);
  }
  
  void resetShift(ComplexD newShift) {
    shift = newShift;
  }
};

int main (int argc, char ** argv)
{
  Grid_init(&argc,&argv);

  // Usage : $ ./Example_spec_kryschur <Nm> <Nk> <maaxiter> <Nstop> <inFile> <outDir>
  std::string NmStr      = argv[1];
  std::string NkStr      = argv[2];
  std::string maxIterStr = argv[3];
  std::string NstopStr   = argv[4];
  std::string file       = argv[5];
  std::string outDir     = argv[6];

  RitzFilter RF;
  if (argc == 8) {
    std::string rf       = argv[7];
    RF = selectRitzFilter(rf);
  } else {
    RF = EvalReSmall;
  }
  std::cout << GridLogMessage << "Sorting eigenvalues using " << rfToString(RF) << std::endl;
  std::cout << GridLogMessage << "Reading gauge field from: " << file << std::endl;

  const int Ls=16;
  // const int Ls = 8;

//   GridCartesian         * UGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd,vComplex::Nsimd()),GridDefaultMpi());
  std::vector<int> lat_size {16, 16, 16, 32};
  // std::vector<int> lat_size {32, 32, 32, 32};
  // std::vector<int> lat_size {8, 8, 8, 8};
  std::cout << GridLogMessage << "Lattice size: " << lat_size << std::endl;
  GridCartesian * UGrid = SpaceTimeGrid::makeFourDimGrid(lat_size, 
								          GridDefaultSimd(Nd,vComplex::Nsimd()),
								          GridDefaultMpi());
  GridRedBlackCartesian * UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);

  GridCartesian         * FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls,UGrid);
  GridRedBlackCartesian * FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGrid);
  std::cout << GridLogMessage << "Grids constructed" << std::endl;

  std::vector<int> seeds4({1,2,3,4});
  std::vector<int> seeds5({5,6,7,8});
  GridParallelRNG          RNG5(FGrid);   RNG5.SeedFixedIntegers(seeds5);
  GridParallelRNG          RNG4(UGrid);   RNG4.SeedFixedIntegers(seeds4);

  LatticeFermion    src(FGrid); random(RNG5,src);
  LatticeGaugeField Umu(UGrid);

  std::cout << GridLogMessage << "Reading configuration" << std::endl;
  FieldMetaData header;
  NerscIO::readConfiguration(Umu,header,file);

  RealD mass=0.01;
  // RealD mass=0.001;
  RealD M5=1.8;

  // DomainWallFermionD Ddwf(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5);
  // DomainWallFermionD Dpv(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,1.0,M5);
  RealD b=1.5;// Scale factor b+c=2, b-c=1
  RealD c=0.5;
  MobiusFermionD Ddwf(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,b,c);
  MobiusFermionD Dpv(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,1.0,M5,b,c);

  std::cout<<GridLogMessage<<std::endl;
  std::cout<<GridLogMessage<<"*******************************************"<<std::endl;
  std::cout<<GridLogMessage<<std::endl;

  // typedef PVdagMLinearOperator<DomainWallFermionD,LatticeFermionD> PVdagM_t;
  // typedef ShiftedPVdagMLinearOperator<DomainWallFermionD,LatticeFermionD> ShiftedPVdagM_t;
  // typedef ShiftedComplexPVdagMLinearOperator<DomainWallFermionD,LatticeFermionD> ShiftedComplexPVdagM_t;
  typedef PVdagMLinearOperator<MobiusFermionD,LatticeFermionD> PVdagM_t;
  typedef ShiftedPVdagMLinearOperator<MobiusFermionD,LatticeFermionD> ShiftedPVdagM_t;
  typedef ShiftedComplexPVdagMLinearOperator<MobiusFermionD,LatticeFermionD> ShiftedComplexPVdagM_t;

  PVdagM_t PVdagM(Ddwf, Dpv);
  ShiftedPVdagM_t ShiftedPVdagM(0.1,Ddwf,Dpv);
  // NonHermitianLinearOperator<DomainWallFermionD, LatticeFermionD> DLinOp (Ddwf);
  NonHermitianLinearOperator<MobiusFermionD, LatticeFermionD> DLinOp (Ddwf);

  int Nm = std::stoi(NmStr);
  int Nk = std::stoi(NkStr);
  int maxIter = std::stoi(maxIterStr);
  int Nstop = std::stoi(NstopStr);

  std::cout << GridLogMessage << "Runnning Krylov Schur. Nm = " << Nm << ", Nk = " << Nk << ", maxIter = " << maxIter 
                  << ", Nstop = " << Nstop << std::endl;
  
  KrylovSchur KrySchur (PVdagM, FGrid, 1e-8, RF);      // use preconditioned PV^\dag D_{dwf}
  // KrylovSchur KrySchur (DLinOp, FGrid, 1e-8, RF);         // use D_{dwf}
  KrySchur(src, maxIter, Nm, Nk, Nstop);

  std::cout<<GridLogMessage << "*******************************************" << std::endl;
  std::cout<<GridLogMessage << "***************** RESULTS *****************" << std::endl;
  std::cout<<GridLogMessage << "*******************************************" << std::endl;

  std::cout << GridLogMessage << "Krylov Schur eigenvalues: " << std::endl << KrySchur.getEvals() << std::endl;

  std::cout << GridLogMessage << "Hessenberg: " << std::endl << KrySchur.getRayleighQuotient() << std::endl;
  std::cout << GridLogMessage << "b: " << std::endl << KrySchur.getB() << std::endl;
  std::cout << GridLogMessage << "beta_k: " << std::endl << KrySchur.getBeta() << std::endl;
  // TODO actually probably need extended Rayleigh quotient, i.e. (S \\ b^\dagger) where S is the matrix printed on the line above

  // writeEigensystem(KrySchur, outDir);

  /*********************************************************************************************
   *                       OVERLAP OPERATOR SPECTRUM (arXiv:2004.07732)                        *
   *********************************************************************************************
   *
   * "Multigrid for Chiral Lattice Fermions: Domain Wall" (Brower, Clark, Weinberg, et al.,
   * arXiv:2004.07732) observes the following.  Write P for the permutation into the "permuted
   * chiral basis", i.e. the 5D transformation built out of the chiral projectors
   * P_\pm = (1 \pm \gamma_5)/2,
   *
   *      P_{s's} = | P_-  P_+   0   ...   0  |
   *                |  0   P_-  P_+  ...   0  |
   *                |            ...          |
   *                | P_+   0   ...  ...  P_- |
   *
   * which moves both domain walls onto the s = 1 slice.  In that basis, the *exactly* Pauli-
   * Villars preconditioned domain wall operator is block lower-triangular,
   *
   *      K_DW(m) = (D_PV P)^{-1} D_DW(m) P
   *              = |     D_ov(m)      0  0  ...  0 |
   *                | -(1-m) \Delta_2  1  0  ...  0 |
   *                |         ...                   |
   *                | -(1-m) \Delta_Ls 0  0  ...  1 |
   *
   * where D_PV = D_DW(1) is the Pauli-Villars operator and the (1,1) block is *exactly* the
   * finite-Ls (truncated-sign-function) overlap operator
   *
   *      D_ov(m) = (1+m)/2 + (1-m)/2 \gamma_5 \epsilon_{Ls}[H] ,
   *
   * with \epsilon_{Ls}[H] -> sign[H] as Ls -> infinity (Neuberger).  The remaining Ls-1
   * diagonal blocks are the identity, so the *bulk* (heavy) modes are mapped exactly onto unit
   * eigenvalues, i.e. onto the cutoff 1/a, independently of Ls, the lattice spacing and the
   * gauge field.
   *
   * Two immediate consequences, both of which we exploit here:
   *
   *   (i) P is unitary (it is a permutation built from orthogonal projectors), so it is a
   *       similarity transformation and drops out of the spectrum.  Therefore
   *
   *          spec( D_PV^{-1} D_DW(m) )  =  spec( D_ov(m) )  U  {1, with multiplicity (Ls-1)}
   *
   *       There is no need to construct P, or to map between 4D and 5D fields: the low end of
   *       the spectrum of the 5D operator D_PV^{-1} D_DW(m) *is* the low end of the overlap
   *       spectrum, since the bulk contamination sits far away at exactly 1.
   *
   *  (ii) Inverting D_PV is expensive, so the paper's actual multigrid target operator replaces
   *       D_PV^{-1} by the (free) D_PV^\dagger, justified by the free-field result
   *       D_PV D_PV^\dagger = 1 + O(p^4).  That is precisely the operator `PVdagM` whose
   *       spectrum was computed above.
   *
   * The paper's claim (their Fig. 2.5) is that the spectrum of D_PV^\dagger D_DW agrees well
   * with the effective overlap spectrum D_PV^{-1} D_DW at the low end, deviations being
   * confined to the large eigenvalues approaching the cutoff scale \pi/a.  Below we test this
   * directly: we run the same Krylov-Schur solve on D_PV^{-1} D_DW(m) and compare.
   *
   * NOTE: every matvec now costs a full 5D Pauli-Villars solve, so this section is *much* more
   * expensive than the D_PV^\dagger run above.  D_PV is very well conditioned (it is the mass
   * = 1 operator), so the solves are short, but there are O(Nm * maxIter) of them.
   */

  /*

  std::cout<<GridLogMessage<<std::endl;
  std::cout<<GridLogMessage<<"*******************************************"<<std::endl;
  std::cout<<GridLogMessage<<"********** OVERLAP OPERATOR SPECTRUM ******"<<std::endl;
  std::cout<<GridLogMessage<<"*******************************************"<<std::endl;

  /*

  /**
   * Applies K_ov = D_PV^{-1} D_dwf(m), the exactly Pauli-Villars preconditioned domain wall
   * operator.  Up to the similarity transformation P (which does not move the spectrum) and up
   * to the (Ls-1)-fold degenerate unit eigenvalues coming from the bulk, this operator has the
   * spectrum of the finite-Ls overlap operator D_ov(m).
   *
   * The operands are held as CayleyFermion5D<WilsonImplD> references so that this works
   * unchanged for either the DomainWallFermionD or the MobiusFermionD choice made above.
   *
   * Only Op() is implemented, since that is all Krylov-Schur (Arnoldi) requires; the Hermitian
   * and adjoint entry points would each need an extra solve against D_PV^\dagger and are not
   * used here.
   */
  /*
  class PVinvMLinearOperator : public LinearOperatorBase<LatticeFermionD> {
    typedef CayleyFermion5D<WilsonImplD>                Cayley_t;
    typedef SchurRedBlackDiagTwoSolve<LatticeFermionD>  SchurSolver_t;

    Cayley_t      &_Mat;        // D_dwf(m)
    Cayley_t      &_PV;         // D_PV = D_dwf(1)
    SchurSolver_t &_PVsolve;    // red-black preconditioned solver used to apply D_PV^{-1}

  public:
    PVinvMLinearOperator(Cayley_t &Mat, Cayley_t &PV, SchurSolver_t &PVsolve)
      : _Mat(Mat), _PV(PV), _PVsolve(PVsolve) {};

    void OpDiag (const LatticeFermionD &in, LatticeFermionD &out) {    assert(0);  }
    void OpDir  (const LatticeFermionD &in, LatticeFermionD &out,int dir,int disp) {    assert(0);  }
    void OpDirAll  (const LatticeFermionD &in, std::vector<LatticeFermionD> &out){    assert(0);  };
    void Op     (const LatticeFermionD &in, LatticeFermionD &out){
      std::cout << "Op: PV^{-1} M "<<std::endl;
      LatticeFermionD tmp(in.Grid());
      _Mat.M(in,tmp);              // tmp = D_dwf(m) in
      out = Zero();                // zero initial guess for the PV solve
      _PVsolve(_PV,tmp,out);       // out = D_PV^{-1} tmp   (exact up to the CG tolerance)
    }
    void AdjOp     (const LatticeFermionD &in, LatticeFermionD &out){    assert(0);  }
    void HermOpAndNorm(const LatticeFermionD &in, LatticeFermionD &out,RealD &n1,RealD &n2){    assert(0);  }
    void HermOp(const LatticeFermionD &in, LatticeFermionD &out){    assert(0);  }
  };

  // The Pauli-Villars solve has to be tighter than the eigensolver tolerance (1e-8), otherwise
  // the "matrix" the Arnoldi process sees is not a fixed linear operator and the Krylov
  // relation degrades.
  // RealD pvTol = 1.0e-12;
  RealD pvTol = 1.0e-8;
  int   pvMaxIter = 10000;
  ConjugateGradient<LatticeFermionD>                CG_PV(pvTol, pvMaxIter);
  SchurRedBlackDiagTwoSolve<LatticeFermionD>        SchurSolver_PV(CG_PV);
  PVinvMLinearOperator                              PVinvM(Ddwf, Dpv, SchurSolver_PV);

  std::cout << GridLogMessage << "Running Krylov Schur on D_PV^{-1} D_dwf (effective overlap). "
            << "PV solver tolerance = " << pvTol << std::endl;

  LatticeFermion srcOv(FGrid); srcOv = src;    // identical starting vector => fair comparison

  // KrylovSchur KrySchurOv (PVinvM, FGrid, 1e-8, RF);
  KrylovSchur KrySchurOv (PVinvM, FGrid, 1e-5, RF);
  KrySchurOv(srcOv, maxIter, Nm, Nk, Nstop);

  // Eigen::VectorXcd evalsPV = KrySchur.getEvals();
  // std::vector<RealD> ritzPV = KrySchur.getRitzEstimates();
  Eigen::VectorXcd evalsOv = KrySchurOv.getEvals();
  std::vector<RealD> ritzOv = KrySchurOv.getRitzEstimates();

  std::cout << GridLogMessage << "Overlap (D_PV^{-1} D_dwf) eigenvalues: " << std::endl
            << evalsOv << std::endl;

  // Write the overlap eigenvalues alongside the D_PV^\dagger ones.  We deliberately do not use
  // writeEigensystem() here: it would also dump Nk five-dimensional eigenvectors, and it would
  // overwrite the evals.txt written for the D_PV^\dagger run above.
  {
    std::string evalPathOv = outDir + "/evals_overlap.txt";
    std::cout << GridLogMessage << "Writes to: " << evalPathOv << std::endl;
    std::ofstream fEvalOv;
    fEvalOv.open(evalPathOv);
    for (int i = 0; i < Nk; i++) {
      fEvalOv << i << " " << evalsOv(i) << " " << ritzOv[i];
      if (i < Nk - 1) { fEvalOv << "\n"; }
    }
    fEvalOv.close();
  }
  */

  /*********************************************************************************************
   *                                     COMPARISON                                            *
   *********************************************************************************************
   *
   * Two views of the same data:
   *
   *   1. Index-by-index.  Both runs sorted their Ritz values with the same RitzFilter, so the
   *      i'th eigenvalue of one list should correspond to the i'th of the other *provided* the
   *      approximation has not reordered anything.
   *
   *   2. Nearest neighbour.  For each overlap eigenvalue we quote the distance to the closest
   *      D_PV^\dagger eigenvalue.  This is the robust measure: it survives any reordering, and
   *      it is small exactly when the paper's claim holds.
   */
  // std::cout<<GridLogMessage<<std::endl;

  /*

  std::cout<<GridLogMessage<<"*******************************************"<<std::endl;
  std::cout<<GridLogMessage<<"*********** SPECTRUM COMPARISON ***********"<<std::endl;
  std::cout<<GridLogMessage<<"*******************************************"<<std::endl;

  std::string cmpPath = outDir + "/spectrum_comparison.txt";
  std::ofstream fCmp;
  fCmp.open(cmpPath);
  fCmp << "# Comparison of the low spectrum of the Pauli-Villars preconditioned domain wall\n"
       << "# operator D_PV^dag D_dwf(m) against the effective overlap operator\n"
       << "# D_ov(m) ~ D_PV^{-1} D_dwf(m)   (cf. arXiv:2004.07732).\n"
       << "# Ls = " << Ls << ", mass = " << mass << ", M5 = " << M5 << ", b = " << b << ", c = " << c << "\n"
       << "# idx  lambda_ov  lambda_PVdag  |lambda_ov - lambda_PVdag|  rel  |lambda_ov - nearest_PVdag|  nearest_idx\n";

  RealD maxAbsDiffIdx  = 0.0;    // worst index-by-index absolute deviation
  RealD maxAbsDiffNear = 0.0;    // worst nearest-neighbour absolute deviation

  std::cout << GridLogMessage
            << " idx |            lambda_ov            |          lambda_PVdag           |   |diff|   |   rel    | nearest |diff| (idx)"
            << std::endl;

  for (int i = 0; i < Nk; i++) {

    ComplexD lov = evalsOv(i);
    ComplexD lpv = evalsPV(i);

    RealD absDiff = abs(lov - lpv);
    RealD relDiff = (abs(lov) > 0.0) ? absDiff / abs(lov) : 0.0;

    // Nearest D_PV^\dagger eigenvalue to this overlap eigenvalue
    int   jNear   = 0;
    RealD nearest = abs(lov - evalsPV(0));
    for (int j = 1; j < Nk; j++) {
      RealD d = abs(lov - evalsPV(j));
      if (d < nearest) { nearest = d; jNear = j; }
    }

    if (absDiff > maxAbsDiffIdx ) maxAbsDiffIdx  = absDiff;
    if (nearest > maxAbsDiffNear) maxAbsDiffNear = nearest;

    std::cout << GridLogMessage
              << std::setw(4) << i << " | " << lov << " | " << lpv << " | "
              << std::scientific << std::setprecision(3) << absDiff << " | " << relDiff << " | "
              << nearest << " (" << jNear << ")" << std::defaultfloat << std::endl;

    fCmp << i << " " << lov << " " << lpv << " " << absDiff << " " << relDiff
         << " " << nearest << " " << jNear << "\n";
  }
  fCmp.close();
  std::cout << GridLogMessage << "Wrote comparison to: " << cmpPath << std::endl;

  std::cout << GridLogMessage << "Max |lambda_ov - lambda_PVdag| (index-matched)      : "
            << maxAbsDiffIdx  << std::endl;
  std::cout << GridLogMessage << "Max |lambda_ov - lambda_PVdag| (nearest-neighbour)  : "
            << maxAbsDiffNear << std::endl;
  std::cout << GridLogMessage
            << "If arXiv:2004.07732 is right, the nearest-neighbour deviation should be tiny for "
            << "the eigenvalues nearest the origin and should grow only as |lambda| approaches "
            << "the cutoff (|lambda| ~ 1)." << std::endl;
  */

  std::cout<<GridLogMessage<<std::endl;
  std::cout<<GridLogMessage<<"*******************************************"<<std::endl;
  std::cout<<GridLogMessage<<std::endl;
  std::cout<<GridLogMessage << "Done "<< std::endl;

  Grid_finalize();
  return 0;
}
