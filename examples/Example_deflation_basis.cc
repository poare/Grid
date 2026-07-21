/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid 

    Source file: ./tests/Test_padded_cell.cc

    Copyright (C) 2023

Author: Peter Boyle <paboyle@ph.ed.ac.uk>

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

// copied here from Test_general_coarse_pvdagm.cc

#include <cstdlib>

#include <Grid/Grid.h>
#include <Grid/lattice/PaddedCell.h>
#include <Grid/stencil/GeneralLocalStencil.h>

#include <Grid/algorithms/iterative/PrecGeneralisedConjugateResidual.h>
#include <Grid/algorithms/iterative/PrecGeneralisedConjugateResidualNonHermitian.h>
#include <Grid/algorithms/iterative/BiCGSTAB.h>

using namespace std;
using namespace Grid;

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
template<class Fobj,class CComplex,int nbasis>
class MGPreconditioner : public LinearFunction< Lattice<Fobj> > {
public:
  using LinearFunction<Lattice<Fobj> >::operator();

  typedef Aggregation<Fobj,CComplex,nbasis> Aggregates;
  typedef typename Aggregation<Fobj,CComplex,nbasis>::FineField    FineField;
  typedef typename Aggregation<Fobj,CComplex,nbasis>::CoarseVector CoarseVector;
  typedef typename Aggregation<Fobj,CComplex,nbasis>::CoarseMatrix CoarseMatrix;
  typedef LinearOperatorBase<FineField>                            FineOperator;
  typedef LinearFunction    <FineField>                            FineSmoother;
  typedef LinearOperatorBase<CoarseVector>                         CoarseOperator;
  typedef LinearFunction    <CoarseVector>                         CoarseSolver;
  Aggregates     & _Aggregates;
  FineOperator   & _FineOperator;
  FineSmoother   & _PreSmoother;
  FineSmoother   & _PostSmoother;
  CoarseOperator & _CoarseOperator;
  CoarseSolver   & _CoarseSolve;

  int    level;  void Level(int lv) {level = lv; };

  MGPreconditioner(Aggregates &Agg,
		   FineOperator &Fine,
		   FineSmoother &PreSmoother,
		   FineSmoother &PostSmoother,
		   CoarseOperator &CoarseOperator_,
		   CoarseSolver &CoarseSolve_)
    : _Aggregates(Agg),
      _FineOperator(Fine),
      _PreSmoother(PreSmoother),
      _PostSmoother(PostSmoother),
      _CoarseOperator(CoarseOperator_),
      _CoarseSolve(CoarseSolve_),
      level(1)  {  }

  virtual void operator()(const FineField &in, FineField & out) 
  {
    GridBase *CoarseGrid = _Aggregates.CoarseGrid;
    //    auto CoarseGrid = _CoarseOperator.Grid();
    CoarseVector Csrc(CoarseGrid);
    CoarseVector Csol(CoarseGrid);
    FineField vec1(in.Grid());
    FineField vec2(in.Grid());

    std::cout<<GridLogMessage << "Calling PreSmoother " <<std::endl;

    //    std::cout<<GridLogMessage << "Calling PreSmoother input residual "<<norm2(in) <<std::endl;
    double t;
    // Fine Smoother
    //    out = in;
    out = Zero();
    t=-usecond();
    _PreSmoother(in,out);
    t+=usecond();

    std::cout<<GridLogMessage << "PreSmoother took "<< t/1000.0<< "ms" <<std::endl;

    // Update the residual
    _FineOperator.Op(out,vec1);  sub(vec1, in ,vec1);   
    //    std::cout<<GridLogMessage <<"Residual-1 now " <<norm2(vec1)<<std::endl;

    // Fine to Coarse 
    t=-usecond();
    _Aggregates.ProjectToSubspace  (Csrc,vec1);
    t+=usecond();
    std::cout<<GridLogMessage << "Project to coarse took "<< t/1000.0<< "ms" <<std::endl;

    // Coarse correction
    t=-usecond();
    Csol = Zero();
    _CoarseSolve(Csrc,Csol);
    //Csol=Zero();
    t+=usecond();
    std::cout<<GridLogMessage << "Coarse solve took "<< t/1000.0<< "ms" <<std::endl;

    // Coarse to Fine
    t=-usecond();  
    //    _CoarseOperator.PromoteFromSubspace(_Aggregates,Csol,vec1);
    _Aggregates.PromoteFromSubspace(Csol,vec1); 
    add(out,out,vec1);
    t+=usecond();
    std::cout<<GridLogMessage << "Promote to this level took "<< t/1000.0<< "ms" <<std::endl;

    // Residual
    _FineOperator.Op(out,vec1);  sub(vec1 ,in , vec1);  
    //    std::cout<<GridLogMessage <<"Residual-2 now " <<norm2(vec1)<<std::endl;

    // Fine Smoother
    t=-usecond();
    //    vec2=vec1;
    vec2=Zero();
    _PostSmoother(vec1,vec2);
    t+=usecond();
    std::cout<<GridLogMessage << "PostSmoother took "<< t/1000.0<< "ms" <<std::endl;

    add( out,out,vec2);
    std::cout<<GridLogMessage << "Done " <<std::endl;
  }
};

/**
 * Computes the coefficients in the Krylov expansion for 1/D ~ \sum_{i=0}^N c_i D^i. 
 * 
 * Parameters
 * ----------
 * std::vector<double> &coeffs
 *    Polynomial coeffients to return, with indexing order (c_0, c_1, c_2, ..., c_n). 
 * LinearOperatorBase<FineField> &DiracOp
 *    Dirac operator D. 
 * FineField src
 *    Source field b. 
 * FineField psiStar
 *    Output approximation for D^{-1} b coming from a Krylov method. 
 * int N
 *    Dimension of the polynomial approximation (Krylov space K_{N-1} = {b, Db, D^2 b, ..., D^{N-1} b}).
 */
void poly_coeffs(std::vector<std::complex<double>> &coeffs, LinearOperatorBase<LatticeFermion> &DiracOp, LatticeFermion src, 
  LatticeFermion psiStar, GridCartesian* FGrid, int N, bool use_herm = false)
{
  // stdBasis = {b, Db, D^2 b, ..., D^N b}, kryBasis = {k0, k1, ..., kN}
  std::vector<LatticeFermion> kryBasis;
  Eigen::VectorXcd psiStarCoeffs (N);

  // Normalize by 1 / ||src||; does not change the polynomial coefficients
  double srcNorm   = 1 / std::sqrt(norm2(src));
  kryBasis.push_back(srcNorm * src);                // normalized source
  psiStar          = srcNorm * psiStar;
  psiStarCoeffs(0) = innerProduct(kryBasis[0], psiStar);

  // orthonormalize canonical Krylov basis {b, Db, D^2 b, ..., D^{N-1} b} <--> {k_i} and compute components <k_i | psi*>
  LatticeFermion tmp (FGrid);
  for (int i = 0; i < N - 1; i++) {               // construct ONB for {b, Db, ..., D^{i+1} b}
    if (use_herm) {
      DiracOp.HermOp(kryBasis.back(), tmp);         // tmp \in span{(D^\dag D)^{i+1} b} \oplus span{(D^\dag D)^i b, ..., D^\dag D b, b}
    } else {
      DiracOp.Op(kryBasis.back(), tmp);             // tmp \in span{D^{i+1} b} \oplus span{D^i b, ..., Db, b}
    }

    for (int j = 0; j < i+1; j++) {               // orthogonalize tmp with previous basis vectors
      ComplexD coeff = innerProduct(kryBasis[j], tmp);      // <k_j | tmp>
      tmp -= coeff * kryBasis[j];                           // subtract off |k_j><k_j | tmp>; now tmp is perp to |k_j>
    }
    double tmpNorm = 1 / std::sqrt(norm2(tmp));
      kryBasis.push_back(
      tmpNorm * tmp
    );                                                      // normalize |k_i> and add to kryBasis
    psiStarCoeffs(i+1) = innerProduct(kryBasis[i+1], psiStar);  // compute < k_i | psi* >
  }

  // To verify the basis is ONB
  // for (int i = 0; i < N; i++) {
  //   for (int j = 0; j < N; j++) {
  //     std::cout << "<ki|kj> for (i,j) = (" << i << ", " << j << ") = "  << innerProduct(kryBasis[i], kryBasis[j]) << std::endl;
  //   }
  // }

  // Compute change of basis matrix
  LatticeFermion tmp2 (FGrid);
  Eigen::MatrixXcd M = Eigen::MatrixXcd::Zero(N, N);
  tmp = kryBasis[0];       // current Krylov vector; starts with tmp = src (normalized)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < i + 1; j++) {    // fill column with components of kryVec. Only need j <= i to get orthonormal components
      M(j, i) = innerProduct(kryBasis[j], tmp);
    }    
    if (use_herm) {     // tmp --> D^\dag D(tmp)
      DiracOp.HermOp(tmp, tmp2);
      tmp = tmp2;
    } else {      // tmp --> D(tmp). Note that DiracOp.Op(tmp, tmp) will cause a bug
      DiracOp.Op(tmp, tmp2);
      tmp = tmp2;
    }
  }

  // Compute M^{-1} @ psiStarCoeffs and copy to coeffs
  Eigen::VectorXcd res (N);
  res = M.inverse() * psiStarCoeffs;
  for (int i = 0; i < N; i++) {
    coeffs[i] = res(i);
  }

}

template<class Field>
void TestSubspace(LinearOperatorBase<Field> &DiracOp, Field &psi, Field &noise, std::vector<std::complex<double>> &coeffs, GridBase* FineGrid, int maxit, std::string path, bool standard = true) 
{

    std::cout << "Writing to file " << path << std::endl;
    PolynomialFile PF;
    // PF.data.push_back(coeffs1);
    // PF.data.push_back(coeffs2);

    RealD scale;
    TrivialPrecon<Field> simple_fine;
    // 10 max iterations, no restarts (max inner iters = 20 I think)
    // PrecGeneralisedConjugateResidualNonHermitian<Field> GCR(0.001, 30, DiracOp, simple_fine, 12, 12);
    // PrecGeneralisedConjugateResidualNonHermitian<Field> GCR(1e-8, 1, DiracOp, simple_fine, maxit, maxit);
    // PGCRPolynomial<Field> GCR(1e-8, 1, DiracOp, simple_fine, maxit, maxit);
    PGCRPolynomial<Field> GCR(1e-8, 1, DiracOp, simple_fine, maxit+1, maxit, PF);
    Field src(FineGrid);
    Field guess(FineGrid);
    Field Mn(FineGrid);

    psi = Zero();
    // gaussian(RNG,noise);
    // scale = std::pow(norm2(noise),-0.5); 
    // noise=noise*scale;
    
    // Mn = D * noise, b = <noise | D | noise>. This tells us how null the vector is-- how good of an approx we have
    DiracOp.Op(noise, Mn); std::cout<<GridLogMessage << "noise <n|Op|n> "<<innerProduct(noise,Mn)<<std::endl;
  
    // First option is Peter's usual method (invert on noise), second option is Christoph's (invert on zero, starting at guess)
    if (standard) {
        std::cout << GridLogMessage << " inverting on noise "<<std::endl;
        src = noise;
        guess=Zero();
        GCR(src, guess);   // solve phi = D^{-1} src to tol 0.001, then iterates this. Solves D^{-3} src, where src ~ Gaussian noise
        psi = guess;
    } else {
        std::cout << GridLogMessage << " inverting on zero "<<std::endl;
        src=Zero();
        guess = noise;
        GCR(src,guess);
        psi = guess;
    }
    // scale = std::pow(norm2(psi),-0.5); 
    // psi = scale * psi;

    DiracOp.Op(psi, Mn); std::cout<<GridLogMessage << "filtered <f|Op|f> "<<innerProduct(psi, Mn)<<std::endl;

    coeffs = GCR.polynomial;

    {
      XmlWriter XW (path);
      write(XW, "PolynomialFile", PF);
    }
}

int main (int argc, char ** argv)
{
  Grid_init(&argc,&argv);

  const int Ls=16;

  assert(argc > 1 && "Must pass in N, the max number of iterations for GCR.");
  // std::cout << "Running with N = " << argv[1] << std::endl;
  int maxit = atoi(argv[1]);
  std::cout << "Running with maxit = " << maxit << std::endl;

//   GridCartesian         * UGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd,vComplex::Nsimd()),GridDefaultMpi());
  std::vector<int> lat_size {16, 16, 16, 32};
  std::cout << "Lattice size: " << lat_size << std::endl;
  GridCartesian * UGrid = SpaceTimeGrid::makeFourDimGrid(lat_size, 
								          GridDefaultSimd(Nd,vComplex::Nsimd()),
								          GridDefaultMpi());
  GridRedBlackCartesian * UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);

  GridCartesian         * FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls,UGrid);
  GridRedBlackCartesian * FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGrid);

  // Construct a coarsened grid
  // poare TODO: replace this with the following line?
  Coordinate clatt = lat_size;
//   Coordinate clatt = GridDefaultLatt();              // [PO] initial line before I edited it
  for(int d=0;d<clatt.size();d++){
    clatt[d] = clatt[d]/2;
    //    clatt[d] = clatt[d]/4;
  }
  GridCartesian *Coarse4d =  SpaceTimeGrid::makeFourDimGrid(clatt, GridDefaultSimd(Nd,vComplex::Nsimd()),GridDefaultMpi());;
  GridCartesian *Coarse5d =  SpaceTimeGrid::makeFiveDimGrid(1,Coarse4d);

  std::vector<int> seeds4({1,2,3,4});
  std::vector<int> seeds5({5,6,7,8});
  std::vector<int> cseeds({5,6,7,8});
  GridParallelRNG          RNG5(FGrid);   RNG5.SeedFixedIntegers(seeds5);
  GridParallelRNG          RNG4(UGrid);   RNG4.SeedFixedIntegers(seeds4);
  GridParallelRNG          CRNG(Coarse5d);CRNG.SeedFixedIntegers(cseeds);

  LatticeFermion    src(FGrid); random(RNG5,src);
  LatticeFermion result(FGrid); result=Zero();
  LatticeFermion    ref(FGrid); ref=Zero();
  LatticeFermion    tmp(FGrid);
  LatticeFermion    err(FGrid);
  LatticeGaugeField Umu(UGrid);

  FieldMetaData header;
//   std::string file("ckpoint_lat.4000");
  std::string file("/Users/patrickoare/libraries/PETSc-Grid/ckpoint_lat.4000");
  NerscIO::readConfiguration(Umu,header,file);

  std::string dir ("/Users/patrickoare/Dropbox (MIT)/research/multigrid/gcr_coeffs/run1/");
  
  RealD mass=0.01;
  RealD M5=1.8;

  DomainWallFermionD Ddwf(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5);
  DomainWallFermionD Dpv(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,1.0,M5);

  // const int nbasis = 20;            // size of approximate basis for low-mode space
  const int nbasis = 3;            // size of approximate basis for low-mode space
  const int cb = 0 ;
  LatticeFermion prom(FGrid);

  typedef GeneralCoarsenedMatrix<vSpinColourVector,vTComplex,nbasis> LittleDiracOperator;
  typedef LittleDiracOperator::CoarseVector CoarseVector;

  NextToNearestStencilGeometry5D geom(Coarse5d);

  std::cout<<GridLogMessage<<std::endl;
  std::cout<<GridLogMessage<<"*******************************************"<<std::endl;
  std::cout<<GridLogMessage<<std::endl;

  typedef PVdagMLinearOperator<DomainWallFermionD,LatticeFermionD> PVdagM_t;
  typedef ShiftedPVdagMLinearOperator<DomainWallFermionD,LatticeFermionD> ShiftedPVdagM_t;
  PVdagM_t PVdagM(Ddwf, Dpv);
  //  ShiftedPVdagM_t ShiftedPVdagM(2.0,Ddwf,Dpv); // 355
  //  ShiftedPVdagM_t ShiftedPVdagM(1.0,Ddwf,Dpv); // 246
  //  ShiftedPVdagM_t ShiftedPVdagM(0.5,Ddwf,Dpv); // 183
  //  ShiftedPVdagM_t ShiftedPVdagM(0.25,Ddwf,Dpv); // 145
  //  ShiftedPVdagM_t ShiftedPVdagM(0.1,Ddwf,Dpv); // 134
  //  ShiftedPVdagM_t ShiftedPVdagM(0.1,Ddwf,Dpv); // 127 -- NULL space via inverse iteration
//    ShiftedPVdagM_t ShiftedPVdagM(0.1,Ddwf,Dpv); // 57 -- NULL space via inverse iteration; 3 iterations
  //  ShiftedPVdagM_t ShiftedPVdagM(0.25,Ddwf,Dpv); // 57 , tighter inversion
  //  ShiftedPVdagM_t ShiftedPVdagM(0.25,Ddwf,Dpv); // nbasis 20 -- 49 iters
  //  ShiftedPVdagM_t ShiftedPVdagM(0.25,Ddwf,Dpv); // nbasis 20 -- 70 iters; asymmetric 
  //  ShiftedPVdagM_t ShiftedPVdagM(0.25,Ddwf,Dpv); // 58; Loosen coarse, tighten fine
  //  ShiftedPVdagM_t ShiftedPVdagM(0.1,Ddwf,Dpv); // 56 ... 
  //  ShiftedPVdagM_t ShiftedPVdagM(0.1,Ddwf,Dpv); // 51 ...  with 24 vecs
  //  ShiftedPVdagM_t ShiftedPVdagM(0.1,Ddwf,Dpv); // 31 ...  with 24 vecs and 2^4 blocking
  //  ShiftedPVdagM_t ShiftedPVdagM(0.1,Ddwf,Dpv); // 43 ...  with 16 vecs and 2^4 blocking, sloppier
  //  ShiftedPVdagM_t ShiftedPVdagM(0.1,Ddwf,Dpv); // 35  ...  with 20 vecs and 2^4 blocking
  //  ShiftedPVdagM_t ShiftedPVdagM(0.1,Ddwf,Dpv); // 35  ...  with 20 vecs and 2^4 blocking, looser coarse
    ShiftedPVdagM_t ShiftedPVdagM(0.1,Ddwf,Dpv); // 64  ...  with 20 vecs, Christoph setup, and 2^4 blocking, looser coarse
//   ShiftedPVdagM_t ShiftedPVdagM(0.01,Ddwf,Dpv); // 

   PowerMethod<LatticeFermion> PM; PM(PVdagM, src);
 
  // Warning: This routine calls PVdagM.Op, not PVdagM.HermOp
//   typedef Aggregation<vSpinColourVector,vTComplex,nbasis> Subspace;
//   Subspace AggregatesPD(Coarse5d,FGrid,cb);

//   // create low-mode subspace with nbasis (default 20) vectors (generate \phi_k's) (takes a while)
//   AggregatesPD.CreateSubspaceGCR(RNG5,
// 				 PVdagM,
// 				 nbasis);

  // TODO move Aggregates into here
//   std::vector<LatticeFermionD> subspace1 (nbasis); std::vector<LatticeFermionD> subspace2 (nbasis);

  LatticeFermionD noise1(FGrid);
  gaussian(RNG5, noise1);
  noise1 = std::pow(norm2(noise1),-0.5) * noise1;

  LatticeFermionD noise2(FGrid);
  gaussian(RNG5, noise2);
  noise2 = std::pow(norm2(noise2),-0.5) * noise2;

  LatticeFermionD psi1 (FGrid); LatticeFermionD psi2 (FGrid);
  // std::vector<std::complex<double>> coeffs1 (maxit);
  // std::vector<std::complex<double>> coeffs2 (maxit + 1);
  std::vector<std::complex<double>> coeffs1;
  std::vector<std::complex<double>> coeffs2;

  // std::string path;

  // // int maxit = 10;
  // // int maxit = 3;
  // int maxit;
  // // std::vector<int> Nlst {3, 4, 5};
  // std::vector<int> Nlst {6, 7, 8};
  // for (int i = 0; i < Nlst.size(); i++)
  // {
  //   maxit = Nlst[i];
  //   std::cout << "Running maxit = " << maxit << std::endl;

  std::string path1 = dir + "invert_" + std::to_string(maxit) + ".xml";
  std::string path2 = dir + "relax_" + std::to_string(maxit) + ".xml";
  // std::cout << "Writing coeffs to file at " << path << std::endl;

  // TestSubspace<LatticeFermionD>(PVdagM, psi1, noise1, coeffs1, FGrid, maxit, path1, true);      // standard
  // TestSubspace<LatticeFermionD>(PVdagM, psi2, noise2, coeffs2, FGrid, maxit, path2, false);     // christoph

  // std::vector<std::complex<double>> coeffs1p (maxit);
  // std::vector<std::complex<double>> coeffs2p (maxit);
  // poly_coeffs(coeffs1p, PVdagM, noise1, psi1, FGrid, maxit);
  // poly_coeffs(coeffs2p, PVdagM, noise2, psi2, FGrid, maxit);       // get coeffs in K_{n+1}(D, b) \supset K_n(D, Db). First coeff should be 0

  // PVdagM.Op(noise2, noise2);               // init resid for this is D(noise2)
  // poly_coeffs(coeffs2, PVdagM, noise2, psi2, FGrid, maxit);        // get coeffs in K_n(D, Db)
  
  std::cout << GridLogMessage << "*******************************************" << std::endl;
  std::cout << GridLogMessage << "******* STANDARD METHOD COEFFICIENTS ******" << std::endl;
  std::cout << GridLogMessage << "*******************************************" << std::endl;
  std::cout << GridLogMessage << coeffs1 << std::endl;
  // std::cout << GridLogMessage << coeffs1p << std::endl;
  
  std::cout << GridLogMessage << "*******************************************" << std::endl;
  std::cout << GridLogMessage << "****** CHRISTOPH METHOD COEFFICIENTS ******" << std::endl;
  std::cout << GridLogMessage << "*******************************************" << std::endl;
  std::cout << GridLogMessage << coeffs2 << std::endl;
  // std::cout << GridLogMessage << coeffs2p << std::endl;

  // write coeffs to file
  // path = dir + "N_" + std::to_string(maxit) + ".xml";
  // std::cout << "Writing coeffs to file at " << path << std::endl;
  // PolynomialFile PF;
  // PF.data.push_back(coeffs1);
  // PF.data.push_back(coeffs2);

  // {
  //   XmlWriter XW (path);
  //   write(XW, "PolynomialFile", PF);
  // }

  // }

  std::cout<<GridLogMessage<<std::endl;
  std::cout<<GridLogMessage<<"*******************************************"<<std::endl;
  std::cout<<GridLogMessage<<std::endl;
  std::cout<<GridLogMessage << "Done "<< std::endl;

  Grid_finalize();
  return 0;
}
