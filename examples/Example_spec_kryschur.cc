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
      ${outDir}/rayleigh.txt = Rayleigh quotient R of the Krylov-Schur factorization
                              D U = U R + u b^\dagger. Each line is `$i $j $re $im`.
      ${outDir}/bvec.txt   = The b vector of the same factorization, one `$i $re $im` per line.
                              Header carries k, norm2(u) and beta_k. Together with rayleigh.txt
                              this is the input to the offline pseudospectra bound
                              s(z) = sigma_min([z I - R ; b^\dagger]) >= sigma_min(z I - D).

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
#include <iomanip>

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
 * Writes the Krylov-Schur factorization D U = U R + u b^\dag to a directory, for offline
 * pseudospectra post-processing. The inner bound
 *
 *    s(z) = sigma_min( [ z I - R ; b^\dag ] )  >=  sigma_min(z I - D)
 *
 * needs R and b only; the basis U never enters, so nothing lattice-sized is written.
 *
 * Parameters
 * ----------
 * KrylovSchur<Field> KS
 *    Krylov-Schur object, after the run has completed.
 * std::string outDir
 *    Directory to write to.
 * RealD normU
 *    norm2(KS.getU()), computed by the caller. norm2 is a global reduction and so must be
 *    called on every rank, whereas this function is boss-only.
 */
template <class Field>
void writeKSFactorization(KrylovSchur<Field> KS, std::string outDir, RealD normU) {

  Eigen::MatrixXcd R = KS.getRayleighQuotient();
  Eigen::VectorXcd b = KS.getB();
  int k = R.rows();

  std::cout << GridLogMessage << "Writing Krylov-Schur factorization, k = " << k
            << ", norm2(u) = " << normU << std::endl;

  std::string rPath = outDir + "/rayleigh.txt";
  std::ofstream fR;
  fR.open(rPath);
  fR << std::scientific << std::setprecision(16);
  fR << "# Krylov-Schur Rayleigh quotient R (Schur form, upper triangular)\n";
  fR << "# k = " << k << "\n";
  fR << "# columns: i j Re Im\n";
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < k; j++) {
      fR << i << " " << j << " " << R(i,j).real() << " " << R(i,j).imag() << "\n";
    }
  }
  fR.close();

  std::string bPath = outDir + "/bvec.txt";
  std::ofstream fB;
  fB.open(bPath);
  fB << std::scientific << std::setprecision(16);
  fB << "# Krylov-Schur b vector\n";
  fB << "# k = " << k << "\n";
  fB << "# norm2_u = " << normU << "\n";
  fB << "# beta_k = " << KS.getBeta() << "\n";
  fB << "# columns: i Re Im\n";
  for (int i = 0; i < k; i++) {
    fB << i << " " << b(i).real() << " " << b(i).imag() << "\n";
  }
  fB.close();
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

  //  Dump R and b for the offline pseudospectra bound, i.e. the extended Rayleigh quotient
  //  (R \\ b^\dagger). norm2 is collective, so it is evaluated on every rank before the
  //  boss-only write.
  {
    LatticeFermionD uvec = KrySchur.getU();
    RealD normU = norm2(uvec);
    if (FGrid->IsBoss()) { writeKSFactorization(KrySchur, outDir, normU); }
  }

  // writeEigensystem(KrySchur, outDir);

  std::cout<<GridLogMessage<<std::endl;
  std::cout<<GridLogMessage<<"*******************************************"<<std::endl;
  std::cout<<GridLogMessage<<std::endl;
  std::cout<<GridLogMessage << "Done "<< std::endl;

  Grid_finalize();
  return 0;
}
