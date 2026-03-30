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
#include <Grid/Grid.h>
#include <Grid/lattice/PaddedCell.h>
#include <Grid/stencil/GeneralLocalStencil.h>

#include <Grid/algorithms/iterative/PrecGeneralisedConjugateResidual.h>
#include <Grid/algorithms/iterative/PrecGeneralisedConjugateResidualNonHermitian.h>
#include <Grid/algorithms/iterative/BiCGSTAB.h>

using namespace std;
using namespace Grid;

template <class T> void readFile(T& out, std::string const fname){  
  #ifdef HAVE_LIME
    // Ref: https://github.com/paboyle/Grid/blob/feature/scidac-wp1/tests/debug/Test_general_coarse_hdcg_phys48.cc#L111
    std::cout << Grid::GridLogMessage << "Reads at: " << fname << std::endl;
    Grid::emptyUserRecord record;
    // Grid::ScidacReader SR(out.Grid()->IsBoss());
    Grid::ScidacReader SR;
    SR.open(fname);
    SR.readScidacFieldRecord(out, record);
    SR.close();
  #endif
}

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
    //    std::cout << GridLogMessage<< "Op: PVdag M "<<std::endl;
    Field tmp(in.Grid());
    _Mat.M(in,tmp);
    _PV.Mdag(tmp,out);
  }
  void AdjOp     (const Field &in, Field &out){
    //    std::cout << GridLogMessage<<"AdjOp: Mdag PV "<<std::endl;
    Field tmp(in.Grid());
    _PV.M(in,tmp);
    _Mat.Mdag(tmp,out);
  }
  void HermOpAndNorm(const Field &in, Field &out,RealD &n1,RealD &n2){
    HermOp(in,out);
    ComplexD dot = innerProduct(in,out);
    n1=real(dot);
    n2=norm2(out);
  }
  void HermOp(const Field &in, Field &out){
    //    std::cout <<GridLogMessage<< "HermOp: Mdag PV PVdag M"<<std::endl;
    Field tmp(in.Grid());
    Op(in,tmp);
    AdjOp(tmp,out);
    //    std::cout << "HermOp done "<<norm2(out)<<std::endl;
  }
};
template<class Matrix,class Field>
class MdagPVLinearOperator : public LinearOperatorBase<Field> {
  Matrix &_Mat;
  Matrix &_PV;
public:
  MdagPVLinearOperator(Matrix &Mat,Matrix &PV): _Mat(Mat),_PV(PV){};

  void OpDiag (const Field &in, Field &out) {    assert(0);  }
  void OpDir  (const Field &in, Field &out,int dir,int disp) {    assert(0);  }
  void OpDirAll  (const Field &in, std::vector<Field> &out){    assert(0);  };
  void Op     (const Field &in, Field &out){
    Field tmp(in.Grid());
    //    std::cout <<GridLogMessage<< "Op: PVdag M "<<std::endl;
    _PV.M(in,tmp);
    _Mat.Mdag(tmp,out);
  }
  void AdjOp     (const Field &in, Field &out){
    //    std::cout <<GridLogMessage<< "AdjOp: Mdag PV "<<std::endl;
    Field tmp(in.Grid());
    _Mat.M(in,tmp);
    _PV.Mdag(tmp,out);
  }
  void HermOpAndNorm(const Field &in, Field &out,RealD &n1,RealD &n2){
    ComplexD dot = innerProduct(in,out);
    n1=real(dot);
    n2=norm2(out);
  }
  void HermOp(const Field &in, Field &out){
    //    std::cout << GridLogMessage<<"HermOp: PVdag M Mdag PV "<<std::endl;
    Field tmp(in.Grid());
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
    //    std::cout << "Op: PVdag M "<<std::endl;
    Field tmp(in.Grid());
    _Mat.M(in,tmp);
    _PV.Mdag(tmp,out);
    out = out + shift * in;
  }
  void AdjOp     (const Field &in, Field &out){
    //    std::cout << "AdjOp: Mdag PV "<<std::endl;
    Field tmp(in.Grid());
    _PV.M(tmp,out);
    _Mat.Mdag(in,tmp);
    out = out + shift * in;
  }
  void HermOpAndNorm(const Field &in, Field &out,RealD &n1,RealD &n2){    assert(0);  }
  void HermOp(const Field &in, Field &out){
    //    std::cout << "HermOp: Mdag PV PVdag M"<<std::endl;
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

int main (int argc, char ** argv)
{
  Grid_init(&argc,&argv);

  const int Ls=16;

  std::vector<int> lat_size {16, 16, 16, 32};

  GridCartesian         * UGrid   = SpaceTimeGrid::makeFourDimGrid(lat_size, GridDefaultSimd(Nd,vComplex::Nsimd()),GridDefaultMpi());
  GridRedBlackCartesian * UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);

  GridCartesian         * FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls,UGrid);
  GridRedBlackCartesian * FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGrid);

  // Construct a coarsened grid
  // Coordinate clatt = GridDefaultLatt();
  Coordinate clatt = lat_size;
  for(int d=0;d<clatt.size();d++){
    clatt[d] = clatt[d]/2;
    //    clatt[d] = clatt[d]/4;
  }
  GridCartesian *Coarse4d =  SpaceTimeGrid::makeFourDimGrid(clatt, GridDefaultSimd(Nd,vComplex::Nsimd()),GridDefaultMpi());
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
  // std::string file("ckpoint_lat.4000");
  std::string file("/sdcc/u/poare/PETSc-Grid/ckpoint_lat.4000");
  NerscIO::readConfiguration(Umu,header,file);

  std::string eigenPath = "/sdcc/u/poare/lqcd/multigrid/spectra/ckpoint_lat.4000/PVdagM_Nm64_Nk32_334102/";

  std::cout << GridLogMessage << "Loading eigenvalues" << std::endl;
  std::ifstream evalFile(eigenPath + "evals.txt");
  std::string str;
  std::vector<ComplexD> evals;
  while (std::getline(evalFile, str)) {
      std::cout << GridLogMessage << "Reading line: " << str << std::endl;
      int i1 = str.find("(") + 1;
      int i2 = str.find(",") + 1;
      int i3 = str.find(")");
      std::cout << "i1,i2,i3 = " << i1 << "," << i2 << "," << i3 << std::endl;
      std::string reStr = str.substr(i1, i2 - i1);
      std::string imStr = str.substr(i2, i3 - i2);
      std::cout << GridLogMessage << "Parsed re = " << reStr << " and im = " << imStr << std::endl;
      // ComplexD z (std::stof(reStr), std::stof(imStr));
      ComplexD z (std::stod(reStr), std::stod(imStr));
      evals.push_back(z);
  }
  std::cout << GridLogMessage << "Eigenvalues: " << evals << std::endl;

  int Nevecs = 32;
  std::vector<LatticeFermion> evecs;
  LatticeFermion evec (FGrid);
  for (int i = 0; i < Nevecs; i++) {
    std::string evecPath = eigenPath + "evec" + std::to_string(i);
    readFile(evec, evecPath);
    evecs.push_back(evec);
  }
  std::cout << GridLogMessage << "Evecs loaded" << std::endl;
  
  RealD mass=0.01;
  RealD M5=1.8;

  DomainWallFermionD Ddwf(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5);
  DomainWallFermionD Dpv(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,1.0,M5);

  // const int nbasis = 30;
  // const int nbasis = 20;
  const int nbasis = 40;
  const int cb = 0 ;
  LatticeFermion prom(FGrid);

  std::cout << GridLogMessage << "nbasis: " << nbasis << std::endl;

  // assert(nbasis <= Nevecs);     // need to have enough evecs

  typedef GeneralCoarsenedMatrix<vSpinColourVector,vTComplex,nbasis> LittleDiracOperator;
  typedef LittleDiracOperator::CoarseVector CoarseVector;

  NextToNearestStencilGeometry5D geom(Coarse5d);

  std::cout<<GridLogMessage<<std::endl;
  std::cout<<GridLogMessage<<"*******************************************"<<std::endl;
  std::cout<<GridLogMessage<<std::endl;

  typedef PVdagMLinearOperator<DomainWallFermionD,LatticeFermionD> PVdagM_t;
  typedef MdagPVLinearOperator<DomainWallFermionD,LatticeFermionD> MdagPV_t;
  typedef ShiftedPVdagMLinearOperator<DomainWallFermionD,LatticeFermionD> ShiftedPVdagM_t;
  PVdagM_t PVdagM(Ddwf,Dpv);
  MdagPV_t MdagPV(Ddwf,Dpv);
  //  ShiftedPVdagM_t ShiftedPVdagM(2.0,Ddwf,Dpv); // 355
  //  ShiftedPVdagM_t ShiftedPVdagM(1.0,Ddwf,Dpv); // 246
  //  ShiftedPVdagM_t ShiftedPVdagM(0.5,Ddwf,Dpv); // 183
  //  ShiftedPVdagM_t ShiftedPVdagM(0.25,Ddwf,Dpv); // 145
  //  ShiftedPVdagM_t ShiftedPVdagM(0.1,Ddwf,Dpv); // 134
  //  ShiftedPVdagM_t ShiftedPVdagM(0.1,Ddwf,Dpv); // 127 -- NULL space via inverse iteration
  //  ShiftedPVdagM_t ShiftedPVdagM(0.1,Ddwf,Dpv); // 57 -- NULL space via inverse iteration; 3 iterations
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
  //  ShiftedPVdagM_t ShiftedPVdagM(0.1,Ddwf,Dpv); // 64  ...  with 20 vecs, Christoph setup, and 2^4 blocking, looser coarse
  ShiftedPVdagM_t ShiftedPVdagM(0.01,Ddwf,Dpv); // 

  // Run power method on HOA??
  PowerMethod<LatticeFermion>       PM;
  //  PM(PVdagM,src);
  //  PM(MdagPV,src);

  // Warning: This routine calls PVdagM.Op, not PVdagM.HermOp
  typedef Aggregation<vSpinColourVector,vTComplex,nbasis> Subspace;
  Subspace AggregatesPD(Coarse5d,FGrid,cb);

  // Breeds right singular vectors with call to HermOp (V)
  // int chebyOrd = 500;
  // V.CreateSubspaceChebyshev(RNG5,PVdagM,
	// 		    nbasis,
	// 		    4000.0,0.003,
	// 		    chebyOrd);
  // AggregatesPD.CreateSubspaceChebyshev(RNG5,
	// 			       PVdagM,
	// 			       nbasis,
	// 			       4000.0,
	// 			       0.003,
	// 			       chebyOrd);
  
  // Subspace testing

  // AggregatesPD.CreateSubspace(RNG5,PVdagM,nbasis);      // Note that CG will automatically call PVdagM.HermOp
  // - nbasis = 20, 36 outer iterations
  // - nbasis = 40, 24 outer iterations
  // - the nbasis = 40 result is the one that Peter had with nbasis = 20

  AggregatesPD.CreateSubspaceGCR(RNG5,PVdagM,nbasis);
  // - nbasis = 20, 35 outer iterations
  // - nbasis = 40, 23 outer iterations

  // AggregatesPD.CreateSubspaceMultishift(RNG5, PVdagM, 0.003, 1e-4, 2000);

  // TODO see if I also want to generate left-singular vectors
  // Subspace LAggregatesPD (Coarse5d, FGrid, cb);

  
  LittleDiracOperator LittleDiracOpPV(geom,FGrid,Coarse5d);
  LittleDiracOpPV.CoarsenOperator(PVdagM,AggregatesPD);

  std::cout<<GridLogMessage<<std::endl;
  std::cout<<GridLogMessage<<"*******************************************"<<std::endl;
  std::cout<<GridLogMessage<<std::endl;
  std::cout<<GridLogMessage<<"Testing coarsened operator "<<std::endl;
  
  CoarseVector c_src (Coarse5d);
  CoarseVector c_res (Coarse5d);
  CoarseVector c_proj(Coarse5d);

  std::vector<LatticeFermion> subspace(nbasis,FGrid);
  subspace=AggregatesPD.subspace;

  Complex one(1.0);
  c_src = one;  // 1 in every element for vector 1.
  blockPromote(c_src,err,subspace);

  prom=Zero();
  for(int b=0;b<nbasis;b++){
    prom=prom+subspace[b];
  }
  err=err-prom; 
  std::cout<<GridLogMessage<<"Promoted back from subspace: err "<<norm2(err)<<std::endl;
  std::cout<<GridLogMessage<<"c_src "<<norm2(c_src)<<std::endl;
  std::cout<<GridLogMessage<<"prom  "<<norm2(prom)<<std::endl;

  PVdagM.Op(prom,tmp);
  blockProject(c_proj,tmp,subspace);
  std::cout<<GridLogMessage<<" Called Big Dirac Op "<<norm2(tmp)<<std::endl;

  LittleDiracOpPV.M(c_src,c_res);
  std::cout<<GridLogMessage<<" Called Little Dirac Op c_src "<< norm2(c_src) << "  c_res "<< norm2(c_res) <<std::endl;

  std::cout<<GridLogMessage<<"Little dop : "<<norm2(c_res)<<std::endl;
  //  std::cout<<GridLogMessage<<" Little "<< c_res<<std::endl;
  std::cout<<GridLogMessage<<"Big dop in subspace : "<<norm2(c_proj)<<std::endl;
  //  std::cout<<GridLogMessage<<" Big "<< c_proj<<std::endl;
  c_proj = c_proj - c_res;
  std::cout<<GridLogMessage<<" ldop error: "<<norm2(c_proj)<<std::endl;
  //  std::cout<<GridLogMessage<<" error "<< c_proj<<std::endl;


  /**********
   * Some solvers
   **********
   */

  ///////////////////////////////////////
  // Coarse grid solver test
  ///////////////////////////////////////

  std::cout<<GridLogMessage<<"******************* "<<std::endl;
  std::cout<<GridLogMessage<<" Coarse Grid Solve -- Level 3 "<<std::endl;
  std::cout<<GridLogMessage<<"******************* "<<std::endl;
  TrivialPrecon<CoarseVector> simple;
  NonHermitianLinearOperator<LittleDiracOperator,CoarseVector> LinOpCoarse(LittleDiracOpPV);
  //  PrecGeneralisedConjugateResidualNonHermitian<CoarseVector>  L2PGCR(1.0e-4, 100, LinOpCoarse,simple,10,10); 
  PrecGeneralisedConjugateResidualNonHermitian<CoarseVector>  L2PGCR(3.0e-2, 100, LinOpCoarse,simple,10,10); 
  L2PGCR.Level(3);
  c_res=Zero();
  L2PGCR(c_src,c_res);

  ////////////////////////////////////////
  // Fine grid smoother
  ////////////////////////////////////////
  std::cout<<GridLogMessage<<"******************* "<<std::endl;
  std::cout<<GridLogMessage<<" Fine Grid Smoother -- Level 2 "<<std::endl;
  std::cout<<GridLogMessage<<"******************* "<<std::endl;
  TrivialPrecon<LatticeFermionD> simple_fine;
  //  NonHermitianLinearOperator<PVdagM_t,LatticeFermionD> LinOpSmooth(PVdagM);
  PrecGeneralisedConjugateResidualNonHermitian<LatticeFermionD> SmootherGCR(0.01,1,ShiftedPVdagM,simple_fine,16,16);
  SmootherGCR.Level(2);
  
  LatticeFermionD f_src(FGrid);
  LatticeFermionD f_res(FGrid);

  f_src = one;  // 1 in every element for vector 1.
  f_res=Zero();
  SmootherGCR(f_src,f_res);

  typedef MGPreconditioner<vSpinColourVector,  vTComplex,nbasis> TwoLevelMG;

  TwoLevelMG TwoLevelPrecon(AggregatesPD,
			    PVdagM,
			    simple_fine,
			    SmootherGCR,
			    LinOpCoarse,
			    L2PGCR);
  
  PrecGeneralisedConjugateResidualNonHermitian<LatticeFermion> L1PGCR(1.0e-8,1000,PVdagM,TwoLevelPrecon,16,16);
  L1PGCR.Level(1);

  f_res=Zero();
  L1PGCR(f_src,f_res);
  
  // same thing but now for the exact Krylov basis
  std::cout<<GridLogMessage<<std::endl;
  std::cout<<GridLogMessage<<"*******************************************"<<std::endl;
  std::cout<<GridLogMessage<<"*********** EXACT EIGENVECTORS ************"<<std::endl;
  std::cout<<GridLogMessage<<"*******************************************"<<std::endl;

  std::cout << GridLogMessage << "Not running exact KS, nbasis too large" << std::endl;

  Subspace AggregatesKS(Coarse5d,FGrid,cb);
  for (int b = 0; b < nbasis; b++) {
    AggregatesKS.subspace[b] = evecs[b];
  }
  
  LittleDiracOperator LittleDiracOperatorKS(geom,FGrid,Coarse5d);
  LittleDiracOperatorKS.CoarsenOperator(PVdagM, AggregatesKS);

  std::cout<<GridLogMessage<<std::endl;
  std::cout<<GridLogMessage<<"*******************************************"<<std::endl;
  std::cout<<GridLogMessage<<std::endl;
  std::cout<<GridLogMessage<<"Testing coarsened operator "<<std::endl;

  subspace = AggregatesKS.subspace;

  c_src = one;  // 1 in every element for vector 1.
  blockPromote(c_src,err,subspace);

  prom=Zero();
  for(int b=0;b<nbasis;b++){
    prom=prom+subspace[b];
  }
  err=err-prom; 
  std::cout<<GridLogMessage<<"Promoted back from subspace: err "<<norm2(err)<<std::endl;
  std::cout<<GridLogMessage<<"c_src_KS "<<norm2(c_src)<<std::endl;
  std::cout<<GridLogMessage<<"prom  "<<norm2(prom)<<std::endl;

  PVdagM.Op(prom,tmp);
  blockProject(c_proj,tmp,subspace);
  std::cout<<GridLogMessage<<" Called Big Dirac Op "<<norm2(tmp)<<std::endl;

  LittleDiracOperatorKS.M(c_src,c_res);
  std::cout<<GridLogMessage<<" Called Little Dirac Op c_src "<< norm2(c_src) << "  c_res "<< norm2(c_res) <<std::endl;

  std::cout<<GridLogMessage<<"Little dop : "<<norm2(c_res)<<std::endl;
  //  std::cout<<GridLogMessage<<" Little "<< c_res<<std::endl;
  std::cout<<GridLogMessage<<"Big dop in subspace : "<<norm2(c_proj)<<std::endl;
  //  std::cout<<GridLogMessage<<" Big "<< c_proj<<std::endl;
  c_proj = c_proj - c_res;
  std::cout<<GridLogMessage<<" ldop error: "<<norm2(c_proj)<<std::endl;
  //  std::cout<<GridLogMessage<<" error "<< c_proj<<std::endl;

  //  **********
  //  * Some solvers
  //  **********

  ///////////////////////////////////////
  // Coarse grid solver test
  ///////////////////////////////////////

  std::cout<<GridLogMessage<<"******************* "<<std::endl;
  std::cout<<GridLogMessage<<" Coarse Grid Solve -- Level 3 "<<std::endl;
  std::cout<<GridLogMessage<<"******************* "<<std::endl;
  NonHermitianLinearOperator<LittleDiracOperator,CoarseVector> LinOpCoarseKS(LittleDiracOperatorKS);
  //  PrecGeneralisedConjugateResidualNonHermitian<CoarseVector>  L2PGCR(1.0e-4, 100, LinOpCoarse,simple,10,10); 
  PrecGeneralisedConjugateResidualNonHermitian<CoarseVector>  L2PGCRKS(3.0e-2, 100, LinOpCoarseKS,simple,10,10); 
  L2PGCRKS.Level(3);
  c_res=Zero();
  L2PGCRKS(c_src,c_res);

  ////////////////////////////////////////
  // Fine grid smoother
  ////////////////////////////////////////
  std::cout<<GridLogMessage<<"******************* "<<std::endl;
  std::cout<<GridLogMessage<<" Fine Grid Smoother -- Level 2 "<<std::endl;
  std::cout<<GridLogMessage<<"******************* "<<std::endl;
  //  NonHermitianLinearOperator<PVdagM_t,LatticeFermionD> LinOpSmooth(PVdagM);
  PrecGeneralisedConjugateResidualNonHermitian<LatticeFermionD> SmootherGCRKS(0.01,1,ShiftedPVdagM,simple_fine,16,16);
  SmootherGCRKS.Level(2);

  f_src = one;  // 1 in every element for vector 1.
  f_res=Zero();
  SmootherGCRKS(f_src,f_res);

  TwoLevelMG TwoLevelPreconKS(AggregatesKS,
			    PVdagM,
			    simple_fine,
			    SmootherGCRKS,
			    LinOpCoarseKS,
			    L2PGCRKS);
  
  PrecGeneralisedConjugateResidualNonHermitian<LatticeFermion> L1PGCRKS(1.0e-8,1000,PVdagM,TwoLevelPreconKS,16,16);
  L1PGCRKS.Level(1);

  f_res=Zero();
  L1PGCRKS(f_src,f_res);

  std::cout<<GridLogMessage<<std::endl;
  std::cout<<GridLogMessage << "Done "<< std::endl;

  Grid_finalize();
  return 0;
}
