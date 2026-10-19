/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./examples/Example_mg_evolution.cc

    Copyright (C) 2026

Author: Patrick Oare

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
//
// Example_mg_evolution.cc
//
// Experiment: how quickly does a multigrid near-null (deflation) subspace go stale
// as the gauge field evolves under HMC, and how cheaply can it be restored?
//
// Motivation: to use multigrid inside HMC we cannot afford to regenerate the
// near-null basis of the Dirac operator at every gauge update. Since each HMC
// trajectory changes U by a smooth O(dt) perturbation, the near-null space of
// the updated operator should be close to that of the old one (local coherence
// means the coarse space only has to capture the low-mode space collectively,
// not vector-by-vector). This program measures the decay of preconditioner
// quality along a trajectory sequence, comparing three reuse strategies:
//
//   (a) FROZEN     : subspace AND coarse (little Dirac) operator both fixed at U_0.
//   (b) RECOARSEN  : subspace fixed at U_0, but the coarse operator matrix elements
//                    are recomputed with the current gauge field (this is cheap and
//                    must be done anyway for the coarse grid to "see" the new U).
//   (c) REFINE     : subspace vectors are polished with a single sloppy GCR
//                    inverse-iteration pass against the CURRENT operator (seeded by
//                    the old vectors, NOT regenerated from noise), then recoarsened.
//
// In all cases the outer solve is a flexible (right-preconditioned) GCR on the
// current operator PV^dag M (U_t) converged to 1e-8, so the preconditioner quality
// only affects the iteration count, never the answer. The figure of merit reported
// per trajectory is the outer GCR step count for each strategy.
//
// The MG construction (PVdagM operator, 2-level preconditioner, GCR subspace
// setup, tolerances, 2^4 blocking) follows examples/Example_pvdagm_svd.cc.
//
// Gauge evolution: by default a QUENCHED Iwasaki HMC (see notes at the action
// setup below) — smooth O(dt) evolution of U is all that is required for this
// test. A dynamical pseudofermion action can be added at the marked location.
//
#include <Grid/Grid.h>
#include <Grid/lattice/PaddedCell.h>
#include <Grid/stencil/GeneralLocalStencil.h>

#include <Grid/algorithms/iterative/PrecGeneralisedConjugateResidual.h>
#include <Grid/algorithms/iterative/PrecGeneralisedConjugateResidualNonHermitian.h>

using namespace std;
using namespace Grid;

//////////////////////////////////////////////////////////////////////////////
// PV^dag M operator and shifted variant (copied from Example_pvdagm_svd.cc).
// PV^dag M is the Pauli-Villars preconditioned operator: its low modes are the
// physical near-null space of the domain wall operator with the bulk (heavy)
// modes cancelled, and it is nearest-neighbour enough to coarsen well.
//////////////////////////////////////////////////////////////////////////////
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
    Field tmp(in.Grid());
    _Mat.M(in,tmp);
    _PV.Mdag(tmp,out);
  }
  void AdjOp     (const Field &in, Field &out){
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
    Field tmp(in.Grid());
    Op(in,tmp);
    AdjOp(tmp,out);
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
    Field tmp(in.Grid());
    _Mat.M(in,tmp);
    _PV.Mdag(tmp,out);
    out = out + shift * in;
  }
  void AdjOp     (const Field &in, Field &out){
    Field tmp(in.Grid());
    _PV.M(tmp,out);
    _Mat.Mdag(in,tmp);
    out = out + shift * in;
  }
  void HermOpAndNorm(const Field &in, Field &out,RealD &n1,RealD &n2){    assert(0);  }
  void HermOp(const Field &in, Field &out){
    Field tmp(in.Grid());
    Op(in,tmp);
    AdjOp(tmp,out);
  }
};

//////////////////////////////////////////////////////////////////////////////
// Two level (fine + coarse) V-cycle preconditioner
// (copied from Example_pvdagm_svd.cc): pre-smooth, coarse-grid correction
// through the aggregated subspace, post-smooth.
//////////////////////////////////////////////////////////////////////////////
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

  // Accumulated wall time (us) per phase, across every application
  RealD tsmooth=0, tcoarse=0, tproj=0, tfine=0;
  int   calls=0;

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
    CoarseVector Csrc(CoarseGrid);
    CoarseVector Csol(CoarseGrid);
    FineField vec1(in.Grid());
    FineField vec2(in.Grid());

    calls++;

    // Pre-smooth on the fine grid
    out = Zero();
    tsmooth-=usecond();
    _PreSmoother(in,out);
    tsmooth+=usecond();

    // Residual after smoothing
    tfine-=usecond();
    _FineOperator.Op(out,vec1);  sub(vec1, in ,vec1);
    tfine+=usecond();

    // Coarse grid correction: project residual to the coarse space,
    // solve the little Dirac equation, prolongate back and add.
    tproj-=usecond();
    _Aggregates.ProjectToSubspace  (Csrc,vec1);
    tproj+=usecond();
    Csol = Zero();
    tcoarse-=usecond();
    _CoarseSolve(Csrc,Csol);
    tcoarse+=usecond();
    tproj-=usecond();
    _Aggregates.PromoteFromSubspace(Csol,vec1);
    tproj+=usecond();
    add(out,out,vec1);

    // Residual after coarse correction, then post-smooth
    tfine-=usecond();
    _FineOperator.Op(out,vec1);  sub(vec1 ,in , vec1);
    tfine+=usecond();
    vec2=Zero();
    tsmooth-=usecond();
    _PostSmoother(vec1,vec2);
    tsmooth+=usecond();
    add( out,out,vec2);
  }
};

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
//////////////////////////////////////////////////////////////////////////////
template<class Aggregates, class Field>
void RefineSubspaceGCR(Aggregates &Agg,
		       LinearOperatorBase<Field> &DiracOp,
		       int nn,
		       RealD tol, int maxouter)
{
  RealD scale;
  TrivialPrecon<Field> simple;
  PrecGeneralisedConjugateResidualNonHermitian<Field> GCR(tol,maxouter,DiracOp,simple,10,10);

  Field src(Agg.FineGrid);
  Field guess(Agg.FineGrid);
  Field Mn(Agg.FineGrid);

  for(int b=0;b<nn;b++){
    scale = std::pow(norm2(Agg.subspace[b]),-0.5);
    src   = Agg.subspace[b]*scale;

    DiracOp.Op(src,Mn);
    std::cout<<GridLogMessage << "refine-in ["<<b<<"] <v|Op|v> "<<innerProduct(src,Mn)<<std::endl;

    guess = Zero();
    GCR(src,guess);

    scale = std::pow(norm2(guess),-0.5);
    Agg.subspace[b] = guess*scale;

    DiracOp.Op(Agg.subspace[b],Mn);
    std::cout<<GridLogMessage << "refine-out["<<b<<"] <v|Op|v> "<<innerProduct(Agg.subspace[b],Mn)<<std::endl;
  }
}

//////////////////////////////////////////////////////////////////////////////
// "Nulliness" of the subspace S = span{v_b} against A = PV^dag M (U_t):
//
//     eta(S) = tr(V^dag A^dag A V)/N ,   V^dag V = I ,
//
// a property of S alone (GL(N)-invariant), bounded below by
// lambda_min(A^dag A) by interlacing. No orthonormal V is formed: with
// G = <v_i|v_j> and M = <A v_i|A v_j> for the stored vectors, the spectrum
// of V^dag A^dag A V is that of the generalised problem M x = mu G x. The
// subspace itself is not modified -- it is the live MG basis.
//
// Per-vector lines are instruments only (basis-dependent): the Rayleigh
// quotient rho = <v|A|v>, |Av|^2, and the pseudo-eigen residual
// eps^2 = |Av|^2 - |rho|^2. Small |rho| is NOT a target; see
// notes/deflation-basis-evolution.md section 1.
//
// Call AFTER CoarsenOperator: it block-orthonormalises the subspace in
// place, which changes the global span, so measuring before it reports on
// a basis the solve never sees. Returns eta(S).
//////////////////////////////////////////////////////////////////////////////
template<class Aggregates, class Field>
RealD ReportNulliness(const std::string &tag,
		      Aggregates &Agg,
		      LinearOperatorBase<Field> &DiracOp,
		      int nn)
{
  std::vector<Field> Av(nn,Agg.FineGrid);
  Eigen::MatrixXcd G(nn,nn);
  Eigen::MatrixXcd M(nn,nn);

  for(int b=0;b<nn;b++){
    DiracOp.Op(Agg.subspace[b],Av[b]);
  }
  for(int i=0;i<nn;i++){
    for(int j=i;j<nn;j++){
      G(i,j) = innerProduct(Agg.subspace[i],Agg.subspace[j]);
      M(i,j) = innerProduct(Av[i],Av[j]);
      G(j,i) = std::conj(G(i,j));
      M(j,i) = std::conj(M(i,j));
    }
  }

  // Per-vector instruments, normalised by |v|^2 = G(b,b)
  for(int b=0;b<nn;b++){
    RealD    vv   = real(G(b,b));
    ComplexD rho  = innerProduct(Agg.subspace[b],Av[b])/vv;  // <v|A|v>/<v|v>
    RealD    aa   = real(M(b,b))/vv;                         // |Av|^2/|v|^2
    RealD    eps2 = aa - norm(rho);                          // pseudo-eigen residual^2

    std::cout<<GridLogMessage<<"MGEVO-NULL "<<tag<<" ["<<b<<"] rho "<<rho
	     <<" |Av|^2 "<<aa<<" eps^2 "<<eps2
	     <<" |Av|/|rho| "<<std::sqrt(aa)/abs(rho)<<std::endl;
  }

  // Conditioning of the stored basis, so a near-dependent set is visible
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> gs(G,Eigen::EigenvaluesOnly);
  std::cout<<GridLogMessage<<"MGEVO-NULL "<<tag<<" Gram eigenvalues ["
	   <<gs.eigenvalues().minCoeff()<<", "<<gs.eigenvalues().maxCoeff()<<"]"<<std::endl;

  // Spectrum of V^dag A^dag A V on an orthonormal basis of S
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXcd> ges(M,G,Eigen::EigenvaluesOnly);
  Eigen::VectorXd mu = ges.eigenvalues();
  for(int b=0;b<nn;b++){
    std::cout<<GridLogMessage<<"MGEVO-ETA "<<tag<<" mu["<<b<<"] "<<mu(b)<<std::endl;
  }
  RealD eta = mu.sum()/nn;
  std::cout<<GridLogMessage<<"MGEVO-ETA "<<tag<<" eta(S) = tr(V^dag A^dag A V)/N = "<<eta<<std::endl;
  return eta;
}

int main (int argc, char ** argv)
{
  Grid_init(&argc,&argv);

  //////////////////////////////////////////////////////////////////////////
  // Parameters
  //////////////////////////////////////////////////////////////////////////
  const int Ls=16;
  std::vector<int> lat_size {16, 16, 16, 32};   // ckpoint_lat.4000 is 16^3 x 32

  RealD mass = 0.001;              // light quark mass
  RealD M5   = 1.8;                // domain wall height
  RealD mob_b= 1.5;                // Mobius kernel: b-c=1 (Shamir-like), b+c=2
  RealD mob_c= 0.5;

  std::string config("/Users/patrickoare/libraries/PETSc-Grid/ckpoint_lat.4000");
  if (GridCmdOptionExists(argv,argv+argc,"--config")) {
    config = GridCmdOptionPayload(argv,argv+argc,"--config");
  }
  std::cout<<GridLogMessage<<"Reading configuration "<<config<<std::endl;

  const int nbasis = 40;           // near-null vectors, as in Example_pvdagm_svd.cc
  const int cb     = 0;

  int   Ntraj   = 40;              // number of HMC trajectories to generate (~28 min each on 8 nodes at nbasis=40)
  RealD beta    = 2.13;            // Iwasaki coupling for the (quenched) evolution
  int   MDsteps = 20;              // MD steps per trajectory
  RealD trajL   = 1.0;             // trajectory length

  RealD refineTol      = 1.0e-2;   // strategy (c): sloppy tolerance ...
  int   refineMaxOuter = 2;        // ... and restart budget for the refresh GCR(10,10)

  //////////////////////////////////////////////////////////////////////////
  // Grids: fine 5d, and a coarse grid blocked 2^4 in the 4d volume with the
  // whole of Ls aggregated (coarse Ls = 1), as in Example_pvdagm_svd.cc.
  //////////////////////////////////////////////////////////////////////////
  GridCartesian         * UGrid   = SpaceTimeGrid::makeFourDimGrid(lat_size, GridDefaultSimd(Nd,vComplex::Nsimd()),GridDefaultMpi());
  GridRedBlackCartesian * UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian         * FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls,UGrid);
  GridRedBlackCartesian * FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGrid);

  Coordinate clatt = lat_size;
  for(int d=0;d<clatt.size();d++) clatt[d] = clatt[d]/2;

  GridCartesian *Coarse4d =  SpaceTimeGrid::makeFourDimGrid(clatt, GridDefaultSimd(Nd,vComplex::Nsimd()),GridDefaultMpi());
  GridCartesian *Coarse5d =  SpaceTimeGrid::makeFiveDimGrid(1,Coarse4d);

  std::vector<int> seeds4({1,2,3,4});
  std::vector<int> seeds5({5,6,7,8});
  std::vector<int> seedsS({9,10,11,12});
  GridParallelRNG RNG5(FGrid); RNG5.SeedFixedIntegers(seeds5);
  GridParallelRNG RNG4(UGrid); RNG4.SeedFixedIntegers(seeds4);
  GridSerialRNG   sRNG;        sRNG.SeedFixedIntegers(seedsS);

  //////////////////////////////////////////////////////////////////////////
  // Gauge field: start from the checkpointed configuration
  //////////////////////////////////////////////////////////////////////////
  LatticeGaugeField Umu(UGrid);
  FieldMetaData header;
  NerscIO::readConfiguration(Umu,header,config);

  //////////////////////////////////////////////////////////////////////////
  // Fixed fermion source, reused for every measurement so that iteration
  // counts are comparable across trajectories and strategies.
  //////////////////////////////////////////////////////////////////////////
  LatticeFermion src(FGrid); gaussian(RNG5,src);
  LatticeFermion result(FGrid);

  //////////////////////////////////////////////////////////////////////////
  // MG types (as Example_pvdagm_svd.cc, with Mobius kernel)
  //////////////////////////////////////////////////////////////////////////
  typedef PVdagMLinearOperator       <MobiusFermionD,LatticeFermionD> PVdagM_t;
  typedef ShiftedPVdagMLinearOperator<MobiusFermionD,LatticeFermionD> ShiftedPVdagM_t;
  typedef Aggregation<vSpinColourVector,vTComplex,nbasis> Subspace;
  typedef GeneralCoarsenedMatrix<vSpinColourVector,vTComplex,nbasis> LittleDiracOperator;
  typedef LittleDiracOperator::CoarseVector CoarseVector;
  typedef MGPreconditioner<vSpinColourVector,vTComplex,nbasis> TwoLevelMG;

  NextToNearestStencilGeometry5D geom(Coarse5d);

  TrivialPrecon<LatticeFermionD> simpleFine;
  TrivialPrecon<CoarseVector>    simpleCoarse;

  //////////////////////////////////////////////////////////////////////////
  // One 2-level preconditioned solve of PV^dag M (U_t) x = src to 1e-8;
  // returns the outer GCR step count (the figure of merit) and the wall
  // time in seconds through tsolve.
  //////////////////////////////////////////////////////////////////////////
  auto RunTwoLevelSolve = [&](const std::string &tag,
			      Subspace            &Aggregates,
			      LittleDiracOperator &LittleDiracOp,
			      PVdagM_t            &FineOp,
			      ShiftedPVdagM_t     &SmootherOp,
			      RealD               &tsolve) -> int
  {
    NonHermitianLinearOperator<LittleDiracOperator,CoarseVector> LinOpCoarse(LittleDiracOp);
    PrecGeneralisedConjugateResidualNonHermitian<CoarseVector>   L2PGCR(3.0e-2,100,LinOpCoarse,simpleCoarse,10,10);
    L2PGCR.Level(3);

    PrecGeneralisedConjugateResidualNonHermitian<LatticeFermionD> SmootherGCR(0.01,1,SmootherOp,simpleFine,16,16);
    SmootherGCR.Level(2);

    TwoLevelMG TwoLevelPrecon(Aggregates,FineOp,
			      simpleFine,SmootherGCR,
			      LinOpCoarse,L2PGCR);

    PrecGeneralisedConjugateResidualNonHermitian<LatticeFermion> L1PGCR(1.0e-8,3000,FineOp,TwoLevelPrecon,16,16);
    L1PGCR.Level(1);

    result = Zero();
    tsolve = -usecond();
    L1PGCR(src,result);
    tsolve = (tsolve+usecond())/1.0e6;

    std::cout<<GridLogMessage<<"MGEVO solve "<<tag<<" : outer steps = "<<L1PGCR.steps
	     <<" time "<<tsolve<<" s"<<std::endl;
    std::cout<<GridLogMessage<<"MGEVO-PRECTIME "<<tag<<" calls "<<TwoLevelPrecon.calls
	     <<" smooth "<<TwoLevelPrecon.tsmooth/1.0e6
	     <<" coarse "<<TwoLevelPrecon.tcoarse/1.0e6
	     <<" proj "  <<TwoLevelPrecon.tproj/1.0e6
	     <<" fineop "<<TwoLevelPrecon.tfine/1.0e6<<std::endl;
    return L1PGCR.steps;
  };

  //////////////////////////////////////////////////////////////////////////
  // Initial setup at U_0: build the operators, breed the near-null subspace
  // with GCR inverse iteration on noise, and coarsen the little Dirac op.
  //
  //  - AggregatesFrozen  : never touched again; used by strategies (a),(b)
  //  - AggregatesRefined : evolves via RefineSubspaceGCR; used by strategy (c)
  //////////////////////////////////////////////////////////////////////////
  MobiusFermionD Ddwf0(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,mob_b,mob_c);
  MobiusFermionD Dpv0 (Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,1.0 ,M5,mob_b,mob_c);
  PVdagM_t        PVdagM0(Ddwf0,Dpv0);
  ShiftedPVdagM_t ShiftedPVdagM0(0.01,Ddwf0,Dpv0);

  Subspace AggregatesFrozen (Coarse5d,FGrid,cb);
  Subspace AggregatesRefined(Coarse5d,FGrid,cb);

  RealD t0_setup = -usecond();
  AggregatesFrozen.CreateSubspaceGCR(RNG5,PVdagM0,nbasis);
  t0_setup = (t0_setup+usecond())/1.0e6;
  AggregatesRefined.subspace = AggregatesFrozen.subspace;   // start from the same basis

  LittleDiracOperator LittleDiracOpFrozen     (geom,FGrid,Coarse5d); // built at U_0, never rebuilt

  RealD t0_coarsen = -usecond();
  LittleDiracOpFrozen.CoarsenOperator(PVdagM0,AggregatesFrozen);
  t0_coarsen = (t0_coarsen+usecond())/1.0e6;

  // Nulliness of the freshly bred basis against the U_0 operator, measured
  // after coarsening has block-orthonormalised it
  RealD t0_eta = -usecond();
  RealD null0 = ReportNulliness("traj 0 (fresh)",AggregatesFrozen,PVdagM0,nbasis);
  t0_eta = (t0_eta+usecond())/1.0e6;

  // Baseline: fresh setup solving at U_0
  RealD t0_solve;
  int steps0 = RunTwoLevelSolve("traj 0 (fresh setup)",AggregatesFrozen,LittleDiracOpFrozen,PVdagM0,ShiftedPVdagM0,t0_solve);

  // Wall times in seconds:
  //   MGEVO-TIME 0 setup coarsen eta solve
  std::cout<<GridLogMessage<<"MGEVO-TIME 0 setup "<<t0_setup<<" coarsen "<<t0_coarsen
	   <<" eta "<<t0_eta<<" solve "<<t0_solve<<std::endl;

  //////////////////////////////////////////////////////////////////////////
  // HMC: quenched Iwasaki evolution.
  //
  // NOTE: for this experiment only the smooth O(dt) evolution of U matters,
  // so we evolve with the pure gauge action. To make the evolution dynamical,
  // push_back a pseudofermion action (e.g. TwoFlavourEvenOddRatioPseudoFermionAction
  // wrapping the Mobius operators, on a separate integrator level) — beware
  // this makes each trajectory vastly more expensive at mass=0.001.
  //
  // We reuse Grid's HybridMonteCarlo driver one trajectory at a time
  // (Trajectories=1) so that control returns here between trajectories.
  //////////////////////////////////////////////////////////////////////////
  typedef PeriodicGimplR Gimpl;
  typedef NoSmearing<Gimpl> SmearingPolicy;
  typedef MinimumNorm2<Gimpl,SmearingPolicy> IntegratorType;

  IwasakiGaugeActionR GaugeAction(beta);
  ActionLevel<LatticeGaugeField> Level1(1);
  Level1.push_back(&GaugeAction);
  ActionSet<LatticeGaugeField,NoHirep> FullSet;
  FullSet.push_back(Level1);

  IntegratorParameters MDpar(MDsteps,trajL);
  SmearingPolicy Smearing;
  IntegratorType MDynamics(UGrid,MDpar,FullSet,Smearing);

  std::vector<HmcObservable<LatticeGaugeField>*> NoObservables;

  std::vector<int>   stepsA, stepsB, stepsC;
  std::vector<RealD> plaqs;
  std::vector<RealD> nullFrozen, nullRefined;

  for(int traj=1;traj<=Ntraj;traj++){

    ////////////////////////////////////////////////////////////////////////
    // One MD trajectory: momentum refresh and MD integration, always
    // accepted (no Metropolis test), so U moves every trajectory. Not a
    // valid Markov chain -- configurations are for this experiment only.
    ////////////////////////////////////////////////////////////////////////
    HMCparameters HMCpar;
    HMCpar.StartTrajectory   = traj;
    HMCpar.Trajectories      = 1;
    HMCpar.NoMetropolisUntil = 0;
    HMCpar.MetropolisTest    = false;  // always accept: only the smooth flow of U matters here
                                       // (dH ~ 4-5 at MDsteps=20 rejected every trajectory, job 424245)
    HMCpar.PerformRandomShift= false;

    RealD t_hmc = -usecond();
    HybridMonteCarlo<IntegratorType> HMC(HMCpar,MDynamics,sRNG,RNG4,NoObservables,Umu);
    HMC.evolve();
    t_hmc = (t_hmc+usecond())/1.0e6;

    RealD plaq = WilsonLoops<Gimpl>::avgPlaquette(Umu);
    plaqs.push_back(plaq);
    std::cout<<GridLogMessage<<"MGEVO traj "<<traj<<" plaquette "<<plaq<<std::endl;

    ////////////////////////////////////////////////////////////////////////
    // Fresh operators on the updated gauge field. These are what we solve
    // with in ALL strategies — only the preconditioner content differs.
    ////////////////////////////////////////////////////////////////////////
    MobiusFermionD Ddwf(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,mob_b,mob_c);
    MobiusFermionD Dpv (Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,1.0 ,M5,mob_b,mob_c);
    PVdagM_t        PVdagM(Ddwf,Dpv);
    ShiftedPVdagM_t ShiftedPVdagM(0.01,Ddwf,Dpv);

    ////////////////////////////////////////////////////////////////////////
    // Nulliness of the FROZEN basis against the current operator.
    // This applies to both strategies (a) and (b): they share the same
    // subspace and differ only in the coarse operator matrix elements.
    ////////////////////////////////////////////////////////////////////////
    RealD t_etaF = -usecond();
    RealD nullF = ReportNulliness("frozen  traj "+std::to_string(traj),
				  AggregatesFrozen,PVdagM,nbasis);
    t_etaF = (t_etaF+usecond())/1.0e6;

    ////////////////////////////////////////////////////////////////////////
    // (a) FROZEN: stale subspace AND stale coarse operator from U_0
    ////////////////////////////////////////////////////////////////////////
    RealD t_solveA;
    int sa = RunTwoLevelSolve("(a) frozen     traj "+std::to_string(traj),
			      AggregatesFrozen,LittleDiracOpFrozen,PVdagM,ShiftedPVdagM,t_solveA);

    ////////////////////////////////////////////////////////////////////////
    // (b) RECOARSEN: stale subspace, coarse operator recomputed at U_t
    ////////////////////////////////////////////////////////////////////////
    // GeneralCoarsenedMatrix is single-use: CoarsenOperator leaves _A on the
    // padded grid, so a second call on the same object fails conformable().
    // Build the (b) and (c) operators afresh every trajectory.
    LittleDiracOperator LittleDiracOpRecoarsened(geom,FGrid,Coarse5d);
    RealD t_coarsenB = -usecond();
    LittleDiracOpRecoarsened.CoarsenOperator(PVdagM,AggregatesFrozen);
    t_coarsenB = (t_coarsenB+usecond())/1.0e6;
    RealD t_solveB;
    int sb = RunTwoLevelSolve("(b) recoarsen  traj "+std::to_string(traj),
			      AggregatesFrozen,LittleDiracOpRecoarsened,PVdagM,ShiftedPVdagM,t_solveB);

    ////////////////////////////////////////////////////////////////////////
    // (c) REFINE: one sloppy inverse-iteration pass at U_t on the evolving
    //             basis, then recoarsen with the refreshed vectors
    ////////////////////////////////////////////////////////////////////////
    RealD t_refineC = -usecond();
    RefineSubspaceGCR(AggregatesRefined,PVdagM,nbasis,refineTol,refineMaxOuter);
    t_refineC = (t_refineC+usecond())/1.0e6;

    LittleDiracOperator LittleDiracOpRefined(geom,FGrid,Coarse5d);
    RealD t_coarsenC = -usecond();
    LittleDiracOpRefined.CoarsenOperator(PVdagM,AggregatesRefined);
    t_coarsenC = (t_coarsenC+usecond())/1.0e6;

    // Nulliness of the refreshed basis: this is the basis actually used in
    // the strategy (c) solve below, now that coarsening has block-orthonormalised it
    RealD t_etaC = -usecond();
    RealD nullR = ReportNulliness("refined traj "+std::to_string(traj),
				  AggregatesRefined,PVdagM,nbasis);
    t_etaC = (t_etaC+usecond())/1.0e6;
    RealD t_solveC;
    int sc = RunTwoLevelSolve("(c) refine     traj "+std::to_string(traj),
			      AggregatesRefined,LittleDiracOpRefined,PVdagM,ShiftedPVdagM,t_solveC);

    // Wall times in seconds, one grep-able line per trajectory:
    //   MGEVO-TIME traj hmc eta_F solve_a coarsen_b solve_b refine_c coarsen_c eta_c solve_c
    std::cout<<GridLogMessage<<"MGEVO-TIME "<<traj<<" hmc "<<t_hmc<<" eta_F "<<t_etaF
	     <<" solve_a "<<t_solveA<<" coarsen_b "<<t_coarsenB<<" solve_b "<<t_solveB
	     <<" refine_c "<<t_refineC<<" coarsen_c "<<t_coarsenC<<" eta_c "<<t_etaC
	     <<" solve_c "<<t_solveC<<std::endl;

    stepsA.push_back(sa);
    stepsB.push_back(sb);
    stepsC.push_back(sc);
    nullFrozen.push_back(nullF);
    nullRefined.push_back(nullR);

    // One grep-able line per trajectory:
    //   MGEVO-SUMMARY traj plaq steps_a steps_b steps_c nullFrozen nullRefined
    std::cout<<GridLogMessage<<"MGEVO-SUMMARY "<<traj<<" "<<plaq<<" "<<sa<<" "<<sb<<" "<<sc
	     <<" "<<nullF<<" "<<nullR<<std::endl;
  }

  //////////////////////////////////////////////////////////////////////////
  // Final table
  //////////////////////////////////////////////////////////////////////////
  std::cout<<GridLogMessage<<"=================================================================="<<std::endl;
  std::cout<<GridLogMessage<<" MG subspace evolution summary (outer GCR steps, fresh setup = "<<steps0
	   <<", fresh-basis eta(S) = "<<null0<<")"<<std::endl;
  std::cout<<GridLogMessage<<" traj   plaquette   (a) frozen  (b) recoarsen  (c) refine   eta(frozen)  eta(refined)"<<std::endl;
  for(int t=0;t<stepsA.size();t++){
    std::cout<<GridLogMessage<<" "<<t+1<<"   "<<plaqs[t]
	     <<"   "<<stepsA[t]<<"   "<<stepsB[t]<<"   "<<stepsC[t]
	     <<"   "<<nullFrozen[t]<<"   "<<nullRefined[t]<<std::endl;
  }
  std::cout<<GridLogMessage<<"=================================================================="<<std::endl;

  Grid_finalize();
  return 0;
}
