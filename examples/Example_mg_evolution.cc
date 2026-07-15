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

    // Pre-smooth on the fine grid
    out = Zero();
    _PreSmoother(in,out);

    // Residual after smoothing
    _FineOperator.Op(out,vec1);  sub(vec1, in ,vec1);

    // Coarse grid correction: project residual to the coarse space,
    // solve the little Dirac equation, prolongate back and add.
    _Aggregates.ProjectToSubspace  (Csrc,vec1);
    Csol = Zero();
    _CoarseSolve(Csrc,Csol);
    _Aggregates.PromoteFromSubspace(Csol,vec1);
    add(out,out,vec1);

    // Residual after coarse correction, then post-smooth
    _FineOperator.Op(out,vec1);  sub(vec1 ,in , vec1);
    vec2=Zero();
    _PostSmoother(vec1,vec2);
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
// "Nulliness" diagnostic: the Ritz estimate <psi|A|psi> of each (normalised)
// subspace vector against the operator A = PV^dag M (U_t). A good deflation
// basis for A has small nulliness. Since A is non-Hermitian the Ritz value is
// complex; we also print |A psi|^2 = <psi|A^dag A|psi>, the nulliness with
// respect to the Hermitian positive operator A^dag A (the PV-preconditioned
// analogue of D^dag D). Returns the average |<psi|A|psi>| over the basis.
//////////////////////////////////////////////////////////////////////////////
template<class Aggregates, class Field>
RealD ReportNulliness(const std::string &tag,
		      Aggregates &Agg,
		      LinearOperatorBase<Field> &DiracOp,
		      int nn)
{
  Field psi(Agg.FineGrid);
  Field Apsi(Agg.FineGrid);

  RealD avg = 0.0;
  for(int b=0;b<nn;b++){
    // Defensive renormalisation so the Ritz value is meaningful even if a
    // caller has not normalised the vectors
    RealD scale = std::pow(norm2(Agg.subspace[b]),-0.5);
    psi = Agg.subspace[b]*scale;

    DiracOp.Op(psi,Apsi);
    ComplexD ritz = innerProduct(psi,Apsi);   // <psi|A|psi>
    RealD    aa   = norm2(Apsi);              // <psi|A^dag A|psi>

    std::cout<<GridLogMessage<<"MGEVO-NULL "<<tag<<" ["<<b<<"] <psi|A|psi> "<<ritz
	     <<" |ritz| "<<abs(ritz)<<" <psi|AdagA|psi> "<<aa<<std::endl;
    avg += abs(ritz);
  }
  avg = avg/nn;
  std::cout<<GridLogMessage<<"MGEVO-NULL "<<tag<<" average |<psi|A|psi>| = "<<avg<<std::endl;
  return avg;
}

int main (int argc, char ** argv)
{
  Grid_init(&argc,&argv);

  //////////////////////////////////////////////////////////////////////////
  // Parameters
  //////////////////////////////////////////////////////////////////////////
  const int Ls=16;
  std::vector<int> lat_size {16, 16, 16, 48};

  RealD mass = 0.001;              // light quark mass
  RealD M5   = 1.8;                // domain wall height
  RealD mob_b= 1.5;                // Mobius kernel: b-c=1 (Shamir-like), b+c=2
  RealD mob_c= 0.5;

  std::string config("/Users/patrickoare/libraries/PETSc-Grid/ckpoint_lat.4000");

  const int nbasis = 4;            // near-null vectors (40 in Example_pvdagm_svd.cc; reduced for local running)
  const int cb     = 0;

  int   Ntraj   = 50;              // number of HMC trajectories to generate
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
  // returns the outer GCR step count (the figure of merit).
  //////////////////////////////////////////////////////////////////////////
  auto RunTwoLevelSolve = [&](const std::string &tag,
			      Subspace            &Aggregates,
			      LittleDiracOperator &LittleDiracOp,
			      PVdagM_t            &FineOp,
			      ShiftedPVdagM_t     &SmootherOp) -> int
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
    L1PGCR(src,result);

    std::cout<<GridLogMessage<<"MGEVO solve "<<tag<<" : outer steps = "<<L1PGCR.steps<<std::endl;
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

  AggregatesFrozen.CreateSubspaceGCR(RNG5,PVdagM0,nbasis);
  AggregatesRefined.subspace = AggregatesFrozen.subspace;   // start from the same basis

  // Nulliness of the freshly bred basis against the U_0 operator
  RealD null0 = ReportNulliness("traj 0 (fresh)",AggregatesFrozen,PVdagM0,nbasis);

  LittleDiracOperator LittleDiracOpFrozen     (geom,FGrid,Coarse5d); // built at U_0, never rebuilt
  LittleDiracOperator LittleDiracOpRecoarsened(geom,FGrid,Coarse5d); // rebuilt every trajectory, frozen basis
  LittleDiracOperator LittleDiracOpRefined    (geom,FGrid,Coarse5d); // rebuilt every trajectory, refined basis

  LittleDiracOpFrozen.CoarsenOperator(PVdagM0,AggregatesFrozen);

  // Baseline: fresh setup solving at U_0
  int steps0 = RunTwoLevelSolve("traj 0 (fresh setup)",AggregatesFrozen,LittleDiracOpFrozen,PVdagM0,ShiftedPVdagM0);

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
    // One HMC trajectory: momentum refresh, MD integration, Metropolis.
    // On rejection Umu is left unchanged (a genuine HMC step).
    ////////////////////////////////////////////////////////////////////////
    HMCparameters HMCpar;
    HMCpar.StartTrajectory   = traj;
    HMCpar.Trajectories      = 1;
    HMCpar.NoMetropolisUntil = 0;
    HMCpar.MetropolisTest    = true;
    HMCpar.PerformRandomShift= false;

    HybridMonteCarlo<IntegratorType> HMC(HMCpar,MDynamics,sRNG,RNG4,NoObservables,Umu);
    HMC.evolve();

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
    RealD nullF = ReportNulliness("frozen  traj "+std::to_string(traj),
				  AggregatesFrozen,PVdagM,nbasis);

    ////////////////////////////////////////////////////////////////////////
    // (a) FROZEN: stale subspace AND stale coarse operator from U_0
    ////////////////////////////////////////////////////////////////////////
    int sa = RunTwoLevelSolve("(a) frozen     traj "+std::to_string(traj),
			      AggregatesFrozen,LittleDiracOpFrozen,PVdagM,ShiftedPVdagM);

    ////////////////////////////////////////////////////////////////////////
    // (b) RECOARSEN: stale subspace, coarse operator recomputed at U_t
    ////////////////////////////////////////////////////////////////////////
    LittleDiracOpRecoarsened.CoarsenOperator(PVdagM,AggregatesFrozen);
    int sb = RunTwoLevelSolve("(b) recoarsen  traj "+std::to_string(traj),
			      AggregatesFrozen,LittleDiracOpRecoarsened,PVdagM,ShiftedPVdagM);

    ////////////////////////////////////////////////////////////////////////
    // (c) REFINE: one sloppy inverse-iteration pass at U_t on the evolving
    //             basis, then recoarsen with the refreshed vectors
    ////////////////////////////////////////////////////////////////////////
    RefineSubspaceGCR(AggregatesRefined,PVdagM,nbasis,refineTol,refineMaxOuter);

    // Nulliness of the refreshed basis: this is the basis actually used in
    // the strategy (c) solve below
    RealD nullR = ReportNulliness("refined traj "+std::to_string(traj),
				  AggregatesRefined,PVdagM,nbasis);

    LittleDiracOpRefined.CoarsenOperator(PVdagM,AggregatesRefined);
    int sc = RunTwoLevelSolve("(c) refine     traj "+std::to_string(traj),
			      AggregatesRefined,LittleDiracOpRefined,PVdagM,ShiftedPVdagM);

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
	   <<", fresh-basis nulliness = "<<null0<<")"<<std::endl;
  std::cout<<GridLogMessage<<" traj   plaquette   (a) frozen  (b) recoarsen  (c) refine   null(frozen)  null(refined)"<<std::endl;
  for(int t=0;t<stepsA.size();t++){
    std::cout<<GridLogMessage<<" "<<t+1<<"   "<<plaqs[t]
	     <<"   "<<stepsA[t]<<"   "<<stepsB[t]<<"   "<<stepsC[t]
	     <<"   "<<nullFrozen[t]<<"   "<<nullRefined[t]<<std::endl;
  }
  std::cout<<GridLogMessage<<"=================================================================="<<std::endl;

  Grid_finalize();
  return 0;
}
