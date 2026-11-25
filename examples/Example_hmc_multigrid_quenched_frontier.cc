/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./examples/Example_hmc_multigrid_quenched_frontier.cc

    Copyright (C) 2026

Author: Patrick Oare

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    See the full license in the file "LICENSE" in the top level distribution
    directory
*************************************************************************************/
/*  END LEGAL */

//
// Quenched HMC with the three-level mrhs PVdagM multigrid attached
// (Grid/algorithms/multigrid/PVdagMMultiGrid.h).  This is diagnostic test 1
// of the deflation-basis-in-HMC plan (research hub, projects/multigrid,
// writeups/deflation_basis_evolution):
//
//   1. generate the near-null basis on the starting configuration (timed, fine
//      matvecs counted) and save it, or load a saved one;
//   2. coarsen, build the dense bottom and certify, as Example_pvdagm_multigrid;
//   3. one mrhs and one single-RHS solve on that basis;
//   4. quenched trajectories: Iwasaki gauge action only, MinimumNorm2,
//      momentum refresh, MD, Metropolis.  No solve is attached to the MD yet;
//      every step logs MD time, plaquette and wall time.  TrajL and MDsteps are
//      paired lists, with Ntraj trajectories per pair.  With Chain 0 each one
//      starts from the starting configuration, with fresh momenta; with Chain 1
//      each continues from the end of the last one;
//   5. with Mode set, MG solves at the gauge field of the MD (the passenger):
//      every SolveEvery steps of the trajectories in SolveTraj (empty: all).
//      The solve does not feed back, so with fixed seeds every Mode sees the
//      same chain.  Sources are Gaussian, refreshed per trajectory.
//
//   Mode none        no solves in the MD
//   Mode frozen      the fine operators follow U; basis and coarse operators
//                    stay at the starting configuration
//   Mode recoarsen   the same, and the coarse operators are rebuilt at U on
//                    the frozen basis (needs Setup.RetainSubspace)
//   Mode refresh     the live basis gets RefreshPasses sloppy inverse-iteration
//                    passes per vector at U (RefreshTol, RefreshMaxOuter restarts
//                    of GCR(10,10)), then the same rebuild.  A rejected
//                    trajectory restores the basis of its start.
//                    RefreshOrthonormalise 1 orthonormalises the basis after
//                    each pass (subspace iteration).  RefreshKick > 0 adds
//                    Gaussian noise of that size relative to |v| to every
//                    vector once per update, before the first pass, from its
//                    own RNG so the MD sources are unchanged.  Large kicks
//                    tend to a fresh setup of RefreshPasses rounds.
//                    RefreshKickShape global: the noise is uniform over sites;
//                    site: it is shaped to |v(x)| site by site, so the kick is
//                    about RefreshKick relative to v everywhere, walls included.
//                    Each solve logs QHMC-SPROFILE, the basis norm per s slice.
//   Mode perturbative  the first-order update v -> v - w, (Q A Q) w = Q dA v,
//                    with dA = A(U) - A(U at the last update) and Q the
//                    projector off the basis: mrhs GCR in batches of CorrNrhs,
//                    preconditioned by the V-cycle from the last update, to
//                    CorrTol, at most CorrMaxRestarts restarts.  Q is rebuilt
//                    from the current basis for each batch (no copy of the
//                    old one), an O(eps^2) change.  Then the same rebuild and
//                    the same restore on rejection.
//                    CorrOperator new (default): A on the left is A(U), and the
//                    V-cycle has its fine level at U and its coarse operators
//                    from the last update.  CorrOperator old: A on the left is
//                    A(U at the last update), the operator the basis and the
//                    V-cycle were built for; equal at first order.
//                    CorrRecoarsen 1 (CorrOperator new only): the coarse
//                    operators are rebuilt on the current basis at U before
//                    the V-cycle is built, so it is consistent with A(U).
//   Mode control     no MG (RunMultiGrid 0): the same system PV^dag M x = b
//                    at U by Krylov.  PV^dag y = b by CG on PV^dag PV (y = PV z),
//                    then M x = y by red-black CG to Outer.Tol on the Schur
//                    system.  The true residual of PV^dag M is logged after.
//
// RunMultiGrid 0 skips 1-3, for an HMC-only scan of the integrator.
//
// Every parameter is in ONE serialisable struct, read from XML with
// --hmc-params, as in Example_pvdagm_multigrid.  The basis:
//
//   MultiGrid.Setup.SubspaceFile set    the file is LOADED, and must exist:
//                                       given a missing file, GetSubspace
//                                       would generate and save over that path.
//   MultiGrid.Setup.SubspaceFile empty  the basis is GENERATED, then saved to
//                                       SaveSubspace if that is set.  An
//                                       existing SaveSubspace stops the run
//                                       before generating.
//
//   NoiseBasis 1 then replaces the basis by normalised Gaussian noise, from
//   its own RNG, before coarsening: with Mode refresh at a fixed field, the
//   solve before the MD has k = 0 rounds of inverse iteration and MD step n
//   has k = n RefreshPasses.
//
#include <Grid/Grid.h>
#include <Grid/lattice/PaddedCell.h>
#include <Grid/stencil/GeneralLocalStencil.h>
#include <Grid/algorithms/iterative/PrecGeneralisedConjugateResidualNonHermitian.h>
#include <Grid/algorithms/multigrid/PVdagMMultiGrid.h>
#include "Example_mg_hmc.h"
#include "Example_mg_refine.h"
#include "Example_basis_diagnostics.h"

using namespace std;
using namespace Grid;

// Q M for an mrhs preconditioner M.  GCR hands it residuals already in the
// range of Q, so only the output needs projecting.
template<class Field>
class MrhsProjectedPreconditioner : public MrhsLinearFunction<Field> {
  SpanProjector<Field>      &_Q;
  MrhsLinearFunction<Field> &_M;
public:
  MrhsProjectedPreconditioner(SpanProjector<Field> &Q, MrhsLinearFunction<Field> &M) : _Q(Q), _M(M) {};
  void operator()(std::vector<Field> &in, std::vector<Field> &out) {
    _M(in,out);
    for(auto &f : out) _Q.Complement(f,f);
  }
};

#ifndef NBASIS
#define NBASIS 60
#endif

#ifdef COARSE_SINGLE
typedef sTComplexF CoarseScalar_t;
#else
typedef sTComplexD CoarseScalar_t;
#endif

struct QuenchedHMCDriverParams : Serializable {
  GRID_SERIALIZABLE_CLASS_MEMBERS(QuenchedHMCDriverParams,
                                  int,         Ls,
                                  RealD,       Mass,
                                  RealD,       M5,
                                  RealD,       MobiusB,
                                  RealD,       MobiusC,
                                  std::string, Config,          // empty: hot start
                                  std::string, SaveSubspace,    // a generated basis is saved here; empty: not saved
                                  int,         Nrhs,
                                  int,         SolveSingleRHS,  // also run Nrhs=1 through the same objects
                                  RealD,       Beta,            // Iwasaki gauge action
                                  int,         RunMultiGrid,    // 0: skip sections 1-3, HMC only
                                  std::vector<RealD>, TrajL,    // one (TrajL,MDsteps) pair per scan point
                                  std::vector<int>,   MDsteps,
                                  int,         Ntraj,           // trajectories per pair
                                  int,         Chain,           // 1: each trajectory continues from the last
                                  std::string, Mode,            // none, frozen, recoarsen, refresh, perturbative or control
                                  int,         SolveEvery,      // MD steps between solves
                                  std::vector<int>, SolveTraj,  // trajectories that get solves; empty: all
                                  RealD,       RefreshTol,      // Mode refresh: GCR tolerance per vector
                                  int,         RefreshMaxOuter, // Mode refresh: restarts of GCR(10,10) per vector
                                  int,         RefreshPasses,   // Mode refresh: inverse-iteration passes per update
                                  int,         RefreshOrthonormalise, // Mode refresh: 1 orthonormalises after each pass
                                  RealD,       RefreshKick,     // Mode refresh: noise once per update, before the first pass, relative to |v|; 0: none
                                  std::string, RefreshKickShape, // Mode refresh: global (uniform noise) or site (noise shaped to |v(x)|)
                                  int,         NoiseBasis,      // 1: replace the basis by Gaussian noise before coarsening (k = 0)
                                  RealD,       CorrTol,         // Mode perturbative: batch GCR tolerance
                                  int,         CorrMaxRestarts, // Mode perturbative: restarts of the Outer cycle per batch
                                  int,         CorrNrhs,        // Mode perturbative: batch size, divides the basis size
                                  std::string, CorrOperator,    // Mode perturbative: new (A(U) on the left) or old (A at the last update)
                                  int,         CorrRecoarsen,   // Mode perturbative, CorrOperator new: 1 rebuilds the coarse operators at U before the V-cycle
                                  std::string, CorrPrecon,      // Mode perturbative: mg (mrhs GCR, V-cycle) or none (plain GCR per vector)
                                  int,         CorrKrylovNstep, // CorrPrecon none: GCR(Nstep,Nstep), CorrMaxRestarts restarts
                                  int,         DiagEvery,       // >0: QHMC-DIAG basis diagnostics after setup and at solves with step%DiagEvery==0
                                  int,         DiagTest,        // DiagEvery: test vectors from fresh noise, made as the setup's; 0: none
                                  int,         DiagTestRegen,   // DiagEvery: 1 makes new test vectors at every diagnostic (moving field); 0 once, after setup
                                  int,         DiagReference,   // DiagEvery: fp32 reference basis for angles, pair_cos and ref_loc: 1 the starting basis; 2 (Mode refresh) the basis before each update; 0 none
                                  std::vector<int>, DiagTestRounds, // DiagTest: depths of the test vectors (rounds of inverse iteration), ascending
                                  int,         DiagBlockGram,   // DiagEvery: 1 adds the block Gram spectrum of the raw basis (QHMC-DIAG-GRAM)
                                  int,         DiagBlockNul,    // DiagEvery: 1 adds the block-resolved nulliness of the basis vectors (QHMC-DIAG-BNUL)
                                  int,         DiagCR,          // DiagEvery: sweeps of compatible relaxation with the fine smoother before each solve (QHMC-CR); 0: none
                                  int,         DiagCRVectors,   // DiagCR: noise vectors
                                  PVdagMMultiGridParams, MultiGrid);
  QuenchedHMCDriverParams()
    : Ls(24), Mass(0.00078), M5(1.8), MobiusB(1.5), MobiusC(0.5),
      Config(""), SaveSubspace(""), Nrhs(12), SolveSingleRHS(1),
      Beta(2.13), RunMultiGrid(1), TrajL({1.0}), MDsteps({20}), Ntraj(1), Chain(0),
      Mode("none"), SolveEvery(5), RefreshTol(0.1), RefreshMaxOuter(1), RefreshPasses(1), RefreshOrthonormalise(0), RefreshKick(0.0), RefreshKickShape("global"), NoiseBasis(0),
      CorrTol(1.0e-2), CorrMaxRestarts(4), CorrNrhs(12), CorrOperator("new"), CorrRecoarsen(0), CorrPrecon("mg"), CorrKrylovNstep(10),
      DiagEvery(0), DiagTest(10), DiagTestRegen(0), DiagReference(1), DiagTestRounds({3}), DiagBlockGram(0), DiagBlockNul(0), DiagCR(0), DiagCRVectors(2) {};
};

int main (int argc, char ** argv)
{
  Grid_init(&argc,&argv);

  QuenchedHMCDriverParams P;
  // Read only when present, outside QuenchedHMCDriverParams: Grid's reader aborts
  // on an absent int tag, and a job submitted before these existed writes a
  // parameter file without them.  Absent, each takes the value below.
  //   TwinUpdate        Mode refresh, DiagReference 2: at each diagnostic step every
  //                     basis vector is also refreshed without the kick, its twin F(v),
  //                     next to the run's F(v + eps eta) (QHMC-TWIN, QHMC-DIAG-TWIN-*)
  //   TwinBaseline      TwinUpdate: that many evenly spaced vectors also refresh their
  //                     noise alone, F(eta); 0: none
  //   TwinBaselineEvery TwinBaseline: at steps with step%TwinBaselineEvery==0
  //   SolveMGEvery      the MG solve at steps with step%SolveMGEvery==0 only; the update,
  //                     rebuild and diagnostics still run at every SolveEvery step
  int TwinUpdate = 0, TwinBaseline = 0, TwinBaselineEvery = 1, SolveMGEvery = 1;
  {
    std::string pfile("");
    if( GridCmdOptionExists(argv,argv+argc,"--hmc-params") )
      pfile = GridCmdOptionPayload(argv,argv+argc,"--hmc-params");
    if ( pfile.length() ) {
      bool good; { std::ifstream f(pfile); good = f.good(); }
      if ( !good ) {
        if ( GlobalSharedMemory::WorldRank == 0 ) {
          XmlWriter WR(pfile+".templ");
          write(WR, "QuenchedHMCDriver", P);
          std::cout << GridLogMessage << pfile << " does not exist; template written to "
                    << pfile << ".templ" << std::endl;
        }
        Grid_finalize(); return 0;
      }
      XmlReader RD(pfile);
      read(RD, "QuenchedHMCDriver", P);
      if ( RD.push("QuenchedHMCDriver") ) {
        std::vector<std::pair<std::string,int *> > optional({{"TwinUpdate",&TwinUpdate},{"TwinBaseline",&TwinBaseline},
                                                             {"TwinBaselineEvery",&TwinBaselineEvery},{"SolveMGEvery",&SolveMGEvery}});
        for(auto &o : optional) if ( RD.push(o.first) ) { RD.pop(); read(RD,o.first,*o.second); }
        RD.pop();
      }
    }
    CheckValidity(P.MultiGrid);
    GRID_ASSERT( P.Mode=="none" || P.Mode=="frozen" || P.Mode=="recoarsen" || P.Mode=="refresh" || P.Mode=="perturbative" || P.Mode=="control" );
    GRID_ASSERT( P.Mode=="none" || P.Mode=="control" || P.RunMultiGrid );
    GRID_ASSERT( (P.Mode!="recoarsen" && P.Mode!="refresh" && P.Mode!="perturbative") || P.MultiGrid.Setup.RetainSubspace );
    GRID_ASSERT( P.Mode!="perturbative" || NBASIS%P.CorrNrhs==0 );
    GRID_ASSERT( P.CorrPrecon=="mg" || P.CorrPrecon=="none" );
    GRID_ASSERT( P.CorrOperator=="new" || P.CorrOperator=="old" );
    GRID_ASSERT( !P.CorrRecoarsen || P.CorrOperator=="new" );
    GRID_ASSERT( !P.DiagEvery || (P.RunMultiGrid && P.MultiGrid.Setup.RetainSubspace) );
    GRID_ASSERT( !P.DiagBlockGram || P.DiagEvery );
    if ( P.RefreshKickShape.empty() ) P.RefreshKickShape = "global";   // a parameter file from before RefreshKickShape
    GRID_ASSERT( P.RefreshKickShape=="global" || P.RefreshKickShape=="site" );
    GRID_ASSERT( P.DiagReference>=0 && P.DiagReference<=2 );
    GRID_ASSERT( (!P.DiagBlockNul && !P.DiagCR) || P.DiagEvery );
    GRID_ASSERT( P.DiagReference!=2 || P.Mode=="refresh" );
    GRID_ASSERT( P.DiagTestRounds.size()>0 && P.DiagTestRounds[0]>=1 );
    for(int d=1;d<P.DiagTestRounds.size();d++) GRID_ASSERT( P.DiagTestRounds[d]>P.DiagTestRounds[d-1] );
    // The twin refreshes each vector on its own, which is the run's refresh only without RefreshOrthonormalise
    GRID_ASSERT( !TwinUpdate || (P.Mode=="refresh" && P.RefreshKick>0.0 && P.DiagReference==2 && !P.RefreshOrthonormalise) );
    GRID_ASSERT( !TwinBaseline || (TwinUpdate && TwinBaseline<=NBASIS && TwinBaselineEvery>=1) );
    GRID_ASSERT( SolveMGEvery>=1 );
    std::cout << GridLogMessage << "QuenchedHMCDriver parameters ("
              << (pfile.length() ? pfile : std::string("defaults")) << "):" << std::endl;
    std::cout << P << std::endl;
    std::cout << GridLogMessage << "QHMC-PARAMS-OPTIONAL TwinUpdate " << TwinUpdate << " TwinBaseline " << TwinBaseline
              << " TwinBaselineEvery " << TwinBaselineEvery << " SolveMGEvery " << SolveMGEvery << std::endl;
  }

  Coordinate latt = GridDefaultLatt();
  Coordinate mpi  = GridDefaultMpi();
  Coordinate fsimd= GridDefaultSimd(Nd,vComplex::Nsimd());

  GridCartesian         * UGrid   = SpaceTimeGrid::makeFourDimGrid(latt,fsimd,mpi);
  GridRedBlackCartesian * UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian         * FGrid   = SpaceTimeGrid::makeFiveDimGrid(P.Ls,UGrid);
  GridRedBlackCartesian * FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(P.Ls,UGrid);

  GridParallelRNG RNG4(UGrid); RNG4.SeedFixedIntegers({1,2,3,4});
  GridParallelRNG RNG5(FGrid); RNG5.SeedFixedIntegers({5,6,7,8});
  GridParallelRNG RNGkick(FGrid); RNGkick.SeedFixedIntegers({9,10,11,12});  // Mode refresh kicks only
  GridParallelRNG RNGnoise(FGrid); RNGnoise.SeedFixedIntegers({13,14,15,16});  // NoiseBasis only
  GridSerialRNG   sRNG;        sRNG.SeedFixedIntegers({9,10,11,12});

  LatticeGaugeField Umu(UGrid);
  if ( P.Config.length() ) {
    std::cout << GridLogMessage << "Reading gauge field " << P.Config << std::endl;
    FieldMetaData header;
    NerscIO::readConfiguration(Umu,header,P.Config);
  } else {
    std::cout << GridLogMessage << "Hot start gauge field" << std::endl;
    SU<Nc>::HotConfiguration(RNG4,Umu);
  }

  //////////////////////////////////////////////////////////////////////
  // Basis files are checked before anything expensive runs
  //////////////////////////////////////////////////////////////////////
  auto FileExists = [&](const std::string &f) -> bool {
    uint64_t e=0;
    if ( FGrid->IsBoss() ){ std::ifstream s(f); e = s.good()?1:0; }
    FGrid->GlobalSum(e);
    return e!=0;
  };
  const std::string &SubspaceFile = P.MultiGrid.Setup.SubspaceFile;
  const int generate = ( SubspaceFile.length() == 0 );
  if ( P.RunMultiGrid && !generate && !FileExists(SubspaceFile) ) {
    std::cout << GridLogError << "SubspaceFile " << SubspaceFile << " does not exist" << std::endl;
    GRID_ASSERT(0);
  }
  if ( P.RunMultiGrid && generate && P.SaveSubspace.length() && FileExists(P.SaveSubspace) ) {
    std::cout << GridLogError << "SaveSubspace " << P.SaveSubspace << " exists; not overwriting" << std::endl;
    GRID_ASSERT(0);
  }

  MobiusFermionD Ddwf(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,P.Mass,P.M5,P.MobiusB,P.MobiusC);
  MobiusFermionD Dpv (Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,1.0,   P.M5,P.MobiusB,P.MobiusC);

  typedef PVdagMLinearOperator<MobiusFermionD,LatticeFermionD> PVdagM_t;
  PVdagM_t PVdagM(Ddwf,Dpv);

  // A at the U of the last update, for dA in Mode perturbative
  MobiusFermionD DdwfLast(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,P.Mass,P.M5,P.MobiusB,P.MobiusC);
  MobiusFermionD DpvLast (Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,1.0,   P.M5,P.MobiusB,P.MobiusC);
  PVdagM_t PVdagMLast(DdwfLast,DpvLast);

  MatvecCounter counter;
  CountingLinearOperator<LatticeFermionD> CountedPVdagM(PVdagM,counter);
  CountingLinearOperator<LatticeFermionD> CountedPVdagMLast(PVdagMLast,counter);

  typedef PVdagMMultiGridCoarsening<vSpinColourVector,CoarseScalar_t,NBASIS> Coarsening_t;
  std::cout << GridLogMessage << "Coarse sector precision (compiled): "
            << (sizeof(typename GridTypeMapper<CoarseScalar_t>::scalar_type)==sizeof(ComplexF) ? "fp32" : "fp64")
            << ", nbasis " << NBASIS << std::endl;

  MGCoarseGrids CGrids(FGrid, P.MultiGrid.Setup);
  MGFineGridsF  FGridsF(FGrid);

  LatticeGaugeFieldF UmuF(FGridsF.UGridF);
  precisionChange(UmuF,Umu);
  MobiusFermionF DdwfF(UmuF,*FGridsF.FGridF,*FGridsF.FrbGridF,*FGridsF.UGridF,*FGridsF.UrbGridF,P.Mass,P.M5,P.MobiusB,P.MobiusC);
  MobiusFermionF DpvF (UmuF,*FGridsF.FGridF,*FGridsF.FrbGridF,*FGridsF.UGridF,*FGridsF.UrbGridF,1.0,   P.M5,P.MobiusB,P.MobiusC);

  // The fp32 operators at the U of the last update: the fine level of the
  // correction's V-cycle under CorrOperator old
  LatticeGaugeFieldF UmuLastF(FGridsF.UGridF);
  UmuLastF = UmuF;
  MobiusFermionF DdwfLastF(UmuLastF,*FGridsF.FGridF,*FGridsF.FrbGridF,*FGridsF.UGridF,*FGridsF.UrbGridF,P.Mass,P.M5,P.MobiusB,P.MobiusC);
  MobiusFermionF DpvLastF (UmuLastF,*FGridsF.FGridF,*FGridsF.FrbGridF,*FGridsF.UGridF,*FGridsF.UrbGridF,1.0,   P.M5,P.MobiusB,P.MobiusC);
  // The last-update operators follow U, in both precisions
  auto ImportLast = [&]()
  {
    DdwfLast.ImportGauge(Umu);
    DpvLast.ImportGauge(Umu);
    precisionChange(UmuLastF,Umu);
    DdwfLastF.ImportGauge(UmuLastF);
    DpvLastF.ImportGauge(UmuLastF);
  };

  Coarsening_t  Coarsening(CGrids, FGridsF, P.MultiGrid.Setup);

  auto RunSolve = [&](int nr)
  {
    PVdagMMultiGridSolver<MobiusFermionD,MobiusFermionF,Coarsening_t> Solver(Ddwf,Dpv,DdwfF,DpvF,Coarsening,P.MultiGrid,nr);
    std::vector<LatticeFermionD> src(nr,FGrid), sol(nr,FGrid);
    for(int r=0;r<nr;r++){ gaussian(RNG5,src[r]); sol[r]=Zero(); }
    Solver.Solve(src,sol);
  };

  auto Rebuild = [&]()
  {
    Coarsening.Coarsen(PVdagM);
    Coarsening.BuildDenseBottom();
  };

  //////////////////////////////////////////////////////////////////////
  // Basis diagnostics (DiagEvery > 0), one QHMC-DIAG line and QHMC-DIAG-LIST
  // lines per call, on the raw basis and the coarsening built from it:
  //   nul, rq       ||A v||/||v|| and |<v|A v>|/|v|^2 of the basis vectors
  //   sin2, cos     principal angles to the reference (the starting basis):
  //                 mean sin^2, smallest cosine, cosines above 0.99 and 0.9
  //   ref_loc/glob  ||(1 - P P^dag) w||/||w|| and ||Q w||/||w|| for the
  //                 reference vectors w, P the current block prolongator
  // and, from the hand-off notes/mg-hmc-peter/BasisColinearityDiagnostic_Handoff.md:
  //   QHMC-DIAG-TEST  section 3.1, one line per depth in DiagTestRounds: DiagTest
  //                 test vectors t made from fresh noise as the setup makes its
  //                 basis, with that many rounds, against the current coarsening:
  //                 defect ||(1 - P P^dag) t||^2/||t||^2, adefect (the same
  //                 with A applied), glob ||Q t||^2/||t||^2, nul ||A t||/||t||,
  //                 and the per-block defect as quantiles over the blocks
  //   QHMC-DIAG-PREV  the same vectors against the coarsening of the last update,
  //                 measured before the update (Mode refresh, perturbative): the
  //                 gain from this update.  DiagTestRegen 1 makes the vectors at
  //                 the field of each diagnostic, before the update
  //   QHMC-DIAG-GRAM  section 3.2 (DiagBlockGram 1): per-block spectrum of the
  //                 column-normalised Gram of the raw basis, as quantiles over blocks,
  //                 with gmean = (det C_b)^{1/N} and the numerical ranks at 1e-2,
  //                 1e-3, 1e-4; QHMC-DIAG-GRAM-SPEC the typical sorted spectrum,
  //                 exp of the mean over blocks of log lambda_k
  // With DiagReference 2 (rolling: the reference is the basis before this update):
  //   QHMC-OVERLAP    during the update, |<v_old|v>|/(|v_old| |v|) per vector after
  //                 the kick and after each pass
  //   pair_cos        the same after the update, on the QHMC-DIAG line
  //   QHMC-DIAG-ANGLES section 3.3 (with DiagBlockGram 1 at the previous step too):
  //                 principal angles between the old and new block spans, per block
  //   QHMC-DIAG-BNUL  (DiagBlockNul 1): ||A v_i||_b/||v_i||_b, the nulliness of each
  //                 basis vector measured on each block (A on the whole vector): per
  //                 block the min, median and max over the vectors, as quantiles over
  //                 blocks, the counts of vectors below 0.02, 0.05, 0.1, and the
  //                 typical sorted local spectrum
  //   QHMC-CR         (DiagCR > 0): compatible relaxation, the solver's fine smoother
  //                 on A e = 0 with (1 - P P^dag) after each sweep, from DiagCRVectors
  //                 noise vectors: ||e_k||/||e_{k-1}|| and ||A e_k||/||A e_{k-1}|| per sweep
  // With TwinUpdate 1 (DiagReference 2), F the refresh, v a basis vector before the
  // update and eta its kick:
  //   QHMC-TWIN-KICK  nul of v, of eta and of v + eps eta, and cos(v, v + eps eta)
  //   QHMC-TWIN       after each pass: cos(F(v + eps eta), F(v)) and the nul and GCR
  //                 steps of both; with the baseline, also cos(F(v + eps eta), F(eta)),
  //                 cos(F(v), F(eta)), and the nul and steps of F(eta)
  //   QHMC-DIAG-TWIN-GRAM, -TWIN-ANGLES, -TWIN-GLOB (DiagBlockGram 1): the block and
  //                 global Gram of the twin basis F(V), the collapse without the kick;
  //                 principal angles between its block spans and the run's; and
  //                 between the two global spans
  // The twin basis is NBASIS fp32 fields; the per-vector work 5 fp64 fields more.
  // Uses PVdagM, not the counted operator, so update_matvecs is unchanged.
  // Device memory: the reference is NBASIS fp32 fields (half the basis), the
  // test vectors DiagTest x (number of depths) fp32 fields and DiagTest fp64
  // fields while they are made, the chunks of the projections 2 x 10 fp64
  // fields more while they run.
  //////////////////////////////////////////////////////////////////////
  GridParallelRNG RNGtest(FGrid); RNGtest.SeedFixedIntegers({17,18,19,20});  // diagnostic test vectors only
  GridParallelRNG RNGcr(FGrid);   RNGcr.SeedFixedIntegers({21,22,23,24});    // compatible relaxation only
  std::vector<LatticeFermionF> DiagRef;
  std::vector<LatticeFermionF> DiagT;      // DiagTest vectors per depth, depth-major
  bool DiagTFresh = false;                 // made at this field, before an update, not yet measured after it
  Eigen::MatrixXcd DiagGW;
  std::vector<Eigen::MatrixXcd> DiagGramKeep;   // per-block Grams of the basis at the last QHMC-DIAG-GRAM
  std::vector<Eigen::MatrixXcd> DiagGramRef;    // the same for the rolling reference, when they match it
  int DiagBasisVersion = 0;                     // counts the updates of the basis
  int DiagGramVersion  = -1;                    // DiagBasisVersion when DiagGramKeep was taken
  std::vector<LatticeFermionF> TwinF;           // TwinUpdate: the twin basis F(V) of the last update
  int TwinVersion      = -1;                    // DiagBasisVersion of the basis TwinF was refreshed next to
  const int DiagChunk = 10;
  GridCartesian *BlockGrid = P.DiagEvery ? BasisBlockGrid(FGrid,CGrids.Coarse5d) : nullptr;
  auto DiagStep = [&](int step) -> bool { return P.DiagEvery && step%P.DiagEvery==0; };
  // Principal cosines as mean sin^2, smallest cosine, and counts above 0.99 and 0.9
  auto PrincipalStats = [&](const std::vector<RealD> &cosv) -> std::string
  {
    RealD sin2 = 0.0;
    int n99 = 0, n90 = 0;
    for(auto c : cosv){ sin2 += (1.0-c*c)/cosv.size(); n99 += (c>0.99); n90 += (c>0.9); }
    std::ostringstream s;
    s.precision(6);
    s << " sin2_mean " << sin2 << " cos_min " << cosv.back() << " n_cos99 " << n99 << " n_cos90 " << n90;
    return s.str();
  };
  // QHMC-DIAG-<tag>, -SPEC, -GSPEC and -LIST: the block and global Gram of a basis W,
  // fp64 or fp32; keep receives its per-block Grams and G its global Gram
  auto ReportGram = [&](const std::string &tag,int traj,int step,auto &W,
                        std::vector<Eigen::MatrixXcd> &keep,Eigen::MatrixXcd &G)
  {
    RealD t_gram = -usecond();
    std::vector<RealD> lmin, kappa, deff, pres, gmean, logspec;
    const std::vector<RealD> taus({1.0e-2,1.0e-3,1.0e-4});
    std::vector<std::vector<RealD> > rank;
    std::vector<RealD> glspec;
    BasisBlockGram<LatticeFermionD>(W,FGrid,BlockGrid,lmin,kappa,deff,pres,gmean,taus,rank,logspec,glspec,&keep,&G);
    RealD glogdet = 0.0;                                   // log det C of the unblocked basis
    for(auto x : glspec) glogdet += std::log(std::max(x,1.0e-16));
    t_gram = (t_gram+usecond())/1.0e6;
    std::vector<RealD> ql = BasisQuantiles(lmin, {0.0,0.1,0.5});
    std::vector<RealD> qk = BasisQuantiles(kappa,{0.5,0.9,1.0});
    std::vector<RealD> qe = BasisQuantiles(deff, {0.0,0.1,0.5,1.0});
    std::vector<RealD> qp = BasisQuantiles(pres, {0.0,0.1,0.5});
    std::vector<RealD> qg = BasisQuantiles(gmean,{0.0,0.1,0.5});
    std::ostringstream gram;
    gram.precision(6);
    gram << "QHMC-DIAG-" << tag << " traj " << traj << " step " << step << " N " << W.size() << " nblocks " << lmin.size()
         << " lmin_p0 " << ql[0] << " lmin_p10 " << ql[1] << " lmin_p50 " << ql[2]
         << " kappa_p50 " << qk[0] << " kappa_p90 " << qk[1] << " kappa_max " << qk[2]
         << " deff_p0 " << qe[0] << " deff_p10 " << qe[1] << " deff_p50 " << qe[2] << " deff_max " << qe[3]
         << " pres_p0 " << qp[0] << " pres_p10 " << qp[1] << " pres_p50 " << qp[2]
         << " gmean_p0 " << qg[0] << " gmean_p10 " << qg[1] << " gmean_p50 " << qg[2];
    const char *tname[3] = {"rank2","rank3","rank4"};
    for(int t=0;t<3;t++){
      std::vector<RealD> qr = BasisQuantiles(rank[t],{0.0,0.1,0.5});
      gram << " " << tname[t] << "_p0 " << qr[0] << " " << tname[t] << "_p10 " << qr[1] << " " << tname[t] << "_p50 " << qr[2];
    }
    gram << " glob_lmin " << glspec.front() << " glob_lmax " << glspec.back()
         << " glob_logdet " << glogdet << " glob_gmean " << std::exp(glogdet/glspec.size());
    for(int t=0;t<3;t++){
      int c = 0;
      for(auto x : glspec) c += (x > taus[t]);
      gram << " glob_" << tname[t] << " " << c;
    }
    gram << " seconds " << t_gram;
    std::cout << GridLogMessage << gram.str() << std::endl;
    std::vector<RealD> spec;
    for(auto x : logspec) spec.push_back(std::exp(x));
    std::cout << GridLogMessage << "QHMC-DIAG-" << tag << "-SPEC traj " << traj << " step " << step
              << " lambda " << BasisList(spec) << std::endl;
    std::cout << GridLogMessage << "QHMC-DIAG-" << tag << "-GSPEC traj " << traj << " step " << step
              << " lambda " << BasisList(glspec) << std::endl;
    std::cout << GridLogMessage << "QHMC-DIAG-" << tag << "-LIST traj " << traj << " step " << step
              << " lmin_worst " << BasisBlockWorst(BlockGrid,lmin,10,false)
              << " pres_worst " << BasisBlockWorst(BlockGrid,pres,10,false) << std::endl;
  };
  // QHMC-DIAG-<tag> and -LIST: principal angles between the block spans of W (its
  // per-block Grams GW) and of the current basis V (its per-block Grams GV)
  auto ReportAngles = [&](const std::string &tag,int traj,int step,std::vector<LatticeFermionF> &W,
                          const std::vector<Eigen::MatrixXcd> &GW,std::vector<LatticeFermionD> &V,
                          const std::vector<Eigen::MatrixXcd> &GV)
  {
    RealD t_ang = -usecond();
    std::vector<RealD> sin2, cmin;
    BasisBlockAngles(W,GW,V,GV,BlockGrid,1.0e-10,sin2,cmin);
    t_ang = (t_ang+usecond())/1.0e6;
    std::vector<RealD> qs = BasisQuantiles(sin2,{0.1,0.5,0.9,1.0});
    std::vector<RealD> qc = BasisQuantiles(cmin,{0.0,0.1,0.5});
    std::cout << GridLogMessage << "QHMC-DIAG-" << tag << " traj " << traj << " step " << step
              << " sin2_p10 " << qs[0] << " sin2_p50 " << qs[1] << " sin2_p90 " << qs[2] << " sin2_max " << qs[3]
              << " cmin_p0 " << qc[0] << " cmin_p10 " << qc[1] << " cmin_p50 " << qc[2]
              << " seconds " << t_ang << std::endl;
    std::cout << GridLogMessage << "QHMC-DIAG-" << tag << "-LIST traj " << traj << " step " << step
              << " sin2_worst " << BasisBlockWorst(BlockGrid,sin2,10,true) << std::endl;
  };
  auto MakeTest = [&]()
  {
    DiagT.assign(P.DiagTest*P.DiagTestRounds.size(),LatticeFermionF(FGridsF.FGridF));
    BasisTestVectors(DiagT,P.DiagTestRounds,PVdagM,FGrid,RNGtest);
    DiagTFresh = true;
  };
  // The test vectors against the current coarsening: one QHMC-DIAG-<tag> line
  // and one -LIST line per depth
  auto MeasureTest = [&](const std::string &tag,int traj,int step,SpanProjector<LatticeFermionD> *Q)
  {
    for(int d=0;d<P.DiagTestRounds.size();d++){
      RealD t_meas = -usecond();
      std::vector<RealD> defect, adefect, glob, nul, p50, p90, bmax;
      std::vector<std::vector<RealD> > blocks;
      BasisDefect<LatticeFermionD,Coarsening_t::CoarseVector>(Coarsening.MrhsProjectorL1,PVdagM,Q,FGrid,CGrids.Coarse5d,BlockGrid,
                                                               DiagT,d*P.DiagTest,P.DiagTest,DiagChunk,
                                                               defect,adefect,glob,nul,blocks);
      std::string worst;
      for(auto &b : blocks){
        std::vector<RealD> q = BasisQuantiles(b,{0.5,0.9,1.0});
        p50.push_back(q[0]); p90.push_back(q[1]); bmax.push_back(q[2]);
        worst += (worst.size() ? "," : "") + BasisBlockWorst(BlockGrid,b,1,true);
      }
      t_meas = (t_meas+usecond())/1.0e6;
      std::ostringstream line, list;
      line << "QHMC-DIAG-" << tag << " traj " << traj << " step " << step << " rounds " << P.DiagTestRounds[d]
           << BasisStat("defect",defect) << BasisStat("adefect",adefect);
      if ( Q ) line << BasisStat("glob",glob);
      line << BasisStat("nul",nul) << BasisStat("blk_p50",p50) << BasisStat("blk_p90",p90) << BasisStat("blk_max",bmax)
           << " seconds " << t_meas;
      list << "QHMC-DIAG-" << tag << "-LIST traj " << traj << " step " << step << " rounds " << P.DiagTestRounds[d]
           << " defect " << BasisList(defect) << " adefect " << BasisList(adefect);
      if ( Q ) list << " glob " << BasisList(glob);
      list << " nul " << BasisList(nul) << " blk_p50 " << BasisList(p50) << " blk_p90 " << BasisList(p90)
           << " blk_max " << BasisList(bmax) << " blk_worst " << worst;
      std::cout << GridLogMessage << line.str() << std::endl;
      std::cout << GridLogMessage << list.str() << std::endl;
    }
  };
  // Before an update at a diagnostic step: the test vectors (made now, at this
  // field, if DiagTestRegen) against the coarsening of the last update
  auto DiagnoseBefore = [&](int traj,int step)
  {
    if ( !DiagStep(step) || P.DiagTest<=0 ) return;
    if ( DiagT.empty() || P.DiagTestRegen ) MakeTest();
    MeasureTest("PREV",traj,step,nullptr);
  };
  auto Diagnose = [&](int traj,int step)
  {
    if ( !DiagStep(step) ) return;
    RealD t_diag = -usecond();
    std::vector<LatticeFermionD> &V = Coarsening.rawNull;
    SpanProjector<LatticeFermionD> Q(V);
    GRID_ASSERT( Q.Ok() );
    std::vector<RealD> nul, rq;
    BasisNulliness(V,PVdagM,nul,rq);
    std::ostringstream line, lists;
    line.precision(6);
    line << "QHMC-DIAG traj " << traj << " step " << step << BasisStat("nul",nul) << BasisStat("rq",rq);
    lists << "QHMC-DIAG-LIST traj " << traj << " step " << step << " nul " << BasisList(nul) << " rq " << BasisList(rq);
    if ( DiagRef.size() ) {
      Eigen::MatrixXcd GV = BasisCrossGram(V,V);
      Eigen::MatrixXcd C  = BasisCrossGram(DiagRef,V);
      std::vector<RealD> cosv = PrincipalCosines(DiagGW,GV,C);
      std::vector<RealD> loc, glob;
      BasisRepresentation<LatticeFermionD,Coarsening_t::CoarseVector>(Coarsening.MrhsProjectorL1,Q,FGrid,CGrids.Coarse5d,
                                                                       DiagRef,DiagChunk,loc,glob);
      line << PrincipalStats(cosv) << BasisStat("ref_loc",loc) << BasisStat("ref_glob",glob);
      lists << " cos " << BasisList(cosv) << " ref_loc " << BasisList(loc);
      if ( P.DiagReference==2 ) {
        std::vector<RealD> pc = BasisPairCosines(DiagRef,V);
        line << BasisStat("pair_cos",pc);
        lists << " pair_cos " << BasisList(pc);
      }
    }
    if ( P.DiagTest>0 ) {
      if ( DiagT.empty() || (P.DiagTestRegen && !DiagTFresh) ) MakeTest();
      MeasureTest("TEST",traj,step,&Q);
      DiagTFresh = false;
    }
    if ( P.DiagBlockGram ) {
      Eigen::MatrixXcd GV;
      ReportGram("GRAM",traj,step,V,DiagGramKeep,GV);
      DiagGramVersion = DiagBasisVersion;
      if ( P.DiagReference==2 && DiagGramRef.size() ) {
        ReportAngles("ANGLES",traj,step,DiagRef,DiagGramRef,V,DiagGramKeep);
        DiagGramRef.clear();
      }
      if ( TwinVersion==DiagBasisVersion ) {
        // The twin basis F(V), refreshed without the kick in this update, against the run's
        std::vector<Eigen::MatrixXcd> GTb;
        Eigen::MatrixXcd GT;
        ReportGram("TWIN-GRAM",traj,step,TwinF,GTb,GT);
        ReportAngles("TWIN-ANGLES",traj,step,TwinF,GTb,V,DiagGramKeep);
        RealD t_glob = -usecond();
        std::ostringstream glob;
        glob.precision(6);
        glob << "QHMC-DIAG-TWIN-GLOB traj " << traj << " step " << step;
        // PrincipalCosines asserts on a Gram that is not positive definite in working precision
        if ( Eigen::LLT<Eigen::MatrixXcd>(GT).info()==Eigen::Success && Eigen::LLT<Eigen::MatrixXcd>(GV).info()==Eigen::Success ) {
          std::vector<RealD> cosv = PrincipalCosines(GT,GV,BasisCrossGram(TwinF,V));
          t_glob = (t_glob+usecond())/1.0e6;
          glob << PrincipalStats(cosv) << " seconds " << t_glob << " cos " << BasisList(cosv);
        } else {
          glob << " singular 1";
        }
        std::cout << GridLogMessage << glob.str() << std::endl;
      }
    }
    if ( P.DiagBlockNul ) {
      RealD t_bn = -usecond();
      const std::vector<RealD> taus({0.02,0.05,0.1});
      std::vector<RealD> bmin, bmed, bmax, logspec;
      std::vector<std::vector<RealD> > count;
      BasisBlockNulliness(V,PVdagM,BlockGrid,taus,bmin,bmed,bmax,count,logspec);
      t_bn = (t_bn+usecond())/1.0e6;
      std::vector<RealD> q1 = BasisQuantiles(bmin,{0.0,0.1,0.5,0.9});
      std::vector<RealD> qm = BasisQuantiles(bmed,{0.1,0.5,0.9});
      std::vector<RealD> qx = BasisQuantiles(bmax,{0.5,0.9,1.0});
      std::ostringstream bn;
      bn.precision(6);
      bn << "QHMC-DIAG-BNUL traj " << traj << " step " << step
         << " bmin_p0 " << q1[0] << " bmin_p10 " << q1[1] << " bmin_p50 " << q1[2] << " bmin_p90 " << q1[3]
         << " bmed_p10 " << qm[0] << " bmed_p50 " << qm[1] << " bmed_p90 " << qm[2]
         << " bmax_p50 " << qx[0] << " bmax_p90 " << qx[1] << " bmax_max " << qx[2];
      const char *cname[3] = {"n002","n005","n01"};
      for(int t=0;t<3;t++){
        std::vector<RealD> qc = BasisQuantiles(count[t],{0.1,0.5,0.9});
        bn << " " << cname[t] << "_p10 " << qc[0] << " " << cname[t] << "_p50 " << qc[1] << " " << cname[t] << "_p90 " << qc[2];
      }
      bn << " seconds " << t_bn;
      std::vector<RealD> spec;
      for(auto x : logspec) spec.push_back(std::exp(x));
      std::cout << GridLogMessage << bn.str() << std::endl;
      std::cout << GridLogMessage << "QHMC-DIAG-BNUL-SPEC traj " << traj << " step " << step
                << " nul " << BasisList(spec) << std::endl;
      std::cout << GridLogMessage << "QHMC-DIAG-BNUL-LIST traj " << traj << " step " << step
                << " bmed_worst " << BasisBlockWorst(BlockGrid,bmed,10,true)
                << " bmax_worst " << BasisBlockWorst(BlockGrid,bmax,10,true) << std::endl;
    }
    t_diag = (t_diag+usecond())/1.0e6;
    std::cout << GridLogMessage << line.str() << " seconds " << t_diag << std::endl;
    std::cout << GridLogMessage << lists.str() << std::endl;
  };

  if ( P.RunMultiGrid ) {
    //////////////////////////////////////////////////////////////////////
    // 1. Basis: generate (or load), then save a generated one
    //////////////////////////////////////////////////////////////////////
    counter.Set("subspace");
    RealD t_subspace = -usecond();
    Coarsening.GetSubspace(RNG5, CountedPVdagM);
    t_subspace = (t_subspace+usecond())/1.0e6;
    std::cout << GridLogMessage << "QHMC-SUBSPACE " << (generate ? "generated" : "loaded")
              << " seconds " << t_subspace << " fine_matvecs " << counter.Get("subspace") << std::endl;

    if ( generate && P.SaveSubspace.length() ) {
      RealD t_save = -usecond();
      saveSubspace(Coarsening.rawNull, P.SaveSubspace);
      t_save = (t_save+usecond())/1.0e6;
      std::cout << GridLogMessage << "QHMC-SAVE " << P.SaveSubspace << " seconds " << t_save << std::endl;
    }

    // The reference for the diagnostics: the basis as loaded or generated,
    // before NoiseBasis replaces it
    if ( P.DiagEvery && P.DiagReference==1 ) {
      DiagRef.assign(NBASIS,LatticeFermionF(FGridsF.FGridF));
      for(int b=0;b<NBASIS;b++) precisionChange(DiagRef[b],Coarsening.rawNull[b]);
      DiagGW = BasisCrossGram(Coarsening.rawNull,Coarsening.rawNull);
    }

    // The A^{-k} scan from k = 0: noise in place of the basis, so that each
    // Mode refresh pass adds one round of inverse iteration
    if ( P.NoiseBasis ) {
      for(auto &v : Coarsening.rawNull){
        gaussian(RNGnoise,v);
        v = v*std::pow(norm2(v),-0.5);
      }
      std::cout << GridLogMessage << "QHMC-NOISEBASIS basis replaced by normalised Gaussian noise" << std::endl;
    }

    //////////////////////////////////////////////////////////////////////
    // 2. Coarsen, dense bottom, certificate
    //////////////////////////////////////////////////////////////////////
    RealD t_coarsen = -usecond();
    Coarsening.Coarsen(PVdagM);
    t_coarsen = (t_coarsen+usecond())/1.0e6;
    RealD t_dense = -usecond();
    Coarsening.BuildDenseBottom();
    t_dense = (t_dense+usecond())/1.0e6;
    Coarsening.CertifyCoarsening(PVdagM);
    std::cout << GridLogMessage << "QHMC-SETUP coarsen_seconds " << t_coarsen
              << " dense_seconds " << t_dense << std::endl;
    Diagnose(0,0);

    //////////////////////////////////////////////////////////////////////
    // 3. Solves: mrhs then (optionally) single RHS through the SAME objects
    //////////////////////////////////////////////////////////////////////
    RunSolve(P.Nrhs);
    if ( P.SolveSingleRHS && P.Nrhs != 1 ) RunSolve(1);

    // The rebuild path against the first solve: same U, same basis
    if ( P.Mode=="recoarsen" ) {
      Rebuild();
      std::cout << GridLogMessage << "QHMC-REBUILD-CHECK solve after a rebuild at the starting configuration" << std::endl;
      RunSolve(P.Nrhs);
    }
  }

  //////////////////////////////////////////////////////////////////////
  // 4. Quenched trajectories, Ntraj per (TrajL,MDsteps) pair
  //////////////////////////////////////////////////////////////////////
  typedef PeriodicGimplR                     Gimpl;
  typedef NoSmearing<Gimpl>                  SmearingPolicy;
  typedef MinimumNorm2<Gimpl,SmearingPolicy> IntegratorType;
  IwasakiGaugeActionR GaugeAction(P.Beta);
  ActionLevel<LatticeGaugeField> Level1(1);
  Level1.push_back(&GaugeAction);
  ActionSet<LatticeGaugeField,NoHirep> FullSet;
  FullSet.push_back(Level1);
  GRID_ASSERT( P.TrajL.size()>0 && P.TrajL.size()==P.MDsteps.size() );

  //////////////////////////////////////////////////////////////////////
  // 5. The passenger solve, at the field of an MD step
  //////////////////////////////////////////////////////////////////////
  std::vector<LatticeFermionD> mdsrc(P.Mode=="none" ? 0 : P.Nrhs,FGrid);
  std::vector<LatticeFermionD> mdsol(P.Mode=="none" ? 0 : P.Nrhs,FGrid);
  const bool updates  = ( P.Mode=="refresh" || P.Mode=="perturbative" );
  const bool rebuilds = ( P.Mode=="recoarsen" || updates );

  // Free device memory in MB on the emptiest rank, so growth shows in the log
  // before an allocation fails.  For the QHMC-MEM line of each solve.
  auto FreeDeviceMB = [&]() -> RealD
  {
    size_t free_t=0,total_t=0;
#if defined(GRID_HIP)
    auto discard = hipMemGetInfo(&free_t,&total_t);
#elif defined(GRID_CUDA)
    cudaMemGetInfo(&free_t,&total_t);
#endif
    RealD neg = -(RealD)(free_t>>20);
    FGrid->GlobalMax(neg);
    return -neg;
  };

  // One first-order update of the basis, at the operators already imported;
  // the coarse operators are still those of the last update.  Returns the mean
  // outer steps per batch.
  auto PerturbativeBasisUpdate = [&](int traj,int step) -> RealD
  {
    std::vector<LatticeFermionD> &V = Coarsening.rawNull;
    // CorrPrecon none needs no MG solver: plain GCR on QAQ per vector, as the
    // refresh's GCR(10,10) but on the correction equation.
    // CorrOperator old: A on the left and the V-cycle's fine level both at the
    // last update's U, where the basis and the coarse operators were built.
    const bool old = ( P.CorrOperator=="old" );
    LinearOperatorBase<LatticeFermionD> &Acorr = old ? (LinearOperatorBase<LatticeFermionD> &)CountedPVdagMLast
                                                     : (LinearOperatorBase<LatticeFermionD> &)CountedPVdagM;
    // CorrRecoarsen: the coarse operators at U on the current basis first, so
    // the V-cycle is consistent with A(U); SolveAtU rebuilds again after the update
    if ( P.CorrRecoarsen && P.CorrPrecon=="mg" ) {
      RealD t_rc = -usecond();
      Rebuild();
      t_rc = (t_rc+usecond())/1.0e6;
      std::cout << GridLogMessage << "QHMC-CORR-RECOARSEN traj " << traj << " step " << step << " seconds " << t_rc << std::endl;
    }
    std::unique_ptr<PVdagMMultiGridSolver<MobiusFermionD,MobiusFermionF,Coarsening_t>> Solver;
    if ( P.CorrPrecon=="mg" ) {
      if ( old ) Solver.reset(new PVdagMMultiGridSolver<MobiusFermionD,MobiusFermionF,Coarsening_t>(DdwfLast,DpvLast,DdwfLastF,DpvLastF,Coarsening,P.MultiGrid,P.CorrNrhs));
      else       Solver.reset(new PVdagMMultiGridSolver<MobiusFermionD,MobiusFermionF,Coarsening_t>(Ddwf,Dpv,DdwfF,DpvF,Coarsening,P.MultiGrid,P.CorrNrhs));
    }
    RealD mem_solver = FreeDeviceMB();

    // No copy of the old basis (60 fine fields): Q is rebuilt from the current
    // V for each batch, so later batches project off a span in which earlier
    // vectors have already moved by O(eps).  w is O(eps), so this changes it at
    // O(eps^2), inside the first-order update.
    std::vector<LatticeFermionD> rhs(P.CorrNrhs,FGrid), w(P.CorrNrhs,FGrid), vold(P.CorrNrhs,FGrid);
    LatticeFermionD Aold(FGrid);
    RealD steps = 0.0;
    for(int b0=0;b0<NBASIS;b0+=P.CorrNrhs){
      SpanProjector<LatticeFermionD> Q(V);
      GRID_ASSERT( Q.Ok() );
      ProjectedLinearOperator<LatticeFermionD> QAQ(Q,Acorr);
      for(int r=0;r<P.CorrNrhs;r++){
        vold[r] = V[b0+r];
        CountedPVdagM.Op(vold[r],rhs[r]);
        PVdagMLast.Op(vold[r],Aold);
        rhs[r] = rhs[r] - Aold;
        Q.Complement(rhs[r],rhs[r]);
      }
      RealD t_batch = -usecond();
      RealD batch_steps = 0.0;
      if ( P.CorrPrecon=="mg" ) {
        MrhsProjectedPreconditioner<LatticeFermionD> QM(Q,Solver->ThreeLevelPrecon);
        MrhsPGCRNonHermitian<LatticeFermionD> GCR(P.CorrTol,P.CorrMaxRestarts,QAQ,QM,P.MultiGrid.Outer.Mmax,P.MultiGrid.Outer.Nstep);
        GCR.Name("Fcorrect");
        GCR.SetZeroGuess(1);
        GCR(rhs,w);
        batch_steps = GCR.steps;
      } else {
        TrivialPrecon<LatticeFermionD> simple;
        PrecGeneralisedConjugateResidualNonHermitian<LatticeFermionD> GCR(P.CorrTol,P.CorrMaxRestarts,QAQ,simple,P.CorrKrylovNstep,P.CorrKrylovNstep);
        for(int r=0;r<P.CorrNrhs;r++){
          w[r] = Zero();
          GCR(rhs[r],w[r]);
          batch_steps += (RealD)GCR.Steps()/P.CorrNrhs;
        }
      }
      t_batch = (t_batch+usecond())/1.0e6;
      steps += batch_steps*P.CorrNrhs/NBASIS;
      RealD wratio = 0.0;
      for(int r=0;r<P.CorrNrhs;r++){
        wratio += std::sqrt(norm2(w[r])/norm2(vold[r]))/P.CorrNrhs;
        V[b0+r] = vold[r] - w[r];
        V[b0+r] = V[b0+r]*std::pow(norm2(V[b0+r]),-0.5);
      }
      std::cout << GridLogMessage << "QHMC-CORRECT traj " << traj << " step " << step << " first " << b0
                << " steps " << batch_steps << " seconds " << t_batch << " wratio " << wratio
                << " free_solver_MB " << mem_solver << " free_batch_MB " << FreeDeviceMB() << std::endl;
    }
    return steps;
  };
  auto SolveAtU = [&](int traj,int step,RealD t_MD)
  {
    RealD t_import = -usecond();
    Ddwf.ImportGauge(Umu);
    Dpv.ImportGauge(Umu);
    precisionChange(UmuF,Umu);
    DdwfF.ImportGauge(UmuF);
    DpvF.ImportGauge(UmuF);
    t_import = (t_import+usecond())/1.0e6;
    if ( updates ) DiagnoseBefore(traj,step);

    RealD mem_start = FreeDeviceMB();
    RealD null_before = InvarianceResidual(Coarsening.rawNull,PVdagM);
    RealD t_update = 0.0;
    uint64_t mv_update = 0;
    RealD update_steps = 0.0;
    if ( updates ) {
      counter.Set("update");
      uint64_t m0 = counter.Get("update");
      t_update = -usecond();
      if ( P.Mode=="refresh" ) {
        // DiagReference 2: the basis before this update becomes the reference, with its
        // block Grams when the last QHMC-DIAG-GRAM was taken on it
        const bool overlap = DiagStep(step) && P.DiagReference==2;
        RealD t_overlap = 0.0;   // kept out of update_seconds
        auto Overlap = [&](const std::string &stage) {
          t_overlap -= usecond();
          std::vector<RealD> c = BasisPairCosines(DiagRef,Coarsening.rawNull);
          std::cout << GridLogMessage << "QHMC-OVERLAP traj " << traj << " step " << step << " stage " << stage
                    << BasisStat("cos",c) << " list " << BasisList(c) << std::endl;
          t_overlap += usecond();
        };
        if ( overlap ) {
          t_overlap -= usecond();
          if ( DiagRef.size()!=NBASIS ) DiagRef.assign(NBASIS,LatticeFermionF(FGridsF.FGridF));
          for(int b=0;b<NBASIS;b++) precisionChange(DiagRef[b],Coarsening.rawNull[b]);
          DiagGW = BasisCrossGram(Coarsening.rawNull,Coarsening.rawNull);
          DiagGramRef.clear();
          if ( DiagGramVersion==DiagBasisVersion ) DiagGramRef = std::move(DiagGramKeep);
          t_overlap += usecond();
        }
        const bool site = P.RefreshKickShape=="site";
        std::vector<int> steps;
        if ( overlap && TwinUpdate ) {
          // Each vector on its own, in index order (RefreshOrthonormalise 0, so this is the
          // same refresh): its kick draws the same noise as KickBasis, and the run's basis
          // is the one it would be without the twin.  The run's vector F(v + eps eta) with
          // the counted operator; its twin F(v), from the unkicked v, and for the baseline
          // vectors the noise alone F(eta), with PVdagM.  Stage 0 is after the kick, p+1
          // after pass p.  Everything but the run's kick and passes is kept out of
          // update_seconds.
          const int  np   = P.RefreshPasses;
          const bool base = TwinBaseline>0 && step%TwinBaselineEvery==0;
          std::vector<int> bidx;                                       // the baseline vectors, evenly spaced
          if ( base ) for(int k=0;k<TwinBaseline;k++) bidx.push_back((k*NBASIS)/TwinBaseline);
          if ( TwinF.size()!=NBASIS ) TwinF.assign(NBASIS,LatticeFermionF(FGridsF.FGridF));
          std::vector<LatticeFermionD> run(1,FGrid), tw(1,FGrid), bl(1,FGrid);
          LatticeFermionD eta(FGrid), Av(FGrid);
          typedef std::vector<std::vector<RealD> > Table;
          Table cos_ref(np+1,std::vector<RealD>(NBASIS)), cos_twin(np,std::vector<RealD>(NBASIS));
          Table nul_run(np,std::vector<RealD>(NBASIS)), nul_twin(np,std::vector<RealD>(NBASIS));
          Table steps_run(np,std::vector<RealD>(NBASIS)), steps_twin(np,std::vector<RealD>(NBASIS));
          Table cos_base(np), cos_tb(np), nul_base(np), steps_base(np);
          std::vector<RealD> nul_v(NBASIS), nul_eta(NBASIS), nul_kick(NBASIS);
          steps.assign(NBASIS,0);
          RealD t_twin = 0.0;
          for(int b=0,kb=0;b<NBASIS;b++){
            LatticeFermionD &v = Coarsening.rawNull[b];
            const bool isb = kb<(int)bidx.size() && bidx[kb]==b;
            t_twin -= usecond();
            tw[0] = v;
            t_twin += usecond();
            KickVector(v,eta,RNGkick,P.RefreshKick,site);
            t_twin -= usecond();
            nul_v[b]    = BasisVectorNulliness(tw[0],PVdagM,Av);
            nul_eta[b]  = BasisVectorNulliness(eta,PVdagM,Av);
            nul_kick[b] = BasisVectorNulliness(v,PVdagM,Av);
            cos_ref[0][b] = BasisCosine(tw[0],v);
            if ( isb ) bl[0] = eta;
            run[0] = v;
            t_twin += usecond();
            for(int p=0;p<np;p++){
              steps_run[p][b] = RefineBasisGCR(run,CountedPVdagM,1,P.RefreshTol,P.RefreshMaxOuter)[0];
              t_twin -= usecond();
              steps_twin[p][b] = RefineBasisGCR(tw,PVdagM,1,P.RefreshTol,P.RefreshMaxOuter)[0];
              BasisFieldCopy(DiagRef[b],Av);
              cos_ref[p+1][b] = BasisCosine(Av,run[0]);
              cos_twin[p][b]  = BasisCosine(run[0],tw[0]);
              nul_run[p][b]   = BasisVectorNulliness(run[0],PVdagM,Av);
              nul_twin[p][b]  = BasisVectorNulliness(tw[0],PVdagM,Av);
              if ( isb ) {
                steps_base[p].push_back(RefineBasisGCR(bl,PVdagM,1,P.RefreshTol,P.RefreshMaxOuter)[0]);
                cos_base[p].push_back(BasisCosine(run[0],bl[0]));
                cos_tb[p].push_back(BasisCosine(tw[0],bl[0]));
                nul_base[p].push_back(BasisVectorNulliness(bl[0],PVdagM,Av));
              }
              t_twin += usecond();
            }
            v = run[0];
            steps[b] = steps_run[np-1][b];
            t_twin -= usecond();
            precisionChange(TwinF[b],tw[0]);
            t_twin += usecond();
            kb += isb;
          }
          t_twin -= usecond();
          std::cout << GridLogMessage << "QHMC-TWIN-KICK traj " << traj << " step " << step << " kick " << P.RefreshKick
                    << BasisStat("nul_v",nul_v) << BasisStat("nul_eta",nul_eta) << BasisStat("nul_kick",nul_kick)
                    << BasisStat("cos",cos_ref[0]) << std::endl;
          std::cout << GridLogMessage << "QHMC-TWIN-KICK-LIST traj " << traj << " step " << step
                    << " nul_v " << BasisList(nul_v) << " nul_eta " << BasisList(nul_eta)
                    << " nul_kick " << BasisList(nul_kick) << " cos " << BasisList(cos_ref[0]) << std::endl;
          for(int k=0;k<=np;k++)                                       // as Overlap: cos(v_old, the run's vector)
            std::cout << GridLogMessage << "QHMC-OVERLAP traj " << traj << " step " << step
                      << " stage " << (k ? "pass"+std::to_string(k) : std::string("kick"))
                      << BasisStat("cos",cos_ref[k]) << " list " << BasisList(cos_ref[k]) << std::endl;
          std::vector<RealD> bidxr(bidx.begin(),bidx.end());
          for(int p=0;p<np;p++){
            std::ostringstream line, list;
            line << "QHMC-TWIN traj " << traj << " step " << step << " pass " << p+1
                 << BasisStat("cos_twin",cos_twin[p]) << BasisStat("nul_run",nul_run[p]) << BasisStat("nul_twin",nul_twin[p])
                 << BasisStat("steps_run",steps_run[p]) << BasisStat("steps_twin",steps_twin[p]) << " baseline " << bidx.size();
            list << "QHMC-TWIN-LIST traj " << traj << " step " << step << " pass " << p+1
                 << " cos_twin " << BasisList(cos_twin[p]) << " nul_run " << BasisList(nul_run[p])
                 << " nul_twin " << BasisList(nul_twin[p]);
            if ( base ) {
              line << BasisStat("cos_base",cos_base[p]) << BasisStat("cos_tb",cos_tb[p]) << BasisStat("nul_base",nul_base[p])
                   << BasisStat("steps_base",steps_base[p]);
              list << " base_idx " << BasisList(bidxr) << " cos_base " << BasisList(cos_base[p])
                   << " cos_tb " << BasisList(cos_tb[p]) << " nul_base " << BasisList(nul_base[p]);
            }
            std::cout << GridLogMessage << line.str() << std::endl;
            std::cout << GridLogMessage << list.str() << std::endl;
          }
          TwinVersion = DiagBasisVersion+1;                            // the basis after this update
          t_twin += usecond();
          t_overlap += t_twin;
        } else {
          for(int p=0;p<P.RefreshPasses;p++){
            if ( p==0 && P.RefreshKick>0.0 ) {
              KickBasis(Coarsening.rawNull,RNGkick,P.RefreshKick,site);
              if ( overlap ) Overlap("kick");
            }
            steps = RefineBasisGCR(Coarsening.rawNull,CountedPVdagM,NBASIS,P.RefreshTol,P.RefreshMaxOuter);
            if ( P.RefreshOrthonormalise ) OrthonormaliseBasis(Coarsening.rawNull);
            if ( overlap ) Overlap("pass"+std::to_string(p+1));
          }
        }
        for(auto s : steps) update_steps += (RealD)s/steps.size();
        t_update -= t_overlap;
      } else {
        update_steps = PerturbativeBasisUpdate(traj,step);
      }
      t_update = (t_update+usecond())/1.0e6;
      mv_update = counter.Get("update") - m0;
      DiagBasisVersion++;
    }
    RealD mem_update = FreeDeviceMB();

    RealD t_rebuild = -usecond();
    if ( rebuilds ) Rebuild();
    t_rebuild = (t_rebuild+usecond())/1.0e6;
    if ( rebuilds ) Coarsening.CertifyCoarsening(PVdagM);
    RealD mem_rebuild = FreeDeviceMB();
    RealD null_resid = InvarianceResidual(Coarsening.rawNull,PVdagM);
    RealD gram_cond  = SpanProjector<LatticeFermionD>(Coarsening.rawNull).GramCondition();
    {
      // Fraction of each basis vector's norm^2 on each s slice (dimension 0 of the 5d grid):
      // the mean over the basis, and vector 0
      std::vector<RealD> sn, mean(P.Ls,0.0), first;
      for(int b=0;b<NBASIS;b++){
        sliceNorm(sn,Coarsening.rawNull[b],0);
        RealD tot = 0.0;
        for(auto x : sn) tot += x;
        for(int s=0;s<P.Ls;s++) sn[s] /= tot;
        for(int s=0;s<P.Ls;s++) mean[s] += sn[s]/NBASIS;
        if ( b==0 ) first = sn;
      }
      std::stringstream line;
      line << "QHMC-SPROFILE traj " << traj << " step " << step << " mean ";
      for(int s=0;s<P.Ls;s++) line << (s ? "," : "") << mean[s];
      line << " v0 ";
      for(int s=0;s<P.Ls;s++) line << (s ? "," : "") << first[s];
      std::cout << GridLogMessage << line.str() << std::endl;
    }
    Diagnose(traj,step);

    const bool mgsolve = step%SolveMGEvery==0;
    const bool cr      = DiagStep(step) && P.DiagCR>0;
    RealD t_solve = 0.0;
    if ( mgsolve || cr ) {
      t_solve = -usecond();
      PVdagMMultiGridSolver<MobiusFermionD,MobiusFermionF,Coarsening_t> Solver(Ddwf,Dpv,DdwfF,DpvF,Coarsening,P.MultiGrid,P.Nrhs);
      if ( cr ) {
        // Compatible relaxation with this solve's own fine smoother; kept out of solve_seconds
        RealD t_cr = -usecond();
        std::vector<RealD> err, res;
        BasisCompatibleRelaxation<LatticeFermionD,Coarsening_t::CoarseVector>(Coarsening.MrhsProjectorL1,PVdagM,Solver.SmootherGCR,
                                                                              FGrid,CGrids.Coarse5d,RNGcr,P.DiagCRVectors,P.DiagCR,err,res);
        t_cr = (t_cr+usecond())/1.0e6;
        t_solve -= t_cr*1.0e6;
        RealD rerr = 1.0, rres = 1.0;              // geometric mean over the second half of the sweeps
        int k0 = P.DiagCR/2, nk = P.DiagCR-k0;
        for(int k=k0;k<P.DiagCR;k++){ rerr *= std::pow(err[k],1.0/nk); rres *= std::pow(res[k],1.0/nk); }
        std::cout << GridLogMessage << "QHMC-CR traj " << traj << " step " << step << " sweeps " << P.DiagCR
                  << " vectors " << P.DiagCRVectors << " rate_err " << rerr << " rate_res " << rres
                  << " err " << BasisList(err) << " res " << BasisList(res) << " seconds " << t_cr << std::endl;
      }
      if ( mgsolve ) {
        for(int r=0;r<P.Nrhs;r++) mdsol[r]=Zero();
        Solver.Solve(mdsrc,mdsol);
      }
      t_solve = (t_solve+usecond())/1.0e6;
    }
    RealD mem_solve = FreeDeviceMB();
    std::cout << GridLogMessage << "QHMC-SOLVE traj " << traj << " step " << step << " t_MD " << t_MD
              << " import_seconds " << t_import << " rebuild_seconds " << t_rebuild
              << " solve_seconds " << t_solve << " null_resid " << null_resid
              << " null_before " << null_before << " update_seconds " << t_update
              << " update_matvecs " << mv_update << " update_steps " << update_steps
              << " gram_cond " << gram_cond << " mg_solve " << mgsolve << std::endl;
    std::cout << GridLogMessage << "QHMC-MEM traj " << traj << " step " << step
              << " free_start_MB " << mem_start << " free_update_MB " << mem_update
              << " free_rebuild_MB " << mem_rebuild << " free_solve_MB " << mem_solve
              << " grid_cache_MB " << (MemoryManager::DeviceCacheBytes()>>20) << std::endl;
    if ( P.Mode=="perturbative" ) ImportLast();
  };
  auto ControlAtU = [&](int traj,int step,RealD t_MD)
  {
    Ddwf.ImportGauge(Umu);
    Dpv.ImportGauge(Umu);
    MdagMLinearOperator<MobiusFermionD,LatticeFermionD> PVdagPV(Dpv);
    ConjugateGradient<LatticeFermionD> CGpv(1.0e-2*P.MultiGrid.Outer.Tol,100000,false);
    ConjugateGradient<LatticeFermionD> CG(P.MultiGrid.Outer.Tol,100000,false);
    SchurRedBlackDiagMooeeSolve<LatticeFermionD> RBCG(CG);
    std::vector<LatticeFermionD> sol(1,FGrid), y(1,FGrid);
    LatticeFermionD z(FGrid), r(FGrid);
    sol[0] = Zero();
    z = Zero();
    RealD t_pv = -usecond();
    CGpv(PVdagPV,mdsrc[0],z);
    Dpv.M(z,y[0]);
    t_pv = (t_pv+usecond())/1.0e6;
    RealD t_solve = -usecond();
    RBCG(Ddwf,y,sol);
    t_solve = (t_solve+usecond())/1.0e6;
    PVdagM.Op(sol[0],r);
    r = r - mdsrc[0];
    std::cout << GridLogMessage << "QHMC-CONTROL traj " << traj << " step " << step << " t_MD " << t_MD
              << " pv_seconds " << t_pv << " pv_iters " << CGpv.IterationsToComplete
              << " solve_seconds " << t_solve << " iters " << CG.IterationsToComplete
              << " true_resid " << std::sqrt(norm2(r)/norm2(mdsrc[0])) << std::endl;
  };
  auto SolvesIn = [&](int traj) -> bool {
    if ( P.Mode=="none" ) return false;
    if ( P.SolveTraj.empty() ) return true;
    return std::find(P.SolveTraj.begin(),P.SolveTraj.end(),traj)!=P.SolveTraj.end();
  };

  std::cout.precision(15);
  LatticeGaugeField U0 = Umu;
  for(int s=0;s<P.TrajL.size();s++){
    RealD tau = P.TrajL[s];
    int   N   = P.MDsteps[s];
    IntegratorParameters MDpar(N,tau);
    SmearingPolicy Smearing;
    IntegratorType MD(UGrid,MDpar,FullSet,Smearing);

    for(int t=0;t<P.Ntraj;t++){
      if ( !P.Chain ) Umu = U0;
      LatticeGaugeField Ustart = Umu;
      const bool solves = SolvesIn(t);
      std::vector<LatticeFermionD> Vstart;
      if ( updates && P.Ntraj>1 ) Vstart = Coarsening.rawNull;
      if ( P.Mode=="perturbative" ) ImportLast();
      if ( solves ) for(int r=0;r<P.Nrhs;r++) gaussian(RNG5,mdsrc[r]);
      RealD t_traj = -usecond();
      MD.refresh(Umu,sRNG,RNG4);
      RealD H0 = MD.Sinitial(Umu);
      std::cout << GridLogMessage << "QHMC-STEP 0 t_MD 0 plaquette "
                << WilsonLoops<Gimpl>::avgPlaquette(Umu) << " seconds 0" << std::endl;

      RealD t_last = usecond();
      IntegrateWithHook(MD,Umu,[&](int step) {
        RealD t_step = (usecond()-t_last)/1.0e6;
        std::cout << GridLogMessage << "QHMC-STEP " << step << " t_MD " << step*tau/N
                  << " plaquette " << WilsonLoops<Gimpl>::avgPlaquette(Umu)
                  << " seconds " << t_step << std::endl;
        if ( solves && step%P.SolveEvery==0 ) {
          if ( P.Mode=="control" ) ControlAtU(t,step,step*tau/N);
          else                     SolveAtU(t,step,step*tau/N);
        }
        t_last = usecond();
      });

      RealD H1 = MD.S(Umu);
      RealD dH = H1 - H0;
      bool accept = MetropolisAccept(dH,sRNG);
      if ( !accept ) {
        Umu = Ustart;
        if ( !Vstart.empty() ) Coarsening.rawNull = Vstart;
      }
      t_traj = (t_traj+usecond())/1.0e6;
      std::cout << GridLogMessage << "QHMC-TRAJ tau " << tau << " N " << N << " traj " << t
                << " H0 " << H0 << " H1 " << H1 << " dH " << dH
                << " accept " << accept << " plaquette " << WilsonLoops<Gimpl>::avgPlaquette(Umu)
                << " seconds " << t_traj << std::endl;
    }
  }

  counter.Report("QHMC-MATVECS");
  Grid_finalize();
}
