/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./examples/Example_pvdagm_inverse_iteration.cc

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

// Inverse iteration on Gaussian noise at a FIXED field, with the three-level
// mrhs PVdagM multigrid built on the basis after every round (research hub,
// projects/multigrid, notes/inverse-iteration-test-design.md).
//
// The question is why deeper inverse iteration makes the MG worse: the
// over-filtering hypothesis is that past the setup's three rounds the basis
// concentrates on the lowest modes of A = PV^dag M and loses the mixture of
// low modes the coarse space needs.
//
// The 60-vector basis starts as normalised Gaussian noise (k = 0).  For each
// k = 0..KMax:
//   1. coarsen and build the dense bottom on V_k, Galerkin certificate;
//   2. solve ONE fixed source, Nrhs 1, zero guess;
//   3. nulliness ||A v||/||v|| and |<v|A v>|/|v|^2 of every vector, and the
//      invariance residual ||Q A V||/||V||;
//   4. (k in CensusK) Peter Boyle's coarse census and coherence checks;
//   5. one round of inverse iteration, v -> A^{-1} v/||A^{-1} v|| by
//      unpreconditioned GCR(10,10) (RefineBasisGCR), skipped after KMax.
// The setup makes its basis with three such rounds, so V_3 is the basis of
// every production run.
//
// Every random number comes from a fixed seed and the field never changes,
// so a rerun reproduces the scan.  One II-* log line per measurement per k;
// lists on II-*-LIST lines, comma separated.  The final GCR residuals of a
// round are not logged here: qhmc_compact.py rebuilds them from the GCR lines.
//
// Parameters: one serialisable struct, read from XML with --ii-params; a
// missing file gets a template written next to it.
//
#include <Grid/Grid.h>
#include <Grid/lattice/PaddedCell.h>
#include <Grid/stencil/GeneralLocalStencil.h>
#include <Grid/algorithms/iterative/PrecGeneralisedConjugateResidualNonHermitian.h>
#include <Grid/algorithms/multigrid/PVdagMMultiGrid.h>
#include "Example_mg_hmc.h"
#include "Example_mg_refine.h"
#include "Example_basis_diagnostics.h"
#include "Example_coarse_coherence.h"

using namespace std;
using namespace Grid;

#ifndef NBASIS
#define NBASIS 60
#endif

#ifdef COARSE_SINGLE
typedef sTComplexF CoarseScalar_t;
#else
typedef sTComplexD CoarseScalar_t;
#endif

struct InverseIterationParams : Serializable {
  GRID_SERIALIZABLE_CLASS_MEMBERS(InverseIterationParams,
                                  int,              Ls,
                                  RealD,            Mass,
                                  RealD,            M5,
                                  RealD,            MobiusB,
                                  RealD,            MobiusC,
                                  std::string,      Config,       // empty: hot start
                                  int,              KMax,         // rounds k = 0..KMax
                                  RealD,            RoundTol,     // GCR target per vector in a round
                                  int,              RoundMaxOuter,// restarts of GCR(10,10) per vector in a round
                                  std::vector<int>, CensusK,      // the k that get the census and the checks
                                  int,              CensusNstop,  // Lanczos: modes wanted
                                  int,              CensusNk,
                                  int,              CensusNm,
                                  RealD,            CensusTol,
                                  int,              CensusMaxIt,  // Lanczos restart cap; unconverged Ritz vectors are used as they are
                                  RealD,            ChebyLo,      // filter low edge in sigma^2
                                  RealD,            ChebyHi,      // filter high edge; 0: 1.1 sigma_max^2 from a power method
                                  int,              ChebyOrder,
                                  int,              Nbasis2,      // CHECK A: census vectors in the blocked basis
                                  PVdagMMultiGridParams, MultiGrid);
  InverseIterationParams()
    : Ls(24), Mass(0.00078), M5(1.8), MobiusB(1.5), MobiusC(0.5), Config(""),
      KMax(20), RoundTol(5.0e-4), RoundMaxOuter(30),
      CensusK({0,1,2,3,4,5,7,10,14,20}), CensusNstop(60), CensusNk(96), CensusNm(192),
      CensusTol(1.0e-5), CensusMaxIt(10), ChebyLo(4.0), ChebyHi(0.0), ChebyOrder(401),
      Nbasis2(48)
  {
    // Peter's P0 cell, as the p0 set of run_hmc_multigrid_quenched_frontier.job;
    // the library defaults differ in the smoother, coarse solver and outer fields
    MultiGrid.Setup.RetainSubspace  = 1;
    MultiGrid.FineSmoother.Mmax     = 4;
    MultiGrid.CoarseSmoother.Shift  = 2.0;
    MultiGrid.CoarseSolver.Mmax     = 8;
    MultiGrid.Outer.MaxIterations   = 20;   // restarts of Outer.Nstep: cap 240
    MultiGrid.Outer.Nstep           = 12;
  };
};

int main (int argc, char ** argv)
{
  Grid_init(&argc,&argv);

  InverseIterationParams P;
  {
    std::string pfile("");
    if( GridCmdOptionExists(argv,argv+argc,"--ii-params") )
      pfile = GridCmdOptionPayload(argv,argv+argc,"--ii-params");
    if ( pfile.length() ) {
      bool good; { std::ifstream f(pfile); good = f.good(); }
      if ( !good ) {
        if ( GlobalSharedMemory::WorldRank == 0 ) {
          XmlWriter WR(pfile+".templ");
          write(WR, "InverseIteration", P);
          std::cout << GridLogMessage << pfile << " does not exist; template written to "
                    << pfile << ".templ" << std::endl;
        }
        Grid_finalize(); return 0;
      }
      XmlReader RD(pfile);
      read(RD, "InverseIteration", P);
    }
    CheckValidity(P.MultiGrid);
    // The basis is set here, never loaded, and must survive every coarsening
    GRID_ASSERT( P.MultiGrid.Setup.SubspaceFile.length() == 0 );
    GRID_ASSERT( P.MultiGrid.Setup.RetainSubspace );
    GRID_ASSERT( P.KMax >= 0 );
    std::cout << GridLogMessage << "InverseIteration parameters ("
              << (pfile.length() ? pfile : std::string("defaults")) << "):" << std::endl;
    std::cout << P << std::endl;
  }

  Coordinate latt = GridDefaultLatt();
  Coordinate mpi  = GridDefaultMpi();
  Coordinate fsimd= GridDefaultSimd(Nd,vComplex::Nsimd());

  GridCartesian         * UGrid   = SpaceTimeGrid::makeFourDimGrid(latt,fsimd,mpi);
  GridRedBlackCartesian * UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian         * FGrid   = SpaceTimeGrid::makeFiveDimGrid(P.Ls,UGrid);
  GridRedBlackCartesian * FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(P.Ls,UGrid);

  // Seeds as Example_hmc_multigrid_quenched_frontier, so k = 0..16 compare
  // with its NoiseBasis scan (job 5853868)
  GridParallelRNG RNG4(UGrid);     RNG4.SeedFixedIntegers({1,2,3,4});
  GridParallelRNG RNG5(FGrid);     RNG5.SeedFixedIntegers({5,6,7,8});        // the source
  GridParallelRNG RNGnoise(FGrid); RNGnoise.SeedFixedIntegers({13,14,15,16}); // the basis at k = 0

  LatticeGaugeField Umu(UGrid);
  if ( P.Config.length() ) {
    std::cout << GridLogMessage << "Reading gauge field " << P.Config << std::endl;
    FieldMetaData header;
    NerscIO::readConfiguration(Umu,header,P.Config);
  } else {
    std::cout << GridLogMessage << "Hot start gauge field" << std::endl;
    SU<Nc>::HotConfiguration(RNG4,Umu);
  }

  MobiusFermionD Ddwf(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,P.Mass,P.M5,P.MobiusB,P.MobiusC);
  MobiusFermionD Dpv (Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,1.0,   P.M5,P.MobiusB,P.MobiusC);

  typedef PVdagMLinearOperator<MobiusFermionD,LatticeFermionD> PVdagM_t;
  PVdagM_t PVdagM(Ddwf,Dpv);

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

  Coarsening_t Coarsening(CGrids, FGridsF, P.MultiGrid.Setup);

  //////////////////////////////////////////////////////////////////////
  // k = 0: the basis is normalised Gaussian noise; the source is drawn once
  //////////////////////////////////////////////////////////////////////
  std::vector<LatticeFermionD> &V = Coarsening.rawNull;
  V.clear();
  V.reserve(NBASIS);
  for(int b=0;b<NBASIS;b++){
    V.push_back(LatticeFermionD(FGrid));
    gaussian(RNGnoise,V[b]);
    V[b] = V[b]*std::pow(norm2(V[b]),-0.5);
  }
  LatticeFermionD src(FGrid);
  gaussian(RNG5,src);
  std::cout << GridLogMessage << "II-START nbasis " << NBASIS << " kmax " << P.KMax << std::endl;

  for(int k=0;k<=P.KMax;k++){

    //////////////////////////////////////////////////////////////////////
    // 1. Coarsen, dense bottom, certificate
    //////////////////////////////////////////////////////////////////////
    RealD t_coarsen = -usecond();
    Coarsening.Coarsen(PVdagM);
    t_coarsen = (t_coarsen+usecond())/1.0e6;
    RealD t_dense = -usecond();
    Coarsening.BuildDenseBottom();
    t_dense = (t_dense+usecond())/1.0e6;
    RealD cert = Coarsening.CertifyCoarsening(PVdagM);
    std::cout << GridLogMessage << "II-SETUP k " << k << " coarsen_seconds " << t_coarsen
              << " dense_seconds " << t_dense << " certificate " << cert << std::endl;

    //////////////////////////////////////////////////////////////////////
    // 2. The solve: the same source at every k, zero guess, Nrhs 1
    //////////////////////////////////////////////////////////////////////
    {
      PVdagMMultiGridSolver<MobiusFermionD,MobiusFermionF,Coarsening_t> Solver(Ddwf,Dpv,DdwfF,DpvF,Coarsening,P.MultiGrid,1);
      std::vector<LatticeFermionD> b(1,src), x(1,FGrid);
      x[0] = Zero();
      RealD t_solve = -usecond();
      Solver.Solve(b,x);
      t_solve = (t_solve+usecond())/1.0e6;
      int iters = Solver.L1PGCR.steps;
      int converged = ( iters < P.MultiGrid.Outer.MaxIterations*P.MultiGrid.Outer.Nstep );
      PVdagM.SloppyComms(0);
      LatticeFermionD Ax(FGrid);
      PVdagM.Op(x[0],Ax);
      Ax = Ax - src;
      RealD true_rel = std::sqrt(norm2(Ax)/norm2(src));
      std::cout << GridLogMessage << "II-SOLVE k " << k << " iters " << iters << " converged " << converged
                << " true_rel " << true_rel << " seconds " << t_solve << std::endl;
    }

    //////////////////////////////////////////////////////////////////////
    // 3. Nulliness of the raw basis V_k
    //////////////////////////////////////////////////////////////////////
    {
      std::vector<RealD> nul, rq;
      BasisNulliness(V,PVdagM,nul,rq);
      RealD inv = InvarianceResidual(V,PVdagM);
      std::cout << GridLogMessage << "II-NUL k " << k << BasisStat("nul",nul) << BasisStat("rq",rq)
                << " inv_resid " << inv << std::endl;
      std::cout << GridLogMessage << "II-NUL-LIST k " << k << " nul " << BasisList(nul)
                << " rq " << BasisList(rq) << std::endl;
    }

    //////////////////////////////////////////////////////////////////////
    // 5. One round of inverse iteration, V_k -> V_{k+1}
    //////////////////////////////////////////////////////////////////////
    if ( k < P.KMax ) {
      RealD t_round = -usecond();
      std::vector<int> steps = RefineBasisGCR(V,PVdagM,NBASIS,P.RoundTol,P.RoundMaxOuter);
      t_round = (t_round+usecond())/1.0e6;
      std::vector<RealD> s(steps.begin(),steps.end());
      std::cout << GridLogMessage << "II-ROUND k " << k << BasisStat("steps",s)
                << " seconds " << t_round << std::endl;
      std::cout << GridLogMessage << "II-ROUND-LIST k " << k << " steps " << BasisList(s) << std::endl;
    }
  }

  std::cout << GridLogMessage << "II-DONE" << std::endl;
  Grid_finalize();
  return 0;
}
