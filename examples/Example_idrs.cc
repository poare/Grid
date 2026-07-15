/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./examples/Example_idrs.cc

Copyright (C) 2015

Author: Patrick Oare <poare@bnl.gov>

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

/*
 * Validates IDR(s) against PrecGCR on a physical WilsonFermionD system at
 * mass m=0.01 on an 8^4 gauge configuration.
 *
 * Wilson fermion at m=0.01 is non-Hermitian and closer to the critical
 * mass than the m=0.1 case, giving a worse-conditioned operator; this
 * exercises how the solvers cope with slower convergence.
 * PrecGCR (= unpreconditioned FGMRES) is the correct non-Hermitian reference;
 * Grid's BiCGSTAB takes Re() of inner products, which assumes Hermitian
 * positive-definiteness and is not theoretically guaranteed to converge on
 * Wilson/DWF, so it is run last (see below) in case it fails to converge.
 */

#include <Grid/Grid.h>
#include <Grid/algorithms/iterative/PrecGeneralisedConjugateResidualNonHermitian.h>

using namespace std;
using namespace Grid;

int main(int argc, char **argv)
{
  Grid_init(&argc, &argv);

  //////////////////////////////////////////////////////////////////
  // 4D grid: 8^4 lattice.
  //////////////////////////////////////////////////////////////////
  std::vector<int> lat_size = {16, 16, 16, 32};

  GridCartesian         *UGrid   = SpaceTimeGrid::makeFourDimGrid(
                                     lat_size,
                                     GridDefaultSimd(Nd, vComplex::Nsimd()),
                                     GridDefaultMpi());
  GridRedBlackCartesian *UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);

  //////////////////////////////////////////////////////////////////
  // Load 8^4 gauge configuration (NERSC format).
  //////////////////////////////////////////////////////////////////
  LatticeGaugeField Umu(UGrid);
  FieldMetaData header;
  std::string config = "/Users/patrickoare/libraries/PETSc-Grid/ckpoint_lat.4000";
  std::cout << GridLogMessage << "Loading configuration: " << config << std::endl;
  NerscIO::readConfiguration(Umu, header, config);

  //////////////////////////////////////////////////////////////////
  // Wilson fermion at mass m = 0.01.
  // Closer to the critical mass than m = 0.1, so the operator is
  // worse-conditioned and unpreconditioned Krylov solvers need more
  // iterations to converge.
  //////////////////////////////////////////////////////////////////
  RealD mass = 0.01;
  WilsonFermionD Dw(Umu, *UGrid, *UrbGrid, mass);
  NonHermitianLinearOperator<WilsonFermionD, LatticeFermionD> WLinOp(Dw);

  //////////////////////////////////////////////////////////////////
  // Random Gaussian source on the 4D fermion grid.
  //////////////////////////////////////////////////////////////////
  GridParallelRNG RNG4(UGrid);
  RNG4.SeedFixedIntegers(std::vector<int>({1, 2, 3, 4, 5}));

  LatticeFermion src(UGrid);
  gaussian(RNG4, src);
  std::cout << GridLogMessage << "Source norm sq: " << norm2(src) << std::endl;

  const RealD tol     = 1.0e-8;
  const int   maxIter = 10000;

  //////////////////////////////////////////////////////////////////
  // Reference solve: unpreconditioned PrecGCR (= FGMRES).
  // Krylov window mmax = nstep = 200 to avoid restart effects.
  //////////////////////////////////////////////////////////////////
  std::cout << GridLogMessage << std::endl;
  std::cout << GridLogMessage << "======================================" << std::endl;
  std::cout << GridLogMessage << "Solving with PrecGCR (reference)" << std::endl;
  std::cout << GridLogMessage << "======================================" << std::endl;

  LatticeFermion psi_gcr(UGrid);
  psi_gcr = Zero();

  IdentityLinearFunction<LatticeFermion> Identity;

  const int mmax  = 200;
  const int nstep = 200;
  PrecGeneralisedConjugateResidualNonHermitian<LatticeFermion>
    gcr(tol, maxIter, WLinOp, Identity, mmax, nstep);

  GridStopWatch timerGCR;
  timerGCR.Start();
  gcr(src, psi_gcr);
  timerGCR.Stop();

  //////////////////////////////////////////////////////////////////
  // IDR(s = 1): should converge similarly to BiCGSTAB (in exact
  // arithmetic IDR(1) and BiCGSTAB are equivalent).
  //////////////////////////////////////////////////////////////////
  std::cout << GridLogMessage << std::endl;
  std::cout << GridLogMessage << "======================================" << std::endl;
  std::cout << GridLogMessage << "Solving with IDR(s=1)" << std::endl;
  std::cout << GridLogMessage << "======================================" << std::endl;

  LatticeFermion psi_idrs1(UGrid);
  psi_idrs1 = Zero();

  IDRs<LatticeFermion> idrs1(WLinOp, 1, tol, maxIter, false);

  GridStopWatch timerIDRS1;
  timerIDRS1.Start();
  idrs1(src, psi_idrs1);
  timerIDRS1.Stop();

  //////////////////////////////////////////////////////////////////
  // IDR(s = 4): larger shadow space, expect faster convergence.
  //////////////////////////////////////////////////////////////////
  std::cout << GridLogMessage << std::endl;
  std::cout << GridLogMessage << "======================================" << std::endl;
  std::cout << GridLogMessage << "Solving with IDR(s=4)" << std::endl;
  std::cout << GridLogMessage << "======================================" << std::endl;

  LatticeFermion psi_idrs4(UGrid);
  psi_idrs4 = Zero();

  IDRs<LatticeFermion> idrs4(WLinOp, 4, tol, maxIter, false);

  GridStopWatch timerIDRS4;
  timerIDRS4.Start();
  idrs4(src, psi_idrs4);
  timerIDRS4.Stop();

  //////////////////////////////////////////////////////////////////
  // IDR(s = 10): even larger shadow space. More matvecs per outer
  // iteration (2s+1 = 21) than s=4, but should further reduce the
  // outer iteration count on this worse-conditioned system.
  //////////////////////////////////////////////////////////////////
  std::cout << GridLogMessage << std::endl;
  std::cout << GridLogMessage << "======================================" << std::endl;
  std::cout << GridLogMessage << "Solving with IDR(s=10)" << std::endl;
  std::cout << GridLogMessage << "======================================" << std::endl;

  LatticeFermion psi_idrs10(UGrid);
  psi_idrs10 = Zero();

  IDRs<LatticeFermion> idrs10(WLinOp, 10, tol, maxIter, false);

  GridStopWatch timerIDRS10;
  timerIDRS10.Start();
  idrs10(src, psi_idrs10);
  timerIDRS10.Stop();

  //////////////////////////////////////////////////////////////////
  // BiCGSTAB: not used as the primary non-Hermitian reference (see
  // header comment) because Grid's BiCGSTAB computes rho/alpha/omega
  // from Re() of complex inner products, which assumes a Hermitian
  // positive-definite operator. Dw at m=0.01 is worse-conditioned than
  // the m=0.1 case, so this solver is even less likely to converge
  // here. It is included for comparison against IDR(s=1) since, in
  // exact arithmetic, IDR(1) and BiCGSTAB generate identical iterates.
  // Run last, after all IDR(s) variants, in case it fails to converge
  // and burns through its full iteration budget.
  //////////////////////////////////////////////////////////////////
  std::cout << GridLogMessage << std::endl;
  std::cout << GridLogMessage << "======================================" << std::endl;
  std::cout << GridLogMessage << "Solving with BiCGSTAB" << std::endl;
  std::cout << GridLogMessage << "======================================" << std::endl;

  LatticeFermion psi_bicgstab(UGrid);
  psi_bicgstab = Zero();

  const int maxIterBiCGSTAB = 5000;
  BiCGSTAB<LatticeFermion> bicgstab(tol, maxIterBiCGSTAB, false);

  GridStopWatch timerBiCGSTAB;
  timerBiCGSTAB.Start();
  bicgstab(WLinOp, src, psi_bicgstab);
  timerBiCGSTAB.Stop();

  // BiCGSTAB does not expose a TrueResidual member; compute it directly.
  LatticeFermion r_bicgstab(UGrid);
  WLinOp.Op(psi_bicgstab, r_bicgstab);
  r_bicgstab = src - r_bicgstab;
  RealD trueResidualBiCGSTAB = std::sqrt(norm2(r_bicgstab) / norm2(src));

  //////////////////////////////////////////////////////////////////
  // Summary.
  //////////////////////////////////////////////////////////////////
  std::cout << GridLogMessage << std::endl;
  std::cout << GridLogMessage << "======================================" << std::endl;
  std::cout << GridLogMessage << "Solver comparison summary" << std::endl;
  std::cout << GridLogMessage << "======================================" << std::endl;
  std::cout << GridLogMessage << "PrecGCR:   steps = " << gcr.steps
            << "  time = " << timerGCR.Elapsed() << std::endl;
  std::cout << GridLogMessage << "IDR(s=1):  outer iters = " << idrs1.IterationsToComplete
            << "  true residual = " << idrs1.TrueResidual
            << "  time = " << timerIDRS1.Elapsed() << std::endl;
  std::cout << GridLogMessage << "IDR(s=4):  outer iters = " << idrs4.IterationsToComplete
            << "  true residual = " << idrs4.TrueResidual
            << "  time = " << timerIDRS4.Elapsed() << std::endl;
  std::cout << GridLogMessage << "IDR(s=10): outer iters = " << idrs10.IterationsToComplete
            << "  true residual = " << idrs10.TrueResidual
            << "  time = " << timerIDRS10.Elapsed() << std::endl;
  std::cout << GridLogMessage << "BiCGSTAB:  outer iters = " << bicgstab.IterationsToComplete
            << "  true residual = " << trueResidualBiCGSTAB
            << "  time = " << timerBiCGSTAB.Elapsed() << std::endl;

  LatticeFermion diff(UGrid);
  diff = psi_idrs1 - psi_gcr;
  std::cout << GridLogMessage << "||psi_IDR(1) - psi_GCR||^2 = " << norm2(diff) << std::endl;
  diff = psi_idrs4 - psi_gcr;
  std::cout << GridLogMessage << "||psi_IDR(4) - psi_GCR||^2 = " << norm2(diff) << std::endl;
  diff = psi_idrs10 - psi_gcr;
  std::cout << GridLogMessage << "||psi_IDR(10) - psi_GCR||^2 = " << norm2(diff) << std::endl;
  diff = psi_bicgstab - psi_gcr;
  std::cout << GridLogMessage << "||psi_BiCGSTAB - psi_GCR||^2 = " << norm2(diff) << std::endl;

  Grid_finalize();
  return 0;
}
