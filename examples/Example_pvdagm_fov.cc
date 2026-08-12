/*************************************************************************************

    Traces the LEFT-HAND boundary of the field of values (numerical range) of the
    Pauli-Villars preconditioned Mobius domain wall operator

        F(m_adj, m_l) = D^dag(m_adj) D(m_l),

    which interpolates between the Hermitian positive definite normal equation
    (m_adj = m_l) and PVdagM (m_adj = 1, the default here).

    Method.  W(F) = { <v|F|v> : |v| = 1 } is convex (Toeplitz-Hausdorff), so it is cut
    out by its supporting lines.  Define

        H_theta = 1/2 ( e^{-i theta} F + e^{+i theta} F^dag ),
        g(theta) = lambda_min(H_theta) = min_{|v|=1} Re( e^{-i theta} <v|F|v> ),

    so that W(F) lies in the half plane Re( e^{-i theta} z ) >= g(theta).  Sweeping
    theta over an arc centred on 0 gives exactly the supporting lines whose inward
    normal points rightward, i.e. the LEFT boundary of W(F).  theta = 0 gives

        g(0) = lambda_min(M),  M = (F + F^dag)/2,

    the leftmost real extent of W(F); the project convention quotes
    lambda_min(F + F^dag) = 2 g(0).  Both are printed.

    g(theta) is an EXTREME (algebraically smallest) eigenvalue of a Hermitian
    operator, not an interior one, so Lanczos handles it well.  Grid's IRL targets the
    algebraically LARGEST eigenvalues -- it sorts Ritz values descending and applies
    the implicit QR shifts at the smallest ones -- so we run it on

        H_{theta + pi} = -H_theta,     g(theta) = -lambda_max(H_{theta+pi}),

    which needs no spectral shift and cannot converge to the wrong end of the
    spectrum.  Note this is a different problem from finding the smallest-MODULUS
    eigenvalues of F itself, which are interior and would need a harmonic or
    shift-invert method (see KrylovSchur's harmonic branch).

    Because IRL returns eigenVECTORS, each angle also yields a genuine point of W(F),

        z(theta) = <v|F|v>,   v the minimising eigenvector,

    which is what to plot.  These points are exact members of W(F) regardless of how
    well IRL converged, and they support a rigorous cross-check: for every pair of
    angles, g(theta') <= Re( e^{-i theta'} z(theta) ).

    Usage :
      $ ./Example_pvdagm_fov --config <file> --outdir <dir> \
            [--nangle N] [--theta DEG] [--madj M] [--nstop N] [--nk N] [--nm N] \
            [Grid options]

    Options are NAMED, not positional, and may be given in any order alongside Grid's
    own flags. This matters: Grid_init() reads its flags out of argv but does NOT
    remove them, so argc still counts --grid, --mpi and the rest and positional
    indexing would read a Grid flag as a parameter.

      --config <file> = Gauge configuration to read in (NERSC format). REQUIRED.
      --outdir <dir>  = Directory to write output to. REQUIRED.
      --nangle N      = Number of angles in the arc. Default 17. Use an ODD value so
                        that theta = 0 is sampled and lambda_min(M) comes out directly.
      --theta DEG     = Half width of the arc IN DEGREES. Default 30. The sweep runs
                        over Nangle points spanning [-DEG, +DEG], so the total arc is
                        2*DEG wide. Because g(theta) = lambda_min(H_theta) is a
                        supporting line whose inward normal points along +theta, this
                        probes W(F) from the directions 180-DEG to 180+DEG, i.e. the
                        left-hand side. --theta 30 sweeps [-30, +30] here and so
                        examines W(F) between 150 and 210 degrees; --theta 90 covers
                        the whole left half, 90 to 270 degrees.
      --madj M        = Mass of the adjoint (Pauli-Villars) factor. Default 1.0, which
                        gives the PVdagM operator.
      --nstop N       = Converged eigenvalues sought per angle. Default 3.
      --nk N          = Lanczos basis kept per restart. Default 6.
      --nm N          = Total Lanczos basis size. Default 12.

    MEMORY: IRL holds Nm five-dimensional fermion fields simultaneously, plus a
    handful of temporaries.  On 16^3x32 with Ls = 16 one LatticeFermionD is about
    0.4 GB, so Nm = 8 is roughly 3.2 GB before temporaries.  The startup banner
    prints the estimate.  Raise Nm for faster convergence only if the memory is
    there; this is why Nm is on the command line rather than hard coded.

    Output:
      ${outDir}/fov_left.txt = One line per angle, formatted as
                            `$idx $theta $g $Re_z $Im_z $Nconv $ok`
                          where ($Re_z, $Im_z) is a point on the left boundary of
                          W(F) and $ok is 0 if any consistency check failed.

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./examples/Example_pvdagm_fov.cc

    Copyright (C) 2026

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

#include <cstdlib>

#include <Grid/Grid.h>

#include <Grid/parallelIO/IldgIOtypes.h>
#include <Grid/parallelIO/IldgIO.h>

using namespace std;
using namespace Grid;

/**
 * F = D_PV^dag D_dwf.  Identical to the operator used in Example_spec_kryschur, minus
 * the per-application diagnostic prints: this driver applies it O(10^4) times and the
 * prints would dominate both the runtime and the log.
 *
 * HermOp is F^dag F, so a power iteration on this operator directly gives
 * lambda_max(F^dag F) = sigma_max(F)^2, which is the denominator of the
 * Eisenstat-Elman-Schultz bound.
 */
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

/**
 * Algebraically smallest eigenvalue of H_theta, and its eigenvector.
 *
 * Runs IRL on H_{theta+pi} = -H_theta, whose algebraically largest eigenvalue is
 * -lambda_min(H_theta).  IRL targets the algebraically largest, so no shift is
 * needed and there is no wrong-end failure mode.
 *
 * The largest eigenvalue is located by an explicit argmax over the converged values
 * rather than by assuming eval[0] is the extremal one, so this does not depend on
 * IRL's internal sort order surviving the final deflation.
 *
 * `vmin` receives the minimising eigenvector of H_theta (the same vector, since
 * H_{theta+pi} = -H_theta shares eigenvectors).  `Nconv` receives IRL's converged
 * count; zero means nothing converged and the returned value is meaningless.
 */
template<class Field>
RealD LambdaMinHermitian(LinearOperatorBase<Field> &Fop, RealD theta, const Field &src,
                         Field &vmin, int &Nconv,
                         int Nstop, int Nk, int Nm, RealD eresid, int MaxIter)
{
  HermitianPartLinearOperator<Field> Hflip(Fop, theta + M_PI);   // = -H_theta
  PlainHermOp<Field> hermop(Hflip);

  ImplicitlyRestartedLanczos<Field> IRL(hermop, hermop, Nstop, Nk, Nm, eresid, MaxIter);

  std::vector<RealD> eval(Nm);
  std::vector<Field> evec(Nm, src.Grid());

  Nconv = 0;
  IRL.calc(eval, evec, src, Nconv);

  if (Nconv < 1) {
    std::cout << GridLogError
              << "CHECK FAILED: IRL converged " << Nconv << " eigenvalues at theta = "
              << theta << ". Increase Nm, Nk or MaxIter." << std::endl;
    return 0.0;
  }

  int imax = 0;
  for (int i = 1; i < Nconv; i++) {
    if (eval[i] > eval[imax]) imax = i;
  }
  vmin = evec[imax];

  return -eval[imax];        // lambda_min(H_theta) = -lambda_max(-H_theta)
}

int main (int argc, char ** argv)
{
  Grid_init(&argc,&argv);

  // Usage : $ ./Example_pvdagm_fov <inFile> <outDir> [Nangle] [theta_deg] [m_adj] [Nstop] [Nk] [Nm]
  /*
   * Options are NAMED rather than positional. Grid_init() parses its own flags out of
   * argv but does not remove them -- it only ever reads through GridCmdOptionExists --
   * so argc still counts --grid, --mpi, --accelerator-threads and friends, and they
   * may appear anywhere in the array. Positional indexing therefore picks up a Grid
   * flag as soon as one is passed, which is what made std::stod("--grid") throw
   * std::invalid_argument. Named lookup is order independent and immune to however
   * many Grid flags are present.
   */
  int   Nangle    = 17;
  RealD theta_deg = 30.0;                   // arc half width in degrees
  RealD m_adj     = 1.0;                    // default: PVdagM
  int   Nstop     = 3;
  int   Nk        = 6;
  int   Nm        = 12;

  std::string file, outDir;
  if (GridCmdOptionExists(argv,argv+argc,"--config")) {
    file = GridCmdOptionPayload(argv,argv+argc,"--config");
  }
  if (GridCmdOptionExists(argv,argv+argc,"--outdir")) {
    outDir = GridCmdOptionPayload(argv,argv+argc,"--outdir");
  }
  if (file.empty() || outDir.empty()) {
    std::cout << GridLogError
              << "usage: Example_pvdagm_fov --config <file> --outdir <dir> "
              << "[--nangle N] [--theta DEG] [--madj M] [--nstop N] [--nk N] [--nm N] "
              << "[Grid options]" << std::endl;
    Grid_finalize();
    return 1;
  }

  if (GridCmdOptionExists(argv,argv+argc,"--nangle")) {
    std::string s = GridCmdOptionPayload(argv,argv+argc,"--nangle");
    GridCmdOptionInt(s, Nangle);
  }
  if (GridCmdOptionExists(argv,argv+argc,"--theta")) {
    std::string s = GridCmdOptionPayload(argv,argv+argc,"--theta");
    GridCmdOptionFloat(s, theta_deg);
  }
  if (GridCmdOptionExists(argv,argv+argc,"--madj")) {
    std::string s = GridCmdOptionPayload(argv,argv+argc,"--madj");
    GridCmdOptionFloat(s, m_adj);
  }
  if (GridCmdOptionExists(argv,argv+argc,"--nstop")) {
    std::string s = GridCmdOptionPayload(argv,argv+argc,"--nstop");
    GridCmdOptionInt(s, Nstop);
  }
  if (GridCmdOptionExists(argv,argv+argc,"--nk")) {
    std::string s = GridCmdOptionPayload(argv,argv+argc,"--nk");
    GridCmdOptionInt(s, Nk);
  }
  if (GridCmdOptionExists(argv,argv+argc,"--nm")) {
    std::string s = GridCmdOptionPayload(argv,argv+argc,"--nm");
    GridCmdOptionInt(s, Nm);
  }

  RealD eresid  = 1.0e-8;
  int   MaxIter = 200;

  assert(theta_deg > 0.0 && theta_deg <= 180.0 && "require 0 < theta_deg <= 180");
  RealD thetaMax = theta_deg * M_PI / 180.0;

  assert(Nstop <= Nk && Nk < Nm && "require Nstop <= Nk < Nm");

  std::cout << GridLogMessage << "Reading gauge field from: " << file << std::endl;
  std::cout << GridLogMessage << "Angles in arc: " << Nangle
            << " over theta in [" << -theta_deg << ", " << theta_deg << "] degrees"
            << " (total arc " << 2.0*theta_deg << " degrees)" << std::endl;
  std::cout << GridLogMessage << "This probes W(F) from directions "
            << 180.0 - theta_deg << " to " << 180.0 + theta_deg
            << " degrees, i.e. its left-hand side." << std::endl;
  std::cout << GridLogMessage << "Adjoint (Pauli-Villars) mass m_adj = " << m_adj << std::endl;
  std::cout << GridLogMessage << "IRL: Nstop = " << Nstop << ", Nk = " << Nk
            << ", Nm = " << Nm << ", eresid = " << eresid << std::endl;

  if (Nangle % 2 == 0) {
    std::cout << GridLogMessage
              << "WARNING: Nangle is even, so theta = 0 is not sampled and lambda_min(M) "
              << "is not obtained directly. Re-run with an odd Nangle." << std::endl;
  }

  const int Ls=16;

  std::vector<int> lat_size {16, 16, 16, 32};
  std::cout << GridLogMessage << "Lattice size: " << lat_size << std::endl;
  GridCartesian * UGrid = SpaceTimeGrid::makeFourDimGrid(lat_size,
                                                         GridDefaultSimd(Nd,vComplex::Nsimd()),
                                                         GridDefaultMpi());
  GridRedBlackCartesian * UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);

  GridCartesian         * FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls,UGrid);
  GridRedBlackCartesian * FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGrid);
  std::cout << GridLogMessage << "Grids constructed" << std::endl;

  // IRL holds Nm fields at once; warn before allocating rather than after failing.
  RealD fieldGB = RealD(FGrid->gSites()) * 12.0 * 2.0 * RealD(sizeof(RealD)) / 1.0e9;
  std::cout << GridLogMessage << "One LatticeFermionD is " << fieldGB << " GB globally; "
            << "IRL will hold Nm = " << Nm << " of them (" << Nm*fieldGB
            << " GB) plus temporaries." << std::endl;

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
  RealD M5=1.8;
  RealD b=1.5;// Scale factor b+c=2, b-c=1
  RealD c=0.5;
  MobiusFermionD Ddwf(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,b,c);
  MobiusFermionD Dpv (Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,m_adj,M5,b,c);

  typedef PVdagMLinearOperator<MobiusFermionD,LatticeFermionD> PVdagM_t;
  PVdagM_t PVdagM(Ddwf, Dpv);

  std::cout<<GridLogMessage<<std::endl;
  std::cout<<GridLogMessage<<"*******************************************"<<std::endl;
  std::cout<<GridLogMessage<<"***** LEFT BOUNDARY OF FIELD OF VALUES ****"<<std::endl;
  std::cout<<GridLogMessage<<"*******************************************"<<std::endl;

  // sigma_max(F)^2 = lambda_max(F^dag F), the denominator of the EES bound. PVdagM's
  // HermOp is exactly F^dag F, which is HPD, so its dominant eigenvalue is also its
  // algebraically largest and a bare power iteration is unambiguous here. IRL is not
  // needed for this one number, and would cost another Nm fields.
  PowerMethod<LatticeFermionD> PM;
  RealD lambdaMaxFdagF = PM(PVdagM, src);
  std::cout << GridLogMessage << "lambda_max(F^dag F) = " << lambdaMaxFdagF << std::endl;

  std::vector<RealD>    theta (Nangle);
  std::vector<RealD>    g     (Nangle);
  std::vector<ComplexD> zpt   (Nangle);
  std::vector<int>      nconv (Nangle, 0);
  std::vector<int>      ok    (Nangle, 1);

  int nFail = 0;

  RealD t0 = usecond();
  for (int i = 0; i < Nangle; i++) {

    theta[i] = (Nangle == 1) ? 0.0
                             : -thetaMax + 2.0*thetaMax*RealD(i)/RealD(Nangle-1);

    LatticeFermion vmin(FGrid);
    int Nconv = 0;
    g[i] = LambdaMinHermitian(PVdagM, theta[i], src, vmin, Nconv,
                              Nstop, Nk, Nm, eresid, MaxIter);
    nconv[i] = Nconv;

    if (Nconv < 1) {
      ok[i] = 0; nFail++;
      zpt[i] = ComplexD(0.0,0.0);
      continue;
    }

    // z = <v|F|v> / <v|v> is a genuine point of W(F) whatever IRL's convergence did.
    LatticeFermion Fv(FGrid);
    PVdagM.Op(vmin, Fv);
    ComplexD z = innerProduct(vmin, Fv) / RealD(norm2(vmin));
    zpt[i] = z;

    // Consistency: Re(e^{-i theta} z) is the Rayleigh quotient of H_theta in the
    // state v, so it must equal lambda_min(H_theta) if v really is the minimiser.
    RealD rq  = real(z)*std::cos(theta[i]) + imag(z)*std::sin(theta[i]);
    RealD tol = 1.0e-4 * (std::abs(g[i]) + 1.0);
    if (std::abs(rq - g[i]) > tol) {
      ok[i] = 0; nFail++;
      std::cout << GridLogError
                << "CHECK FAILED: at theta = " << theta[i] << " the eigenvector's Rayleigh "
                << "quotient " << rq << " disagrees with the reported lambda_min "
                << g[i] << ". The eigenpair is not converged." << std::endl;
    }

    std::cout << GridLogMessage
              << "angle " << std::setw(4) << i
              << "  theta = " << theta[i] << " rad (" << theta[i]*180.0/M_PI << " deg)"
              << "  lambda_min(H_theta) = " << g[i]
              << "  z = " << z
              << "  (Nconv = " << Nconv << ")" << std::endl;
  }
  RealD t1 = usecond();
  std::cout << GridLogMessage << "Sweep took " << (t1-t0)/1.0e6 << " s" << std::endl;

  /*
   * Pairwise support check.  Each z(theta) is an actual element of W(F), and every
   * supporting line must lie weakly to its left:
   *
   *     g(theta') <= Re( e^{-i theta'} z(theta) )   for all theta, theta'.
   *
   * This is a rigorous consequence of g being an infimum over all unit vectors, so a
   * violation proves that the angle theta' under-converged. It costs no matvecs.
   */
  int   nViolate = 0;
  RealD worst    = 0.0;
  for (int i = 0; i < Nangle; i++) {
    if (!ok[i]) continue;
    for (int j = 0; j < Nangle; j++) {
      if (!ok[j]) continue;
      RealD proj = real(zpt[i])*std::cos(theta[j]) + imag(zpt[i])*std::sin(theta[j]);
      RealD slack = proj - g[j];                       // must be >= 0
      RealD tol   = 1.0e-4 * (std::abs(g[j]) + 1.0);
      if (slack < worst) worst = slack;
      if (slack < -tol) {
        nViolate++;
        std::cout << GridLogError
                  << "CHECK FAILED: supporting line at theta = " << theta[j]
                  << " (g = " << g[j] << ") cuts off the point z = " << zpt[i]
                  << " found at theta = " << theta[i] << " by " << -slack
                  << ". Angle " << j << " is under-converged." << std::endl;
      }
    }
  }
  std::cout << GridLogMessage << "Support check: worst slack = " << worst
            << " over " << Nangle*Nangle << " ordered pairs" << std::endl;

  if (nFail || nViolate) {
    std::cout << GridLogError
              << "*** " << nFail << " per-angle failures and " << nViolate
              << " support violations. Results below are NOT trustworthy. ***" << std::endl;
  } else {
    std::cout << GridLogMessage << "All consistency checks passed." << std::endl;
  }

  std::cout<<GridLogMessage<<"*******************************************"<<std::endl;
  std::cout<<GridLogMessage<<"***************** RESULTS *****************"<<std::endl;
  std::cout<<GridLogMessage<<"*******************************************"<<std::endl;

  std::cout << GridLogMessage << "m_adj                = " << m_adj << std::endl;

  bool  haveZero = (Nangle % 2 == 1);
  RealD lambdaMinM = 0.0;
  if (haveZero) {
    lambdaMinM = g[Nangle/2];                 // theta = 0 sits at the midpoint
    std::cout << GridLogMessage << "lambda_min(M)        = " << lambdaMinM << std::endl;
    std::cout << GridLogMessage << "lambda_min(F + Fdag) = " << 2.0*lambdaMinM
              << "   <-- the quantity quoted in PROJECT.md" << std::endl;

    if (lambdaMinM > 0.0) {
      RealD ess = lambdaMinM*lambdaMinM / lambdaMaxFdagF;
      std::cout << GridLogMessage << "EES ratio lambda_min(M)^2 / lambda_max(Fdag F) = "
                << ess << std::endl;
      std::cout << GridLogMessage
                << "Field of values is in the open right half plane: the Eisenstat-Elman-"
                << "Schultz bound applies, and by field-of-values containment under "
                << "block-orthonormal restriction it applies on every coarse level too."
                << std::endl;
    } else {
      std::cout << GridLogMessage
                << "Field of values straddles the origin: the Eisenstat-Elman-Schultz "
                << "bound is unavailable. Note this does NOT by itself imply slow Krylov "
                << "convergence (cf. Nachtigal, Reddy & Trefethen 1992, section 6)."
                << std::endl;
    }
  }

  // Write the sweep
  std::string fovPath = outDir + "/fov_left.txt";
  std::cout << GridLogMessage << "Writes to: " << fovPath << std::endl;
  std::ofstream fFov;
  fFov.open(fovPath);
  fFov << "# Left boundary of the field of values of F(m_adj, m_l) = D^dag(m_adj) D(m_l).\n"
       << "# g(theta) = lambda_min(H_theta) = min_{|v|=1} Re( e^{-i theta} <v|F|v> ), so\n"
       << "# W(F) lies in every half plane Re( e^{-i theta} z ) >= g(theta).\n"
       << "# (Re_z, Im_z) is <v|F|v> for the minimising eigenvector: an exact point of W(F).\n"
       << "# Ls = " << Ls << ", m_l = " << mass << ", m_adj = " << m_adj
       << ", M5 = " << M5 << ", b = " << b << ", c = " << c << "\n"
       << "# config = " << file << "\n"
       << "# arc: theta in [" << -theta_deg << ", " << theta_deg << "] degrees over "
       << Nangle << " points, probing W(F) from " << 180.0 - theta_deg << " to "
       << 180.0 + theta_deg << " degrees\n"
       << "# IRL Nstop = " << Nstop << ", Nk = " << Nk << ", Nm = " << Nm
       << ", eresid = " << eresid << "\n"
       << "# consistency: " << nFail << " per-angle failures, "
       << nViolate << " support violations, worst slack " << worst << "\n"
       << "# idx  theta  g(theta)  Re_z  Im_z  Nconv  ok\n";
  for (int i = 0; i < Nangle; i++) {
    fFov << i << " " << theta[i] << " " << g[i] << " "
         << real(zpt[i]) << " " << imag(zpt[i]) << " "
         << nconv[i] << " " << ok[i] << "\n";
  }
  if (haveZero) {
    fFov << "# lambda_min(M) = " << lambdaMinM << "\n";
    fFov << "# lambda_min(F + Fdag) = " << 2.0*lambdaMinM << "\n";
  }
  fFov << "# lambda_max(Fdag F) = " << lambdaMaxFdagF << "\n";
  if (haveZero && lambdaMinM > 0.0) {
    fFov << "# EES ratio = " << lambdaMinM*lambdaMinM / lambdaMaxFdagF << "\n";
  }
  fFov.close();

  std::cout<<GridLogMessage<<std::endl;
  std::cout<<GridLogMessage<<"*******************************************"<<std::endl;
  std::cout<<GridLogMessage<<std::endl;
  std::cout<<GridLogMessage << "Done "<< std::endl;

  Grid_finalize();
  return 0;
}
