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
            [--eresid R] [--maxiter N] \
            [--cheby-order N] [--cheby-lo R] [--cheby-hi R] [--write-vecs L] \
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
      --nstop N       = Converged eigenvalues sought per angle. Default 2. Only the
                        extremal eigenvalue is needed for the field of values, and the
                        convergence test requires ALL probed indices to pass, so a
                        larger Nstop makes convergence strictly harder. With the
                        Chebyshev filter on, the extra modes are cheap and the argmin
                        over them is more robust; unfiltered, keep this small.
      --nk N          = Lanczos basis kept per restart. Default 4.
      --nm N          = Total Lanczos basis size. Default 48. Raise this first when
                        angles fail to converge -- at 0.4 GB per vector.
      --eresid R      = Lanczos convergence tolerance, a RELATIVE RESIDUAL NORM.
                        Default 1e-4. Note IRL compares it squared: the printed
                        "target" is eresid^2, so 1e-8 means 1e-16 and will not be
                        reached. Too tight a value does not merely waste time -- IRL
                        calls abort() when MaxIter runs out.
      --maxiter N     = Maximum IRL restarts per angle. Default 500. With the filter
                        on, 20 is ample (see Example_pvdagm_halfplane.cc).
      --cheby-order N = Chebyshev filter order. Default 0, which DISABLES the filter
                        and runs unaccelerated Lanczos on -H_theta. Any N > 0 turns
                        the filter on. MUST BE ODD and is forced odd if not: Grid
                        stores N coefficients with Coeffs[N-1] = 1, so the polynomial
                        is T_{N-1}, and only an even degree is positive below the
                        window. An odd degree is negative there, IRL targets away from
                        the filtered modes, and it never converges. 61 is a good start.
      --cheby-lo R    = Lower edge of the filter window. Default 0.1. Modes BELOW this
                        are amplified, everything inside [lo, hi] is damped to |T| <= 1.
                        It does not need to separate lambda_min from lambda_2 -- it only
                        needs to sit above lambda_min(H_theta) and below the bulk, and
                        Lanczos resolves within the amplified block. There is roughly a
                        decade of slack. If it is set BELOW lambda_min nothing is
                        amplified and IRL locks onto an arbitrary Ritz value; that case
                        is detected per angle and reported as a CHECK FAILED.
      --cheby-hi R    = Upper edge of the filter window. Default 0 = auto, which measures
                        1.1*|lambda_max(H_theta)| PER ANGLE with a power iteration
                        (FieldOfValues.h). The angle-independent alternative
                        1.1*sigma_max(F) is also a valid upper edge -- H_theta =
                        Re(e^{-i theta} F) gives ||H_theta|| <= sigma_max(F) for every
                        theta -- but for a strongly non-normal operator it is far larger
                        than the true lambda_max(H_theta), and the resulting window is so
                        wide that the filter is nearly inert. The per-angle measurement
                        costs about 200 extra operator applications per angle and is only
                        performed when the filter is on.
      --write-vecs L  = Comma separated list of ANGLE INDICES (the first column of
                        fov_left.txt) whose minimising eigenvector should also be written,
                        as ${outDir}/fov${idx} in SCIDAC format. They are named fov, not
                        evec: these are eigenvectors of H_theta, not Ritz vectors of F,
                        which for a non-Hermitian F are different objects. Default empty,
                        which writes none. Giving any index makes the sweep retain ALL
                        eigenvectors until the end -- they are produced one angle at a
                        time and cannot be selected retrospectively -- so budget Nangle
                        times the per-field size, not the size of the list. Needed to do
                        anything with a point of W(F) beyond plotting it: e.g. the unit
                        vector whose Rayleigh quotient is exactly zero, which stagnates the
                        first step of GMRES/GCR, is built from these.

    MEMORY: IRL holds Nm five-dimensional fermion fields simultaneously, plus a
    handful of temporaries.  On 16^3x32 with Ls = 16 one LatticeFermionD is about
    0.4 GB, so Nm = 8 is roughly 3.2 GB before temporaries.  The startup banner
    prints the estimate.  Raise Nm for faster convergence only if the memory is
    there; this is why Nm is on the command line rather than hard coded.

    The sweep itself is operator agnostic and lives in
    Grid/algorithms/iterative/FieldOfValues.h; this driver only builds F and reports.

    Output:
      ${outDir}/fov_left.txt = One line per angle, formatted as
                            `$idx $theta $g $Re_z $Im_z $normFv $Nconv $ok`
                          where ($Re_z, $Im_z) is a point on the left boundary of
                          W(F) and $ok is 0 if any consistency check failed.
                          $normFv = ||F v|| for the minimising unit vector v, so one
                          step of GMRES/GCR from r_0 = v reduces the residual by
                          ||r_1||^2/||r_0||^2 = 1 - |z|^2/$normFv^2.

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
 * Writes one field in SCIDAC format. Handed to FieldOfValues::VectorWriter, which cannot
 * name ScidacWriter itself: Grid/algorithms is parsed before Grid/parallelIO.
 */
template <class T> void writeFile(T& in, std::string const fname){
  #ifdef HAVE_LIME
    std::cout << Grid::GridLogMessage << "Writes to: " << fname << std::endl;
    Grid::emptyUserRecord record;
    Grid::ScidacWriter WR(in.Grid()->IsBoss());
    WR.open(fname);
    WR.writeScidacFieldRecord(in,record,0); // Lexico
    WR.close();
  #else
    std::cout << Grid::GridLogError
              << "Grid was built without LIME; cannot write " << fname << std::endl;
  #endif
}

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
  // int   Nstop     = 2;
  // int   Nk        = 6;
  // int   Nm        = 12;
  int   Nstop     = 2;
  int   Nk        = 4;
  int   Nm        = 48;
  RealD eresid    = 1.0e-4;
  int   MaxIter   = 500;
  int   chebyOrder = 0;                     // 0 disables the filter
  RealD chebyLo    = 0.1;                   // window lower edge; modes below are amplified
  RealD chebyHi    = 0.0;                   // 0 => auto, per-angle 1.1*|lambda_max(H_theta)|
  std::vector<int> writeVecs;               // angle indices whose eigenvector to write

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
              << "[--eresid R] [--maxiter N] "
              << "[--cheby-order N] [--cheby-lo R] [--cheby-hi R] [--write-vecs L] "
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
  if (GridCmdOptionExists(argv,argv+argc,"--eresid")) {
    std::string s = GridCmdOptionPayload(argv,argv+argc,"--eresid");
    GridCmdOptionFloat(s, eresid);
  }
  if (GridCmdOptionExists(argv,argv+argc,"--cheby-order")) {
    std::string s = GridCmdOptionPayload(argv,argv+argc,"--cheby-order");
    GridCmdOptionInt(s, chebyOrder);
  }
  if (GridCmdOptionExists(argv,argv+argc,"--cheby-lo")) {
    std::string s = GridCmdOptionPayload(argv,argv+argc,"--cheby-lo");
    GridCmdOptionFloat(s, chebyLo);
  }
  if (GridCmdOptionExists(argv,argv+argc,"--cheby-hi")) {
    std::string s = GridCmdOptionPayload(argv,argv+argc,"--cheby-hi");
    GridCmdOptionFloat(s, chebyHi);
  }
  if (GridCmdOptionExists(argv,argv+argc,"--maxiter")) {
    std::string s = GridCmdOptionPayload(argv,argv+argc,"--maxiter");
    GridCmdOptionInt(s, MaxIter);
  }
  if (GridCmdOptionExists(argv,argv+argc,"--write-vecs")) {
    std::string s = GridCmdOptionPayload(argv,argv+argc,"--write-vecs");
    GridCmdOptionIntVector(s, writeVecs);
  }

  // theta_deg is validated, Nstop <= Nk < Nm is asserted, and an even --cheby-order is
  // forced odd, all in the FieldOfValues constructor below.  Nothing to repeat here.

  std::cout << GridLogMessage << "Reading gauge field from: " << file << std::endl;
  std::cout << GridLogMessage << "Angles in arc: " << Nangle
            << " over theta in [" << -theta_deg << ", " << theta_deg << "] degrees"
            << " (total arc " << 2.0*theta_deg << " degrees)" << std::endl;
  std::cout << GridLogMessage << "This probes W(F) from directions "
            << 180.0 - theta_deg << " to " << 180.0 + theta_deg
            << " degrees, i.e. its left-hand side." << std::endl;
  std::cout << GridLogMessage << "Adjoint (Pauli-Villars) mass m_adj = " << m_adj << std::endl;
  
  std::cout << GridLogMessage << "IRL: Nstop = " << Nstop << ", Nk = " << Nk
            << ", Nm = " << Nm << ", eresid = " << eresid
            << " (convergence target is eresid^2 = " << eresid*eresid << ")"
            << ", MaxIter = " << MaxIter << std::endl;
  std::cout << GridLogMessage
            << "NOTE: IRL aborts the job if MaxIter is exhausted without convergence."
            << std::endl;

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

  RealD sigmaMax = std::sqrt(lambdaMaxFdagF);
  std::cout << GridLogMessage << "sigma_max(F) = " << sigmaMax << std::endl;
  if (chebyOrder > 0) {
    std::cout << GridLogMessage << "Chebyshev filter: T_" << chebyOrder-1
              << " with lower edge " << chebyLo;
    if (chebyHi > chebyLo) { std::cout << " and upper edge " << chebyHi << std::endl; }
    else { std::cout << " and per-angle auto upper edge" << std::endl; }
  } else {
    std::cout << GridLogMessage << "Chebyshev filter DISABLED (--cheby-order 0): "
              << "unaccelerated Lanczos on -H_theta." << std::endl;
  }

  // Everything above this point is operator specific; everything below is not.  The
  // sweep, its consistency checks, the Chebyshev filtering and the output all live in
  // Grid/algorithms/iterative/FieldOfValues.h and touch only Op and AdjOp.
  FieldOfValues<LatticeFermionD> FoV(theta_deg, Nangle, Nstop, Nk, Nm, eresid, MaxIter,
                                     chebyOrder, chebyLo, chebyHi);

  // Everything the sweep cannot know about itself, because none of it is a property of a
  // LinearOperatorBase. Written verbatim into the head of fov_left.txt.
  {
    std::ostringstream prov;
    prov << "# F(m_adj, m_l) = D^dag(m_adj) D(m_l), Mobius domain wall\n"
         << "# Ls = " << Ls << ", m_l = " << mass << ", m_adj = " << m_adj
         << ", M5 = " << M5 << ", b = " << b << ", c = " << c << "\n"
         << "# config = " << file << "\n"
         << "# lambda_max(Fdag F) = " << lambdaMaxFdagF
         << ", sigma_max(F) = " << sigmaMax << "\n";
    FoV.Provenance = prov.str();
  }

  // FieldOfValues.h sits below parallelIO in Grid's include order and so cannot name
  // ScidacWriter itself; it writes fields through this hook.
  FoV.VectorWriter = [](LatticeFermionD &v, const std::string &f) {
    writeFile(v, f);
  };
  FoV.KeepVectors = !writeVecs.empty();
  if (FoV.KeepVectors) {
    std::cout << GridLogMessage << "Retaining all " << Nangle
              << " minimising eigenvectors for output (" << Nangle*fieldGB << " GB)"
              << std::endl;
  }

  std::vector<FieldOfValuesPoint> pts = FoV(PVdagM, src);

  int   nFail    = FoV.nFail;
  int   nViolate = FoV.SupportCheck(pts);
  RealD worst    = FoV.worstSlack;

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
    lambdaMinM = pts[Nangle/2].g;             // theta = 0 sits at the midpoint
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

  // Write the sweep, and any requested eigenvectors alongside it.
  FoV.Write(pts, outDir, writeVecs);

  std::cout<<GridLogMessage<<std::endl;
  std::cout<<GridLogMessage<<"*******************************************"<<std::endl;
  std::cout<<GridLogMessage<<std::endl;
  std::cout<<GridLogMessage << "Done "<< std::endl;

  Grid_finalize();
  return 0;
}
