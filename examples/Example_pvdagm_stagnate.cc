/*************************************************************************************

    Constructs a right-hand side that stagnates GCR on the Pauli-Villars preconditioned
    Mobius domain wall operator

        F(m_adj, m_l) = D^dag(m_adj) D(m_l),

    and demonstrates the stagnation against a random right-hand side as a control.

    WHY THIS WORKS.  One step of GMRES/GCR from residual r_0 minimises ||r_0 - a F r_0||
    over a, whose optimum is a = <F r_0, r_0>/||F r_0||^2.  The reduction is therefore

        ||r_1||^2 / ||r_0||^2 = 1 - |<v|F|v>|^2 / ||F v||^2 ,     v = r_0/||r_0||,

    so the step does NOTHING precisely when <v|F|v> = 0.  What stalls a solve is the
    Rayleigh quotient being ZERO, not its real part being negative: the sign never enters.
    A vector at the leftmost point of W(F) is the right probe for positive-realness and the
    wrong probe for stagnation.

    Such a v exists iff 0 lies in the field of values W(F) = { <v|F|v> : |v| = 1 }, and
    that is decidable from a completed FoV sweep with no further operator applications.
    Each recorded z(theta) = <v_theta|F|v_theta> is an EXACT point of W(F), and W(F) is
    convex (Toeplitz-Hausdorff), so if the origin lies inside the convex hull of the
    recorded points then 0 is in W(F) and a stagnating vector exists in their span.  For
    ckpoint_lat.4000 at m_adj = 1 the three angles theta = -30, 0, +30 degrees already
    enclose the origin, which is why those three suffice.

    THE CONSTRUCTION.  Let V be an orthonormal basis of span{v_1, v_2, v_3} (3 fields) and
    B = V^dag F V the 3x3 compression.  W(B) is contained in W(F) and contains each
    z(theta), hence contains their hull, hence contains 0.  So it is enough to find a unit
    y in C^3 with y^dag B y = 0 and set r_0 = V y -- a dense 3x3 problem with no operator
    applications in it at all.

    Solving that 3x3 problem follows the constructive proof of Toeplitz-Hausdorff, in two
    two-dimensional steps.  The key fact is that for any two unit vectors a, b the
    numerical range of the 2x2 compression onto span{a, b} is an ELLIPSE, hence convex,
    hence contains the whole chord between <a|F|a> and <b|F|b>: any point of that chord is
    the Rayleigh quotient of some explicit unit vector in span{a, b}.

    Write z_i for the Rayleigh quotient of input vector i and solve the real 3x3 system

        w_1 z_1 + w_2 z_2 + w_3 z_3 = 0 ,    w_1 + w_2 + w_3 = 1

    for the barycentric coordinates of the origin.  All three w_i are non-negative exactly
    when the origin lies inside the triangle, which is the condition to check first.  Then

      step A:  p = (1 - lam) z_1 + lam z_3 with lam = w_3/(w_1 + w_3) is the point where
               the line from z_2 through the origin meets the opposite edge.  Find a unit
               u in span{v_1, v_3} with <u|F|u> = p.
      step B:  0 lies on the chord from z_2 to p, so find a unit y in span{v_2, u} with
               <y|F|y> = 0.  That is r_0.

    Each step solves RQ(t, phi) = tau for a unit vector cos(t) e_1 + e^{i phi} sin(t) e_2 in
    an orthonormalised pair -- two real equations in two real unknowns -- by a coarse grid
    followed by Newton with analytic derivatives.  It converges to roundoff.

    A NOTE ON AN APPROACH THAT DOES NOT WORK, since it is the obvious one.  Splitting
    B = H + iK into Hermitian parts and looking for y with y^dag H y = 0 and y^dag K y = 0
    suggests taking eigenvectors h_p, h_n of H with eigenvalues of opposite sign and using
    y = alpha h_p + beta e^{i psi} h_n, which kills the H part exactly for every psi and
    leaves one phase to kill the K part.  That fails: y^dag K y = A + C cos(psi + arg kappa)
    has a root only when |A| <= C, and restricting to a 2-plane spanned by two H
    eigenvectors discards too much of the space.  Tested on normal 3x3 matrices whose
    eigenvalues are exactly the three measured z values -- where 0 is provably in W(B) --
    it returns |y^dag B y| ~ 0.5 rather than 0.  The chord construction above returns
    ~1e-16 on the same matrices.

    WHAT IT DOES AND DOES NOT SHOW.  A zero Rayleigh quotient stalls exactly ONE step:
    GMRES stagnates completely for k steps iff <F^j r_0, r_0> = 0 for j = 1..k, and this
    construction kills only the j = 1 moment.  This is an ENGINEERED right-hand side, so it
    demonstrates that stagnation is possible and that the Eisenstat-Elman-Schultz bound has
    no content here -- that bound needs dist(0, W(F)) > 0, and the distance is zero.  It
    does not show that physical sources converge slowly.  The random right-hand side is run
    alongside precisely to make that contrast visible.

    Usage :
      $ ./Example_pvdagm_stagnate --config <file> --indir <dir> \
            [--outdir <dir>] [--vecs L] [--madj M] \
            [--gcr-tol R] [--gcr-maxit N] [--mmax N] [--nstep N] \
            [Grid options]

      --config <file> = Gauge configuration to read in (NERSC format). REQUIRED. MUST be
                        the same configuration the field-of-values sweep was run on.
      --indir <dir>   = Directory holding a completed sweep: fov_left.txt and the fov${idx}
                        vectors written by Example_pvdagm_fov --write-vecs. REQUIRED.
      --outdir <dir>  = Where to write the constructed vector, as ${outdir}/stagnate_r0 in
                        SCIDAC format. Default empty, which writes nothing.
      --vecs L        = Comma separated ANGLE INDICES to build the hull from. Default is
                        the first, middle and last angle of the sweep, which for the
                        standard 17-point arc is theta = -30, 0, +30 degrees. Exactly three
                        are expected; more are accepted and the compression grows to match.
      --madj M        = Mass of the adjoint (Pauli-Villars) factor. Default 1.0. MUST match
                        the value the sweep used, or B is a compression of a different
                        operator than the one whose field of values was measured and the
                        origin is no longer guaranteed to be inside.
      --gcr-tol R     = GCR relative residual target. Default 1e-8.
      --gcr-maxit N   = GCR maximum iterations. Default 200.
      --mmax N        = GCR search direction depth. Default 8.
      --nstep N       = GCR steps per restart. Default 8.

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./examples/Example_pvdagm_stagnate.cc

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
#include <iomanip>

#include <Grid/Grid.h>

#include <Grid/parallelIO/IldgIOtypes.h>
#include <Grid/parallelIO/IldgIO.h>

using namespace std;
using namespace Grid;

/** Writes one field in SCIDAC format. */
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

/** Reads one field in SCIDAC format. Handed to ReadFieldOfValuesVectors. */
template <class T> void readFile(T& out, std::string const fname){
  #ifdef HAVE_LIME
    std::cout << Grid::GridLogMessage << "Reads from: " << fname << std::endl;
    Grid::emptyUserRecord record;
    Grid::ScidacReader SR;
    SR.open(fname);
    SR.readScidacFieldRecord(out, record);
    SR.close();
  #else
    std::cout << Grid::GridLogError
              << "Grid was built without LIME; cannot read " << fname << std::endl;
  #endif
}

/**
 * F = D_PV^dag D_dwf.  Same operator as Example_pvdagm_fov.cc and Example_spec_kryschur.cc.
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

/** Rayleigh quotient of cos(t) e_1 + e^{i phi} sin(t) e_2 under a 2x2 compression. */
inline ComplexD RQ2(const Eigen::Matrix2cd &M, RealD t, RealD phi)
{
  RealD    c = std::cos(t), s = std::sin(t);
  ComplexD e = std::exp(ComplexD(0.0,1.0)*phi);
  return c*c*M(0,0) + s*s*M(1,1) + c*s*(e*M(0,1) + std::conj(e)*M(1,0));
}

/**
 * Unit y in C^2 with y^dag M y = tau, for tau inside the numerical range of M.
 *
 * Two real equations in the two real unknowns (t, phi).  A coarse grid locates the basin
 * -- the map is not injective and Newton alone would wander -- and Newton with analytic
 * derivatives then converges to roundoff.  `resid` receives the achieved |RQ - tau|, which
 * is the honest report when tau is outside the ellipse and no exact solution exists.
 */
Eigen::Vector2cd Solve2D(const Eigen::Matrix2cd &M, ComplexD tau, RealD &resid,
                         int ngrid = 128, int nNewton = 60)
{
  RealD tBest = 0.0, pBest = 0.0, best = -1.0;
  for (int i = 0; i < ngrid; i++) {
    RealD t = 0.5*M_PI*RealD(i)/RealD(ngrid-1);
    for (int j = 0; j < ngrid; j++) {
      RealD p = 2.0*M_PI*RealD(j)/RealD(ngrid);
      RealD v = std::abs(RQ2(M,t,p) - tau);
      if (best < 0.0 || v < best) { best = v; tBest = t; pBest = p; }
    }
  }

  RealD t = tBest, p = pBest;
  for (int it = 0; it < nNewton; it++) {
    ComplexD r  = RQ2(M,t,p) - tau;
    RealD    c  = std::cos(t), s = std::sin(t);
    ComplexD e  = std::exp(ComplexD(0.0,1.0)*p);
    ComplexD g  = e*M(0,1) + std::conj(e)*M(1,0);
    ComplexD dt = -2.0*c*s*M(0,0) + 2.0*s*c*M(1,1) + (c*c - s*s)*g;
    ComplexD dp = c*s*ComplexD(0.0,1.0)*(e*M(0,1) - std::conj(e)*M(1,0));

    RealD det = real(dt)*imag(dp) - imag(dt)*real(dp);
    if (std::abs(det) < 1.0e-300) break;
    RealD dts = ( imag(dp)*real(r) - real(dp)*imag(r)) / det;
    RealD dps = (-imag(dt)*real(r) + real(dt)*imag(r)) / det;
    t -= dts;
    p -= dps;
  }

  resid = std::abs(RQ2(M,t,p) - tau);
  Eigen::Vector2cd y;
  y(0) = std::cos(t);
  y(1) = std::exp(ComplexD(0.0,1.0)*p)*std::sin(t);
  return y / y.norm();
}

/** Barycentric coordinates of the origin with respect to the triangle z_0 z_1 z_2. */
Eigen::Vector3d BarycentricOrigin(const Eigen::Vector3cd &z)
{
  Eigen::Matrix3d A;
  for (int i = 0; i < 3; i++) { A(0,i) = real(z(i)); A(1,i) = imag(z(i)); A(2,i) = 1.0; }
  Eigen::Vector3d rhs(0.0, 0.0, 1.0);
  return A.colPivHouseholderQr().solve(rhs);
}

int main (int argc, char ** argv)
{
  Grid_init(&argc,&argv);

  RealD m_adj    = 1.0;
  RealD gcrTol   = 1.0e-8;
  int   gcrMaxIt = 200;
  int   mmax     = 8;
  int   nstep    = 8;
  std::vector<int> vecIdx;

  std::string file, inDir, outDir;
  if (GridCmdOptionExists(argv,argv+argc,"--config")) {
    file = GridCmdOptionPayload(argv,argv+argc,"--config");
  }
  if (GridCmdOptionExists(argv,argv+argc,"--indir")) {
    inDir = GridCmdOptionPayload(argv,argv+argc,"--indir");
  }
  if (GridCmdOptionExists(argv,argv+argc,"--outdir")) {
    outDir = GridCmdOptionPayload(argv,argv+argc,"--outdir");
  }
  if (file.empty() || inDir.empty()) {
    std::cout << GridLogError
              << "usage: Example_pvdagm_stagnate --config <file> --indir <dir> "
              << "[--outdir <dir>] [--vecs L] [--madj M] "
              << "[--gcr-tol R] [--gcr-maxit N] [--mmax N] [--nstep N] "
              << "[Grid options]" << std::endl;
    Grid_finalize();
    return 1;
  }

  if (GridCmdOptionExists(argv,argv+argc,"--madj")) {
    std::string s = GridCmdOptionPayload(argv,argv+argc,"--madj");
    GridCmdOptionFloat(s, m_adj);
  }
  if (GridCmdOptionExists(argv,argv+argc,"--vecs")) {
    std::string s = GridCmdOptionPayload(argv,argv+argc,"--vecs");
    GridCmdOptionIntVector(s, vecIdx);
  }
  if (GridCmdOptionExists(argv,argv+argc,"--gcr-tol")) {
    std::string s = GridCmdOptionPayload(argv,argv+argc,"--gcr-tol");
    GridCmdOptionFloat(s, gcrTol);
  }
  if (GridCmdOptionExists(argv,argv+argc,"--gcr-maxit")) {
    std::string s = GridCmdOptionPayload(argv,argv+argc,"--gcr-maxit");
    GridCmdOptionInt(s, gcrMaxIt);
  }
  if (GridCmdOptionExists(argv,argv+argc,"--mmax")) {
    std::string s = GridCmdOptionPayload(argv,argv+argc,"--mmax");
    GridCmdOptionInt(s, mmax);
  }
  if (GridCmdOptionExists(argv,argv+argc,"--nstep")) {
    std::string s = GridCmdOptionPayload(argv,argv+argc,"--nstep");
    GridCmdOptionInt(s, nstep);
  }

  // ---- The recorded sweep -------------------------------------------------------------
  std::vector<FieldOfValuesPoint> pts = ReadFieldOfValues(inDir);
  if (pts.empty()) {
    std::cout << GridLogError << "No field-of-values data in " << inDir
              << "; nothing to build from." << std::endl;
    Grid_finalize();
    return 1;
  }

  if (vecIdx.empty()) {
    vecIdx.push_back(0);
    vecIdx.push_back((int)pts.size()/2);
    vecIdx.push_back((int)pts.size()-1);
  }
  int Nv = (int)vecIdx.size();

  std::cout << GridLogMessage << "Building the hull from " << Nv << " angles:" << std::endl;
  for (int i = 0; i < Nv; i++) {
    int k = vecIdx[i];
    assert(k >= 0 && k < (int)pts.size() && "angle index out of range");
    std::cout << GridLogMessage << "   idx " << k
              << "  theta = " << pts[k].theta*180.0/M_PI << " deg"
              << "  z = " << pts[k].z
              << "  |z| = " << std::abs(pts[k].z)
              << "  ||Fv|| = " << pts[k].normFv << std::endl;
  }

  // ---- Lattice and operator -----------------------------------------------------------
  const int Ls=16;

  std::vector<int> lat_size {16, 16, 16, 32};
  std::cout << GridLogMessage << "Lattice size: " << lat_size << std::endl;
  GridCartesian * UGrid = SpaceTimeGrid::makeFourDimGrid(lat_size,
                                                         GridDefaultSimd(Nd,vComplex::Nsimd()),
                                                         GridDefaultMpi());
  GridRedBlackCartesian * UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian         * FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls,UGrid);
  GridRedBlackCartesian * FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGrid);

  std::vector<int> seeds5({5,6,7,8});
  GridParallelRNG RNG5(FGrid);  RNG5.SeedFixedIntegers(seeds5);

  LatticeGaugeField Umu(UGrid);
  std::cout << GridLogMessage << "Reading configuration " << file << std::endl;
  FieldMetaData header;
  NerscIO::readConfiguration(Umu,header,file);

  RealD mass=0.01;
  RealD M5=1.8;
  RealD b=1.5;
  RealD c=0.5;
  MobiusFermionD Ddwf(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,b,c);
  MobiusFermionD Dpv (Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,m_adj,M5,b,c);

  typedef PVdagMLinearOperator<MobiusFermionD,LatticeFermionD> PVdagM_t;
  PVdagM_t PVdagM(Ddwf, Dpv);
  std::cout << GridLogMessage << "m_adj = " << m_adj
            << " (must match the sweep that produced " << inDir << ")" << std::endl;

  // ---- Read the eigenvectors and orthonormalise ---------------------------------------
  std::vector<LatticeFermionD> vin;
  ReadFieldOfValuesVectors(inDir, vecIdx, vin, FGrid,
      [](LatticeFermionD &v, const std::string &f) { readFile(v, f); });
  assert((int)vin.size() == Nv && "failed to read the requested vectors");

  // Modified Gram-Schmidt. Only Nv = O(3) fields, so a dense QR would be overkill; the
  // rank check below is the part that matters, since two nearly parallel FoV vectors would
  // make the compression a rank-deficient fake rather than fail outright.
  std::vector<LatticeFermionD> V;
  for (int i = 0; i < Nv; i++) {
    LatticeFermionD w = vin[i];
    for (int j = 0; j < (int)V.size(); j++) {
      ComplexD ov = innerProduct(V[j], w);
      w = w - ov*V[j];
    }
    RealD nw = std::sqrt(norm2(w));
    std::cout << GridLogMessage << "Gram-Schmidt: vector " << vecIdx[i]
              << " retains norm " << nw << " after projection" << std::endl;
    if (nw < 1.0e-6) {
      std::cout << GridLogError
                << "CHECK FAILED: vector " << vecIdx[i] << " is numerically inside the span "
                << "of the earlier ones. The hull is degenerate; choose angles further apart."
                << std::endl;
      Grid_finalize();
      return 1;
    }
    V.push_back((1.0/nw)*w);
  }

  // ---- The compression B = V^dag F V --------------------------------------------------
  Eigen::MatrixXcd B(Nv,Nv);
  {
    LatticeFermionD FV(FGrid);
    for (int j = 0; j < Nv; j++) {
      PVdagM.Op(V[j], FV);                       // Nv operator applications in total
      for (int i = 0; i < Nv; i++) {
        B(i,j) = innerProduct(V[i], FV);
      }
    }
  }
  std::cout << GridLogMessage << "Compression B = V^dag F V =" << std::endl << B << std::endl;

  // Coefficient vectors of the inputs in the orthonormal basis. Each is a unit vector of
  // C^Nv whose Rayleigh quotient under B is the recorded z, which is what makes the dense
  // problem below faithful to the measured points.
  Eigen::MatrixXcd Cf(Nv,Nv);
  Eigen::Vector3cd zz;
  for (int i = 0; i < Nv; i++) {
    Eigen::VectorXcd ci(Nv);
    for (int j = 0; j < Nv; j++) ci(j) = innerProduct(V[j], vin[i]);
    ci.normalize();
    Cf.col(i) = ci;
    ComplexD zi = ComplexD(ci.adjoint()*B*ci);
    zz(i) = zi;

    // A wrong --madj or --config shows up here rather than as a mysteriously non-zero
    // Rayleigh quotient at the end.
    RealD dev = std::abs(zi - pts[vecIdx[i]].z);
    std::cout << GridLogMessage << "Recovered z for angle " << vecIdx[i] << " = " << zi
              << " against recorded " << pts[vecIdx[i]].z << ", deviation " << dev << std::endl;
    if (dev > 1.0e-6*(std::abs(pts[vecIdx[i]].z) + 1.0)) {
      std::cout << GridLogError
                << "CHECK FAILED: the compression does not reproduce the recorded point. "
                << "Wrong --madj, wrong --config, or the vectors do not match fov_left.txt."
                << std::endl;
    }
  }

  // ---- Solve the 3x3 problem ----------------------------------------------------------
  std::cout<<GridLogMessage<<"*******************************************"<<std::endl;
  std::cout<<GridLogMessage<<"******** ZERO RAYLEIGH QUOTIENT ***********"<<std::endl;
  std::cout<<GridLogMessage<<"*******************************************"<<std::endl;

  assert(Nv == 3 && "the chord construction is written for exactly three vectors");

  Eigen::Vector3d w = BarycentricOrigin(zz);
  std::cout << GridLogMessage << "Barycentric coordinates of the origin: "
            << w.transpose() << std::endl;
  if (w.minCoeff() < 0.0) {
    std::cout << GridLogError
              << "CHECK FAILED: the origin is OUTSIDE the triangle of the three recorded "
              << "points, so nothing in their span has zero Rayleigh quotient. Widen the "
              << "arc or choose angles whose points enclose the origin." << std::endl;
    Grid_finalize();
    return 1;
  }

  // Step A: u in span{v_1, v_3} with <u|F|u> = p, the point where the line from z_2 through
  // the origin meets the opposite edge.
  RealD    lam = w(2)/(w(0) + w(2));
  ComplexD p   = (1.0 - lam)*zz(0) + lam*zz(2);
  std::cout << GridLogMessage << "Edge point p = " << p << " (lambda = " << lam << ")"
            << std::endl;

  Eigen::VectorXcd e1 = Cf.col(0);
  Eigen::VectorXcd e2 = Cf.col(2) - (e1.adjoint()*Cf.col(2))(0,0)*e1;
  e2.normalize();
  Eigen::Matrix2cd M2;
  M2 << ComplexD(e1.adjoint()*B*e1), ComplexD(e1.adjoint()*B*e2),
        ComplexD(e2.adjoint()*B*e1), ComplexD(e2.adjoint()*B*e2);
  RealD residA;
  Eigen::Vector2cd ya = Solve2D(M2, p, residA);
  Eigen::VectorXcd u  = ya(0)*e1 + ya(1)*e2;
  std::cout << GridLogMessage << "Step A residual |<u|F|u> - p| = " << residA << std::endl;

  // Step B: y in span{v_2, u} with <y|F|y> = 0, which lies on the chord from z_2 to p.
  e1 = Cf.col(1);
  e2 = u - (e1.adjoint()*u)(0,0)*e1;
  e2.normalize();
  M2 << ComplexD(e1.adjoint()*B*e1), ComplexD(e1.adjoint()*B*e2),
        ComplexD(e2.adjoint()*B*e1), ComplexD(e2.adjoint()*B*e2);
  RealD residB;
  Eigen::Vector2cd yb = Solve2D(M2, ComplexD(0.0,0.0), residB);
  Eigen::VectorXcd y  = yb(0)*e1 + yb(1)*e2;
  y.normalize();
  std::cout << GridLogMessage << "Step B residual |<y|F|y>|     = " << residB << std::endl;

  RealD predicted = std::abs(ComplexD(y.adjoint()*B*y));
  std::cout << GridLogMessage << "|y^dag B y| from the dense problem = " << predicted
            << (predicted < 1.0e-12 ? "   (exact: 0 is in W(B), hence in W(F))"
                                    : "   (NOT exact -- see the step residuals above)")
            << std::endl;

  // ---- Assemble r_0 and verify against the real operator ------------------------------
  LatticeFermionD r0(FGrid); r0 = Zero();
  for (int j = 0; j < Nv; j++) r0 = r0 + y(j)*V[j];
  r0 = (1.0/std::sqrt(norm2(r0)))*r0;

  LatticeFermionD Fr0(FGrid);
  PVdagM.Op(r0, Fr0);
  ComplexD zr0    = innerProduct(r0, Fr0);
  RealD    nFr0   = std::sqrt(norm2(Fr0));
  RealD    reduce = 1.0 - std::norm(zr0)/(nFr0*nFr0);

  std::cout << GridLogMessage << "<r0|F|r0>   = " << zr0
            << "   |.| = " << std::abs(zr0) << std::endl;
  std::cout << GridLogMessage << "||F r0||    = " << nFr0 << std::endl;
  std::cout << GridLogMessage << "Predicted first-step reduction ||r1||^2/||r0||^2 = "
            << reduce << std::endl;
  std::cout << GridLogMessage
            << "A value of 1 means the first GCR step makes no progress at all."
            << std::endl;

  if (!outDir.empty()) {
    std::string rName = outDir + "/stagnate_r0";
    writeFile(r0, rName);
  }

  // ---- GCR on the engineered source, then on a random one -----------------------------
  TrivialPrecon<LatticeFermionD> Prec;
  LatticeFermionD psi(FGrid);

  std::cout<<GridLogMessage<<"*******************************************"<<std::endl;
  std::cout<<GridLogMessage<<"****** GCR ON THE ENGINEERED SOURCE *******"<<std::endl;
  std::cout<<GridLogMessage<<"*******************************************"<<std::endl;
  {
    // b = r0 with x0 = 0 makes the initial residual exactly r0. PGCR does NOT zero psi
    // itself (the psi=Zero() at PrecGeneralisedConjugateResidualNonHermitian.h:77 is
    // commented out), so it has to be done here or the initial residual is garbage.
    PrecGeneralisedConjugateResidualNonHermitian<LatticeFermionD>
      GCR(gcrTol, gcrMaxIt, PVdagM, Prec, mmax, nstep);
    psi = Zero();
    GCR(r0, psi);
  }

  std::cout<<GridLogMessage<<"*******************************************"<<std::endl;
  std::cout<<GridLogMessage<<"******** GCR ON A RANDOM SOURCE ***********"<<std::endl;
  std::cout<<GridLogMessage<<"*******************************************"<<std::endl;
  {
    LatticeFermionD rnd(FGrid); random(RNG5, rnd);
    rnd = (1.0/std::sqrt(norm2(rnd)))*rnd;

    LatticeFermionD Frnd(FGrid);
    PVdagM.Op(rnd, Frnd);
    ComplexD zr = innerProduct(rnd, Frnd);
    RealD    nF = std::sqrt(norm2(Frnd));
    std::cout << GridLogMessage << "Control: <v|F|v> = " << zr
              << ", ||Fv|| = " << nF
              << ", predicted first-step reduction = "
              << 1.0 - std::norm(zr)/(nF*nF) << std::endl;

    PrecGeneralisedConjugateResidualNonHermitian<LatticeFermionD>
      GCR(gcrTol, gcrMaxIt, PVdagM, Prec, mmax, nstep);
    psi = Zero();
    GCR(rnd, psi);
  }

  std::cout<<GridLogMessage<<std::endl;
  std::cout<<GridLogMessage << "Done "<< std::endl;

  Grid_finalize();
  return 0;
}
