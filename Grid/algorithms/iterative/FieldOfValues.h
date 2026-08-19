/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./lib/algorithms/iterative/FieldOfValues.h

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
#ifndef GRID_FIELD_OF_VALUES_H
#define GRID_FIELD_OF_VALUES_H

NAMESPACE_BEGIN(Grid);

/**
 * One supporting line of the field of values, and the point of W(F) that realises it.
 *
 * `z` is an EXACT member of W(F) regardless of how well the eigensolve converged, since
 * it is the Rayleigh quotient of an actual unit vector.  `g` is only as good as the
 * eigensolve.  That asymmetry is why both are kept: the pairwise support check below
 * tests the (trustworthy) points against the (approximate) lines.
 *
 * `normFv` = ||F v|| turns each point into a first-step GMRES/GCR residual reduction,
 *
 *     ||r_1||^2 / ||r_0||^2 = 1 - |z|^2 / normFv^2 ,     r_0 = v,
 *
 * which is the quantity that decides whether a given point of W(F) actually stalls a
 * solve.  It cannot be reconstructed from z alone, so it is measured here (one extra
 * operator application per angle).
 */
struct FieldOfValuesPoint {
  RealD    theta;      // supporting direction, radians
  RealD    g;          // lambda_min(H_theta) = min_{|v|=1} Re( e^{-i theta} <v|F|v> )
  ComplexD z;          // <v|F|v> for the minimising v: an exact point of W(F)
  RealD    normFv;     // ||F v||
  int      Nconv;      // eigenvalues IRL converged at this angle
  int      ok;         // 0 if any consistency check failed here
};

/**
 * Left-hand boundary of the field of values (numerical range) of a general linear
 * operator F,
 *
 *     W(F) = { <v|F|v> : |v| = 1 }.
 *
 * W(F) is convex (Toeplitz-Hausdorff), so it is cut out by its supporting lines.  With
 *
 *     H_theta  = 1/2 ( e^{-i theta} F + e^{+i theta} F^dag ) = Re( e^{-i theta} F ),
 *     g(theta) = lambda_min(H_theta) = min_{|v|=1} Re( e^{-i theta} <v|F|v> ),
 *
 * W(F) lies in the half plane Re( e^{-i theta} z ) >= g(theta).  Sweeping theta over an
 * arc centred on 0 gives exactly the supporting lines whose inward normal points
 * rightward, i.e. the LEFT boundary of W(F).  theta = 0 gives
 *
 *     g(0) = lambda_min( (F + F^dag)/2 ),
 *
 * the leftmost real extent of W(F), whose sign decides whether the Eisenstat-Elman-
 * Schultz bound is available at all.
 *
 * g(theta) is an EXTREME (algebraically smallest) eigenvalue of a Hermitian operator,
 * not an interior one, so Lanczos handles it well.  Grid's IRL targets the algebraically
 * LARGEST eigenvalue -- it sorts Ritz values descending and applies the implicit QR
 * shifts at the smallest ones -- so the unfiltered path runs it on
 *
 *     H_{theta + pi} = -H_theta,     g(theta) = -lambda_max(H_{theta+pi}),
 *
 * which needs no spectral shift and cannot converge to the wrong end of the spectrum.
 * Note this is a different problem from finding the smallest-MODULUS eigenvalues of F
 * itself, which are interior and would need a harmonic or shift-invert method (see
 * KrylovSchur's harmonic branch).
 *
 * OPERATOR AGNOSTIC.  Only Op and AdjOp of the supplied operator are ever used, both via
 * HermitianPartLinearOperator.  In particular nothing here assumes that the operator's
 * own HermOp is F^dag F, and no mass or discretisation parameter appears: build the
 * operator in the driver and pass it in.
 *
 * Usage:
 *
 *     FieldOfValues<LatticeFermionD> FoV(30.0, 17);          // +/- 30 degrees, 17 angles
 *     auto pts = FoV(PVdagM, src);
 *     FoV.SupportCheck(pts);
 *
 * The tuning parameters are constructor arguments rather than operator() arguments,
 * following ConjugateGradient and the other Krylov solvers in this directory: there are
 * ten of them, they are mostly RealD and int, and a positional call would be a long
 * unlabelled list that silently does the wrong thing when two are transposed.  What
 * operator() takes is what changes per call -- the operator and the source.
 */
template<class Field>
class FieldOfValues
{
public:

  int   Nangle;        // number of supporting directions in the arc
  RealD thetaMax;      // arc half width, RADIANS (the constructor takes degrees)

  int   Nstop;         // converged eigenvalues sought per angle
  int   Nk;            // Lanczos basis kept per restart
  int   Nm;            // total Lanczos basis size
  RealD eresid;        // IRL tolerance. NOTE IRL compares this SQUARED
  int   MaxIter;       // maximum IRL restarts per angle

  int   chebyOrder;    // 0 disables the filter. Forced ODD if positive, see below
  RealD chebyLo;       // window lower edge; modes BELOW this are amplified
  RealD chebyHi;       // window upper edge; <= chebyLo means auto, see LambdaMin

  RealD checkTol;      // relative tolerance for the consistency checks

  bool  KeepVectors;   // retain the minimising eigenvector of every angle in `vecs`
  std::vector<Field> vecs;

  /**
   * Extra `#` comment lines for the head of the output file, supplied by the driver.
   *
   * Everything this class knows about is operator agnostic, so nothing here can name a
   * mass, an Ls, a gauge configuration or sigma_max(F).  Those are exactly what makes a
   * sweep reproducible, so the driver writes them into this string -- each line already
   * beginning with "# " and ending with "\n" -- and Write() emits it verbatim.
   */
  std::string Provenance;

  /**
   * How to write one Field to one path.  Must be set by the driver before Write() is
   * called with a non-empty `nvecs`; left unset, vector output is refused with an error
   * and the text output still happens.
   *
   * It is a hook rather than a direct ScidacWriter call because of Grid's include order:
   * this header is reached through GridCore.h -> Algorithms.h, which is parsed BEFORE
   * GridQCDcore.h and the parallelIO/IldgIO.h that defines ScidacWriter.  ScidacWriter is
   * a non-dependent name, so referring to it here would be looked up at template
   * definition time and fail to compile.  The IO layer sits above the algorithms layer
   * and cannot be pulled downwards.
   *
   * The drivers already have exactly the right function; in Example_pvdagm_fov.cc,
   *
   *     FoV.VectorWriter = [](LatticeFermionD &v, const std::string &f) {
   *       writeFile(v, f);
   *     };
   *
   * The Field is taken by NON-const reference because ScidacWriter::writeScidacFieldRecord
   * does (IldgIO.h:520).
   */
  std::function<void(Field &, const std::string &)> VectorWriter;

  // Diagnostics from the last operator() / SupportCheck call.
  int   nFail;
  int   nViolate;
  RealD worstSlack;

  // Captured from src in operator(), so Write() can restrict itself to the boss rank.
  GridBase *_grid;

  /**
   * Defaults reproduce the settings Example_pvdagm_fov.cc ran the 16^3x32 sweep with.
   *
   * thetaMaxDeg is in DEGREES and is converted here; every other angle in this class is
   * in radians.  Nangle should be ODD so that theta = 0 is sampled and lambda_min of the
   * Hermitian part comes out directly rather than by interpolation.
   */
  FieldOfValues(RealD thetaMaxDeg = 30.0,
                int   _Nangle     = 17,
                int   _Nstop      = 2,
                int   _Nk         = 4,
                int   _Nm         = 48,
                RealD _eresid     = 1.0e-4,
                int   _MaxIter    = 500,
                int   _chebyOrder = 0,
                RealD _chebyLo    = 0.1,
                RealD _chebyHi    = 0.0)
    : Nangle(_Nangle), thetaMax(thetaMaxDeg * M_PI / 180.0),
      Nstop(_Nstop), Nk(_Nk), Nm(_Nm), eresid(_eresid), MaxIter(_MaxIter),
      chebyOrder(_chebyOrder), chebyLo(_chebyLo), chebyHi(_chebyHi),
      checkTol(1.0e-4), KeepVectors(false), vecs(), Provenance(), VectorWriter(),
      nFail(0), nViolate(0), worstSlack(0.0), _grid(nullptr)
  {
    assert(thetaMaxDeg > 0.0 && thetaMaxDeg <= 180.0 && "require 0 < thetaMaxDeg <= 180");
    assert(Nangle >= 1 && "require at least one angle");
    assert(Nstop <= Nk && Nk < Nm && "require Nstop <= Nk < Nm");

    // Grid stores `order` coefficients with Coeffs[order-1] = 1 (Chebyshev.h:94), so the
    // polynomial is T_{order-1} and only an EVEN degree is positive below the window.  An
    // odd degree is negative there, IRL targets away from the modes the filter selected,
    // and it never converges.  Forced rather than asserted: the failure mode is silent
    // non-convergence, which is far harder to diagnose than a nudge here.
    if (chebyOrder > 0 && chebyOrder % 2 == 0) {
      chebyOrder++;
      std::cout << GridLogMessage << "FieldOfValues: chebyOrder forced odd -> "
                << chebyOrder << std::endl;
    }
  }

  /** Supporting direction i, radians. Nangle == 1 degenerates to theta = 0. */
  RealD Theta(int i) const {
    if (Nangle == 1) return 0.0;
    return -thetaMax + 2.0*thetaMax*RealD(i)/RealD(Nangle-1);
  }

  /**
   * Algebraically smallest eigenvalue of H_theta, and its eigenvector.
   *
   * PolyOp selects which end IRL targets; HermOp is what the tester measures.
   *
   *   chebyOrder == 0 : PolyOp = -H_theta.  IRL targets the algebraically largest, so
   *                     negating points it at the bottom.  Slow on a clustered bottom --
   *                     this is the unaccelerated reference path.
   *   chebyOrder >  0 : PolyOp = T_{chebyOrder-1} on [chebyLo, chebyHi] applied to
   *                     H_theta.  Below chebyLo the argument passes -1 and |T| grows like
   *                     cosh(m acosh|y|).  An even degree m is positive there, so the
   *                     filter targets the bottom by itself and no negation is used.
   *
   * HermOp is plain H_theta in both cases, so eval[] comes back as eigenvalues of H_theta
   * with the natural sign, and evalMaxApprox (ImplicitlyRestartedLanczos.h:245, built from
   * HermOp) stays rho(H_theta) instead of scaling with the filter's large gain -- eresid
   * keeps the same meaning on both paths.
   *
   * Krylov spaces are invariant under negation, K(-H,v) = K(H,v), so on the unfiltered
   * path the sign changes only which Ritz values the restart shifts discard.
   *
   * The minimum is located by an explicit argmin over the converged values rather than by
   * assuming eval[0] is extremal, so this does not depend on IRL's internal sort order
   * surviving the final deflation.
   *
   * WARNING: IRL calls abort() rather than returning if it exhausts MaxIter without
   * converging (ImplicitlyRestartedLanczos.h:406-407), so the Nconv guard below is
   * defensive only and cannot in practice be reached.  The job simply dies.  The only
   * defence is parameters that actually converge.
   */
  RealD LambdaMin(LinearOperatorBase<Field> &Fop, RealD theta, const Field &src,
                  Field &vmin, int &Nconv)
  {
    HermitianPartLinearOperator<Field> Htheta(Fop, theta);          //  H_theta
    HermitianPartLinearOperator<Field> Hflip (Fop, theta + M_PI);   // -H_theta

    PlainHermOp<Field> hermop   (Htheta);   // tester: reports eigenvalues of H_theta
    PlainHermOp<Field> plainpoly(Hflip);    // unfiltered PolyOp

    // Window upper edge.  chebyHi <= chebyLo requests auto, which is 1.1*lambda_max
    // measured PER ANGLE by a power iteration on H_theta.
    //
    // This is deliberately not the angle-independent sigma_max(F) bound: ||H_theta|| <=
    // sigma_max(F) holds for every theta and is rigorous, but for a strongly non-normal
    // operator it is so much larger than the actual lambda_max(H_theta) that the window
    // swamps the spectrum and the filter is left nearly inert.
    //
    // H_theta is INDEFINITE, so the power iteration converges to the eigenvalue of
    // largest MODULUS, which is lambda_min when |lambda_min| > |lambda_max|.  Taking the
    // absolute value keeps the result a valid upper edge either way, since
    // |lambda_dominant| >= |lambda_max| >= lambda_max.
    //
    // Only evaluated when the filter is actually on: when chebyOrder == 0 the window is
    // never used and the ~200 extra operator applications would be pure waste.
    RealD hi = chebyHi;
    if (chebyOrder > 0 && hi <= chebyLo) {
      PowerMethod<Field> PM;
      hi = 1.1 * std::abs(PM(Htheta, src));
      std::cout << GridLogMessage << "FieldOfValues: theta = " << theta
                << ", auto Chebyshev window [" << chebyLo << ", " << hi << "]" << std::endl;
    }
    if (hi <= chebyLo) hi = chebyLo + 1.0;   // filter disabled; Chebyshev still needs hi > lo

    // Constructed unconditionally: it holds only coefficients, no fields.
    Chebyshev<Field>      Cheby(chebyLo, hi, chebyOrder > 2 ? chebyOrder : 3);
    FunctionHermOp<Field> chebypoly(Cheby, Htheta);

    LinearFunction<Field> &polyop = (chebyOrder > 0)
        ? static_cast<LinearFunction<Field>&>(chebypoly)
        : static_cast<LinearFunction<Field>&>(plainpoly);

    ImplicitlyRestartedLanczos<Field> IRL(polyop, hermop, Nstop, Nk, Nm, eresid, MaxIter);

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

    // The tester overwrites eval[] with Rayleigh quotients of H_theta while IRL's sort
    // order follows PolyOp, so an explicit argmin is right whatever ordering survives.
    int imin = 0;
    for (int i = 1; i < Nconv; i++) {
      if (eval[i] < eval[imin]) imin = i;
    }
    vmin = evec[imin];

    // The filter only amplifies what lies BELOW chebyLo.  If the mode we landed on sits
    // inside the window, nothing was amplified and IRL locked onto an essentially
    // arbitrary Ritz value -- a wrong answer, not a slow one.
    if (chebyOrder > 0 && eval[imin] >= chebyLo) {
      std::cout << GridLogError
                << "CHECK FAILED: at theta = " << theta << " the minimum eigenvalue "
                << eval[imin] << " is not below the Chebyshev window lower edge "
                << chebyLo << ". The filter did not bracket the bottom of the spectrum; "
                << "this value is untrustworthy. Lower chebyLo." << std::endl;
    }

    return eval[imin];        // already lambda_min(H_theta): no sign flip
  }

  /**
   * Sweeps the arc and returns one FieldOfValuesPoint per angle.
   *
   * `src` is the Lanczos start vector, reused at every angle.  Sets nFail; call
   * SupportCheck on the result for the cross-angle test.
   */
  std::vector<FieldOfValuesPoint> operator()(LinearOperatorBase<Field> &Fop, const Field &src)
  {
    std::vector<FieldOfValuesPoint> pts(Nangle);

    _grid = src.Grid();
    nFail = 0;
    if (KeepVectors) { vecs.clear(); vecs.reserve(Nangle); }

    if (Nangle % 2 == 0) {
      std::cout << GridLogMessage
                << "FieldOfValues: WARNING Nangle is even, so theta = 0 is not sampled "
                << "and lambda_min of the Hermitian part is not obtained directly."
                << std::endl;
    }

    RealD t0 = usecond();
    for (int i = 0; i < Nangle; i++) {

      FieldOfValuesPoint &p = pts[i];
      p.theta = Theta(i);
      p.ok    = 1;

      std::cout << GridLogMessage << "FieldOfValues: angle " << i << " of " << Nangle
                << ", theta = " << p.theta << " rad ("
                << p.theta*180.0/M_PI << " deg)" << std::endl;

      Field vmin(src.Grid());
      p.g = LambdaMin(Fop, p.theta, src, vmin, p.Nconv);

      if (p.Nconv < 1) {
        p.ok = 0; nFail++;
        p.z = ComplexD(0.0,0.0); p.normFv = 0.0;
        if (KeepVectors) vecs.push_back(vmin);
        continue;
      }

      // z = <v|F|v>/<v|v> is a genuine point of W(F) whatever IRL's convergence did.
      Field Fv(src.Grid());
      Fop.Op(vmin, Fv);
      p.z      = innerProduct(vmin, Fv) / RealD(norm2(vmin));
      p.normFv = std::sqrt(norm2(Fv) / norm2(vmin));

      // Re(e^{-i theta} z) is the Rayleigh quotient of H_theta in the state v, so it must
      // equal lambda_min(H_theta) if v really is the minimiser.
      RealD rq  = real(p.z)*std::cos(p.theta) + imag(p.z)*std::sin(p.theta);
      RealD tol = checkTol * (std::abs(p.g) + 1.0);
      if (std::abs(rq - p.g) > tol) {
        p.ok = 0; nFail++;
        std::cout << GridLogError
                  << "CHECK FAILED: at theta = " << p.theta << " the eigenvector's Rayleigh "
                  << "quotient " << rq << " disagrees with the reported lambda_min "
                  << p.g << ". The eigenpair is not converged." << std::endl;
      }

      if (KeepVectors) vecs.push_back(vmin);

      std::cout << GridLogMessage
                << "FieldOfValues: theta = " << p.theta
                << "  lambda_min(H_theta) = " << p.g
                << "  z = " << p.z
                << "  ||Fv|| = " << p.normFv
                << "  (Nconv = " << p.Nconv << ")" << std::endl;
    }
    RealD t1 = usecond();
    std::cout << GridLogMessage << "FieldOfValues: sweep took "
              << (t1-t0)/1.0e6 << " s" << std::endl;

    return pts;
  }

  /**
   * Pairwise support check.  Each z(theta) is an actual element of W(F), and every
   * supporting line must lie weakly to its left:
   *
   *     g(theta') <= Re( e^{-i theta'} z(theta) )   for all theta, theta'.
   *
   * This is a rigorous consequence of g being an infimum over all unit vectors, so a
   * violation proves that the angle theta' under-converged.  It costs no operator
   * applications.
   *
   * The i == j pairs are EXCLUDED.  There the inequality is an equality by construction
   * -- it is the same comparison LambdaMin already made against its own eigenvector -- so
   * including them pins worstSlack at machine epsilon and hides the cross-angle number,
   * which is the only one carrying information.
   *
   * Returns the number of violations and sets nViolate, worstSlack.
   */
  int SupportCheck(const std::vector<FieldOfValuesPoint> &pts)
  {
    nViolate   = 0;
    worstSlack = 0.0;
    int npair  = 0;

    for (int i = 0; i < (int)pts.size(); i++) {
      if (!pts[i].ok) continue;
      for (int j = 0; j < (int)pts.size(); j++) {
        if (i == j || !pts[j].ok) continue;
        npair++;
        RealD proj  = real(pts[i].z)*std::cos(pts[j].theta)
                    + imag(pts[i].z)*std::sin(pts[j].theta);
        RealD slack = proj - pts[j].g;                    // must be >= 0
        RealD tol   = checkTol * (std::abs(pts[j].g) + 1.0);
        if (slack < worstSlack) worstSlack = slack;
        if (slack < -tol) {
          nViolate++;
          std::cout << GridLogError
                    << "CHECK FAILED: supporting line at theta = " << pts[j].theta
                    << " (g = " << pts[j].g << ") cuts off the point z = " << pts[i].z
                    << " found at theta = " << pts[i].theta << " by " << -slack
                    << ". Angle " << j << " is under-converged." << std::endl;
        }
      }
    }

    std::cout << GridLogMessage << "FieldOfValues: support check, worst slack = "
              << worstSlack << " over " << npair << " ordered pairs with i != j"
              << std::endl;
    return nViolate;
  }

  /**
   * Writes the sweep to outDir/fov_left.txt, one line per angle:
   *
   *     idx  theta  g(theta)  Re_z  Im_z  normFv  Nconv  ok
   *
   * The `#` header carries the generic description, whatever the driver put in
   * Provenance, the arc and IRL parameters, and the consistency counters.  Call
   * SupportCheck first if the support-check line is to mean anything: nViolate and
   * worstSlack are read from the last call, and are zero if none was made.
   *
   * `nvecs` optionally lists ANGLE INDICES whose minimising eigenvector should also be
   * written, as outDir/fov${idx} in SCIDAC format, labelled by the same index that appears
   * in the first column of fov_left.txt.  Empty (the default) writes no vectors.  This
   * requires KeepVectors to have been set before operator() ran, and a VectorWriter; both
   * are diagnosed rather than assumed.
   *
   * The files are `fov${idx}`, NOT `evec${idx}`: these are eigenvectors of H_theta, not
   * Ritz vectors of F.  For a non-Hermitian F the two are quite different objects and the
   * project reserves `evec` for the latter.
   *
   * Vectors are how a point of W(F) becomes usable rather than merely plottable -- e.g.
   * building a unit vector whose Rayleigh quotient is exactly zero, which stagnates the
   * first step of GMRES/GCR, needs the vectors themselves and not the z values.
   */
  void Write(const std::vector<FieldOfValuesPoint> &pts, std::string outDir,
             const std::vector<int> &nvecs = std::vector<int>())
  {
    int N = (int)pts.size();

    // ScidacWriter is collective and does its own boss handling; the text file is not,
    // so without this guard every rank opens and writes the same path.
    bool boss = (_grid == nullptr) || _grid->IsBoss();

    if (boss) {
      std::string fovPath = outDir + "/fov_left.txt";
      std::cout << GridLogMessage << "FieldOfValues: writes to " << fovPath << std::endl;

      std::ofstream fFov;
      fFov.open(fovPath);
      fFov << "# Left boundary of the field of values W(F) = { <v|F|v> : |v| = 1 }.\n"
           << "# g(theta) = lambda_min(H_theta) = min_{|v|=1} Re( e^{-i theta} <v|F|v> ), so\n"
           << "# W(F) lies in every half plane Re( e^{-i theta} z ) >= g(theta).\n"
           << "# (Re_z, Im_z) is <v|F|v> for the minimising eigenvector: an exact point of\n"
           << "# W(F), trustworthy whatever the eigensolve did. g(theta) is not.\n"
           << "# normFv = ||F v||, so one step of GMRES/GCR from r_0 = v reduces the\n"
           << "# residual by ||r_1||^2/||r_0||^2 = 1 - |z|^2/normFv^2.\n";
      fFov << Provenance;
      fFov << "# arc: theta in [" << -thetaMax*180.0/M_PI << ", " << thetaMax*180.0/M_PI
           << "] degrees over " << N << " points, probing W(F) from "
           << 180.0 - thetaMax*180.0/M_PI << " to " << 180.0 + thetaMax*180.0/M_PI
           << " degrees\n"
           << "# IRL Nstop = " << Nstop << ", Nk = " << Nk << ", Nm = " << Nm
           << ", eresid = " << eresid << ", MaxIter = " << MaxIter << "\n";
      if (chebyOrder > 0) {
        fFov << "# Chebyshev T_" << chebyOrder-1 << " on [" << chebyLo << ", ";
        if (chebyHi > chebyLo) { fFov << chebyHi << "]\n"; }
        else { fFov << "per-angle 1.1*|lambda_max(H_theta)|]\n"; }
      } else {
        fFov << "# Chebyshev filter disabled\n";
      }
      fFov << "# consistency: " << nFail << " per-angle failures, "
           << nViolate << " support violations, worst slack " << worstSlack << "\n"
           << "# idx  theta  g(theta)  Re_z  Im_z  normFv  Nconv  ok\n";
      for (int i = 0; i < N; i++) {
        fFov << i << " " << pts[i].theta << " " << pts[i].g << " "
             << real(pts[i].z) << " " << imag(pts[i].z) << " "
             << pts[i].normFv << " "
             << pts[i].Nconv << " " << pts[i].ok << "\n";
      }
      if (N % 2 == 1) {
        fFov << "# lambda_min((F + Fdag)/2) = " << pts[N/2].g << "\n";
        fFov << "# lambda_min(F + Fdag) = " << 2.0*pts[N/2].g << "\n";
      }
      fFov.close();
    }

    if (nvecs.empty()) return;

    if ((int)vecs.size() < N) {
      std::cout << GridLogError
                << "FieldOfValues: cannot write eigenvectors, only " << vecs.size()
                << " are held. Set KeepVectors = true BEFORE calling operator()."
                << std::endl;
      return;
    }

    if (!VectorWriter) {
      std::cout << GridLogError
                << "FieldOfValues: eigenvector output requested but VectorWriter is not "
                << "set, so there is no way to write a field from here. See the comment "
                << "on VectorWriter." << std::endl;
      return;
    }

    for (int n = 0; n < (int)nvecs.size(); n++) {
      int idx = nvecs[n];
      if (idx < 0 || idx >= N) {
        std::cout << GridLogError
                  << "FieldOfValues: eigenvector index " << idx << " is out of range [0, "
                  << N-1 << "]; skipped." << std::endl;
        continue;
      }
      std::string fName = outDir + "/fov" + std::to_string(idx);
      std::cout << GridLogMessage << "FieldOfValues: writes eigenvector for angle "
                << idx << " (theta = " << pts[idx].theta << ") to " << fName << std::endl;
      VectorWriter(vecs[idx], fName);
    }
  }
};

/**
 * Reads back the sweep written by FieldOfValues::Write.
 *
 * Takes the SAME outDir that was passed to Write and parses outDir/fov_left.txt.  All `#`
 * lines are skipped, including the provenance block, so nothing about the operator comes
 * back -- only the eight data columns.  Ordered by angle index, so the returned vector
 * lines up with the `fov${idx}` field files.
 *
 * Not a member of FieldOfValues: reading a completed sweep has nothing to do with holding
 * the parameters that produced one, and the consumer is typically a different driver
 * doing something with the points rather than re-running the sweep.
 */
inline std::vector<FieldOfValuesPoint> ReadFieldOfValues(std::string outDir)
{
  std::string fovPath = outDir + "/fov_left.txt";
  std::vector<FieldOfValuesPoint> pts;

  std::ifstream fFov(fovPath);
  if (!fFov.is_open()) {
    std::cout << GridLogError << "ReadFieldOfValues: cannot open " << fovPath << std::endl;
    return pts;
  }

  std::string line;
  int lineno = 0;
  while (std::getline(fFov, line)) {
    lineno++;
    if (line.empty() || line[0] == '#') continue;

    FieldOfValuesPoint p;
    int   idx;
    RealD re, im;
    std::istringstream ss(line);
    if (!(ss >> idx >> p.theta >> p.g >> re >> im >> p.normFv >> p.Nconv >> p.ok)) {
      // The pre-2026-08-17 format had no normFv column, so a seven-field line parses
      // partially and would otherwise be returned with garbage in the tail.
      std::cout << GridLogError
                << "ReadFieldOfValues: " << fovPath << " line " << lineno
                << " does not have the expected 8 columns "
                << "(idx theta g Re_z Im_z normFv Nconv ok). Older files written before "
                << "normFv was added have 7 and must be re-read by hand." << std::endl;
      return std::vector<FieldOfValuesPoint>();
    }
    p.z = ComplexD(re, im);

    if (idx != (int)pts.size()) {
      std::cout << GridLogError
                << "ReadFieldOfValues: " << fovPath << " line " << lineno
                << " has index " << idx << " where " << pts.size()
                << " was expected; the file is out of order or has a gap." << std::endl;
      return std::vector<FieldOfValuesPoint>();
    }
    pts.push_back(p);
  }

  std::cout << GridLogMessage << "ReadFieldOfValues: read " << pts.size()
            << " angles from " << fovPath << std::endl;
  return pts;
}

/**
 * Reads the minimising eigenvectors written by FieldOfValues::Write, i.e. outDir/fov${idx}
 * for each idx in `nvecs`, appending them to `vecs` in the order given.
 *
 * `grid` is the Grid to construct each field on; `reader` is the driver's SCIDAC read
 * helper, e.g.
 *
 *     ReadFieldOfValuesVectors(dir, {0, 8, 16}, vecs, FGrid,
 *         [](LatticeFermionD &v, const std::string &f) { readFile(v, f); });
 *
 * `reader` is a template parameter rather than a std::function so that the call is a
 * dependent expression: this header is parsed before parallelIO/IldgIO.h, so ScidacReader
 * cannot be named here (see FieldOfValues::VectorWriter for the same problem on the write
 * side).  Passing it in is not ceremony -- it is the only way the algorithms layer can
 * reach the IO layer.
 *
 * These are eigenvectors of H_theta.  They are NOT Ritz vectors of F, and F does not act
 * diagonally on them; what each one gives is an exact point of W(F), <v|F|v>, which is
 * already in the text file.  The reason to want the vector itself is to build something
 * out of it -- most immediately a unit vector whose Rayleigh quotient is exactly zero,
 * which stagnates the first step of GMRES/GCR and lives in the span of a few of these.
 */
template<class Field, class Reader>
void ReadFieldOfValuesVectors(std::string outDir, const std::vector<int> &nvecs,
                              std::vector<Field> &vecs, GridBase *grid, Reader reader)
{
  for (int n = 0; n < (int)nvecs.size(); n++) {
    std::string fName = outDir + "/fov" + std::to_string(nvecs[n]);
    std::cout << GridLogMessage << "ReadFieldOfValuesVectors: reads " << fName << std::endl;
    Field v(grid);
    reader(v, fName);
    vecs.push_back(v);
  }
}

NAMESPACE_END(Grid);
#endif
