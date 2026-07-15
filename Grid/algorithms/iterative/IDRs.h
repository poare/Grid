/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./Grid/algorithms/iterative/IDRs.h

Copyright (C) 2015

Author: Peter Boyle <paboyle@ph.ed.ac.uk>
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
#ifndef GRID_IDRS_H
#define GRID_IDRS_H

NAMESPACE_BEGIN(Grid);

///////////////////////////////////////////////////////////////////////
// IDR(s): Induced Dimension Reduction algorithm
//
// Solves the non-Hermitian linear system D psi = src.
//
// References:
//   [1] P. Sonneveld and M. B. van Gijzen, "IDR(s): A family of
//       simple and fast algorithms for solving large nonsymmetric
//       systems of linear equations," SIAM J. Sci. Comput. 31(2),
//       pp. 1035-1062, 2008.
//   [2] M. B. van Gijzen and P. Sonneveld, "Algorithm 913: An
//       elegant IDR(s) variant that efficiently exploits
//       biorthogonality properties," ACM Trans. Math. Softw. 38(1),
//       Article 5, 2011.
//
// The core idea is to reduce the residual through a nested sequence
// of subspaces G_0 ⊇ G_1 ⊇ ... → {0}, where
//
//   G_j = (I - omega_j D)(G_{j-1} ∩ S^⊥)
//
// and S is a fixed "shadow space" of dimension s, chosen randomly.
//
// The parameter s controls the trade-off between cost and convergence:
//   s = 1  gives behaviour similar to BiCGSTAB
//   s = 4  or s = 8 are common practical choices
// Larger s converges faster per matrix-vector product at the cost of
// s extra vectors of storage.
///////////////////////////////////////////////////////////////////////

template <class Field>
class IDRs : public LinearFunction<Field> {
public:
  using LinearFunction<Field>::operator();

  RealD   Tolerance;
  Integer MaxIterations;
  Integer IterationsToComplete;   // number of outer iterations taken; set on exit
  RealD   TrueResidual;           // true residual at exit; set on exit
  int     s;                      // IDR recurrence length (must be >= 1)
  bool    ErrorOnNoConverge;      // if true, assert(0) when solver fails to converge

  LinearOperatorBase<Field> &Linop;

  IDRs(LinearOperatorBase<Field> &_Linop, int _s, RealD tol, Integer maxit, bool err_on_no_conv = true)
    : Tolerance(tol),
      MaxIterations(maxit),
      s(_s),
      ErrorOnNoConverge(err_on_no_conv),
      Linop(_Linop)
  {
    assert(s >= 1);
  };

  void operator()(const Field &src, Field &psi)
  {
    psi.Checkerboard() = src.Checkerboard();
    conformable(psi, src);

    GridBase *grid = src.Grid();

    /////////////////////////////////////////////////////////////////////
    // Small (s-dimensional) scalars and matrices, held in Eigen.
    /////////////////////////////////////////////////////////////////////
    ComplexD         omega;     // polynomial step parameter (outer step)
    Eigen::MatrixXcd M(s, s);  // M(j,k) = <p_j, g_k>, updated column-by-column
    Eigen::VectorXcd f(s);     // f(j)   = <p_j, r>, maintained throughout
    Eigen::VectorXcd c(s);     // solution of M c = f at each inner step

    /////////////////////////////////////////////////////////////////////
    // Lattice fields.
    /////////////////////////////////////////////////////////////////////
    Field r    (grid);              // residual
    Field v    (grid);              // v = r - G c (inner loop work vector)
    Field t    (grid);              // work vector: D*v (inner) or D*r (outer)
    Field u_new(grid);              // new search direction before storing into U[k]

    // Shadow space P: s fixed random vectors, orthonormalised.
    // U[k] and G[k] = D*U[k] are the k-th column of the search direction
    // matrix U and its image G; they are updated at each inner step.
    std::vector<Field> P(s, grid);
    std::vector<Field> U(s, grid);
    std::vector<Field> G(s, grid);

    /////////////////////////////////////////////////////////////////////
    // Initialise residual r = src - D psi.
    /////////////////////////////////////////////////////////////////////
    RealD ssq = norm2(src);
    if (ssq == 0.0) {
      psi = Zero();
      IterationsToComplete = 0;
      TrueResidual = 0.0;
      return;
    }

    if (norm2(psi) == 0.0) {
      r = src;
    } else {
      Linop.Op(psi, t);
      r = src - t;
    }

    RealD cp  = norm2(r);
    RealD rsq = Tolerance * Tolerance * ssq;

    std::cout << GridLogIterative << std::setprecision(8)
              << "IDR(s=" << s << "): src " << ssq
              << " initial residual " << cp
              << " target " << rsq << std::endl;

    if (cp <= rsq) {
      TrueResidual         = std::sqrt(cp / ssq);
      IterationsToComplete = 0;
      std::cout << GridLogMessage << "IDR(s=" << s << "): initial guess already converged." << std::endl;
      return;
    }

    /////////////////////////////////////////////////////////////////////
    // Initialise shadow space P with Gaussian random vectors and
    // orthonormalise via modified Gram-Schmidt.
    //
    // A fixed seed is used so that runs are reproducible; the shadow
    // space just needs to be linearly independent, not specially chosen.
    /////////////////////////////////////////////////////////////////////
    {
      GridParallelRNG RNG(grid);
      RNG.SeedFixedIntegers(std::vector<int>({45, 12, 81, 9}));
      for (int j = 0; j < s; j++) {
        gaussian(RNG, P[j]);
      }
    }
    for (int j = 0; j < s; j++) {
      for (int i = 0; i < j; i++) {
        ComplexD ip = innerProduct(P[i], P[j]);
        P[j] = P[j] - ip * P[i];
      }
      RealD nrm = std::sqrt(norm2(P[j]));
      assert(nrm > 0.0);
      P[j] = (1.0 / nrm) * P[j];
    }

    /////////////////////////////////////////////////////////////////////
    // Initialise U, G to zero; M to identity; omega to 1.
    /////////////////////////////////////////////////////////////////////
    for (int j = 0; j < s; j++) {
      U[j] = Zero();
      G[j] = Zero();
    }
    M     = Eigen::MatrixXcd::Identity(s, s);
    omega = ComplexD(1.0, 0.0);

    for (int j = 0; j < s; j++) {
      f(j) = innerProduct(P[j], r);
    }

    /////////////////////////////////////////////////////////////////////
    // Main IDR(s) iteration.
    //
    // Each outer iteration consists of:
    //   (a) s inner steps: each step requires 2 matrix-vector products
    //       (D*v to form u_k, then D*u_k to get g_k) and reduces the
    //       residual into G_{j+1} ⊂ G_j.
    //   (b) One outer polynomial step: 1 matrix-vector product (D*r),
    //       choosing omega to minimise ||r - omega*(D r)||.
    //
    // Total per outer iteration: 2s + 1 matrix-vector products.
    /////////////////////////////////////////////////////////////////////
    int  iter      = 0;
    bool converged = false;

    for (iter = 0; iter < MaxIterations && !converged; iter++) {

      /////////////////////////////////////////////////////////////////
      // (a) Inner loop: s steps.
      /////////////////////////////////////////////////////////////////
      for (int k = 0; k < s && !converged; k++) {

        // Solve the s x s system M c = f.
        // M is initialised to the identity and its k-th column is
        // overwritten at inner step k, so after k steps M has k
        // updated columns; a general LU decomposition handles this.
        c = M.lu().solve(f);

        // Form v = r - G c. This vector lies (approximately) in S^⊥,
        // i.e. <p_j, v> ≈ 0, which is the subspace the IDR
        // construction requires.
        v = r;
        for (int j = 0; j < s; j++) {
          v = v - c(j) * G[j];
        }

        // Compute the new search direction:
        //   u_k = omega * v + U c           (direction in solution space)
        //   g_k = D u_k = omega*(D v) + G c (image; built as linear combination,
        //                                    exploiting D(U c) = G c, saving a matvec)
        //
        // We compute t = D v once (the only matvec in the inner loop), then:
        //   u_new = omega*v + U*c   stored in u_new to avoid aliasing with U[k]
        //   g_new = omega*t + G*c   stored in u_new (reused) to avoid aliasing with G[k]
        Linop.Op(v, t);                          // t = D v  (only matvec this inner step)
        u_new = omega * v;
        for (int j = 0; j < s; j++) {
          u_new = u_new + c(j) * U[j];
        }
        U[k] = u_new;
        // g_k = omega*(D v) + G*c = omega*t + G*c; reuse u_new as scratch
        u_new = omega * t;
        for (int j = 0; j < s; j++) {
          u_new = u_new + c(j) * G[j];
        }
        G[k] = u_new;

        // Update the k-th column of M: M(j,k) = <p_j, G[k]>.
        for (int j = 0; j < s; j++) {
          M(j, k) = innerProduct(P[j], G[k]);
        }

        // Breakdown check: if M(k,k) is too small, beta = f(k)/M(k,k) would
        // blow up. Break to the outer polynomial step; f = P^H r is recomputed
        // there, effectively restarting the inner loop from the current residual.
        if (std::abs(M(k, k)) < 1.0e-10 * std::sqrt(norm2(G[k]))) {
          std::cout << GridLogMessage << "IDR(s=" << s << "): breakdown at outer "
                    << iter << " inner " << k
                    << " |M(k,k)| = " << std::abs(M(k, k))
                    << " — refreshing shadow space" << std::endl;
          // The shadow space P has become (nearly) orthogonal to D*G[k], so
          // biorthogonalisation has broken down. Re-draw P randomly (using a
          // seed that varies with the iteration so successive refreshes are
          // genuinely different), re-orthonormalise, and restart the IDR
          // subspace from the current residual r.
          {
            GridParallelRNG RNG(grid);
            RNG.SeedFixedIntegers(std::vector<int>({45 + iter, 12, 81, 9}));
            for (int j = 0; j < s; j++) gaussian(RNG, P[j]);
          }
          for (int j = 0; j < s; j++) {
            for (int i = 0; i < j; i++) {
              ComplexD ip = innerProduct(P[i], P[j]);
              P[j] = P[j] - ip * P[i];
            }
            RealD nrm = std::sqrt(norm2(P[j]));
            P[j] = (1.0 / nrm) * P[j];
          }
          for (int j = 0; j < s; j++) { U[j] = Zero(); G[j] = Zero(); }
          M = Eigen::MatrixXcd::Identity(s, s);
          for (int j = 0; j < s; j++) f(j) = innerProduct(P[j], r);
          break;
        }

        // Step length: beta = f_k / M_{kk}.
        // Updating psi by +beta*u_k and r by -beta*g_k maintains
        // the invariant r = src - D psi.
        ComplexD beta = f(k) / M(k, k);
        psi = psi + beta * U[k];
        r   = r   - beta * G[k];

        cp = norm2(r);
        std::cout << GridLogIterative << std::setprecision(8)
                  << "IDR(s=" << s << "): outer " << iter
                  << " inner " << k
                  << " residual " << std::sqrt(cp / ssq)
                  << " target " << Tolerance << std::endl;

        if (cp <= rsq) { converged = true; break; }

        // Update f for all j: after r → r - beta*g_k,
        //   f_new(j) = <p_j, r_new> = f(j) - beta * M(j,k)
        // All entries must be updated so the LU solve at the next inner step
        // uses the correct f = P^H r throughout the loop.
        for (int j = 0; j < s; j++) {
          f(j) = f(j) - beta * M(j, k);
        }

      } // end inner loop

      if (converged) break;

      /////////////////////////////////////////////////////////////////
      // (b) Outer (polynomial) step.
      //
      // Compute t = D r, then choose omega to minimise ||r - omega t||:
      //   omega = <t, r> / <t, t>
      // This is the same minimal-residual (BiCGSTAB-style) update used
      // to ensure the residual decreases at each outer step.
      /////////////////////////////////////////////////////////////////
      Linop.Op(r, t);                           // t = D r
      ComplexD tr = innerProduct(t, r);
      RealD    tt = norm2(t);

      if (tt == 0.0) {
        // D r = 0 implies r = 0 (for non-singular D); treat as converged.
        converged = true;
        break;
      }

      omega = tr / tt;   // complex omega: minimises ||r - omega*t||^2 over all complex omega
      psi   = psi + omega * r;
      r     = r   - omega * t;

      cp = norm2(r);
      std::cout << GridLogIterative << std::setprecision(8)
                << "IDR(s=" << s << "): outer " << iter
                << " (poly) residual " << std::sqrt(cp / ssq)
                << " target " << Tolerance << std::endl;

      if (cp <= rsq) { converged = true; break; }

      // Recompute f = P^H r from scratch for the next inner loop.
      for (int j = 0; j < s; j++) {
        f(j) = innerProduct(P[j], r);
      }

    } // end outer loop

    /////////////////////////////////////////////////////////////////////
    // Report outcome.
    /////////////////////////////////////////////////////////////////////
    if (converged) {
      Linop.Op(psi, t);
      Field err(grid);
      err          = src - t;
      TrueResidual = std::sqrt(norm2(err) / ssq);

      std::cout << GridLogMessage
                << "IDR(s=" << s << ") converged in " << iter
                << " outer iterations."
                << " Computed residual " << std::sqrt(cp / ssq)
                << " True residual " << TrueResidual
                << " Target " << Tolerance << std::endl;
      IterationsToComplete = iter;
    } else {
      std::cout << GridLogMessage
                << "IDR(s=" << s << ") did NOT converge after " << MaxIterations
                << " iterations. Residual " << std::sqrt(cp / ssq)
                << " target " << Tolerance << std::endl;
      if (ErrorOnNoConverge) assert(0);
      IterationsToComplete = MaxIterations;
    }
  }

};

NAMESPACE_END(Grid);
#endif
