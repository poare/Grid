/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid 

Source file: ./lib/algorithms/iterative/Arnoldi.h

Copyright (C) 2015

Author: Peter Boyle <paboyle@ph.ed.ac.uk>
Author: paboyle <paboyle@ph.ed.ac.uk>
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
#ifndef GRID_ARNOLDI_H
#define GRID_ARNOLDI_H

NAMESPACE_BEGIN(Grid); 

/**
 * Implementation of the Arnoldi algorithm.
 */
template<class Field> 
class Arnoldi {

  private:
  
    std::string cname = std::string("Arnoldi");
    int MaxIter;   // Max iterations
    RealD Tolerance;
    RealD ssq;
    RealD rtol;
    int Nm;           // Number of basis vectors to track (equals MaxIter if no restart)
    int Nk;           // Number of basis vectors to keep every restart (equals -1 if no restart)
    int Nstop;       // Stop after converging Nstop eigenvectors.

    LinearOperatorBase<Field> &Linop;
    GridBase *Grid;

    RealD approxLambdaMax;
    RealD beta_k;
    Field f;
    std::vector<Field> basis;               // orthonormal Arnoldi basis
    Eigen::MatrixXcd Hess;                  // Hessenberg matrix of size Nbasis (after construction)
    Eigen::MatrixXcd Qt;                    // Transpose of basis rotation which projects out high modes.

    Eigen::VectorXcd evals;                 // evals of Hess
    Eigen::MatrixXcd littleEvecs;           // Nm x Nm evecs matrix
    std::vector<Field> evecs;               // Vector of evec fields

    RitzFilter ritzFilter;                        // how to sort evals

  public:       

    Arnoldi(LinearOperatorBase<Field> &_Linop, GridBase *_Grid, RealD _Tolerance, RitzFilter filter = EvalReSmall)
      : Linop(_Linop), Grid(_Grid), Tolerance(_Tolerance), ritzFilter(filter), f(_Grid), MaxIter(-1), Nm(-1), Nk(-1), 
          Nstop (-1), evals (0), evecs (), ssq (0.0), rtol (0.0), beta_k (0.0), approxLambdaMax (0.0)
    {
      f = Zero();
    };

    /**
     * Runs the Arnoldi loop with(out) implicit restarting. For each iteration:
     *   - Runs an Arnoldi step.
     *   - Computes the eigensystem of the Hessenberg matrix.
     *   - Performs implicit restarting.
     */
    void operator()(const Field& v0, int _maxIter, int _Nm, int _Nk, int _Nstop, bool doubleOrthog = false) {
      MaxIter = _maxIter;
      Nm = _Nm; Nk = _Nk;
      Nstop = _Nstop;

      // BUGFIX: clear state from any previous call to operator(). Without this a second call
      // appends to the stale basis and trips the assert(start == basis.size()) in
      // arnoldiIteration.
      basis.clear();
      evecs.clear();

      ssq = norm2(v0);
      // BUGFIX: the RealD here declared a local that shadowed the class member of the same
      // name, so the member stayed at 0.0 (a latent trap for anything else using it, e.g. the
      // breakdown guard in arnoldiIteration).
      // RealD approxLambdaMax = approxMaxEval(v0);
      approxLambdaMax = approxMaxEval(v0);
      rtol = Tolerance * approxLambdaMax;

      ComplexComparator compareComplex (ritzFilter);
      std::cout << GridLogMessage << "Comparing Ritz values with: " << ritzFilter << std::endl;

      int start = 1;
      Field startVec = v0;
      littleEvecs = Eigen::MatrixXcd::Zero(Nm, Nm);
      for (int i = 0; i < MaxIter; i++) {
        std::cout << GridLogMessage << "Restart Iteration " << i << std::endl;

        // Perform Arnoldi steps to compute Krylov basis and Rayleigh quotient (Hess)
        arnoldiIteration(startVec, Nm, start, doubleOrthog);
        startVec = f;

        // compute eigensystem and sort evals
        // compute_eigensystem();
        compute_eigensystem(Hess);
        std::cout << GridLogMessage << "Eigenvalues after Arnoldi step: " << std::endl << evals << std::endl;

        // BUGFIX: sorting evals alone breaks the pairing between evals[k], littleEvecs.col(k)
        // and evecs[k] (Eigen returns eigenvalues and eigenvectors in a matched but arbitrary
        // order). The implicit-restart shifts evals[Nk..Nm-1] were still correct, but the
        // convergence test and the returned (eval, evec) pairs were mismatched. Sort an index
        // permutation and apply it to all three objects together instead.
        // std::sort(evals.begin(), evals.end(), compareComplex);
        sortRitzPairs(compareComplex);
        std::cout << GridLogMessage << "Ritz values after sorting (first Nk preserved): " << std::endl << evals << std::endl;
        // SU(N)::tepidConfiguration

        // Implicit restart to de-weight unwanted eigenvalues
        implicitRestart(_Nm, _Nk);      // probably can delete _Nm and _Nk from function args
        start = Nk;

        // check convergence and return if needed.
        int Nconv = converged();
        std::cout << GridLogMessage << "Number of evecs converged: " << Nconv << std::endl;
        if (Nconv >= Nstop || i == MaxIter - 1) {
          // BUGFIX: report failure honestly when we fall out of the loop on MaxIter rather
          // than claiming convergence.
          if (Nconv >= Nstop) {
            std::cout << GridLogMessage << "Converged with " << Nconv << " / " << Nstop << " eigenvectors on iteration "
                          << i << "." << std::endl;
          } else {
            std::cout << GridLogMessage << "NOT converged: only " << Nconv << " / " << Nstop
                          << " eigenvectors after " << MaxIter << " restart iterations." << std::endl;
          }
          // BUGFIX: evecs already holds the Ritz vectors V*S built in compute_eigensystem
          // (and sorted by sortRitzPairs). The restart rotation Qt maps the old Arnoldi basis
          // to the new one -- applying it to Ritz vectors has no mathematical meaning, and by
          // this point it would have been the second rotation applied to them. The first
          // Nstop entries of evecs/evals are the answer as-is.
          // basisRotate(evecs, Qt, 0, Nk, 0, Nk, Nm);
          std::cout << GridLogMessage << "Eigenvalues [first " << Nconv << " converged]: " << std::endl << evals << std::endl;
          return;
        }
      }      
    }

    /**
     * Approximates the maximum eigenvalue of Linop.Op to normalize the residual and test for convergence. 
     * 
     * Parameters
     * ----------
     * Field& v0
     *  Source field to start with. Must have non-zero norm.
     * int MAX_ITER (default = 50)
     *  Maximum number of iterations for power approximation. 
     * 
     * Returns
     * -------
     * RealD lamApprox
     *  Approximation of largest eigenvalue. 
     */
    RealD approxMaxEval(const Field& v0, int MAX_ITER = 50) {
      assert (norm2(v0) > 1e-8);                        // must have relatively large source norm to start
      RealD lamApprox = 0.0;
      Field v0cp (Grid); Field tmp (Grid);
      // BUGFIX: keep the power-iteration vector at unit norm. The previous version let
      // |A^n v0| grow like lambda_max^n, which overflows double precision after ~50
      // applications for operators with large spectral radius:
      // RealD denom = 1.0; RealD num = 1.0;
      // v0cp = v0;
      // denom = std::sqrt(norm2(v0cp));
      // for (int i = 0; i < MAX_ITER; i++) {
      //   Linop.Op(v0cp, tmp);                            // CAREFUL: do not do Op(tmp, tmp)
      //   v0cp = tmp;
      //   num = std::sqrt(norm2(v0cp));                   // num = |A^{n+1} v0|
      //   lamApprox = num / denom;                        // lam = |A^{n+1} v0| / |A^n v0|
      //   denom = num;                                    // denom = |A^{n} v0|
      // }
      v0cp = (1.0 / std::sqrt(norm2(v0))) * v0;            // unit-norm starting vector
      for (int i = 0; i < MAX_ITER; i++) {
        Linop.Op(v0cp, tmp);                               // CAREFUL: do not do Op(tmp, tmp)
        lamApprox = std::sqrt(norm2(tmp));                 // |A v| for unit v -> lambda_max
        v0cp = (1.0 / lamApprox) * tmp;                    // renormalize for the next application
        std::cout << GridLogDebug << "Approx for max eval: " << lamApprox << std::endl;
      }
      return lamApprox;
    }

    /**
     * Constructs the Arnoldi basis for the Krylov space K_n(D, src). (TODO make private)
     * 
     * Parameters
     * ----------
     * v0 : Field&
     *  Source to generate Krylov basis. 
     * Nm : int
     *  Final size of the basis desired. If the basis becomes complete before a basis of size Nm is constructed 
     *  (determined by relative tolerance Tolerance), stops iteration there. 
     * doubleOrthog : bool (default = false)
     *  Whether to double orthogonalize the basis (for numerical cancellations) or not. 
     * start        : int (default = 0)
     *  If non-zero, assumes part of the Arnoldi basis has already been constructed. 
     */
    void arnoldiIteration(const Field& v0, int Nm, int start = 1, bool doubleOrthog = false)
    {

      ComplexD coeff;
      Field w (Grid);           // A acting on last Krylov vector. 

      if (start == 1) {       // initialize everything that we need.
        RealD v0Norm = 1 / std::sqrt(ssq);
        basis.push_back(v0Norm * v0);                // normalized source

        Hess = Eigen::MatrixXcd::Zero(Nm, Nm);
        f = Zero();
      } else {
        assert( start == basis.size() );      // should be starting at the end of basis (start = Nk)
        Eigen::MatrixXcd HessCp = Hess;
        Hess = Eigen::MatrixXcd::Zero(Nm, Nm);
        Hess(Eigen::seqN(0, Nk), Eigen::seqN(0, Nk)) = HessCp;
      }

      // Construct next Arnoldi vector by normalizing w_i = Dv_i - \sum_j v_j h_{ji}
      for (int i = start - 1; i < Nm; i++) {

        Linop.Op(basis.back(), w);
        for (int j = 0; j < basis.size(); j++) {
          coeff = innerProduct(basis[j], w);       // coeff = h_{ij}. Note that since {vi} is ONB it's OK to subtract it off after. 
          Hess(j, i) = coeff;
          w -= coeff * basis[j];
        }

        // BUGFIX (was an empty TODO): second orthogonalization pass. A single Gram-Schmidt
        // pass loses orthogonality when the new direction is nearly contained in the span of
        // the current basis; the standard cure ("twice is enough", Kahan/Parlett) is to
        // orthogonalize a second time and accumulate the (small) corrections into the same
        // Hessenberg column.
        if (doubleOrthog) {
          for (int j = 0; j < basis.size(); j++) {
            coeff = innerProduct(basis[j], w);
            Hess(j, i) += std::complex<double>(coeff);
            w -= coeff * basis[j];
          }
        }

        // add w_i to the pile
        if (i < Nm - 1) {
          coeff = std::sqrt(norm2(w));
          // BUGFIX: guard against "happy breakdown" (||w|| ~ 0 means an exact invariant
          // subspace has been found). Dividing by ~0 below would inject a garbage vector into
          // the basis; deflating the invariant subspace is not implemented, so stop loudly
          // rather than silently corrupting the factorization. (approxLambdaMax is 0 if
          // arnoldiIteration is called outside operator(), in which case this only catches an
          // exact zero.)
          assert(abs(coeff) > 1e-14 * approxLambdaMax);
          Hess(i+1, i) = coeff;
          basis.push_back(
            (1.0/coeff) * w
          );
        }

        // after iterations, update f and beta_k = ||f||
        f = w;                                // make sure f is not normalized
        beta_k = std::sqrt(norm2(f));         // beta_k = ||f_k|| determines convergence.
      }

      std::cout << GridLogMessage << "|f|^2 after Arnoldi step = " << norm2(f) << std::endl;
      std::cout << GridLogDebug << "Computed Hessenberg matrix = " << std::endl << Hess << std::endl;

      return;
    }

    /**
     * Approximates the eigensystem of the linear operator by computing the eigensystem of 
     * the Hessenberg matrix. Assumes that the Hessenberg matrix has already been constructed (by 
     * calling the operator() function).
     * 
     * TODO implement in parent class eventually.
     * 
     * Parameters
     * ----------
     * Eigen::MatrixXcd& S
     *  Schur matrix (upper triangular) similar to original Rayleigh quotient.
     */
    void compute_eigensystem(Eigen::MatrixXcd& S)
    {

      std::cout << GridLogMessage << "Computing eigenvalues." << std::endl;

      evecs.clear();

      Eigen::ComplexEigenSolver<Eigen::MatrixXcd> es;
      es.compute(S);
      evals = es.eigenvalues();
      littleEvecs = es.eigenvectors();

      // Convert evecs to lattice fields
      for (int k = 0; k < evals.size(); k++) {
        Eigen::VectorXcd vec = littleEvecs.col(k);
        Field tmp (basis[0].Grid());
        tmp = Zero();
        for (int j = 0; j < basis.size(); j++) {
          tmp = tmp + vec[j] * basis[j];
        }
        evecs.push_back(tmp);
      }

      std::cout << GridLogMessage << "Eigenvalues: " << std::endl << evals << std::endl;

    }

    /**
     * BUGFIX (new function): sorts the Ritz pairs (evals[k], littleEvecs.col(k), evecs[k])
     * simultaneously with the given comparator, preserving the pairing between eigenvalues and
     * eigenvectors. Replaces the bare std::sort of evals in operator(), which silently
     * decoupled the eigenvalues from their eigenvectors. After this call the first Nk pairs
     * are the "wanted" ones and evals[Nk..Nm-1] are the unwanted values used as exact shifts
     * in the implicit restart.
     */
    void sortRitzPairs(ComplexComparator& compare) {
      // Sort a permutation of the indices rather than the eigenvalues themselves.
      std::vector<int> perm(evals.size());
      for (int k = 0; k < evals.size(); k++) perm[k] = k;
      std::sort(perm.begin(), perm.end(),
                [&](int a, int b) { return compare(evals[a], evals[b]); });

      // Apply the permutation to all three objects together.
      Eigen::VectorXcd sortedEvals(evals.size());
      Eigen::MatrixXcd sortedLittleEvecs(littleEvecs.rows(), littleEvecs.cols());
      std::vector<Field> sortedEvecs;
      sortedEvecs.reserve(evecs.size());
      for (int k = 0; k < evals.size(); k++) {
        sortedEvals[k] = evals[perm[k]];
        sortedLittleEvecs.col(k) = littleEvecs.col(perm[k]);
        sortedEvecs.push_back(evecs[perm[k]]);
      }
      evals = sortedEvals;
      littleEvecs = sortedLittleEvecs;
      evecs = sortedEvecs;
    }

    /**
     * Verifies the factorization DV = V^\dag H + f e^\dag with the last-computed
     * V, H, f.
     */
    // RealD verifyFactorization() {
    //   int k = basis.size();         // number of basis vectors, also the size of H.
    //   std::vector<Field> factorized (k, Zero());
    //   Field tmp (FGrid); tmp = Zero();
    //   for (int i = 0; i < basis.size(); i++) {
    //     Linop.Op(basis[i], tmp);
    //   }
    //   // basisRotate(basis, Q, 0, Nk, 0, Nk, Nm);
    //   // Linop.Op(, )
    // }

    /* Getters */
    Eigen::MatrixXcd    getHessenbergMat()  { return Hess; }
    Field               getF()              { return f; }
    std::vector<Field>  getBasis()          { return basis; }
    Eigen::VectorXcd    getEvals()          { return evals; }
    std::vector<Field>  getEvecs()          { return evecs; }

    /**
     * Implements implicit restarting for Arnoldi. Assumes eigenvalues are sorted. 
     * 
     * Parameters
     * ----------
     * int _Nm
     *  Size of basis to keep (Hessenberg is MxM).
     * int Nk
     *  Number of basis vectors to keep at each restart.
     */
    void implicitRestart(int _Nm, int _Nk) {
      assert ( _Nk <= _Nm );
      Nm = _Nm; Nk = _Nk;
      int Np = Nm - Nk;       // keep Nk smallest (or largest, depends on sort function) evecs
      
      std::cout << GridLogMessage << "Computing QR Factorizations." << std::endl;

      Eigen::MatrixXcd Q = Eigen::MatrixXcd::Identity(Nm, Nm);
      Eigen::MatrixXcd Qi (Nm, Nm);
      Eigen::MatrixXcd R (Nm, Nm);

      for (int i = Nk; i < Nm; i++) {        // keep the first Nk eigenvalues and iterate through the last Np. Should loop Np times

        // Useful debugging output
        std::cout << GridLogDebug << "Computing QR factorization for i = " << i << std::endl;
        std::cout << GridLogDebug << "Eval shift = " << evals[i] << std::endl;
        std::cout << GridLogDebug << "Hess before rotation: " << Hess << std::endl;

        // QR factorize 
        Eigen::HouseholderQR<Eigen::MatrixXcd> QR (Hess - evals[i] * Eigen::MatrixXcd::Identity(Nm, Nm));
        Qi = QR.householderQ();
        Q = Q * Qi;
        Hess = Qi.adjoint() * Hess * Qi;

        std::cout << GridLogDebug << "Qt up to i = " << Q.transpose() << std::endl;

      }

      std::cout << GridLogDebug << "Hess after all rotations: " << std::endl << Hess << std::endl; 

      // Restarted residual: f_+ = (V Q) e_{k+1} * beta + f * sigma  (Sorensen), with
      // beta = Hhat(k+1, k) and sigma = Q(m, k) in 1-based notation.
      std::complex<double> beta = Hess(Nk, Nk-1);
      std::complex<double> sigma = Q(Nm-1, Nk-1);

      // Rotate basis by Qt: basisRotate sets basis[j] <- sum_k Qt(j,k) basis[k] = (V Q)(:, j),
      // for the leading Nk+1 columns (we need column Nk for f_+ below).
      Qt = Q.transpose();
      basisRotate(basis, Qt, 0, Nk + 1, 0, Nm, Nm);

      // BUGFIX: f_+ must be built from the ROTATED vector (V Q) e_{k+1}, so this update has to
      // come after the basisRotate above. Previously it ran before the rotation and used the
      // un-rotated basis[Nk]:
      // f = basis[Nk] * beta + f * sigma;
      f = basis[Nk] * beta + f * sigma;
      // BUGFIX: removed a dead local that shadowed nothing and was never read:
      // RealD betak = std::sqrt(norm2(f));
      // Note that the member beta_k is deliberately NOT updated here: the Ritz estimates in
      // converged() pair ||f_m|| from the m-step factorization (set in arnoldiIteration) with
      // the last components littleEvecs(Nm-1, k) of the m x m Hessenberg eigenvectors -- both
      // pre-restart quantities. Overwriting beta_k with ||f_+|| would corrupt that test.
      std::cout << GridLogMessage << "|f|^2 after implicit restart = " << norm2(f) << std::endl;

      // BUGFIX: do not rotate the Ritz vectors. Q maps the old Arnoldi basis onto the new one;
      // applying it to evecs = V*S is not a meaningful operation. evecs is rebuilt from
      // scratch in compute_eigensystem on the next iteration anyway, and at convergence the
      // (sorted) evecs from compute_eigensystem are already the answer.
      // basisRotate(evecs, Qt, 0, Nk + 1, 0, Nm, Nm);

      // Truncate the basis and restart
      basis = std::vector<Field> (basis.begin(), basis.begin() + Nk);
      // evecs = std::vector<Field> (evecs.begin(), evecs.begin() + Nk);
      // BUGFIX: added .eval(). Assigning a block of Hess to itself resizes the destination
      // while the block expression still references the old storage -- undefined behaviour in
      // Eigen (may work on one platform and corrupt memory on another). .eval() forces the
      // block into a temporary before the assignment.
      // Hess = Hess(Eigen::seqN(0, Nk), Eigen::seqN(0, Nk));
      Hess = Hess(Eigen::seqN(0, Nk), Eigen::seqN(0, Nk)).eval();

      std::cout << GridLogDebug << "evecs size: " << evecs.size() << std::endl;

    }
  
    /**
     * Computes the number of Arnoldi eigenvectors that have converged. An eigenvector s is considered converged 
     * for a tolerance epsilon if 
     *    r(s) := |\beta e_m^T s| < epsilon
     * where beta is the norm of f_{m+1}.
     * 
     * Parameters
     * ----------
     * 
     * Returns
     * -------
     * int : Number of converged eigenvectors.
     */
    int converged() {
      int Nconv = 0;
      // BUGFIX: only test the wanted (leading, post-sort) Ritz pairs. Looping over all Nm
      // pairs let converged *unwanted* Ritz values -- typically exterior ones, which converge
      // fastest -- count towards the Nconv >= Nstop stopping test, so the iteration could
      // terminate before any wanted pair had actually converged.
      // for (int k = 0; k < evecs.size(); k++) {
      int Nwanted = std::min((int)evecs.size(), Nstop);
      for (int k = 0; k < Nwanted; k++) {
        RealD emTs = abs(littleEvecs(Nm - 1, k));           // e_m^T s
        RealD ritzEstimate = beta_k * emTs;                      // ||A x - theta x|| = ||f_m|| |e_m^T s|
        std::cout << GridLogMessage << "Ritz estimate for evec " << k << " = " << ritzEstimate << std::endl;
        if (ritzEstimate < rtol) {
          Nconv++;
        }
      }
      return Nconv;
    }

};
    
NAMESPACE_END(Grid);
#endif
