*********************************************************
* Orignal BAYESPACK source code (author: Alan Genz)
*********************************************************
*
*    This file contains RANRTH and supporting functions and subroutines.
*
      SUBROUTINE RANRTH( M, NF, MXVALS, F, EPSABS, EPSREL, RS,
     &                   KEY, VALUE, ERROR, INTVLS, INFORM, WK )
****BEGIN PROLOGUE RANRTH
****AUTHOR
*            Alan Genz, Department of Mathematics, Washington State
*            University, Pullman, WA 99164-3113, USA
*              Email: AlanGenz@wsu.edu
*
*        Reference:
*            Genz, A., and Monahan, J. (1998),
*             Stochastic Integration Rules for Infinite Regions,
*             {\it SIAM J. Sci. Com.}, {\bf 19}, pp. 426--439.
*
****KEYWORDS automatic multidimensional integrator,
*            n-dimensional region ( -infin, infin )^n, Gaussian weight
****PURPOSE  The routine calculates an approximation to a given
*            vector of definite integrals
*
*
*      infin     infin
*     I    ...  I     w(X)( F ,F ,...,F   ) DX(M)...DX(2)DX(1),
*     -infin    -infin       1  2      NF
*
*       where F = F ( X ,X ,...,X  ), I = 1,2,...,NF,
*              I   I   1  2      M
*
*       w(X) = (2PI)^(-M/2)EXP(-( X(1)**2 + ... + X(M)**2 )/2),
*
*            hopefully satisfying for K = 1, 2, ... NF
*            ABS( I(K)-VALUE(K) ) .LE. MAX( EPSABS, EPSREL*ABS(I(K)) )
*
****DESCRIPTION Computation of integrals over infinite regions with
*               Gaussian weight function.
*
*   ON ENTRY
*
*     M  Integer number of variables, M > 1.
*     NF Integer number of components of the integral NF >= 1.
*     MXVALS Integer maximum number of F calls.
*            When RS > 0, this is the maximum number of new F calls.
*     F Externally declared subroutine for computing all components
*            of the integrand at the given evaluation point.
*            It must have parameters ( M, X, NF, FUNS )
*            Input parameters:
*              M   Integer number of variables.
*              X   Real array of length M, the evaluation point.
*              NF Integer number of components for I.
*            Output parameter:
*              FUNS Real array of length NF, components of the integrand
*               evaluated at the point X.
*     EPSABS Real requested absolute accuracy.
*     EPSREL Real requested relative accuracy.
*     RS Integer.
*            If RS = 0, this is the first attempt to compute the integral(s).
*            If RS = 1, then a previous calculation is continued. In
*              this case, the only parameters that may be changed (with
*              respect to the previous call of the subroutine) are
*              MXVALS, EPSABS, EPSREL and KEY.
*     KEY  Integer, determines degree of integration rules.
*            If KEY = 1, a degree 1 rule is used, and a minimum of
*                20 calls of F are used.
*            If KEY = 2, a degree 3 rule is used, and a minimum of
*                6( M + 1 ) + 1 calls of F are used.
*            If KEY = 3, a degree 5 rule is used, and a minimum of
*                4( M + 1 )( M + 2 ) + 1 calls of F are used.
*            If KEY = 4, a degree 7 spherical rule is used,
*                a degree 5 radial rule is used , and a minimum of
*                4( M + 1 )( M^2 +8M + 6 )/3 + 1 calls of F are used.
*     WK   Real work array, with length at least 6NF + 2M for KEY = 1,
*          and with length at least 6NF + M( M + 2 ) for KEY > 1.
*
*   ON RETURN
*
*     VALUE Real array of length NF of approximations to the
*            components of the integral.
*     ERROR Real array of length NF of estimates of absolute accuracies.
*     INTVLS Integer number of F calls used.
*            When RS > 0, this is the number of new F calls.
*     INFORM  Integer.
*            INFORM = 0 for normal exit, when
*              ERROR(K) <= MAX( EPSABS, ABS( VALUE(K) )EPSREL ) for
*              0 < K <= NF, with <= MXVALS function values.
*            INFORM = 1 if MXVALS was too small to obtain the required
*              accuracy. In this case values of VALUE are returned
*              with estimated absolute accuracies ERROR.
*            INFORM = 2 if MXVALS was less than the minimum required for
*              the rule type specified by KEY. In this case all
*              components of VALUE are set to 0 and all components of
*              ERROR are set to -1.
*
****END PROLOGUE RANRTH
*
*   Global variables.
*
      EXTERNAL F
      INTEGER M, NF, MXVALS, RS, KEY
      INTEGER INTVLS, INFORM
      DOUBLE PRECISION EPSABS, EPSREL
      DOUBLE PRECISION VALUE(*), ERROR(*), WK(*)

*   Local variables.
*
      INTEGER I, I1, I2, I3, I4, I5, I6, I7
      DOUBLE PRECISION WEIGHT
*
*   Divide up work space and call RANBAS
*
      I1 =  1 + NF
      I2 = I1 + NF
      I3 = I2 + M
      I4 = I3 + NF
      I5 = I4 + NF
      I6 = I5 + NF
      I7 = I6 + NF
      IF ( RS .EQ. 0 ) THEN
         DO I = 1, NF
            VALUE(I) = 0
            ERROR(I) = -1
         END DO
      END IF
      CALL RANBAS( M, NF, F, MXVALS, KEY, WK, WK(I1), INTVLS,
     &             WK(I2), WK(I3), WK(I4), WK(I5), WK(I6), WK(I7) )
      IF ( INTVLS .LE. MXVALS ) THEN
         INFORM = 0
         DO I = 1, NF
            IF ( ERROR(I) .GT. 0 ) THEN
               WEIGHT = 1/( 1 + WK(NF+I)/ERROR(I)**2 )
            ELSE
               WEIGHT = 1
            END IF
            VALUE(I) = VALUE(I) + WEIGHT*( WK(I) - VALUE(I) )
            ERROR(I) = SQRT( WEIGHT*WK(NF+I) )
            IF ( ERROR(I) .GT. MAX( EPSABS, EPSREL*ABS(VALUE(I) ) ) )
     &           INFORM = 1
         END DO
      ELSE
         INTVLS = 0
         INFORM = 2
      END IF
*
****END RANRTH
*
      END
*
      SUBROUTINE RANBAS( M, NF, F, MXVALS, KEY, VALUE, ERROR,
     &                   INTVLS, X, FUNS, FUNC, FUNV, WK, V )
*
*     Stochastic Spherical Radial Rule, for
*
*                         INF
*        (2PI)^(-M/2) I  I   exp(-r^2/2) r^(M-1) G(rZ) dr dZ.
*                      U  0
*
*     U is the surface of unit M-sphere,
*     Z is an M-vector and G is an NF-vector valued function.
*     In this subroutine, F is a subroutine with calling sequence:
*               CALL F( M, X, NF, G ).
*
*        Author:
*            Alan Genz, Department of Mathematics, Washington State
*            University, Pullman, WA 99164-3113, USA
*              Email: alangenz@wsu.edu
*
*        References:
*            Genz, A., and Monahan, J. (1997),
*             A Stochastic Algorithm for High Dimensional Multiple
*             Integrals over Unbounded Regions with Gaussian Weight,
*             to appear in {\it J. Comp. Appl. Math.}.
*            Genz, A., and Monahan, J. (1998),
*             Stochastic Integration Rules for Infinite Regions,
*             {\it SIAM J. Sci. Com.}, {\bf 19}, pp. 426--439.
*
*
*   ON ENTRY
*
*     M  Integer number of variables, M > 1.
*     NF Integer number of components of the integral, NF >= 1.
*     MXVALS Integer maximum number of F calls.
*            When RS > 0, this is the maximum number of new F calls.
*     F Externally declared subroutine for computing all components
*            of the integrand at the given evaluation point.
*            It must have parameters ( M, X, NF, FUNS )
*            Input parameters:
*              M   Integer number of variables.
*              X   Real array of length M, the evaluation point.
*              NF Integer number of components for I.
*            Output parameter:
*              FUNS Real array of length NF, the components of the
*                      integrand evaluated at the point X.
*     KEY  Integer, determines degree of integration rules.
*            If KEY = 1, a degree 1 rule is used, and a minimum of
*                20 calls of F are used.
*            If KEY = 2, a degree 3 rule is used, and a minimum of
*                6( M + 1 ) + 1 calls of F are used.
*            If KEY = 3, a degree 5 rule is used, and a minimum of
*                4( M + 1 )( M + 2 ) + 1 calls of F are used.
*            If KEY = 4, a degree 7 spherical rule is used,
*                a degree 5 radial rule is used , and a minimum of
*                4( M + 1 )( M^2 +8M + 6 )/3 + 1 calls of F are used.
*     X, FUNS, FUNC, FUNV, WK and V Real work arrays, with respective
*                lengths M, NF, NF, NF, NF, and M( M + 1 ).
*
*   ON RETURN
*
*     VALUE Real array of length NF of approximations to the
*            components of the integral.
*     ERROR Real array of length NF of squares of standard errors.
*     INTVLS Integer number of F calls used.
*
      EXTERNAL F
      INTEGER M, NF, MXVALS, KEY, INTVLS, I, L, NS, NV
      DOUBLE PRECISION VALUE(*), ERROR(*)
      DOUBLE PRECISION X(*), FUNS(*), FUNC(*), FUNV(*), V(*), WK(*)
      DOUBLE PRECISION R, Q, RM, TH, WC, WV, DIFFER
      DOUBLE PRECISION NORRAN, BETRAN, GAMRAN
      IF ( KEY .EQ. 2 ) THEN
         NS = MAX( 3, ( MXVALS - 1 )/( 2*( M + 1 ) ) )
         INTVLS = 1 + NS*2*( M + 1 )
      ELSE IF ( KEY .EQ. 3 ) THEN
         NV = 2*( M + 1 )*( M + 2 )
         IF ( M .EQ. 2 ) NV = 12
         IF ( M .EQ. 3 ) NV = 28
         NS = MAX( 2, ( MXVALS - 1 )/NV )
         INTVLS = 1 + NS*NV
      ELSE IF ( KEY .EQ. 4 ) THEN
         NV = 2*( M + 1 )*( M*M +8*M + 6 )/3
         IF ( M .EQ. 2 ) NV = 36
         IF ( M .EQ. 3 ) NV = 76
         IF ( M .EQ. 4 ) NV = 140
         IF ( M .EQ. 5 ) NV = 244
         NS = MAX( 2, ( MXVALS - 1 )/NV )
         INTVLS = 1 + NS*NV
      ELSE
         NS = MAX( 10, MXVALS/2 )
         INTVLS = 2*NS
         WV = 1
      END IF
      IF ( INTVLS .LE. MXVALS ) THEN
         DO I = 1,NF
            VALUE(I) = 0
            ERROR(I) = 0
         END DO
         IF ( 2 .LE. KEY .AND. KEY .LE. 4 ) THEN
            DO I = 1, M
               X(I) = 0
            END DO
            CALL F( M, X, NF, FUNC )
         END IF
         DO L = 1, NS
            IF ( KEY .LE. 1 .OR. 5 .LE. KEY ) THEN
               DO I = 1, M
                  V(I) = NORRAN()
               END DO
               CALL RNSRUL( KEY, M, NF, F, R, V, X, WK, FUNV )
            ELSE
               RM = M + 2
               IF( KEY .EQ. 2 ) THEN
                  R = SQRT( 2*GAMRAN( RM/2 ) )
                  WV = M/R**2
                  WC = 1 - WV
               ELSE
                  TH = 3
                  TH = ASIN( BETRAN( RM, TH/2 ) )/2
                  RM = 2*M + 7
                  R = SQRT( 2*GAMRAN( RM/2 ) )
                  Q = R*COS(TH)
                  R = R*SIN(TH)
                  WC = 1 + M*( M + 2 - R**2 - Q**2 )/( R*Q )**2
                  WV = M*( M + 2 - Q**2 )/( R**2*( R**2 - Q**2 ) )
               END IF
               DO I = 1,NF
                  FUNV(I) = WC*FUNC(I)
               END DO
               CALL RNSIMP( M, V, X )
*
*              Compute integrand average
*
               CALL RNSRUL( KEY, M, NF, F, R, V, X, WK, FUNS )
               DO I = 1, NF
                  FUNV(I) = FUNV(I) + WV*FUNS(I)
               END DO
               IF ( KEY .GT. 2 ) THEN
                  WV = M*( M + 2 - R**2 )/( Q**2*( Q**2 - R**2 ) )
                  CALL RNSRUL( KEY, M, NF, F, Q, V, X, WK, FUNS )
                  DO I = 1, NF
                     FUNV(I) = FUNV(I) + WV*FUNS(I)
                  END DO
               END IF
            END IF
            DO I = 1, NF
               DIFFER = ( FUNV(I) - VALUE(I) )/L
               VALUE(I) = VALUE(I) + DIFFER
               ERROR(I) = ( L - 2 )*ERROR(I)/L + DIFFER**2
            END DO
         END DO
      END IF
      END
*
      SUBROUTINE RNSRUL( KEY, M, NF, F, R, V, X, RESR, INTV )
*
*     Spherical Radial Rule, for
*
*         I  G(Z) dZ .
*          U
*
*     U is the surface of a radius R M-sphere, Z is an M-vector,
*     G is an NF-vector valued function.
*       In this subroutine, F is a subroutine with calling sequence:
*               CALL F( M, X, NF, G ).
*     KEY is an integer rule degree parameter with 1 <= KEY <= 4.
*       A degree 2*KEY-1 stochastic spherical rule is used.
*     Output INTV is an NF-vector of integral estimates.
*     Work vectors V, X and RESR must have respective lengths at
*      least M*(M+1), M and NF.
*
      EXTERNAL F
      INTEGER KEY, M, NF, I, IS, J, K, N
      DOUBLE PRECISION R, V( M, * ), X(*), RESR(*), INTV(*)
      DOUBLE PRECISION MP, WV, WM, WC, WT, RM, RC, RT
*
*     Determine Weights
*
      MP = M + 1
      IF ( KEY .EQ. 2 ) THEN
         WV = 1/( 2*MP )
      ELSE IF ( KEY .EQ. 3 ) THEN
         WV =   ( 7 - M )*M /( 2*( M + 2 )*MP**2 )
         WM = 2*( M - 1 )**2/( M*( M + 2 )*MP**2 )
         RM = R/SQRT( 2*( MP - 2 )/M )
         IF ( M .EQ. 2 ) WV = WV + WM
         IF ( M .EQ. 3 ) WM = 2*WM
      ELSE IF ( KEY .EQ. 4 ) THEN
         WV =   1/( 36*M*MP**3*( M + 2 )*( M + 4 ) )
         WM = 144*( M - 1)**3*( 4 - M )*WV
         WC = 486*( M - 2 )**3*WV
         WT =  ( 10*M - 6 )**3*WV
         WV = M**3*( 1800 - 793*M + 9*M*M )*WV
         RM = R/SQRT( 2*( MP - 2 )/M )
         IF ( M .GT. 2 ) RC = R/SQRT( 3*( MP - 3 )/M )
         RT = R/SQRT( ( 10*MP - 16 )/M )
         IF ( M .EQ. 2 ) WV = WV + WM
         IF ( M .EQ. 3 ) WV = WV + WC
         IF ( M .EQ. 3 ) WM = 2*WM
         IF ( M .EQ. 4 ) WM = WC
         IF ( M .EQ. 5 ) WC = 2*WC
      ELSE
         WV = 1
         WV = WV/2
      ENDIF
      DO I = 1,NF
         INTV(I) = 0
      END DO
*
*     Compute integrand average
*
      DO IS = -1, 1, 2
         IF ( KEY .LE. 1 .OR. 5 .LE. KEY ) THEN
            DO I = 1, M
               X(I) = IS*V(I,1)
            END DO
            CALL F( M, X, NF, RESR )
            DO I = 1, NF
               INTV(I) = INTV(I)+ WV*RESR(I)
            END DO
         ELSE
            DO K = 1, M+1
               DO I = 1, M
                  X(I) = IS*R*V(I,K)
               END DO
               CALL F( M, X, NF, RESR )
               DO I = 1, NF
                  INTV(I) = INTV(I) + WV*RESR(I)
               END DO
            END DO
         END IF
         IF ( ( KEY .EQ. 3 .OR. KEY .EQ. 4 ) .AND.
     &        (   M .GT. 3 .OR.   M .EQ. 3 .AND. IS .EQ. 1 ) ) THEN
            DO K = 1, M
               DO J = K+1, M+1
                  DO I = 1, M
                     X(I) = IS*RM*( V(I,K) + V(I,J) )
                  END DO
                  CALL F( M, X, NF, RESR )
                  DO I = 1, NF
                     INTV(I) = INTV(I) + WM*RESR(I)
                  END DO
               END DO
            END DO
         END IF
         IF ( KEY .EQ. 4 ) THEN
            DO K = 1, M+1
               DO J = 1, M+1
                  IF ( J .NE. K ) THEN
                     DO I = 1, M
                        X(I) = IS*RT*( V(I,K) + 3*V(I,J) )
                     END DO
                     CALL F( M, X, NF, RESR )
                     DO I = 1, NF
                        INTV(I) = INTV(I) + WT*RESR(I)
                     END DO
                  END IF
               END DO
            END DO
            IF ( M .GT. 5 .OR. M .EQ. 5 .AND. IS .EQ. 1 ) THEN
               DO K = 1, M-1
                  DO J = K+1, M
                     DO N = J+1, M+1
                        DO I = 1, M
                           X(I) = IS*RC*( V(I,K) + V(I,J) + V(I,N) )
                        END DO
                        CALL F( M, X, NF, RESR )
                        DO I = 1, NF
                           INTV(I) = INTV(I) + WC*RESR(I)
                        END DO
                     END DO
                  END DO
               END DO
            END IF
         END IF
      END DO
      END
*
      SUBROUTINE RNSIMP( M, V, X )
*
*     Determine random M-simplex with vertices V and work vector X.
*
      INTEGER I, J, K, M
      DOUBLE PRECISION V( M, * ), X(*), AL, BT, RV, MP, NORRAN
      MP = M + 1
*
*     Determine standard unit simplex centered at origin
*
      DO I = 1, M
         DO J = 1, I-1
            V(I,J) = 0
         END DO
         RV = SQRT( MP/( ( M - I + 1 )*M*( M - I + 2 ) ) )
         V(I,I) = ( M - I + 1 )*RV
         DO J = I+1, M+1
            V(I,J) = -RV
         END DO
      END DO
*
*     Replace V with (random orthogonal matrix)*V
*     This generates the random orthogonal matrix using a
*     sequence of random Householder transformations.
*     Reference: Stewart, G.W. (1980),
*     "The Efficient Generation of Random Orthogonal Matrices
*      with an Application to Condition Estimaors",
*     SIAM J Numer. Anal. 17, pp. 403-408.
*
      DO K = M - 1, 1, -1
         AL = 0
         DO I = K, M
            X(I) = NORRAN()
            AL = AL + X(I)**2
         END DO
         AL = -SQRT(AL)
         BT = 1/( AL*( AL + X(K) ) )
         X(K) = X(K) + AL
         DO J = K, M+1
            AL = 0
            DO I = K, M
               AL = AL + X(I)*V(I,J)
            END DO
            AL = BT*AL
            DO I = K, M
               V(I,J) = V(I,J) - X(I)*AL
            END DO
         END DO
      END DO
      END
*
