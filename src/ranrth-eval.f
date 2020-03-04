      SUBROUTINE RANRTHEVAL(M, MXVALS, EPSABS, EPSREL, KEY, VALUE,
     &                      ERROR, INTVLS, INFORM, WK, NF)

      EXTERNAL EVALINTEGRAND
      INTEGER M, MXVALS, KEY
      INTEGER INTVLS, INFORM
      DOUBLE PRECISION EPSABS, EPSREL
      DOUBLE PRECISION VALUE(1), ERROR(1), WK(*)

*   Local variables.
      INTEGER NF, RS

      RS = 0

      CALL RANRTH(M, NF, MXVALS, EVALINTEGRAND, EPSABS, EPSREL, RS,
     &            KEY, VALUE, ERROR, INTVLS, INFORM, WK)

      END
