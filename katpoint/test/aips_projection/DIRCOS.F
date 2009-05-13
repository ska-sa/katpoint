      SUBROUTINE DIRCOS (TYPE, RA0, DEC0, RA, DEC, L, M, IERR)
C-----------------------------------------------------------------------
C! determines direction cosines between ref position and test position
C# Coordinates
C-----------------------------------------------------------------------
C;  Copyright (C) 1995, 1999, 2002
C;  Associated Universities, Inc. Washington DC, USA.
C;
C;  This program is free software; you can redistribute it and/or
C;  modify it under the terms of the GNU General Public License as
C;  published by the Free Software Foundation; either version 2 of
C;  the License, or (at your option) any later version.
C;
C;  This program is distributed in the hope that it will be useful,
C;  but WITHOUT ANY WARRANTY; without even the implied warranty of
C;  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
C;  GNU General Public License for more details.
C;
C;  You should have received a copy of the GNU General Public
C;  License along with this program; if not, write to the Free
C;  Software Foundation, Inc., 675 Massachusetts Ave, Cambridge,
C;  MA 02139, USA.
C;
C;  Correspondence concerning AIPS should be addressed as follows:
C;         Internet email: aipsmail@nrao.edu.
C;         Postal address: AIPS Project Office
C;                         National Radio Astronomy Observatory
C;                         520 Edgemont Road
C;                         Charlottesville, VA 22903-2475 USA
C-----------------------------------------------------------------------
C   DIRCOS determines the direction cosines (L,M) corresponding to
C   the separation from (RA0,DEC0), the reference pixel, to
C   (RA,DEC).  The direction cosine L is assumed to be positive to the
C   east; M is positive to the north.  Several projective geometries
C   are supported.
C   Inputs:
C      TYPE   I    Type of projection: 2 => SIN, 3 => TAN, 4 => Arc,
C                  5 => WSRT (rel north pole), 6 Global sinusoidal,
C                  7 Mercator, 8 Aitoff
C      RA0    D    Coordinate reference right ascension/longitude
C      DEC0   D    Coordinate reference declination/latitude
C      RA     D    right ascension/longitude of point
C      DEC    D    declination/latitude of point
C   Outputs:
C      L      D    Cosine angle of displacement to east
C      M      D    Cosine angle of displacement to north
C      IERR   I    0 ok, 1 out of range, 2 bad type, 3 undefined
C   ALL ANGLES ARE IN RADIANS.
C-----------------------------------------------------------------------
      INTEGER   IERR, TYPE
      DOUBLE PRECISION    RA0, DEC0, RA, DEC, L, M
C
      DOUBLE PRECISION    DEPS, COSS, SINS, DT, DA, DD, SINT
      INCLUDE 'DMSG.INC'
      INCLUDE 'DLOC.INC'
      INCLUDE 'PSTD.INC'
      DATA DEPS /1.0D-5/
Cf2py intent(in) type
Cf2py intent(in) ra0
Cf2py intent(in) dec0
Cf2py intent(in) ra
Cf2py intent(in) dec
Cf2py intent(out) l
Cf2py intent(out) m
Cf2py intent(out) ierr
C-----------------------------------------------------------------------
C                                       Use full accuracy
 10   IERR = 2
      IF ((TYPE.LT.2) .OR. (TYPE.GT.9)) GO TO 999
      IF (ABS(DEC).GT.PI/2.0D0) GO TO 990
      IF (ABS(DEC0).GT.PI/2.0D0) GO TO 990
      IERR = 0
      COSS = COS (DEC)
      SINS = SIN (DEC)
      DT = RA - RA0
      IF (DT.GT.PI) DT = DT - TWOPI
      IF (DT.LT.-PI) DT = DT + TWOPI
      L = SIN(DT) * COSS
      SINT = SINS * SIN(DEC0) + COSS * COS(DEC0) * COS(DT)
      GO TO (20, 20, 30, 40, 50, 60, 70, 80, 90), TYPE
C                                       SIN projection
 20   CONTINUE
         IF (SINT.LT.0.0D0) IERR = 1
         M = SINS * COS(DEC0) - COSS * SIN(DEC0) * COS(DT)
         GO TO 999
C                                       TAN projection
 30   CONTINUE
         IF (SINT.LE.0.0D0) THEN
            IERR = 1
         ELSE
            M = SINS * SIN(DEC0) + COSS * COS(DEC0) * COS(DT)
            L = L / M
            M = (SINS * COS(DEC0) - COSS * SIN(DEC0) * COS(DT)) / M
            END IF
         GO TO 999
C                                       Arc projection
 40   CONTINUE
         M = SINS * SIN(DEC0) + COSS * COS(DEC0) * COS(DT)
         M =  MIN (1.0D0, MAX (-1.0D0, M))
         M = ACOS (M)
         IF (M.NE.0.0D0) THEN
            M = M / SIN(M)
         ELSE
            M = 1.0D0
            END IF
         L = L * M
         M = (SINS * COS(DEC0) - COSS * SIN(DEC0) * COS(DT)) * M
         GO TO 999
C                                       WSRT projection
 50   CONTINUE
         IF (DEC0.NE.0.0D0) GO TO 55
            WRITE (MSGTXT,1050)
C            CALL MSGWRT (6)
            L = 0.0D0
            M = 0.0D0
            GO TO 990
 55      M = (COS(DEC0) - COSS * COS(DT)) / SIN(DEC0)
         GO TO 999
C                                       Global sinusoidal
 60   CONTINUE
         M = DEC - DEC0
         L = DT * COSS
         GO TO 999
C                                       Mercator
 70   CONTINUE
         L = GEOMD1(LOCNUM) * DT
         DT = DEC / 2.0D0 + PI / 4.0D0
         DT = TAN (DT)
         IF (DT.LT.DEPS) GO TO 990
         M = GEOMD2(LOCNUM) * LOG (DT) - GEOMD3(LOCNUM)
         GO TO 999
C                                       Aitoff
 80   CONTINUE
         L = 0.0D0
         M = 0.0D0
         DA = DT / 2.0D0
         DT = SQRT ((1.0D0 + COS(DEC) * COS(DA))/2.0D0)
         IF (ABS(DT).LT.DEPS) GO TO 990
         L = 2.0D0 * GEOMD1(LOCNUM) * COS(DEC) * SIN(DA) / DT
         M = GEOMD2(LOCNUM) * SIN(DEC) / DT - GEOMD3(LOCNUM)
         DA = (L/4.0D0)**2 + (M/2.0D0)**2
         IF (DA.GT.0.5000009D0) IERR = 3
         GO TO 999
C                                       Stereographic
 90   CONTINUE
         DD = 1.0D0 + SINS * SIN(DEC0) + COSS * COS(DEC0) * COS(DT)
         IF (ABS(DD).LT.DEPS) GO TO 990
         DD = 2.0D0 / DD
         L = L * DD
         M = DD * (SINS * COS(DEC0) - COSS * SIN(DEC0) * COS(DT))
         GO TO 999
C                                       Undefined
 990  IERR = 3
C
 999  RETURN
C-----------------------------------------------------------------------
 1050 FORMAT ('DIRCOS: WSRT COORDINATES CANNOT HAVE DEC0 = 0')
      END
