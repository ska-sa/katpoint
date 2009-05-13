      SUBROUTINE NEWPOS (TYPE, RA0, DEC0, L, M, RAOUT, DECOUT, IERR)
C-----------------------------------------------------------------------
C! returns astronomical coordinates given direction cosines, projection
C# Coordinates
C-----------------------------------------------------------------------
C;  Copyright (C) 1995-1996, 1999-2000, 2002
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
C   NEWPOS determines the coordinates (RAOUT,DECOUT) corresponding to a
C   displacement (L,M) given by the direction cosines from coordinates
C   (RA0,DEC0).  The direction cosine L is assumed to be positive to
C   the east; M is positive to the north.  The routine works for
C   4 kinds of projective geometries and for Celestial, Ecliptic, or
C   Galactic coordinate systems.
C   This subroutine always uses an accurate computation.
C   Inputs:
C      TYPE  I    2 = SIN projection, 3 = TAN projection, 4 = arc
C                 projection, 5 = projection to north pole oriented
C                 plane (i.e. WSRT)
C      RA0   D    Coordinate reference right ascension (longitude)
C      DEC0  D    Coordinate reference declination (latitude)
C      L     D    Cosine angle of displacement to east
C      M     D    Cosine angle of displacement to north
C   Outputs:
C      RAOUT  D    right ascension or longitude at (L,M)
C      DECOUT D    declination or latitude at (L,M)
C      IERR   I    Error condition: 0 = ok, 1 = L,M crazy, 2 = bad
C                        type,  3 = answer undefined
C   ALL ANGLES IN THIS SUBROUTINE ARE IN RADIANS.
C-----------------------------------------------------------------------
      INTEGER   TYPE, IERR
      DOUBLE PRECISION RA0, DEC0, L, M, RAOUT, DECOUT
C
      DOUBLE PRECISION SINS, COSS, TWOPI, DECT, RAT, DT, DEPS, MG, DA,
     *   DD, DZ, COS0, SIN0
      INCLUDE 'DMSG.INC'
      INCLUDE 'DLOC.INC'
      DATA TWOPI, DEPS /6.28318530717959D0, 1.D-5/
Cf2py intent(in) type
Cf2py intent(in) ra0
Cf2py intent(in) dec0
Cf2py intent(in) l
Cf2py intent(in) m
Cf2py intent(out) raout
Cf2py intent(out) decout
Cf2py intent(out) ierr
C-----------------------------------------------------------------------
      IERR = 2
      IF ((TYPE.GE.2) .AND. (TYPE.LE.9)) GO TO 5
         WRITE (MSGTXT,1000) TYPE
C         CALL MSGWRT (6)
         GO TO 999
 5    IERR = 1
      SINS = L*L + M*M
      IF (SINS.LE.1.0D0) GO TO 10
      IF (TYPE.EQ.9) GO TO 10
      IF (TYPE.EQ.8) GO TO 10
      IF (TYPE.EQ.6) GO TO 10
      IF ((TYPE.NE.4) .OR. (SINS.GT.TWOPI*TWOPI/4.0D0)) GO TO 999
 10   IERR = 3
      DECOUT = 0.D0
      RAOUT = 0.D0
      COS0 = COS(DEC0)
      SIN0 = SIN(DEC0)
C                                       Use accurate solution
      GO TO (20, 20, 30, 40, 50, 60, 70, 80, 90), TYPE
C                                       SIN projection
 20   CONTINUE
         COSS = SQRT (1.0D0 - SINS)
         DT = SIN0 * COSS + COS0 * M
         IF (ABS(DT).GT.1.0D0) GO TO 999
         DECT = ASIN (DT)
         RAT = COS0 * COSS - SIN0 * M
         IF ((RAT.EQ.0.D0) .AND. (L.EQ.0.0D0)) GO TO 999
         RAT = ATAN2 (L, RAT) + RA0
         GO TO 900
C                                       TAN projection
 30   CONTINUE
         DECT = COS0 - M * SIN0
         IF (DECT.EQ.0.0D0) GO TO 999
         RAT = RA0 + ATAN2 (L, DECT)
         DECT = ATAN (COS(RAT-RA0) * (M * COS0 + SIN0) / DECT)
         GO TO 900
C                                       Arc projection
 40   CONTINUE
         SINS = SQRT(SINS)
         COSS = COS (SINS)
         IF (SINS.NE.0.0D0) THEN
            SINS = SIN (SINS) / SINS
         ELSE
            SINS = 1.0D0
            END IF
         DT = M * COS0 * SINS + SIN0 * COSS
         IF (ABS(DT).GT.1.0D0) GO TO 999
         DECT = ASIN (DT)
         DA = COSS - DT * SIN0
         DT = L * SINS * COS0
         IF ((DA.EQ.0.0D0) .AND. (DT.EQ.0.0D0)) GO TO 999
         RAT = RA0 + ATAN2 (DT, DA)
         GO TO 900
C                                       WSRT (North pole projection)
 50   CONTINUE
         DECT = COS0 - M * SIN0
         IF (DECT.EQ.0.0D0) GO TO 999
         RAT = RA0 + ATAN2 (L, DECT)
         DT = COS (RAT-RA0)
         IF (DT.EQ.0.0D0) GO TO 999
         DECT = DECT / DT
         IF (ABS(DECT).GT.1.0D0) GO TO 999
         DECT = ACOS (DECT)
         IF (DEC0.LT.0.0D0) DECT = -DECT
         GO TO 900
C                                       Global sinusoid
 60   CONTINUE
         DECT = DEC0 + M
         IF (ABS(DECT).GT.TWOPI/4.0D0) GO TO 999
         COSS = COS (DECT)
         IF (ABS(L).GT.TWOPI*COSS/2.0D0) GO TO 999
         RAT = RA0
         IF (COSS.GT.DEPS) RAT = RAT + L / COSS
         GO TO 900
C                                       Mercator
 70   CONTINUE
         RAT = L / GEOMD1(LOCNUM) + RA0
         IF (ABS(RAT-RA0).GT.TWOPI) GO TO 999
         DT = 0.0D0
         IF (GEOMD2(LOCNUM).NE.0.0D0) DT = (M + GEOMD3(LOCNUM)) /
     *      GEOMD2(LOCNUM)
         DT = EXP (DT)
         DECT = 2.0D0 * ATAN (DT) - TWOPI / 4.0D0
         GO TO 900
C                                       Aitoff
 80   CONTINUE
         RAT = RA0
         DECT = DEC0
         IF ((L.EQ.0.0D0) .AND. (M.EQ.0.0D0)) GO TO 900
         DZ = 4.0D0 - L*L/(4.0D0*GEOMD1(LOCNUM)*GEOMD1(LOCNUM)) -
     *      ((M+GEOMD3(LOCNUM))/GEOMD2(LOCNUM))**2
         IF ((DZ.GT.4.0D0) .OR. (DZ.LT.2.0D0)) GO TO 999
         DZ = 0.5D0 * SQRT (DZ)
         DD = (M+GEOMD3(LOCNUM)) * DZ / GEOMD2(LOCNUM)
         IF (ABS(DD).GT.1.0D0) GO TO 999
         DD = ASIN (DD)
         IF (ABS(COS(DD)).LT.DEPS) GO TO 999
         DA = L * DZ / (2.0D0 * GEOMD1(LOCNUM) * COS(DD))
         IF (ABS(DA).GT.1.0D0) GO TO 999
         DA = ASIN (DA)
         RAT = RA0 + 2.0D0 * DA
         DECT = DD
         IF (ABS(DECT).GT.TWOPI/4.0D0) GO TO 999
         IF (ABS(DA).GT.TWOPI/4.0D0) GO TO 999
         GO TO 900
C                                       Stereographic
 90   CONTINUE
         DZ = (4.0D0 - SINS) / (4.0D0 + SINS)
         IF (ABS(DZ).GT.1.0D0) GO TO 999
         DECT = DZ * SIN0 + M * COS0 * (1.0D0+DZ) / 2.0D0
         IF (ABS(DECT).GT.1.0D0) GO TO 999
         DECT = ASIN (DECT)
         RAT = COS(DECT)
         IF (ABS(RAT).LT.DEPS) GO TO 999
         RAT = L * (1.0D0+DZ) / (2.0D0 * RAT)
         IF (ABS(RAT).GT.1.0D0) GO TO 999
         RAT = ASIN (RAT)
         MG = 1.0D0 + SIN(DECT) * SIN0 +
     *      COS(DECT) * COS0 * COS(RAT)
         IF (ABS(MG).LT.DEPS) GO TO 999
         MG = 2.0D0 * (SIN(DECT) * COS0 - COS(DECT) * SIN0
     *      * COS(RAT)) / MG
         IF (ABS(MG-M).GT.DEPS) RAT = TWOPI/2.0D0 - RAT
         RAT = RA0 + RAT
         GO TO 900
C                                       Return: in range RA
 900  RAOUT = RAT
      DECOUT = DECT
      IERR = 0
      IF (RAOUT-RA0.GT.TWOPI/2.0D0) RAOUT = RAOUT - TWOPI
      IF (RAOUT-RA0.LT.-TWOPI/2.0D0) RAOUT = RAOUT + TWOPI
C
 999  RETURN
C-----------------------------------------------------------------------
 1000 FORMAT ('NEWPOS: ILLEGAL PROJECTION TYPE',I4,' INPUT')
 1005 FORMAT ('NEWPOS: ANGLE IS TOO LARGE. L,M=',2E11.3)
      END
