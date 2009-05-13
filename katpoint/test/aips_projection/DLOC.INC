C                                                          Include DLOC.
C-----------------------------------------------------------------------
C;  Copyright (C) 1995
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
C                                       Position labeling common
C                                       Dimensionality limit
      INTEGER   NUMLOC
      PARAMETER (NUMLOC = 3)
      DOUBLE PRECISION RPVAL(4,NUMLOC), COND2R, AXDENU(NUMLOC),
     *   GEOMD1(NUMLOC), GEOMD2(NUMLOC), GEOMD3(NUMLOC), GEOMD4(NUMLOC)
      CHARACTER CTYP(4,NUMLOC)*20, CPREF(2,NUMLOC)*5,
     *   SAXLAB(2,NUMLOC)*20
      REAL      RPLOC(4,NUMLOC), AXINC(4,NUMLOC), ROT(NUMLOC),
     *   REPOCH(NUMLOC)
      INTEGER   ZDEPTH(5,NUMLOC), ZAXIS(NUMLOC), AXTYP(NUMLOC),
     *   CORTYP(NUMLOC), LABTYP(NUMLOC), SGNROT(NUMLOC),
     *   AXFUNC(7,NUMLOC), KLOCL(NUMLOC), KLOCM(NUMLOC), KLOCF(NUMLOC),
     *   KLOCS(NUMLOC), KLOCA(NUMLOC), KLOCB(NUMLOC), NCHLAB(2,NUMLOC),
     *   LOCNUM
      COMMON /LOCATC/ CTYP, CPREF, SAXLAB
      COMMON /LOCATI/ RPVAL, COND2R, AXDENU, GEOMD1, GEOMD2, GEOMD3,
     *   GEOMD4, RPLOC, AXINC, ROT, ZDEPTH,
     *   ZAXIS, AXTYP, CORTYP, LABTYP, SGNROT, AXFUNC, KLOCL, KLOCM,
     *   KLOCF, KLOCS, KLOCA, KLOCB, NCHLAB, REPOCH, LOCNUM
C                                                          End DLOC.
