C                                                          Include PSTD
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
C                                       Parameters related trigonometry
      DOUBLE PRECISION PI, TWOPI, DG2RAD, RAD2DG, AS2RAD, RAD2AS
      PARAMETER (PI = 3.14159265358979323846D0)
      PARAMETER (TWOPI = 2.0D0 * PI)
      PARAMETER (DG2RAD = PI / 180.0D0,      RAD2DG = 180.0D0 / PI)
      PARAMETER (AS2RAD = DG2RAD / 3600.0D0, RAD2AS =3600.0D0 * RAD2DG)
C                                       speed of light (meters/sec)
      DOUBLE PRECISION VELITE
      PARAMETER (VELITE = 2.997924562D8)
C                                                          End PSTD.
