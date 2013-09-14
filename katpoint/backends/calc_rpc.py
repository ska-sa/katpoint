import rpc

# A useful way to debug the RPC request and reply is to run rpc.test()
# to obtain the CalcServer port (associated with program 536871744) 
# and then sniff the traffic with "sudo tcpdump -X -ilo0 port <port>"

class CalcPacker(rpc.Packer):

    def pack_int(self, x):
        """Convert int32 to uint32 since xdrlib only supports uints (see Python issue #9696, recently fixed)."""
        self.pack_uint((4294967296 + (x)) % 4294967296)

    def pack_calc_args(self, kwargs):
        self.pack_int(kwargs['request_id'])
        self.pack_int(kwargs['date'])
        self.pack_double(kwargs['time'])
        self.pack_int(kwargs['ref_frame'])
        for n in range(64):
            self.pack_int(kwargs['kflags'][n])
        self.pack_double(kwargs['a_x'])
        self.pack_double(kwargs['a_y'])
        self.pack_double(kwargs['a_z'])
        self.pack_double(kwargs['axis_off_a'])
        self.pack_double(kwargs['b_x'])
        self.pack_double(kwargs['b_y'])
        self.pack_double(kwargs['b_z'])
        self.pack_double(kwargs['axis_off_b'])
        self.pack_double(kwargs['ra'])
        self.pack_double(kwargs['dec'])
        self.pack_double(kwargs['dra'])
        self.pack_double(kwargs['ddec'])
        self.pack_double(kwargs['depoch'])
        self.pack_double(kwargs['parallax'])
        self.pack_double(kwargs['pressure_a'])
        self.pack_double(kwargs['pressure_b'])

        self.pack_string(kwargs['station_a'])
        self.pack_string(kwargs['axis_type_a'])
        self.pack_string(kwargs['station_b'])
        self.pack_string(kwargs['axis_type_b'])
        self.pack_string(kwargs['source'])

        for n in range(5):
            self.pack_double(kwargs['EOP_time'][n])
            self.pack_double(kwargs['tai_utc'][n])
            self.pack_double(kwargs['ut1_utc'][n])
            self.pack_double(kwargs['xpole'][n])
            self.pack_double(kwargs['ypole'][n])

class CalcUnpacker(rpc.Unpacker):

    def unpack_calc_reply(self):
        error = self.unpack_int()
        request_id = self.unpack_int()
        date = self.unpack_int()
        time = self.unpack_double()
        uvw = [0.0, 0.0, 0.0]
        riseset = [0.0, 0.0]
        delay = self.unpack_double()
        uvw[0] = self.unpack_double()
        riseset[0] = self.unpack_double()
        rate = self.unpack_double()
        uvw[1] = self.unpack_double()
        riseset[1] = self.unpack_double()
        uvw[2] = self.unpack_double()
        dry_atmos = [0.0, 0.0, 0.0, 0.0]
        wet_atmos = [0.0, 0.0, 0.0, 0.0]
        az = [0.0, 0.0, 0.0, 0.0]
        el = [0.0, 0.0, 0.0, 0.0]
        for n in range(4):
            dry_atmos[n] = self.unpack_double()
            wet_atmos[n] = self.unpack_double()
            el[n] = self.unpack_double()
            az[n] = self.unpack_double()
        return request_id, date, time, delay, rate, uvw, riseset, dry_atmos, wet_atmos, az, el

class CalcClient(rpc.TCPClient):

    def __init__(self, host=None):
        host = '' if host is None else host
        rpc.TCPClient.__init__(self, host, 0x20000340, 1)

    def addpackers(self):
        self.packer = CalcPacker()
        self.unpacker = CalcUnpacker('')

    def get_calc(self, **kwargs):
        return self.make_call(1, kwargs, \
                self.packer.pack_calc_args, \
                self.unpacker.unpack_calc_reply)

import ephem
import numpy as np

def dut1(mjd, leap_secs):
    obs = ephem.Observer()
    obs.date = mjd - (2415020 - 2400000.5)
    return 32.184 + leap_secs - ephem.delta_t(obs)

def do_calc(target, ant2, timestamp, ant1):
    """Used to compare original katpoint with CALC."""
    mjd = timestamp.to_mjd()
    xyz1, xyz2 = ant1.position_ecef, ant2.position_ecef
    radec = target.radec(timestamp, ant1)
    mjd_range = np.arange(int(mjd) - 1, int(mjd) + 4, dtype=np.float).tolist()
    leap_secs = 35.0 # Correct after July 2012
    args = {
        'date' : int(mjd),
        'time' : mjd - int(mjd),
        'request_id' : 100,
        'ref_frame' : 0,
        'kflags' : 64 * [-1],
        'station_a' : ant1.name,
        'a_x' : xyz1[0],
        'a_y' : xyz1[1],
        'a_z' : xyz1[2],
        'axis_type_a' : 'altz',
        'axis_off_a' : 0.00,
        'station_b' : ant2.name,
        'b_x' : xyz2[0],
        'b_y' : xyz2[1],
        'b_z' : xyz2[2],
        'axis_type_b' : 'altz',
        'axis_off_b' : 0.00,
        'source' : target.name,
        'ra' : radec[0],
        'dec' : radec[1],
        'dra' : 0.0,
        'ddec' : 0.0,
        'depoch' : 0.0,
        'parallax' : 0.0,
        'pressure_a' : 0.0,
        'pressure_b' : 0.0,
        'EOP_time' : mjd_range,
        'tai_utc' : 5 * [leap_secs],
        'ut1_utc' : [dut1(d, leap_secs) for d in mjd_range],
        'xpole' : 5 * [0.],
        'ypole' : 5 * [0.],
    }
    c = CalcClient()
    res = c.get_calc(**args)
    return res[3], res[4], res[5], res[9][0], res[10][0]

two_pi = 6.2831853071795864769

args = {
    'date' : 50774,
    'time' : 22.0/24.0 + 2.0/(24.0*60.0),
    'request_id' : 150,
    'ref_frame' : 0,
    'kflags' : 64 * [-1],
    'station_a' : 'EC',
    'a_x' : 0.000,
    'a_y' : 0.000,
    'a_z' : 0.000,
    'axis_type_a' : 'altz',
    'axis_off_a' : 0.00,
    'station_b' : 'KP',
    'b_x' : -1995678.4969,
    'b_y' : -5037317.8209,
    'b_z' : 3357328.0825,
    'axis_type_b' : 'altz',
    'axis_off_b' : 2.1377,
    'source' : 'B1937+21',
    'ra' : (two_pi/24.0)*(19.0 + 39.0/60.0 + 38.560210/3600.0),
    'dec' : (two_pi/360.)*(21.0 + 34.0/60.0 + 59.141000/3600.0),
    'dra' : 0.0,
    'ddec' : 0.0,
    'depoch' : 0.0,
    'parallax' : 0.0,
    'pressure_a' : 0.0,
    'pressure_b' : 0.0,
    'EOP_time' : [50773.0, 50774.0, 50775.0, 50776.0, 50777.0],
    'tai_utc' : [31.0, 31.0, 31.0, 31.0, 31.0],
    'ut1_utc' : [0.285033, 0.283381, 0.281678, 0.280121, 0.278435],
    'xpole' : [0.19744, 0.19565, 0.19400, 0.19244, 0.19016],
    'ypole' : [0.24531, 0.24256, 0.24000, 0.23700, 0.23414],
}

c = CalcClient()
print c.get_calc(**args)

import katpoint

target = katpoint.Target('1934-638, radec, 19:39:25.03, -63:42:45.7,  (200.0 12000.0 -11.11 7.777 -1.231)')
ant1 = katpoint.Antenna('ant7, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -87.9881 75.7543 0.138305, , 1.22')
ant2 = katpoint.Antenna('ant2, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 90.2844 26.3804 -0.22636, , 1.22')
timestamp = katpoint.Timestamp('2013-09-13 15:49')

delay, rate, uvw, az, el = do_calc(target, ant2, timestamp, ant1)
calc_delay_rate = np.array([delay, rate])
calc_uvw = -np.array(uvw)
calc_azel = np.array([az, el])
katpoint_delay_rate = np.array(target.geometric_delay(ant2, timestamp, ant1))
katpoint_uvw = np.array(target.uvw(ant2, timestamp, ant1))
katpoint_azel = np.array(target.azel(timestamp, ant1))

print "CALC delay + rate:", calc_delay_rate
print "katpoint delay + rate:", katpoint_delay_rate
print "diff (s, s/s):", katpoint_delay_rate - calc_delay_rate

print "CALC uvw:", calc_uvw
print "katpoint uvw:", katpoint_uvw
print "diff (m):", katpoint_uvw - calc_uvw
print "angle (uvw arcsec):", np.arccos(np.dot(katpoint_uvw, calc_uvw) /
                                       np.sqrt(np.dot(katpoint_uvw, katpoint_uvw) *
                                               np.dot(calc_uvw, calc_uvw))) * 180 / np.pi * 3600
print "angle (uv arcsec):", np.arccos(np.dot(katpoint_uvw[:2], calc_uvw[:2]) /
                                      np.sqrt(np.dot(katpoint_uvw[:2], katpoint_uvw[:2]) *
                                              np.dot(calc_uvw[:2], calc_uvw[:2]))) * 180 / np.pi * 3600

print "CALC azel:", calc_azel
print "katpoint azel:", katpoint_azel
cazel = katpoint.construct_azel_target(*calc_azel)
kazel = katpoint.construct_azel_target(*katpoint_azel)
print "separation (sky arcsec)", cazel.separation(kazel, timestamp, ant1) * 180 / np.pi * 3600
