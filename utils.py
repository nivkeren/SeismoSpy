import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.cm as cm

import obspy
from obspy import UTCDateTime
from obspy.signal.trigger import recursive_sta_lta, trigger_onset

from pyrocko import obspy_compat

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

obspy_compat.plant()

AVG_EARTH_RADIUS_KM = 6371.0088


# Module that download waveforms
def download_event_waveforms(client, t0, t1, networks="*", stations="*", locations="*", channels="*"):
    st = client.get_waveforms(network=networks, station=stations, location=locations, channel=channels, starttime=t0,
                              endtime=t1, attach_response=True)
    st = st.slice(starttime=t0, endtime=t1)
    st.write(filename=f"{t0}-{t1}.mseed", format="MSEED")
    return st


def haversine(points1, points2):
    """ Calculate the great-circle distance between two points on the Earth surface.
    :inp: two 2-tuples, containing the latitude and longitude of each point
    in decimal degrees.
    Example: haversine((45.7597, 4.8422), (48.8567, 2.3508))
    :output: Returns the distance between the two points in km.
    """
    # get earth radius in required units
    avg_earth_radius = AVG_EARTH_RADIUS_KM

    # unpack latitude/longitude
    lats1, lons1 = points1
    lats2, lons2 = points2

    # convert all latitudes/longitudes from decimal degrees to radians
    lats1, lons1, lats2, lons2 = map(np.radians, (lats1, lons1, lats2, lons2))

    # calculate haversine

    distance = np.arcsin(np.sqrt(
        np.sin((lats2 - lats1) * 0.5) ** 2
        + (np.cos(lats1) * np.cos(lats2) * np.sin((lons2 - lons1) * 0.5) ** 2)))
    return 2 * avg_earth_radius * distance


def bearing(points1, points2):
    # unpack latitude/longitude
    lats1, lons1 = points1
    lats2, lons2 = points2
    # convert all latitudes/longitudes from decimal degrees to radians
    lats1, lons1, lats2, lons2 = map(np.radians, (lats1, lons1, lats2, lons2))

    # calculate bearing
    dlons = lons2 - lons1
    y = np.sin(dlons) * np.cos(lats2)
    x = np.cos(lats1) * np.sin(lats2) - np.sin(lats1) * np.cos(lats2) * np.cos(dlons)
    return np.arctan2(y, x)


def absolute2relative(points1, points2):
    dist = haversine(points1, points2)
    azim = bearing(points1, points2)
    return dist * np.sin(azim), dist * np.cos(azim)


def relative2absolute(lat, lon, x, y):
    # φ2 = asin( sin φ1 ⋅ cos δ + cos φ1 ⋅ sin δ ⋅ cos θ )
    # λ2 = λ1 + atan2( sin θ ⋅ sin δ ⋅ cos φ1, cos δ − sin φ1 ⋅ sin φ2 )
    distance = (x**2 + y**2)**0.5
    azimuth = np.arctan2(x, y)
    lat1, lon1 = map(np.radians, (lat, lon))
    distance /= AVG_EARTH_RADIUS_KM
    lat2 = np.arcsin(np.sin(lat1) * np.cos(distance) + np.cos(lat1) * np.sin(distance) * np.cos(azimuth))
    lon2 = lon1 + np.arctan2(np.sin(azimuth) * np.sin(distance) * np.cos(lat1),
                             np.cos(distance) - np.sin(lat1) * np.sin(lat))
    # lon = (lon1 - dlon + np.pi) % (2 * np.pi) - np.pi
    lat2, lon2 = map(np.degrees, (lat2, lon2))
    return lat2, lon2


def conversite(latitudes, longitudes, grid_size=100):
    min_lat = latitudes.min()
    max_lat = latitudes.max()

    min_lon = longitudes.min()
    max_lon = longitudes.max()

    grid_max = absolute2relative((min_lat, min_lon), (max_lat, max_lon))
    rstations = absolute2relative(np.tile((min_lat, min_lon), (len(latitudes), 1)).T,
                                  (latitudes, longitudes))

    return rstations, grid_max[0] / grid_size, grid_max[1] / grid_size


def download_inv(client, events_times, networks, stations, channels):
    day_sec = 24 * 3600
    level = "response"
    inv_start = UTCDateTime(events_times.min()) - day_sec
    inv_end = UTCDateTime(events_times.max()) + day_sec

    inv = client.get_stations(starttime=inv_start, endtime=inv_end, network=networks, station=stations,
                              channel=channels, level=level)
    params = {"start": inv_start,
              "end": inv_end,
              "networks": networks,
              "stations": stations,
              "channels": channels,
              "level": level}
    return inv, params


def download_catalog(client, events_times, lat, lon, max_rad, min_mag, inc_arrivals=True, inc_picks=True):
    cat_start = UTCDateTime(events_times.min()) - 1
    cat_end = UTCDateTime(events_times.max()) + 1
    cat_events = client.get_events(starttime=cat_start, endtime=cat_end,
                                   latitude=lat, longitude=lon, maxradius=max_rad, minmagnitude=min_mag,
                                   includearrivals=inc_arrivals, includepicks=inc_picks)

    matched = []
    not_matched = []
    catalog = obspy.Catalog()
    for i, event_time in enumerate(events_times):
        for cat_event in cat_events.events[::-1]:
            origin = cat_event.preferred_origin() or cat_event.origins[-1]
            if origin is None:
                continue
            if event_time == str(origin.time):
                catalog.append(cat_event)
                matched.append(i)
                break
        else:
            not_matched.append(i)
    params = {"start": cat_start,
              "end": cat_end,
              "matched": matched,
              "not_matched": not_matched}
    return catalog, params


def download_inv_cat(client, events_times, networks, stations, channels, lat, lon, max_rad, min_mag, inc_arrivals=True,
                     inc_picks=True, write=True):
    inv, inv_params = download_inv(client, events_times, networks, stations, channels)
    catalog, cat_params = download_catalog(client, events_times, lat, lon, max_rad, min_mag, inc_arrivals, inc_picks)
    print(f"strat time: inventory-{inv_params['start']}, catalog-{cat_params['start']}")
    print(f"end time:   inventory-{inv_params['end']}, catalog-{cat_params['end']}")
    print(f"networks: {inv_params['networks']}")
    print(f"stations: {inv_params['stations']}")
    print(f"channels: {inv_params['channels']}")
    print(f"matched events: {cat_params['matched']}")
    print(f"not matched events: {cat_params['not_matched']}")

    if write:
        catalog.write("catalog.xml", format="QUAKEML")
        inv.write("inventory.xml", format="STATIONXML")
    return inv, catalog


def download_waveforms(client, events, networks, stations, channels, before=5, after=30, min_freq=1, max_freq=10):
    waveforms_raw = []
    waveforms_detrented = []
    waveforms_filtered = []
    for idx, event in events.iterrows():
        t = UTCDateTime(event["event time"])
        start_time = t - before
        end_time = t + after
        print(f"downloading mseed wavform file from : {start_time} to : {end_time}")
        st = download_event_waveforms(client=client, t0=start_time, t1=end_time, networks=networks, stations=stations,
                                      channels=channels)
        st_detrened, st_filtered = procces_waveform(st, min_freq, max_freq)
        waveforms_raw.append(st)
        waveforms_detrented.append(st_detrened)
        waveforms_filtered.append(st_filtered)
    return waveforms_raw, waveforms_detrented, waveforms_filtered


def load_waveforms(dir_path='.', min_freq=1, max_freq=10):
    waveforms_raw = []
    waveforms_detrented = []
    waveforms_filtered = []
    mseed_paths = [os.path.join(dir_path, path) for path in os.listdir(dir_path)
                   if os.path.isfile(os.path.join(dir_path, path)) and path.lower().endswith('.mseed')]
    mseed_paths.sort()
    for path in mseed_paths:
        print(f"loading mseed file: {path}")
        st = obspy.read(path, format="MSEED")
        st_detrened, st_filtered = procces_waveform(st, min_freq, max_freq)
        waveforms_raw.append(st)
        waveforms_detrented.append(st_detrened)
        waveforms_filtered.append(st_filtered)
    return waveforms_raw, waveforms_detrented, waveforms_filtered


def procces_waveform(st, min_freq=1, max_freq=10):
    st_detrened = st.copy().detrend("linear")
    st_filtered = st_detrened.copy().filter("bandpass", freqmin=min_freq, freqmax=max_freq)
    return st_detrened, st_filtered


def distance_from_event(event, points, elevations, local_depths):
    origin = event.preferred_origin() or event.origins[-1]
    hypo_dist = haversine(points, np.tile((origin.latitude, origin.longitude), (points.shape[1], 1)).T)
    depth = (elevations - local_depths + origin.depth) / 1000
    epi_dist = np.sqrt((hypo_dist ** 2 + depth ** 2))
    return epi_dist, hypo_dist


def catalog_pick(event, network, station):
    cat_tp = None
    cat_ts = None

    for pick in event.picks:
        if (pick.waveform_id.network_code, pick.waveform_id.station_code) == (network, station):
            t = pick.time
            if pick.phase_hint in ('P', 'p'):
                cat_tp = t
            elif pick.phase_hint in ('S', 's'):
                cat_ts = t
    return cat_tp, cat_ts


def sta_lta_pick(tr, sta, lta, thres1, thres2, max_len=1):
    tp_sta = None

    sr = tr.stats.sampling_rate
    cft = recursive_sta_lta(tr.data, int(sta * sr), int(lta * sr))
    p = trigger_onset(cft, thres1, thres2, max_len=max_len)
    if len(p) > 0:
        tp_sta = UTCDateTime((tr.stats.starttime + tr.times()[p[0][0]]).timestamp)
    return tp_sta


def hand_pick(st, inv, freqmin=1, freqmax=10):
    tp = None
    ts = None

    st_filtered = st.copy().detrend("linear").filter("bandpass", freqmin=freqmin, freqmax=freqmax)

    for tr_pick in st_filtered:
        tr_pick.stats.location = f"{freqmin}{freqmax}"
    st_pick = st_filtered + st.copy()

    picks = st_pick.snuffle(inventory=inv)
    for mark in picks[1]:
        mark.get_phasename()
        if mark.get_phasename() == "P":
            tp = UTCDateTime(mark.get_tmin())
        if mark.get_phasename() == "S":
            ts = UTCDateTime(mark.get_tmin())
    return tp, ts


def event_df(st, inv, event):
    cols = ["lat", "lon", "elevation", "local_depth", "sampling_rate", "distance_catalog", "tp_catalog", "ts_catalog",
            "tp_sta_lta", "tp", "ts"]
    index_cols = ["station", "channel"]
    df = pd.DataFrame(columns=index_cols + cols).set_index(index_cols)
    stz = st.select(component="Z")
    for tr in stz:
        network = tr.stats.network
        station = tr.stats.station
        channel = tr.stats.channel
        sr = tr.stats.sampling_rate
        coord = inv.get_coordinates(f"{network}.{station}.{tr.stats.location}.{channel}",
                                    tr.stats.starttime)
        lat, lon, elevation, local_depth = coord.values()
        st_station = st.select(station=station, channel=channel[0: 2] + '?').sort(reverse=True)

        tp_sta = np.nan
        tp = np.nan
        ts = np.nan
        cat_tp = np.nan
        cat_ts = np.nan
        epi_dist, hypo_dist = distance_from_event(event, np.array((lat, lon)).reshape((2, -1)),
                                                  elevation, local_depth)

        # catalog picks
        cat_tp, cat_ts = catalog_pick(event, network, station)

        # p pick with sta/lta
        tp_sta = sta_lta_pick(tr, 1, 5, 3, 1)

        print(f"distance form hypocenter: {hypo_dist} km")
        print(f"time between p and s waves: {hypo_dist / 8} s")
        #         print(f"distance form epicenter: {epi_dist} km")
        #         print(f"time between p and s waves: {epi_dist / 8} s")
        print("----")
        tp, ts = hand_pick(st_station, inv)
        # cols = ["lat", "lon", "elevation", "local_depth", "sampling_rate", "distance_catalog",
        # "tp_catalog", "ts_catalog", "tp_sta_lta", "tp", "ts"]
        df.loc[(station, channel[0: 2]), cols] = lat, lon, elevation, local_depth, sr, hypo_dist, \
                                                 cat_tp, cat_ts, tp_sta, tp, ts


def load_location(dir_path, event_id, methods, speeds):
    X, Y, Z, V = 3, 2, 1, 0
    results = {}
    for method in methods:
        residuals_list = [0] * len(speeds)
        for j, vp in enumerate(speeds):
            npy_paths = [os.path.join(dir_path, path) for path in os.listdir(dir_path)
                         if os.path.isfile(os.path.join(dir_path, path))
                         and path.lower().endswith(f"{vp}.npy")
                         and path.startswith(f"{event_id}-{method}")]
            npy_paths.sort()
            npy_paths.append(npy_paths.pop(1))
            residuals_list[j] = np.stack([np.load(path) for path in npy_paths])
        residuals = np.stack(residuals_list)
        loc = np.unravel_index(residuals.argmin(), residuals.shape)
        results[method] = {"mrs": residuals[loc[V], loc[Z], :, :].copy(),
                           "mrs_min": residuals[loc],
                           "location": (loc[Y], loc[X]),
                           "depth": loc[Z] + 1,
                           "vp": speeds[loc[V]]}

    return results

def plot_map(ax, event_data):
    lon_margin, lat_margin = 0.1, 0.025
    ticks_diff = 0.05
    station_size, location_size = 55, 100
    label_offset_x, label_offset_y = 0, 0.005

    #     projection = ccrs.PlateCarree()
    #     ax = plt.subplot(projection=projection)

    ax.set_extent([event_data["lon"].min() - lon_margin,
                   event_data["lon"].max() + lon_margin,
                   event_data["lat"].min() - lat_margin,
                   event_data["lat"].max() + lat_margin],
                  crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.OCEAN.with_scale('50m'), edgecolor='k', facecolor=(0, 0.85, 1))
    ax.add_feature(cfeature.LAND.with_scale('10m'), edgecolor='k', alpha=0.0, facecolor='none')
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='k')
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), edgecolor='k')
    ax.add_feature(cfeature.RIVERS.with_scale('10m'), edgecolor=(0, 0.5, 1), alpha=0.5)
    ax.add_feature(cfeature.LAKES.with_scale('10m'), edgecolor='k', alpha=1)

    ax.grid(True)
    ax.set_xticks(
        np.arange(event_data["lon"].min().round(1) - lon_margin, event_data["lon"].max().round(1) + lon_margin,
                  ticks_diff),
        crs=ccrs.PlateCarree())
    ax.set_yticks(
        np.arange(event_data["lat"].min().round(1) - lat_margin, event_data["lat"].max().round(1) + lat_margin,
                  ticks_diff),
        crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    ax.scatter(event_data["lon"], event_data["lat"], marker='^', s=station_size, c='k', zorder=2)
    for station in event_data.iterrows():
        ax.text(station[1]["lon"] + label_offset_x, station[1]["lat"] + label_offset_y, station[0][0])
    return ax


def plot_stations_xy(ax, event_data, title="", x_lim_add=5, y_lim_add=5):
    x_min, y_min = -2, -2
    marg_x, marg_y = 0, 0
    station_size = 55
    label_offset_x, label_offset_y = 0, 0.4

    ax.set_xlim(x_min, event_data["x"].max() + x_lim_add)

    ax.set_ylim(y_min, event_data["y"].max() + y_lim_add)

    ax.scatter(event_data["x"], event_data["y"], marker='^', s=station_size, c='k', zorder=2)
    for station in event_data.iterrows():
        ax.text(station[1]["x"] + label_offset_x, station[1]["y"] + label_offset_y, station[0][0])

    ax.axis('equal')
    ax.grid(True, zorder=1)
    ax.set_xticks(np.arange(-(5 * marg_x), (np.ceil(event_data["x"].max()) // 5 + marg_x + 2) * 5, 5))
    ax.set_yticks(np.arange(-(5 * marg_y), (np.ceil(event_data["y"].max()) // 5 + marg_y + 2) * 5, 5))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)


def plot_location(ax, fig, event_data, mrs, wx, wy, delta_x, delta_y, title, percents=(1, 5, 10, 15, 50)):
    location_size = 100

    plot_stations_xy(ax, event_data, title=title)

    xp = np.linspace(0, wx, wx) * delta_x
    yp = np.linspace(0, wy, wy) * delta_y
    res = ax.pcolor(xp, yp, mrs, cmap=cm.RdBu, vmin=abs(mrs).min(), vmax=abs(mrs).max())
    cb = fig.colorbar(res, ax=ax)

    min_y, min_x = np.unravel_index(np.argmin(mrs, axis=None), mrs.shape)
    ax.scatter(min_x * delta_x, min_y * delta_y, color='yellow', marker='*', s=location_size)

    xx, yy = np.meshgrid(xp, yp)
    cs = ax.contour(xx, yy, mrs, np.percentile(mrs, percents), colors='w')

    # Recast levels to new class
    cs.levels = percents

    # # Label levels with specially formatted floats
    if plt.rcParams["text.usetex"]:
        fmt = r'%r\%%'
    else:
        fmt = '%r%%'

    ax.clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=10)

    return ax


class nf(float):
    def __repr__(self):
        s = f'{self * 100:2.1f}'
        return f'{self:2.0f}' if s[-1] == '0' else s
