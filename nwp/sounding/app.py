import bz2
import concurrent
import datetime
import json
import logging
import os
import os.path
import re
from concurrent.futures._base import ALL_COMPLETED
from pathlib import Path
import numpy as np

import pytz
import requests
import requests.sessions
import xarray as xr
from flask import Flask, request, make_response
from flask_json import FlaskJSON
from opencensus.ext.stackdriver import trace_exporter as stackdriver_exporter
import opencensus.trace.tracer
from opencensus.trace import config_integration

import nwp.sounding.config as config

app = Flask(__name__)
FlaskJSON(app)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# Metrics
exporter = stackdriver_exporter.StackdriverExporter()
tracer = opencensus.trace.tracer.Tracer(
    exporter=exporter,
    sampler=opencensus.trace.tracer.samplers.AlwaysOnSampler()
)
config_integration.trace_integrations(['requests'])


def download_bz2(url, target_file, session=requests.sessions.Session()):
    r = session.get(url)
    r.raise_for_status()

    decompressor = bz2.BZ2Decompressor()

    with open(target_file, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(decompressor.decompress(chunk))


def download_content():
    Path("tmp").mkdir(parents=True, exist_ok=True)
    download_bz2(config.content_log_url, "tmp/content.log")


def latest_run(model, valid_time):
    download_content()

    pattern = re.compile("./%s" \
                         "/grib" \
                         "\\/(\d{2})" \
                         "/t" \
                         "/icon-eu_europe_regular-lat-lon_model-level_(\d{10})_(\d{3})_1_T.grib2.bz2" % model)

    max_t: int = 0
    result = None

    for i, line in enumerate(open('tmp/content.log')):
        for match in re.finditer(pattern, line):
            matches = match.groups()

            match_valid_at = datetime.datetime.strptime(matches[1], "%Y%m%d%H")
            match_valid_at = pytz.timezone('UTC').localize(match_valid_at)
            match_valid_at = match_valid_at + datetime.timedelta(hours=int(matches[2]))

            delta_t = abs((match_valid_at - valid_time).total_seconds())

            if delta_t <= 30 * 60 and int(matches[1]) > max_t:
                result = matches
                max_t = int(matches[1])

    return result


def download_file(path, session=requests.sessions.Session()):
    """
    Load grib files for a single level from the local disk or from the OpenData server
    """
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        download_url = config.dwd_base_url + path[2:] + ".bz2"
        logging.info("Download file from: " + download_url)
        download_bz2(download_url, path, session)

    return path


def level_path(model, run_hour, run_datetime, timestep, parameter, level, level_type):
    if level_type == "model_level":
        path = f"./{model}" \
               f"/grib" \
               f"/{run_hour:02d}" \
               f"/{parameter.lower()}" \
               f"/icon-eu_europe_regular-lat-lon_model-level_{run_datetime}_{timestep:03d}_{level}_{parameter.upper()}.grib2"
    elif level_type == "time_invariant":
        path = f"./{model}" \
               f"/grib" \
               f"/{run_hour:02d}" \
               f"/{parameter.lower()}" \
               f"/icon-eu_europe_regular-lat-lon_time-invariant_{run_datetime}_{level}_{parameter.upper()}.grib2"
    else:
        raise AttributeError("Invalid level type")
    return path


class AllLevelDataResult:
    def __init__(self, data, model_time, valid_time):
        self.data = data.tolist()
        self.model_time = str(np.datetime_as_string(model_time))
        self.valid_time = str(np.datetime_as_string(valid_time))


def parameter_all_levels(model, latitude, longitude, run_hour, run_datetime, timestep,
                         parameter, level_type="model_level", base_level=60, top_level=1):
    with tracer.span(name="download"):
        levels = list(range(base_level, top_level - 1, -1))
        paths = [level_path(model, run_hour, run_datetime, timestep, parameter, level, level_type) for level in levels]

        session = requests.sessions.Session()
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.http_download_pool) as executor:
            futures = list(executor.submit(download_file(path, session)) for path in paths)
            concurrent.futures.wait(futures, timeout=None, return_when=ALL_COMPLETED)

    with tracer.span(name="parsing"):
        data_set = xr.open_mfdataset(paths, engine="cfgrib", concat_dim="generalVerticalLayer",
                                     combine='nested', parallel=config.cfgrib_parallel)
        interpolated = data_set.to_array()[0].interp(latitude=latitude, longitude=longitude)
        data = AllLevelDataResult(interpolated.values, interpolated.time.values, interpolated.valid_time.values)
        data_set.close()
    return data


@app.route("/<float:latitude>/<float(signed=True):longitude>/<int:run_hour>/<int:run_datetime>/<int:timestep>/<parameter>")
def sounding(latitude, longitude, run_hour, run_datetime, timestep, parameter):
    with tracer.span(name="sounding") as span:
        span.add_attribute("latitude", str(latitude))
        span.add_attribute("longitude", str(longitude))
        span.add_attribute("run_hour", str(run_hour))
        span.add_attribute("run_datetime", str(run_datetime))
        span.add_attribute("timestep", str(timestep))
        span.add_attribute("parameter", str(parameter))

        level_type = request.args.get("level_type", "model_level")
        base_level = int(request.args.get("base", "60"))
        top_level = int(request.args.get("top", "1"))

        sounding = parameter_all_levels(config.model, latitude, longitude,
                                        run_hour, run_datetime, int(timestep),
                                        parameter, level_type, base_level, top_level)

        response = make_response(json.dumps(sounding.__dict__))
        response.mimetype = 'application/json'
        return response


if __name__ == "__main__":
    app.run(port=5001)
