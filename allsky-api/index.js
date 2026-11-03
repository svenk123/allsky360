const http = require('http');
const fs = require('fs');
const path = require('path');
const mime = require('mime-types');
const sqlite3 = require('sqlite3');
const { execFile } = require('child_process');
const { promisify } = require('util');
const crypto = require('crypto');
const { pipeline } = require('stream');

// Simple parser for key=value config files
function parseConfig(file) {
    const config = {};
    const lines = fs.readFileSync(file, "utf8").split("\n");
    for (let line of lines) {
      line = line.trim();

      if (!line || line.startsWith("#")) 
        continue; // skip empty and comments

      // Split on the FIRST '=' only, so values may contain '=' themselves
      // (base64 padding in api-token, query strings in paths, ...).
      const sep = line.indexOf("=");
      if (sep <= 0)
        continue;

      const key = line.slice(0, sep).trim();
      const value = line.slice(sep + 1).trim();

      if (key && value) {
        config[key] = value;
      }
    }

    return config;
  }

// Default path
let configPath = "/home/allskyuser/config.ini";

// Look for --config parameter
const args = process.argv;
const configIndex = args.indexOf("--config");
if (configIndex !== -1 && args[configIndex + 1]) {
  configPath = args[configIndex + 1];
}

// Parse config
const config = parseConfig(configPath);

// Port 3000 (not root)
const PORT = process.env.PORT || config["api-port"] || 3000;

// Listen address. Default 0.0.0.0 (all interfaces) keeps the historic behaviour;
// set api-bind=127.0.0.1 in config.ini to expose the API only to a local reverse proxy.
const BIND_ADDR = process.env.BIND_ADDR || config["api-bind"] || "0.0.0.0";

// Shared secret for the privileged /api/services/{start,stop,restart} endpoints.
// Empty (the default) disables those endpoints entirely - fail closed.
const API_TOKEN = process.env.API_TOKEN || config["api-token"] || "";

// Directories
const PUBLIC_PATH = path.resolve(__dirname, 'public');
const DB_PATH = path.join(__dirname, 'database');
const IMAGE_PATH = path.join(__dirname, 'images');
const VIDEO_PATH = path.join(__dirname, 'videos');

// Allows extensions for public/
const allowedExtensions = ['.html', '.js', '.css', '.ico', '.png', '.jpg', '.jpeg', '.json', '.mp4'];

// Promisify execFile for async/await usage
const execFileAsync = promisify(execFile);

// Detect absolute path to systemctl
const SYSTEMCTL_PATH = fs.existsSync('/usr/bin/systemctl') ? '/usr/bin/systemctl' : '/bin/systemctl';

// Allowed systemd services
const ALLOWED_SERVICES = [
    'allsky360-camera.service',
    'allsky360-api.service',
    'allsky360-meteo.service',
    'allsky360-aurora.service',
    'allsky360-mount-position.service',
    'allsky360-sensors.service'
];

// Services of Type=oneshot that are started by a timer. Between two runs the
// service itself is "inactive"; whether it is switched on is the state of the
// timer. start/stop act on the timer, restart runs the service once now.
const TIMER_UNITS = {
    'allsky360-meteo.service': 'allsky360-meteo.timer',
    'allsky360-aurora.service': 'allsky360-aurora.timer',
    'allsky360-mount-position.service': 'allsky360-mount-position.timer',
    'allsky360-sensors.service': 'allsky360-sensors.timer'
};

// Services whose binary may also run outside systemd ("make camera-run"
// stops the service and runs the camera in the foreground). Matched against
// the full command line, the process name (comm) is cut to 15 characters.
const PROCESS_PATTERNS = {
    'allsky360-camera.service': '^([^ ]*/)?allsky360-camera( |$)'
};

const ALLOWED_UNITS = ALLOWED_SERVICES.concat(Object.values(TIMER_UNITS));

// GET /api/system/status: health of services, timers, data, disk, temperatures
const { createSystemStatus } = require('./system_status');
const systemStatus = createSystemStatus({
    systemctl: SYSTEMCTL_PATH,
    execFileAsync,
    services: ALLOWED_SERVICES,
    timerUnits: TIMER_UNITS,
    processPatterns: PROCESS_PATTERNS,
    publicPath: PUBLIC_PATH,
    dbPath: DB_PATH
});

// Helper function to execute systemctl commands
async function executeSystemctl(command, service, extraArgs = []) {
    try {
        if (!ALLOWED_UNITS.includes(service)) {
            throw new Error(`Service ${service} is not allowed`);
        }

        // No sudo: start/stop/restart are authorised by the polkit rule
        // etc/polkit-1/rules.d/50-allsky360.rules, is-active needs no rights.
        // sudo cannot work in the hardened unit anyway (NoNewPrivileges).
        // execFile, not exec: no shell is involved, the arguments are passed
        // as they are. --no-ask-password: fail instead of waiting for an agent.
        const { stdout, stderr } = await execFileAsync(
            SYSTEMCTL_PATH, ['--no-ask-password', ...extraArgs, command, service], { timeout: 30000 });

        return {
            success: true,
            stdout: stdout.trim(),
            stderr: stderr.trim()
        };
    } catch (error) {
        // 'is-active' exits with code 3 when the service is inactive/failed — that is expected,
        // stdout still contains the state string ("inactive", "failed", …).
        if (command === 'is-active' && error.stdout) {
            return {
                success: true,
                stdout: error.stdout.trim(),
                stderr: (error.stderr || '').trim()
            };
        }
        return {
            success: false,
            error: error.message,
            stdout: (error.stdout || '').trim(),
            stderr: (error.stderr || '').trim()
        };
    }
}

/**
 * State of a service for the web page: "output" is the is-active state of the
 * service, "timer" the state of its timer (timer-driven services only),
 * "process" whether its binary runs outside systemd (see PROCESS_PATTERNS).
 */
async function serviceStatus(service) {
    const result = await executeSystemctl('is-active', service);
    if (!result.success) return result;

    result.output = result.stdout;
    if (TIMER_UNITS[service]) {
        const timer = await executeSystemctl('is-active', TIMER_UNITS[service]);
        result.timer = timer.stdout || 'unknown';
    }
    if (PROCESS_PATTERNS[service]) {
        try {
            await execFileAsync('pgrep', ['-f', PROCESS_PATTERNS[service]], { timeout: 5000 });
            result.process = true;
        } catch {
            result.process = false;    // exit code 1: no matching process
        }
    }
    return result;
}

/**
 * systemctl verb, unit and options for a start/stop/restart request.
 * Timer-driven services: start/stop switch the timer, restart triggers one
 * run of the service now (--no-block: do not wait for the oneshot to finish).
 */
function controlCommand(action, service) {
    const timer = TIMER_UNITS[service];
    if (!timer) return { verb: action, unit: service, args: [] };
    if (action === 'restart') return { verb: 'start', unit: service, args: ['--no-block'] };
    return { verb: action, unit: timer, args: [] };
}

/**
 * Gate for the privileged service-control endpoints.
 * Requires POST plus a matching X-API-Token header; without a configured
 * api-token the endpoints stay disabled. Returns false once a response was sent.
 */
function requireServiceAuth(req, res) {
    if (!API_TOKEN) {
        res.writeHead(403, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: "Service control is disabled (no api-token configured)" }));
        return false;
    }

    if (req.method !== 'POST') {
        res.writeHead(405, { 'Content-Type': 'application/json', 'Allow': 'POST' });
        res.end(JSON.stringify({ error: "Method not allowed (use POST)" }));
        return false;
    }

    const provided = Buffer.from(String(req.headers['x-api-token'] || ''), 'utf8');
    const expected = Buffer.from(API_TOKEN, 'utf8');
    const ok = provided.length === expected.length &&
               crypto.timingSafeEqual(provided, expected);

    if (!ok) {
        res.writeHead(401, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: "Unauthorized" }));
        return false;
    }

    return true;
}

/**
 * Parse a single HTTP Range header ("bytes=start-end", "bytes=start-",
 * "bytes=-suffix") against a file size.
 * Returns {start, end} (inclusive), null if the header should be ignored
 * (absent, multiple ranges, other units) or 'unsatisfiable'.
 */
function parseRange(header, size) {
    if (!header) return null;
    const m = /^bytes=(\d*)-(\d*)$/.exec(header.trim());
    if (!m) return null;                       // multipart/other units: send the whole file
    if (m[1] === '' && m[2] === '') return null;

    let start, end;
    if (m[1] === '') {                         // suffix range: last N bytes
        const suffix = parseInt(m[2], 10);
        if (suffix === 0) return 'unsatisfiable';
        start = Math.max(0, size - suffix);
        end = size - 1;
    } else {
        start = parseInt(m[1], 10);
        end = m[2] === '' ? size - 1 : Math.min(parseInt(m[2], 10), size - 1);
    }
    if (start >= size || start > end) return 'unsatisfiable';
    return { start, end };
}

/**
 * Stream a file to the client instead of reading it into memory
 * (videos are large, review.md REL-14). Supports:
 *  - GET and HEAD
 *  - Range requests (206 Partial Content / 416), needed by browsers to seek
 *    in videos and required by Safari/iOS to play MP4 at all
 *  - ETag / Last-Modified validation (304), If-Range
 *  - aborted connections (the read stream is closed)
 * Cache-Control is 'no-cache': files such as the latest image keep their
 * name, so the browser revalidates every time - cheap thanks to the 304.
 */
function serveFile(req, res, filePath) {
    if (req.method !== 'GET' && req.method !== 'HEAD') {
        res.writeHead(405, { 'Content-Type': 'text/plain', 'Allow': 'GET, HEAD' });
        return res.end("Method not allowed.");
    }

    fs.stat(filePath, (err, stat) => {
        if (err || !stat.isFile()) {
            res.writeHead(404, { 'Content-Type': 'text/plain' });
            return res.end("Not found.");
        }

        const size = stat.size;
        const etag = `W/"${size.toString(16)}-${Math.floor(stat.mtimeMs).toString(16)}"`;
        const lastModified = stat.mtime.toUTCString();
        const headers = {
            'Content-Type': mime.lookup(filePath) || 'application/octet-stream',
            'Accept-Ranges': 'bytes',
            'ETag': etag,
            'Last-Modified': lastModified,
            'Cache-Control': 'no-cache',
            'X-Content-Type-Options': 'nosniff',
        };

        // Conditional GET: unchanged file -> 304 without body
        const inm = req.headers['if-none-match'];
        const ims = req.headers['if-modified-since'];
        if ((inm && inm.split(',').map(t => t.trim()).includes(etag)) ||
            (!inm && ims && Date.parse(ims) >= Math.floor(stat.mtimeMs / 1000) * 1000)) {
            res.writeHead(304, headers);
            return res.end();
        }

        // If-Range: only honour the Range header if the file is still the same
        let range = parseRange(req.headers['range'], size);
        const ifRange = req.headers['if-range'];
        if (range && ifRange && ifRange !== etag && ifRange !== lastModified)
            range = null;

        if (range === 'unsatisfiable') {
            headers['Content-Range'] = `bytes */${size}`;
            res.writeHead(416, headers);
            return res.end();
        }

        let status = 200;
        let streamOptions = {};
        if (range) {
            status = 206;
            headers['Content-Range'] = `bytes ${range.start}-${range.end}/${size}`;
            headers['Content-Length'] = range.end - range.start + 1;
            streamOptions = { start: range.start, end: range.end };
        } else {
            headers['Content-Length'] = size;
        }

        res.writeHead(status, headers);
        if (req.method === 'HEAD' || headers['Content-Length'] === 0)
            return res.end();

        // pipeline() closes the file when the client disconnects (seeking in
        // a video aborts the previous request all the time)
        pipeline(fs.createReadStream(filePath, streamOptions), res, (pipeErr) => {
            if (pipeErr && pipeErr.code !== 'ERR_STREAM_PREMATURE_CLOSE')
                console.error(`Error streaming ${filePath}: ${pipeErr.message}`);
        });
    });
}

const server = http.createServer((req, res) => {
    // WHATWG URL API instead of the deprecated url.parse() (DEP0169).
    // The base is only needed to parse the relative request target.
    // Query values are plain strings (last one wins for repeated keys).
    // decodeURIComponent() throws URIError on malformed percent-encoding.
    // Uncaught, that would terminate the whole server process.
    let parsed;
    let pathname;
    try {
        const requestUrl = new URL(req.url, 'http://localhost');
        parsed = { query: Object.fromEntries(requestUrl.searchParams) };
        pathname = decodeURIComponent(requestUrl.pathname);
    } catch (err) {
        res.writeHead(400, { 'Content-Type': 'text/plain' });
        return res.end("Bad request.");
    }

    // A NUL byte can truncate the path inside native fs calls.
    if (pathname.indexOf('\0') !== -1) {
        res.writeHead(400, { 'Content-Type': 'text/plain' });
        return res.end("Bad request.");
    }

    // API call: /api/image?date=YYYY-MM-DD
    if (pathname === '/api/image') {
        const date = parsed.query.date;
        if (!date || !/^\d{4}-\d{2}-\d{2}$/.test(date)) {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            return res.end(JSON.stringify({ error: "Invalid or missing date (expected YYYY-MM-DD)" }));
        }

        const yyyymmdd = date.replace(/-/g, '');
        const dbFile = `database_${yyyymmdd}.db`;
        const dbFullPath = path.join(DB_PATH, dbFile);

        if (!fs.existsSync(dbFullPath)) {
            res.writeHead(404, { 'Content-Type': 'application/json' });
            return res.end(JSON.stringify({ error: "Database not found for date " + date }));
        }

        const db = new sqlite3.Database(dbFullPath, sqlite3.OPEN_READONLY);
        const sql = `SELECT timestamp, timezone_offset, exposure, gain, brightness, mean_r, mean_g, mean_b, noise, hdr, night_mode, stars, sqm, focus FROM image ORDER BY timestamp ASC`;

        db.all(sql, [], (err, rows) => {
            if (err) {
                res.writeHead(500, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: err.message }));
            } else {
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify(rows));
            }
            db.close();
        });

    // API call: /api/meteo?date=YYYY-MM-DD
    } else if (pathname === '/api/meteo') {
        const date = parsed.query.date;
        if (!date || !/^\d{4}-\d{2}-\d{2}$/.test(date)) {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            return res.end(JSON.stringify({ error: "Invalid or missing date (expected YYYY-MM-DD)" }));
        }

        const yyyymmdd = date.replace(/-/g, '');
        const dbFile = `database_${yyyymmdd}.db`;
        const dbFullPath = path.join(DB_PATH, dbFile);

        if (!fs.existsSync(dbFullPath)) {
            res.writeHead(404, { 'Content-Type': 'application/json' });
            return res.end(JSON.stringify({ error: "Database not found for date " + date }));
        }

        const db = new sqlite3.Database(dbFullPath, sqlite3.OPEN_READONLY);
        const sql = `SELECT timestamp, temperature, humidity, dew_point, pressure_msl, surface_pressure, cloud_cover, cloud_low, cloud_mid, cloud_high, visibility, wind_speed_10m, wind_dir_10m, wind_speed_300hPa FROM meteo ORDER BY timestamp ASC`;

        db.all(sql, [], (err, rows) => {
            if (err) {
                res.writeHead(500, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: err.message }));
            } else {
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify(rows));
            }
            db.close();
        });

    // API call: /api/sensors?date=YYYY-MM-DD (RS485/Modbus sensors, allsky360-sensors)
    } else if (pathname === '/api/sensors') {
        const date = parsed.query.date;
        if (!date || !/^\d{4}-\d{2}-\d{2}$/.test(date)) {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            return res.end(JSON.stringify({ error: "Invalid or missing date (expected YYYY-MM-DD)" }));
        }

        const yyyymmdd = date.replace(/-/g, '');
        const dbFile = `database_${yyyymmdd}.db`;
        const dbFullPath = path.join(DB_PATH, dbFile);

        if (!fs.existsSync(dbFullPath)) {
            res.writeHead(404, { 'Content-Type': 'application/json' });
            return res.end(JSON.stringify({ error: "Database not found for date " + date }));
        }

        const db = new sqlite3.Database(dbFullPath, sqlite3.OPEN_READONLY);
        const sql = `SELECT timestamp, temperature, humidity, dew_point, pressure, ws_wind_speed, ws_wind_direction, noise, pm25, pm10, illuminance, rainfall, wind_speed, wind_direction, cabinet_temperature, cabinet_humidity, cabinet_dew_point FROM sensors ORDER BY timestamp ASC`;

        db.all(sql, [], (err, rows) => {
            if (err) {
                // Days before the sensor tool was installed have no table
                const missing = /no such table/.test(err.message);
                res.writeHead(missing ? 404 : 500, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: missing ? "No sensor data for date " + date : err.message }));
            } else {
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify(rows));
            }
            db.close();
        });

    // API call: /api/aurora?date=YYYY-MM-DD
    } else if (pathname === '/api/aurora') {
        const date = parsed.query.date;
        if (!date || !/^\d{4}-\d{2}-\d{2}$/.test(date)) {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            return res.end(JSON.stringify({ error: "Invalid or missing date (expected YYYY-MM-DD)" }));
        }

        const yyyymmdd = date.replace(/-/g, '');
        const dbFile = `database_${yyyymmdd}.db`;
        const dbFullPath = path.join(DB_PATH, dbFile);

        if (!fs.existsSync(dbFullPath)) {
            res.writeHead(404, { 'Content-Type': 'application/json' });
            return res.end(JSON.stringify({ error: "Database not found for date " + date }));
        }

        const db = new sqlite3.Database(dbFullPath, sqlite3.OPEN_READONLY);
        const sql = `SELECT timestamp, probability_percent, probability_max, probability_avg, kp_index, bt, bz, density, speed, temperature FROM aurora ORDER BY timestamp ASC`;

        db.all(sql, [], (err, rows) => {
            if (err) {
                res.writeHead(500, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: err.message }));
            } else {
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify(rows));
            }
            db.close();
        });

    // API call: /api/thumbnails?date=YYYY-MM-DD&type=type
    } else if (pathname === '/api/thumbnails') {
	const date = parsed.query.date;
	const type = parsed.query.type;

	if (!date || !/^\d{4}-\d{2}-\d{2}$/.test(date)) {
    	    res.writeHead(400, { 'Content-Type': 'application/json' });
    	    return res.end(JSON.stringify({ error: "Invalid or missing date (expected YYYY-MM-DD)" }));
	}
	if (!['image', 'video'].includes(type)) {
    	    res.writeHead(400, { 'Content-Type': 'application/json' });
    	    return res.end(JSON.stringify({ error: "Invalid type (expected image or video)" }));
	}

	const yyyymmdd = date.replace(/-/g, '');

	if (type === 'video') {
	    const videoDir = path.join(PUBLIC_PATH, 'videos');
	    fs.readdir(videoDir, (err, files) => {
    		if (err) {
        	    res.writeHead(404, { 'Content-Type': 'application/json' });
        	    return res.end(JSON.stringify({ error: "Video directory not found" }));
    		}
    		const result = files
        	    .filter(name => name.endsWith(`_${yyyymmdd}.mp4`))
        	    .map(name => name.slice(0, -4))
        	    .sort();
    		res.writeHead(200, { 'Content-Type': 'application/json' });
    		res.end(JSON.stringify(result));
	    });
	} else {
	    const thumbDir = path.join(PUBLIC_PATH, 'images', yyyymmdd, 'thumbnails');
	    fs.readdir(thumbDir, (err, files) => {
    		if (err) {
        	    res.writeHead(404, { 'Content-Type': 'application/json' });
        	    return res.end(JSON.stringify({ error: "Thumbnail directory not found" }));
    		}
    		const prefix = `image-${yyyymmdd}`;
    		const result = files
        	    .filter(name => name.startsWith(prefix) && name.endsWith('.jpg'))
        	    .map(name => name.slice('image'.length + 1, -4));
    		res.writeHead(200, { 'Content-Type': 'application/json' });
    		res.end(JSON.stringify(result));
	    });
	}

    // API call: /api/config
    } else if (pathname === '/api/config') {
        const configPath = path.join(PUBLIC_PATH, 'images', 'config.json');

        fs.readFile(configPath, (err, content) => {
            if (err) {
                res.writeHead(500, { 'Content-Type': 'application/json' });
                return res.end(JSON.stringify({ error: "Failed to load config.json." }));
            }

            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(content);
        });

    // API call: /api/latest_image
    } else if (pathname === '/api/latest_image') {
        const configPath = path.join(PUBLIC_PATH, 'images', 'latest_image.json');

        fs.readFile(configPath, (err, content) => {
            if (err) {
                res.writeHead(500, { 'Content-Type': 'application/json' });
                return res.end(JSON.stringify({ error: "Failed to load latest_image.json." }));
            }

            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(content);
        });

    // API call: /api/aircraft — serve /run/dump1090-mutability/aircraft.json
    } else if (pathname === '/api/aircraft') {
        const aircraftFile = '/run/dump1090-mutability/aircraft.json';
        fs.readFile(aircraftFile, (err, content) => {
            if (err) {
                res.writeHead(503, { 'Content-Type': 'application/json' });
                return res.end(JSON.stringify({ error: 'aircraft.json not available', detail: err.message }));
            }
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(content);
        });

    // API call: /api/aircraft/status — check if dump1090-mutability is running
    } else if (pathname === '/api/aircraft/status') {
        execFileAsync('pgrep', ['-a', 'dump1090']).then(({ stdout }) => {
            const lines = stdout.trim().split('\n').filter(Boolean);
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ running: lines.length > 0, processes: lines }));
        }).catch(() => {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ running: false, processes: [] }));
        });

    // API call: /api/system/status (read only, no token: like /api/services/status)
    } else if (pathname === '/api/system/status') {
        systemStatus().then(status => {
            res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
            res.end(JSON.stringify(status));
        }).catch(err => {
            res.writeHead(500, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'Failed to read the system status', detail: err.message }));
        });

    // API call: /api/services/status?service=SERVICE_NAME
    } else if (pathname === '/api/services/status') {
        const service = parsed.query.service;
        
        if (!service) {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            return res.end(JSON.stringify({ error: "Missing service parameter" }));
        }

        serviceStatus(service).then(result => {
            if (result.success) {
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({
                    service: service,
                    status: 'success',
                    output: result.output,   // e.g. "active", "inactive", "failed"
                    timer: result.timer,     // timer-driven services only
                    process: result.process  // camera: runs outside systemd
                }));
            } else {
                res.writeHead(500, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({
                    service: service,
                    status: 'error',
                    error: result.error,
                    output: result.stdout,
                    stderr: result.stderr
                }));
            }
        });

    // API call: /api/services/start?service=SERVICE_NAME
    } else if (pathname === '/api/services/start') {
        if (!requireServiceAuth(req, res)) return;

        const service = parsed.query.service;
        
        if (!service) {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            return res.end(JSON.stringify({ error: "Missing service parameter" }));
        }

        const cmd = controlCommand('start', service);
        executeSystemctl(cmd.verb, cmd.unit, cmd.args).then(result => {
            if (result.success) {
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({
                    service: service,
                    action: 'start',
                    status: 'success',
                    output: result.stdout
                }));
            } else {
                res.writeHead(500, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({
                    service: service,
                    action: 'start',
                    status: 'error',
                    error: result.error,
                    output: result.stdout,
                    stderr: result.stderr
                }));
            }
        });

    // API call: /api/services/stop?service=SERVICE_NAME
    } else if (pathname === '/api/services/stop') {
        if (!requireServiceAuth(req, res)) return;

        const service = parsed.query.service;
        
        if (!service) {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            return res.end(JSON.stringify({ error: "Missing service parameter" }));
        }

        const cmd = controlCommand('stop', service);
        executeSystemctl(cmd.verb, cmd.unit, cmd.args).then(result => {
            if (result.success) {
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({
                    service: service,
                    action: 'stop',
                    status: 'success',
                    output: result.stdout
                }));
            } else {
                res.writeHead(500, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({
                    service: service,
                    action: 'stop',
                    status: 'error',
                    error: result.error,
                    output: result.stdout,
                    stderr: result.stderr
                }));
            }
        });

    // API call: /api/services/restart?service=SERVICE_NAME
    } else if (pathname === '/api/services/restart') {
        if (!requireServiceAuth(req, res)) return;

        const service = parsed.query.service;

        if (!service) {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            return res.end(JSON.stringify({ error: "Missing service parameter" }));
        }

        const cmd = controlCommand('restart', service);
        executeSystemctl(cmd.verb, cmd.unit, cmd.args).then(result => {
            if (result.success) {
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({
                    service: service,
                    action: 'restart',
                    status: 'success',
                    output: result.stdout
                }));
            } else {
                res.writeHead(500, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({
                    service: service,
                    action: 'restart',
                    status: 'error',
                    error: result.error,
                    output: result.stdout,
                    stderr: result.stderr
                }));
            }
        });

    // Root: / -> dashboard.html
    } else if (pathname === '/' || pathname === '/dashboard.html') {
        serveFile(req, res, path.join(PUBLIC_PATH, 'dashboard.html'));

    // All other static files from public/
    } else {
        // Resolve inside the docroot and reject anything that escapes it.
        // path.join() alone does not help: decodeURIComponent() turns
        // %2e%2e%2f back into '../' even after URL normalisation.
        const filePath = path.resolve(PUBLIC_PATH, '.' + pathname);
        if (filePath !== PUBLIC_PATH && !filePath.startsWith(PUBLIC_PATH + path.sep)) {
            res.writeHead(403, { 'Content-Type': 'text/plain' });
            return res.end("Access denied.");
        }

        // Blockiere .db-Dateien sowie nicht erlaubte Endungen
        const ext = path.extname(filePath).toLowerCase();
        if (!allowedExtensions.includes(ext) || filePath.endsWith('.db')) {
            res.writeHead(403, { 'Content-Type': 'text/plain' });
            return res.end("Access denied.");
        }

        serveFile(req, res, filePath);
    }
});


server.listen(PORT, BIND_ADDR, () => {
    console.log(`Allsky360 API and Webserver running at http://${BIND_ADDR}:${PORT}`);
    if (!API_TOKEN) {
        console.log("Service control endpoints are DISABLED (set api-token in config.ini to enable them).");
    }
});
