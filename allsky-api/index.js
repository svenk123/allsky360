const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');
const url = require('url');
const mime = require('mime-types');
const sqlite3 = require('sqlite3');
const { exec, execFile } = require('child_process');
const { promisify } = require('util');

// Simple parser for key=value config files
function parseConfig(file) {
    const config = {};
    const lines = fs.readFileSync(file, "utf8").split("\n");
    for (let line of lines) {
      line = line.trim();

      if (!line || line.startsWith("#")) 
        continue; // skip empty and comments
      const [key, value] = line.split("=");

      if (key && value) {
        config[key.trim()] = value.trim();
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

// Directories
const PUBLIC_PATH = path.join(__dirname, 'public');
const DB_PATH = path.join(__dirname, 'database');
const IMAGE_PATH = path.join(__dirname, 'images');
const VIDEO_PATH = path.join(__dirname, 'videos');

// Allows extensions for public/
const allowedExtensions = ['.html', '.js', '.css', '.ico', '.png', '.jpg', '.jpeg', '.json', '.mp4'];

// Promisify exec for async/await usage
const execAsync = promisify(exec);

// Detect absolute path to systemctl
const SYSTEMCTL_PATH = fs.existsSync('/usr/bin/systemctl') ? '/usr/bin/systemctl' : '/bin/systemctl';

// Allowed systemd services
const ALLOWED_SERVICES = [
    'allsky360-aurora.service',
    'allsky360-meteo.service', 
    'allsky360-camera.service',
    'allsky360-location.service'
];

// Helper function to get disk usage for a path
function getDiskUsage(path, callback) {
    // -P = POSIX-Format, gut parsbar
    execFile('df', ['-P', path], (err, stdout, stderr) => {
        if (err) {
            return callback(err);
        }

        const lines = stdout.trim().split('\n');
        if (lines.length < 2) {
            return callback(new Error('Unexpected df output'));
        }

        // Beispiel: Filesystem 1024-blocks  Used Available Capacity Mounted on
        //           /dev/sda1   100000000  ... ...
        const parts = lines[1].trim().split(/\s+/);
        if (parts.length < 6) {
            return callback(new Error('Could not parse df line'));
        }

        const fsName     = parts[0];
        const blocks1024 = parseInt(parts[1], 10);
        const used1024   = parseInt(parts[2], 10);
        const avail1024  = parseInt(parts[3], 10);
        const capacity   = parts[4]; // z.B. "85%"
        const mountpoint = parts[5];

        const totalBytes = blocks1024 * 1024;
        const usedBytes  = used1024   * 1024;
        const freeBytes  = avail1024  * 1024;
        const freePercent = totalBytes > 0
            ? (freeBytes / totalBytes) * 100
            : 0;

        callback(null, {
            path,
            filesystem: fsName,
            mountpoint,
            totalBytes,
            usedBytes,
            freeBytes,
            usedPercent: 100 - freePercent,
            freePercent,
            capacityString: capacity
        });
    });
}

// Helper function to execute systemctl commands
async function executeSystemctl(command, service) {
    try {
        if (!ALLOWED_SERVICES.includes(service)) {
            throw new Error(`Service ${service} is not allowed`);
        }
        
        // -n ensures sudo is non-interactive (no password prompt). Requires proper sudoers setup.
        const extra = command === 'status' ? '--no-pager' : '';
        const cmd = `sudo -u allskyuser -n ${SYSTEMCTL_PATH} ${command} ${service} ${extra}`.trim();
        const { stdout, stderr } = await execAsync(cmd);
        
        return {
            success: true,
            stdout: stdout.trim(),
            stderr: stderr.trim()
        };
    } catch (error) {
        return {
            success: false,
            error: error.message,
            stdout: error.stdout || '',
            stderr: error.stderr || ''
        };
    }
}

// Helper function to convert YYYYMMDDhhmmss to Unix timestamp
function parseDateTimeString(dateTimeStr) {
    if (!dateTimeStr || !/^\d{14}$/.test(dateTimeStr)) {
        return null;
    }
    
    const year = parseInt(dateTimeStr.substring(0, 4), 10);
    const month = parseInt(dateTimeStr.substring(4, 6), 10) - 1; // Month is 0-indexed
    const day = parseInt(dateTimeStr.substring(6, 8), 10);
    const hour = parseInt(dateTimeStr.substring(8, 10), 10);
    const minute = parseInt(dateTimeStr.substring(10, 12), 10);
    const second = parseInt(dateTimeStr.substring(12, 14), 10);
    
    const date = new Date(Date.UTC(year, month, day, hour, minute, second));
    return Math.floor(date.getTime() / 1000); // Convert to Unix timestamp in seconds
}

// Helper function to get all database files between start and end dates
function getDatabaseFiles(startTimestamp, endTimestamp) {
    const files = [];
    const startDate = new Date(startTimestamp * 1000);
    const endDate = new Date(endTimestamp * 1000);
    
    // Get all database files in the directory
    if (!fs.existsSync(DB_PATH)) {
        return files;
    }
    
    const allFiles = fs.readdirSync(DB_PATH);
    const dbPattern = /^database_(\d{8})\.db$/;
    
    for (const file of allFiles) {
        const match = file.match(dbPattern);
        if (match) {
            const dbDateStr = match[1];
            const dbYear = parseInt(dbDateStr.substring(0, 4), 10);
            const dbMonth = parseInt(dbDateStr.substring(4, 6), 10) - 1;
            const dbDay = parseInt(dbDateStr.substring(6, 8), 10);
            const dbDate = new Date(Date.UTC(dbYear, dbMonth, dbDay));
            
            // Check if database date is within range (or overlaps)
            const dbStartOfDay = Math.floor(dbDate.getTime() / 1000);
            const dbEndOfDay = dbStartOfDay + 86400; // 24 hours in seconds
            
            // Include database if it overlaps with the time range
            if (dbEndOfDay >= startTimestamp && dbStartOfDay <= endTimestamp) {
                files.push({
                    file: file,
                    path: path.join(DB_PATH, file),
                    date: dbDate
                });
            }
        }
    }
    
    // Sort by date
    files.sort((a, b) => a.date - b.date);
    return files;
}

// Helper function to query database for timestamps with optional filters
function queryDatabaseTimestamps(dbPath, startTimestamp, endTimestamp, filters = {}) {
    return new Promise((resolve, reject) => {
        if (!fs.existsSync(dbPath)) {
            resolve([]);
            return;
        }
        
        const db = new sqlite3.Database(dbPath, sqlite3.OPEN_READONLY);
        
        // Build WHERE clause
        let whereConditions = ['timestamp >= ?', 'timestamp <= ?'];
        const params = [startTimestamp, endTimestamp];
        
        // Add optional filters
        if (filters.night_mode !== undefined) {
            whereConditions.push('night_mode = ?');
            params.push(filters.night_mode);
        }
        
        if (filters.hdr !== undefined) {
            whereConditions.push('hdr = ?');
            params.push(filters.hdr);
        }
        
        if (filters.sqm_min !== undefined) {
            whereConditions.push('sqm >= ?');
            params.push(filters.sqm_min);
        }
        
        if (filters.sqm_max !== undefined) {
            whereConditions.push('sqm <= ?');
            params.push(filters.sqm_max);
        }
        
        if (filters.stars_min !== undefined) {
            whereConditions.push('stars >= ?');
            params.push(filters.stars_min);
        }
        
        if (filters.stars_max !== undefined) {
            whereConditions.push('stars <= ?');
            params.push(filters.stars_max);
        }
        
        // Note: moon_altitude is not in the image table, it would need to be joined
        // or queried separately. For now, we'll prepare the structure for it.
        if (filters.moon_altitude_min !== undefined) {
            // This would require a JOIN or separate query - placeholder for future
            // whereConditions.push('moon_altitude >= ?');
            // params.push(filters.moon_altitude_min);
        }
        
        const sql = `SELECT timestamp, timezone_offset FROM image WHERE ${whereConditions.join(' AND ')} ORDER BY timestamp ASC`;
        
        db.all(sql, params, (err, rows) => {
            db.close();
            if (err) {
                reject(err);
            } else {
                // Return timestamps with timezone_offset
                const results = rows.map(row => ({
                    timestamp: row.timestamp,
                    timezone_offset: row.timezone_offset
                }));
                resolve(results);
            }
        });
    });
}

const server = http.createServer((req, res) => {
    const parsed = url.parse(req.url, true);
    const pathname = decodeURIComponent(parsed.pathname);

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
	const baseDir = type === 'video' ? 'videos' : 'images';
	const thumbDir = path.join(PUBLIC_PATH, baseDir, yyyymmdd, 'thumbnails');

	fs.readdir(thumbDir, (err, files) => {
    	    if (err) {
        	res.writeHead(404, { 'Content-Type': 'application/json' });
        	return res.end(JSON.stringify({ error: "Thumbnail directory not found" }));
    	    }

	    const prefix = `${type}-${yyyymmdd}`;
    	    const result = files
        	.filter(name => name.startsWith(prefix) && name.endsWith('.jpg'))
        	.map(name => name.slice(type.length + 1, -4));

    	    res.writeHead(200, { 'Content-Type': 'application/json' });
    	    res.end(JSON.stringify(result));
	});

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

    // API call: /api/services/status?service=SERVICE_NAME
    } else if (pathname === '/api/services/status') {
        const service = parsed.query.service;
        
        if (!service) {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            return res.end(JSON.stringify({ error: "Missing service parameter" }));
        }

        executeSystemctl('status', service).then(result => {
            if (result.success) {
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({
                    service: service,
                    status: 'success',
                    output: result.stdout
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
        const service = parsed.query.service;
        
        if (!service) {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            return res.end(JSON.stringify({ error: "Missing service parameter" }));
        }

        executeSystemctl('start', service).then(result => {
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
        const service = parsed.query.service;
        
        if (!service) {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            return res.end(JSON.stringify({ error: "Missing service parameter" }));
        }

        executeSystemctl('stop', service).then(result => {
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

    // API call: /api/history?start=YYYYMMDDhhmmss&end=YYYYMMDDhhmmss[&night_mode=0|1][&hdr=0|1]
    } else if (pathname === '/api/history') {
        const startStr = parsed.query.start;
        const endStr = parsed.query.end;
        
        if (!startStr || !endStr) {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            return res.end(JSON.stringify({ error: "Missing start or end parameter (expected YYYYMMDDhhmmss)" }));
        }
        
        const startTimestamp = parseDateTimeString(startStr);
        const endTimestamp = parseDateTimeString(endStr);
        
        if (startTimestamp === null || endTimestamp === null) {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            return res.end(JSON.stringify({ error: "Invalid date format (expected YYYYMMDDhhmmss)" }));
        }
        
        if (startTimestamp > endTimestamp) {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            return res.end(JSON.stringify({ error: "Start timestamp must be before end timestamp" }));
        }
        
        // Build filters object from query parameters
        const filters = {};
        if (parsed.query.night_mode !== undefined) {
            const nightMode = parseInt(parsed.query.night_mode, 10);
            if (nightMode === 0 || nightMode === 1) {
                filters.night_mode = nightMode;
            }
        }
        
        if (parsed.query.hdr !== undefined) {
            const hdr = parseInt(parsed.query.hdr, 10);
            if (hdr === 0 || hdr === 1) {
                filters.hdr = hdr;
            }
        }
        
        if (parsed.query.sqm_min !== undefined) {
            const sqmMin = parseFloat(parsed.query.sqm_min);
            if (!isNaN(sqmMin)) {
                filters.sqm_min = sqmMin;
            }
        }
        
        if (parsed.query.sqm_max !== undefined) {
            const sqmMax = parseFloat(parsed.query.sqm_max);
            if (!isNaN(sqmMax)) {
                filters.sqm_max = sqmMax;
            }
        }
        
        if (parsed.query.stars_min !== undefined) {
            const starsMin = parseInt(parsed.query.stars_min, 10);
            if (!isNaN(starsMin)) {
                filters.stars_min = starsMin;
            }
        }
        
        if (parsed.query.stars_max !== undefined) {
            const starsMax = parseInt(parsed.query.stars_max, 10);
            if (!isNaN(starsMax)) {
                filters.stars_max = starsMax;
            }
        }
        
        // Get all relevant database files
        const dbFiles = getDatabaseFiles(startTimestamp, endTimestamp);
        
        if (dbFiles.length === 0) {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            return res.end(JSON.stringify([]));
        }
        
        // Query all databases and collect timestamps
        const allPromises = dbFiles.map(dbFile => 
            queryDatabaseTimestamps(dbFile.path, startTimestamp, endTimestamp, filters)
        );
        
        Promise.all(allPromises)
            .then(results => {
                // Flatten and sort all results by timestamp
                const allResults = results.flat().sort((a, b) => a.timestamp - b.timestamp);
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify(allResults));
            })
            .catch(err => {
                res.writeHead(500, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: err.message }));
            });

    // API call: /api/storage
    } else if (pathname === '/api/storage') {
        // Get disk usage for both IMAGE_PATH and VIDEO_PATH
        const results = {};
        let completed = 0;
        const total = 2;

        function checkComplete() {
            completed++;
            if (completed === total) {
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify(results));
            }
        }

        // Get disk usage for IMAGE_PATH
        // Check if path exists first
        const imagesPath = path.join(PUBLIC_PATH, 'images');
        if (!fs.existsSync(imagesPath)) {
            results.images = { 
                path: imagesPath,
                error: 'Path does not exist' 
            };
            checkComplete();
        } else {
            getDiskUsage(imagesPath, (err, imageData) => {
                if (err) {
                    results.images = { 
                        path: IMAGE_PATH,
                        error: err.message 
                    };
                } else {
                    results.images = imageData;
                }
                checkComplete();
            });
        }

        // Get disk usage for VIDEO_PATH
        // Check if path exists first
        const videosPath = path.join(PUBLIC_PATH, 'videos');
        if (!fs.existsSync(videosPath)) {
            results.videos = { 
                path: videosPath,
                error: 'Path does not exist' 
            };
            checkComplete();
        } else {
            getDiskUsage(videosPath, (err, videoData) => {
                if (err) {
                    results.videos = { 
                        path: videosPath,
                        error: err.message 
                    };
                } else {
                    results.videos = videoData;
                }
                checkComplete();
            });
        }

    // API call: /api/comets?date=YYYY-MM-DD&lat=LAT&long=LON&alt=ALT
    } else if (pathname === '/api/comets') {
        const date = parsed.query.date;
        const lat = parsed.query.lat;
        const long = parsed.query.long;
        const alt = parsed.query.alt;

        if (!date || !lat || !long || alt === undefined) {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            return res.end(JSON.stringify({ error: "Missing required parameters: date, lat, long, alt" }));
        }

        // Proxy-Anfrage an die COBS API
        const apiUrl = `https://cobs.si/api/planner.api?date=${encodeURIComponent(date)}&lat=${lat}&long=${long}&alt=${alt}`;
        
        https.get(apiUrl, (apiRes) => {
            let data = '';

            apiRes.on('data', (chunk) => {
                data += chunk;
            });

            apiRes.on('end', () => {
                // CORS-Header setzen
                res.writeHead(200, {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                });
                res.end(data);
            });
        }).on('error', (err) => {
            res.writeHead(500, {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            });
            res.end(JSON.stringify({ error: err.message }));
        });

    // API call: /api/videos
    } else if (pathname === '/api/videos') {
        const videosPath = path.join(PUBLIC_PATH, 'videos');
        
        // Helper function to recursively find all MP4 files
        function findMp4Files(dir, fileList = []) {
            if (!fs.existsSync(dir)) {
                return fileList;
            }
            
            const files = fs.readdirSync(dir);
            
            for (const file of files) {
                const filePath = path.join(dir, file);
                const stat = fs.statSync(filePath);
                
                if (stat.isDirectory()) {
                    // Recursively search subdirectories
                    findMp4Files(filePath, fileList);
                } else if (path.extname(file).toLowerCase() === '.mp4') {
                    // Get relative path from videos directory
                    const relativePath = path.relative(videosPath, filePath);
                    fileList.push(relativePath);
                }
            }
            
            return fileList;
        }
        
        try {
            const mp4Files = findMp4Files(videosPath);
            // Sort files alphabetically
            mp4Files.sort();
            
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(mp4Files));
        } catch (err) {
            res.writeHead(500, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: err.message }));
        }

    // Serve video files from videos directory
    } else if (pathname.startsWith('/videos/')) {
        // Remove /videos/ prefix to get the relative path
        const videoRelativePath = pathname.substring(8); // Remove '/videos/'
        // Videos are stored in public/videos (same as /api/videos uses)
        const videosPath = path.join(PUBLIC_PATH, 'videos');
        const videoFilePath = path.join(videosPath, videoRelativePath);
        
        // Security: Prevent directory traversal
        const resolvedPath = path.resolve(videoFilePath);
        const resolvedVideosPath = path.resolve(videosPath);
        if (!resolvedPath.startsWith(resolvedVideosPath)) {
            res.writeHead(403, { 'Content-Type': 'text/plain' });
            return res.end("Access denied.");
        }
        
        // Check if file exists and is an allowed file type
        const ext = path.extname(videoFilePath).toLowerCase();
        const allowedVideoExtensions = ['.mp4', '.png', '.jpg', '.jpeg'];
        if (!allowedVideoExtensions.includes(ext)) {
            res.writeHead(403, { 'Content-Type': 'text/plain' });
            return res.end("Access denied.");
        }
        
        // Determine content type based on file extension
        let contentType;
        switch (ext) {
            case '.mp4':
                contentType = 'video/mp4';
                break;
            case '.png':
                contentType = 'image/png';
                break;
            case '.jpg':
            case '.jpeg':
                contentType = 'image/jpeg';
                break;
            default:
                contentType = 'application/octet-stream';
        }
        
        fs.readFile(videoFilePath, (err, content) => {
            if (err) {
                res.writeHead(404, { 'Content-Type': 'text/plain' });
                res.end("File not found.");
            } else {
                res.writeHead(200, { 
                    'Content-Type': contentType,
                    'Content-Length': content.length
                });
                res.end(content);
            }
        });

    // Root: / -> index.html
    } else if (pathname === '/' || pathname === '/dashboard.html') {
        const filePath = path.join(PUBLIC_PATH, 'dashboard.html');
        fs.readFile(filePath, (err, content) => {
            if (err) {
                res.writeHead(500, { 'Content-Type': 'text/plain' });
                res.end("Failed to load landing page.");
            } else {
                res.writeHead(200, { 'Content-Type': 'text/html' });
                res.end(content);
            }
        });

    // All other static files from public/
    } else {
        const filePath = path.join(PUBLIC_PATH, pathname);

        // Blockiere .db-Dateien sowie nicht erlaubte Endungen
        const ext = path.extname(filePath).toLowerCase();
        if (!allowedExtensions.includes(ext) || filePath.endsWith('.db')) {
            res.writeHead(403, { 'Content-Type': 'text/plain' });
            return res.end("Access denied.");
        }

        fs.readFile(filePath, (err, content) => {
            if (err) {
                res.writeHead(404, { 'Content-Type': 'text/plain' });
                res.end("Not found.");
            } else {
		const ext = path.extname(filePath).toLowerCase();
		const contentType = mime.lookup(ext) || 'application/octet-stream';
                res.writeHead(200, { 'Content-Type': contentType });
                res.end(content);
            }
        });
    }
});


server.listen(PORT, () => {
    console.log(`Allsky360 API and Webserver running at http://localhost:${PORT}`);
});
